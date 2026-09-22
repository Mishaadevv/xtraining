"""Inference runtime for trained models.

Runs as a long-lived subprocess. The shell writes JSON requests on stdin (one
per line) and reads events on stdout, so a model is loaded once and reused
across prompts instead of being reloaded per message.

Requests:

    {"type": "load", "model_dir": "..."}
    {"type": "generate", "prompt": "...", "max_new_tokens": 256,
     "temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.1, "stream": true}
    {"type": "unload"}
    {"type": "ping"}
    {"type": "quit"}
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

from . import events
from .errors import humanize


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

THINKING_INSTRUCTION = (
    "Before answering, reason step by step inside <think> and </think> tags. "
    "Keep the reasoning brief and concrete, then give the final answer after "
    "the closing tag. Only the final answer goes outside the tags."
)


class ThinkingSplitter:
    """Splits a token stream into reasoning (<think>…</think>) and the answer.

    Tokens are fed in as they arrive; the splitter yields ``(kind, text)``
    pairs where ``kind`` is ``"thinking"`` or ``"answer"``. A tag split across
    two chunks is handled by holding back a short tail until it can be
    classified, so the UI never shows half a tag.
    """

    def __init__(self) -> None:
        self.in_think = False
        self.tail = ""

    def feed(self, chunk: str):
        buffer = self.tail + chunk
        self.tail = ""
        while buffer:
            if self.in_think:
                end = buffer.find(THINK_CLOSE)
                if end == -1:
                    # Hold back a possible partial closing tag.
                    keep = self._partial_suffix(buffer, THINK_CLOSE)
                    emit, self.tail = buffer[: len(buffer) - keep], buffer[len(buffer) - keep:]
                    if emit:
                        yield "thinking", emit
                    return
                emit, buffer = buffer[:end], buffer[end + len(THINK_CLOSE):]
                if emit:
                    yield "thinking", emit
                self.in_think = False
            else:
                start = buffer.find(THINK_OPEN)
                if start == -1:
                    keep = self._partial_suffix(buffer, THINK_OPEN)
                    emit, self.tail = buffer[: len(buffer) - keep], buffer[len(buffer) - keep:]
                    if emit:
                        yield "answer", emit
                    return
                emit, buffer = buffer[:start], buffer[start + len(THINK_OPEN):]
                if emit:
                    yield "answer", emit
                self.in_think = True

    @staticmethod
    def _partial_suffix(text: str, tag: str) -> int:
        """Length of the longest suffix of ``text`` that is a prefix of ``tag``."""
        for size in range(min(len(text), len(tag) - 1), 0, -1):
            if text.endswith(tag[:size]):
                return size
        return 0


class InferenceRuntime:
    """Holds one loaded model. Everything is real state — no placeholders."""

    def __init__(self) -> None:
        self.model_dir: str | None = None
        self.mode: str | None = None
        self.metadata: dict[str, Any] = {}
        self.model = None
        self.tokenizer = None
        self.device = "cpu"

    # ------------------------------------------------------------------ load
    def load(self, model_dir: str) -> dict[str, Any]:
        self.unload()
        path = Path(model_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        metadata_path = path / "metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                metadata = {}

        base_model = metadata.get("base_model") or str(path)
        needs_base = bool(metadata.get("adapter", False))

        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        events.stage("inference", f"Loading model into {self.device.upper()}")

        tokenizer = None
        tokenizer_error: Exception | None = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(path) if not needs_base else base_model,
                trust_remote_code=False,
            )
        except Exception as exc:  # noqa: BLE001 - a from-scratch run brings its own
            tokenizer_error = exc

        if tokenizer is None and not needs_base:
            # A model trained from scratch carries the character-level tokenizer
            # learned from the dataset, and it is used like any other tokenizer —
            # otherwise such a model could be trained but never talked to.
            from .backends.scratch import CharTokenizer  # noqa: PLC0415

            if CharTokenizer.has_vocab(path):
                tokenizer = CharTokenizer.load_from_dir(path)
                events.log(
                    "Using the character-level tokenizer saved with this model "
                    f"({tokenizer.vocab_size:,} symbols)."
                )

        if tokenizer is None:
            raise RuntimeError(
                f"No tokenizer could be loaded for '{path.name}'. "
                "A from-scratch run keeps its own tokenizer next to the weights; an adapter "
                "needs its base model's tokenizer to be reachable."
            ) from tokenizer_error
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = getattr(tokenizer, "eos_token", None)

        dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) \
            else (torch.float16 if self.device == "cuda" else torch.float32)

        if needs_base:
            # Trained model = frozen base weights + a saved LoRA adapter.
            from peft import PeftModel  # noqa: PLC0415

            base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
            model = PeftModel.from_pretrained(base, str(path))
            self.mode = "adapter"
        else:
            model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=dtype)
            self.mode = "full"

        model.eval()
        if self.device == "cuda":
            model = model.to("cuda")

        self.model = model
        self.tokenizer = tokenizer
        self.model_dir = str(path)
        self.metadata = metadata

        total = sum(p.numel() for p in model.parameters())
        return {
            "model_dir": self.model_dir,
            "mode": self.mode,
            "base_model": base_model,
            "method": metadata.get("train_mode"),
            "params": total,
            "device": self.device,
            "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
        }

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_dir = None
        self.mode = None
        self.metadata = {}
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # -------------------------------------------------------------- generate
    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model is loaded. Send a 'load' request first.")

        import torch  # noqa: PLC0415
        from transformers import TextIteratorStreamer  # noqa: PLC0415

        prompt = str(request.get("prompt") or "")
        system = str(request.get("system") or "").strip()
        messages = request.get("messages")
        thinking = bool(request.get("thinking", False))
        max_new_tokens = int(request.get("max_new_tokens") or 256)
        temperature = float(request.get("temperature") or 0.7)
        top_p = float(request.get("top_p") or 0.9)
        repetition_penalty = float(request.get("repetition_penalty") or 1.1)
        stream = bool(request.get("stream", True))

        tokenizer = self.tokenizer
        if thinking:
            if messages:
                messages = [*messages]
                if messages and messages[0].get("role") == "system":
                    messages[0] = {
                        **messages[0],
                        "content": f"{messages[0].get('content', '')}\n\n{THINKING_INSTRUCTION}",
                    }
                else:
                    messages.insert(0, {"role": "system", "content": THINKING_INSTRUCTION})
                text = self._render_messages(messages)
            else:
                base_system = f"{system}\n\n{THINKING_INSTRUCTION}" if system else THINKING_INSTRUCTION
                text = self._render_messages([
                    {"role": "system", "content": base_system},
                    {"role": "user", "content": prompt},
                ])
        elif messages:
            text = self._render_messages(messages)
        elif system:
            text = self._render_messages([
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ])
        else:
            text = prompt

        encoded = tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-5),
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "streamer": streamer,
        }

        started = time.time()
        error_box: list[BaseException] = []

        def run_generation() -> None:
            try:
                with torch.no_grad():
                    self.model.generate(**generation_kwargs)
            except BaseException as exc:  # noqa: BLE001 - forwarded to the caller
                error_box.append(exc)

        thread = threading.Thread(target=run_generation, daemon=True)
        thread.start()

        pieces: list[str] = []
        answer_pieces: list[str] = []
        thinking_pieces: list[str] = []
        splitter = ThinkingSplitter() if thinking else None
        for chunk in streamer:
            if error_box:
                break
            pieces.append(chunk)
            if not stream:
                continue
            if splitter is None:
                events.emit("inference-token", {"token": chunk})
                continue
            for kind, piece in splitter.feed(chunk):
                if kind == "thinking":
                    thinking_pieces.append(piece)
                    events.emit("inference-thinking", {"token": piece})
                else:
                    answer_pieces.append(piece)
                    events.emit("inference-token", {"token": piece})
        thread.join(timeout=1)

        if error_box:
            raise error_box[0]

        if splitter is None:
            reasoning = ""
            text_out = "".join(pieces).strip()
        else:
            for kind, piece in splitter.feed(""):
                if kind == "thinking":
                    thinking_pieces.append(piece)
                else:
                    answer_pieces.append(piece)
            reasoning = "".join(thinking_pieces).strip()
            text_out = "".join(answer_pieces).strip()
            raw = "".join(pieces)
            # If the model never opened the tag, everything it wrote *is* the
            # reasoning and there is no separate answer.
            if THINK_OPEN not in raw and reasoning:
                reasoning = raw.strip()
                text_out = ""
        elapsed = max(0.001, time.time() - started)
        output_tokens = 0
        try:
            output_tokens = len(tokenizer(text_out)["input_ids"]) if text_out else 0
        except Exception:
            output_tokens = 0

        thinking_tokens = 0
        if reasoning:
            try:
                thinking_tokens = len(tokenizer(reasoning)["input_ids"])
            except Exception:
                thinking_tokens = max(1, len(reasoning) // 4)

        return {
            "text": text_out,
            "thinking": reasoning,
            "thinking_tokens": thinking_tokens,
            "tokens": output_tokens,
            "seconds": round(elapsed, 3),
            "tokens_per_second": round(output_tokens / elapsed, 2),
            "prompt_tokens": int(input_ids.shape[-1]),
            "model_dir": self.model_dir,
            "mode": self.mode,
        }

    def _render_messages(self, messages: list[dict[str, Any]]) -> str:
        tokenizer = self.tokenizer
        template = getattr(tokenizer, "chat_template", None)
        if template:
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        lines = []
        for message in messages:
            lines.append(f"{message.get('role', 'user')}: {message.get('content', '')}")
        lines.append("assistant:")
        return "\n".join(lines)


def serve() -> int:
    """Read requests on stdin, answer on stdout. Blocks until quit or EOF."""
    events.emit("ready", {"command": "infer"})
    runtime = InferenceRuntime()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            events.fail("Malformed request from the app.", code="bad_request")
            continue

        kind = request.get("type")
        request_id = request.get("id")

        def respond(event: str, detail: dict[str, Any]) -> None:
            events.emit(event, {**detail, "request_id": request_id})

        try:
            if kind == "ping":
                respond("inference-result", {"ok": True, "loaded": runtime.model is not None})
            elif kind == "load":
                info = runtime.load(str(request.get("model_dir")))
                respond("inference-result", {"op": "load", **info})
            elif kind == "generate":
                result = runtime.generate(request)
                respond("inference-result", {"op": "generate", **result})
            elif kind == "unload":
                runtime.unload()
                respond("inference-result", {"op": "unload", "ok": True})
            elif kind == "quit":
                runtime.unload()
                respond("inference-result", {"op": "quit", "ok": True})
                return 0
            else:
                respond("inference-result", {"error": f"Unknown request type '{kind}'"})
        except Exception as exc:
            info = humanize(exc, stage="inference")
            events.emit("inference-error", {
                "request_id": request_id,
                "message": info["message"],
                "hint": info["hint"],
                "code": info["code"],
                "traceback": info.get("traceback", ""),
            })

    runtime.unload()
    return 0
