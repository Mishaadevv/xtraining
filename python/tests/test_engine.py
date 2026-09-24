"""Engine test suite.

These tests do real work: they write datasets, train a real model with real
gradients, checkpoint it, resume it, generate from it and evaluate it. Nothing
here is mocked, which is the point — the suite fails if the engine only pretends
to do something.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zxtrain import datasets as ds  # noqa: E402
from zxtrain import estimator, hardware, storage, trainer  # noqa: E402
from zxtrain import models as model_tools  # noqa: E402
from zxtrain.backends import registry, tiny  # noqa: E402
from zxtrain.bpe import BPETokenizer, train_bpe  # noqa: E402
from zxtrain.cli import METHODS, main, m_engine_capabilities  # noqa: E402
from zxtrain.errors import ZxError  # noqa: E402

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox is quick and the dog is lazy",
    "a lazy dog sleeps while the quick fox runs",
    "quick foxes jump over lazy dogs in the yard",
    "the dog and the fox are friends in this short story",
    "story time: the fox runs, the dog sleeps, the yard is quiet",
] * 8


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="zxtrain-test-"))
        self.rows = [
            {"text": row} for row in CORPUS[:20]
        ] + [{"text": CORPUS[0]}, {"text": ""}, {"text": "   "}]

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, name: str, rows: list[dict]) -> Path:
        path = self.tmp / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def test_inspect_reports_real_statistics(self) -> None:
        path = self._write("data.jsonl", self.rows)
        report = ds.inspect_dataset(path)
        self.assertEqual(report["record_count"], len(self.rows))
        self.assertIn("text", report["field_names"])
        self.assertGreaterEqual(report["duplicates"], 1)
        self.assertGreaterEqual(report["empty_records"], 1)
        self.assertEqual(report["detected_mapping"].get("text"), "text")
        self.assertGreater(report["token_estimate"]["total"], 0)
        self.assertEqual(report["token_estimate"]["labelled"], "estimated")

    def test_clean_removes_duplicates_and_empty_records(self) -> None:
        path = self._write("dirty.jsonl", self.rows)
        output = self.tmp / "clean.jsonl"
        result = ds.clean_dataset(path, {"text": "text"},
                                  [{"type": "drop_empty"}, {"type": "drop_duplicates"},
                                   {"type": "trim_whitespace"}],
                                  output)
        self.assertLess(result["output_records"], result["input_records"])
        self.assertTrue(output.exists())
        audit = json.loads(Path(f"{output}.audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["output_records"], result["output_records"])

    def test_split_is_deterministic_with_the_same_seed(self) -> None:
        path = self._write("split.jsonl", self.rows * 4)
        first = ds.split_dataset(path, {"text": "text"}, self.tmp / "a", seed=7)
        second = ds.split_dataset(path, {"text": "text"}, self.tmp / "b", seed=7)
        self.assertEqual(first["counts"], second["counts"])
        self.assertEqual(sum(first["counts"].values()), len(self.rows) * 4)

    def test_csv_and_folder_shards_are_readable(self) -> None:
        csv_path = self.tmp / "table.csv"
        csv_path.write_text("prompt,response\nhello,hi there\ngood day,likewise\n", encoding="utf-8")
        mapping = ds.detect_mapping(["prompt", "response"], {"prompt": "hello", "response": "hi"})["mapping"]
        self.assertEqual(mapping.get("user"), "prompt")
        self.assertEqual(mapping.get("assistant"), "response")
        text = ds.record_to_text({"prompt": "hello", "response": "hi there"}, mapping)
        self.assertIn("hello", text)
        shard_dir = self.tmp / "shards"
        shard_dir.mkdir()
        (shard_dir / "part-1.jsonl").write_text('{"text": "one"}\n', encoding="utf-8")
        (shard_dir / "part-2.jsonl").write_text('{"text": "two"}\n', encoding="utf-8")
        records = list(ds.iter_records(shard_dir))
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["__source"], "part-1.jsonl")

    def test_unsupported_format_explains_itself(self) -> None:
        path = self.tmp / "binary.xyz"
        path.write_bytes(b"\x00\x01")
        with self.assertRaises(ZxError) as caught:
            list(ds.iter_records(path))
        self.assertEqual(caught.exception.code, "unsupported_format")


class TokenizerTests(unittest.TestCase):
    def test_bpe_roundtrip_and_specials(self) -> None:
        payload = train_bpe(CORPUS, vocab_size=330)
        tokenizer = BPETokenizer(payload)
        self.assertGreater(tokenizer.vocab_size, 256)
        text = "<|user|>\nhello quick fox\n<|assistant|>\nrun dog"
        ids = tokenizer.encode(text)
        self.assertTrue(all(isinstance(token, int) for token in ids))
        decoded = tokenizer.decode(ids)
        self.assertIn("quick fox", decoded)
        self.assertLess(len(ids), len(text))
        self.assertEqual(tokenizer.decode([tokenizer.vocab_size - 1]), "<|end|>")

    def test_save_and_load_roundtrip(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="zxtrain-tok-"))
        try:
            tokenizer = BPETokenizer(train_bpe(CORPUS, vocab_size=300))
            path = tokenizer.save(tmp / "tokenizer.json")
            reloaded = BPETokenizer.load(path)
            self.assertEqual(reloaded.encode("quick fox"), tokenizer.encode("quick fox"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TrainingTests(unittest.TestCase):
    """Trains a real model, resumes it, generates and evaluates — no mocks."""

    workspace: Path
    dataset: Path
    mapping: dict

    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = Path(tempfile.mkdtemp(prefix="zxtrain-ws-"))
        cls.dataset = cls.workspace / "datasets" / "corpus.jsonl"
        cls.dataset.parent.mkdir(parents=True, exist_ok=True)
        with cls.dataset.open("w", encoding="utf-8") as handle:
            for index in range(600):
                handle.write(json.dumps({"text": CORPUS[index % len(CORPUS)]}) + "\n")
        cls.mapping = {"text": "text"}

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.workspace, ignore_errors=True)

    def _spec(self, job_id: str, **overrides) -> Path:
        job_dir = self.workspace / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        spec = {
            "job_id": job_id,
            "kind": "train",
            "job_dir": str(job_dir),
            "output_dir": str(job_dir / "output"),
            "backend": "tiny",
            "method": "scratch",
            "dataset_paths": [str(self.dataset)],
            "dataset_weights": [1.0],
            "mapping": self.mapping,
            "vocab_size": 320,
            "hidden_size": 12,
            "context_length": 4,
            "batch_size": 4,
            "gradient_accumulation": 1,
            "max_steps": 25,
            "learning_rate": 0.05,
            "logging_every": 5,
            "save_every": 10,
            "checkpoint_limit": 3,
            "eval_every": 10,
            "eval_ratio": 0.1,
            "seed": 11,
        }
        spec.update(overrides)
        spec_path = job_dir / "spec.json"
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        return spec_path

    def test_training_reduces_loss_and_writes_checkpoints(self) -> None:
        spec_path = self._spec("run-main")
        runner = trainer.JobRunner(spec_path)
        result = runner.run()
        self.assertEqual(result["status"], "completed", msg=str(result.get("error")))
        history = result["history"]
        self.assertGreaterEqual(len(history), 3)
        first_loss = history[0]["loss"]
        final_loss = history[-1]["loss"]
        self.assertLess(final_loss, first_loss, "loss must actually decrease during training")
        self.assertGreater((result["model_dir"] and Path(result["model_dir"]).exists()), False)
        checkpoints = [item for item in result["checkpoints"] if item["kind"] == "checkpoint"]
        self.assertTrue(checkpoints, "at least one checkpoint must be written")
        status = json.loads((spec_path.parent / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["state"], "completed")
        events = (spec_path.parent / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertGreater(len(events), 5)
        self.assertTrue((spec_path.parent / "metrics.jsonl").exists())
        self.assertTrue((spec_path.parent / "lineage.json").exists())

    def test_resume_restores_state_and_continues(self) -> None:
        spec_path = self._spec("run-resume")
        first = trainer.JobRunner(spec_path).run()
        checkpoint = next(item for item in first["checkpoints"] if item["kind"] == "checkpoint")
        resume_spec = self._spec(
            "run-resume-2",
            method="continued_training",
            parent_checkpoint=checkpoint["path"],
            resume_from=checkpoint["path"],
            max_steps=12,
            tokenizer_path=str(Path(first["output_dir"]) / "tokenizer.json"),
        )
        resumed_step = int(checkpoint["step"])
        result = trainer.JobRunner(resume_spec).run()
        self.assertEqual(result["status"], "completed", msg=str(result.get("error")))
        self.assertEqual(result["steps"], resumed_step + 12,
                         "a resumed run must continue from the checkpoint step and add the requested steps")
        status = json.loads((resume_spec.parent / "status.json").read_text(encoding="utf-8"))
        event_lines = (resume_spec.parent / "events.jsonl").read_text(encoding="utf-8")
        self.assertIn("Resumed from", event_lines)
        self.assertEqual(status["state"], "completed")

    def test_continuation_from_parent_model_changes_lineage(self) -> None:
        spec_path = self._spec("run-continue")
        parent = trainer.JobRunner(spec_path).run()
        continue_spec = self._spec(
            "run-continue-2",
            method="continued_pretraining",
            base_model=parent["model_dir"],
            tokenizer_path=str(Path(parent["output_dir"]) / "tokenizer.json"),
            max_steps=8,
        )
        result = trainer.JobRunner(continue_spec).run()
        self.assertEqual(result["status"], "completed", msg=str(result.get("error")))
        lineage = json.loads((continue_spec.parent / "lineage.json").read_text(encoding="utf-8"))
        self.assertIsNotNone(lineage["parent"])
        self.assertFalse(lineage["resumed"])
        self.assertEqual(lineage["child"]["method"], "continued_pretraining")

    def test_generation_produces_text_and_respects_limits(self) -> None:
        spec_path = self._spec("run-generate")
        run = trainer.JobRunner(spec_path).run()
        model, tokenizer = tiny.load_model(run["model_dir"])
        self.assertEqual(model.context_length, 4)
        events: list[dict] = []
        result = tiny.generate(run["model_dir"], {
            "prompt": "the quick fox ",
            "max_tokens": 12,
            "temperature": 0.7,
            "seed": 5,
        }, events.append)
        self.assertGreater(len(result["text"]), 0)
        self.assertLessEqual(result["completion_tokens"], 12)
        self.assertTrue(any(event.get("type") == "token" for event in events))
        self.assertTrue(any(event.get("type") == "done" for event in events))

    def test_evaluation_reports_real_metrics(self) -> None:
        spec_path = self._spec("run-evaluate")
        run = trainer.JobRunner(spec_path).run()
        result = tiny.evaluate(run["model_dir"], [str(self.dataset)],
                               {"mapping": self.mapping, "limit": 20}, lambda _event: None)
        metrics = result["metrics"]
        self.assertIn("loss", metrics)
        self.assertIn("perplexity", metrics)
        self.assertGreater(metrics["perplexity"], 0)
        self.assertGreater(result["evaluated_tokens"], 0)

    def test_failed_run_records_a_usable_diagnosis(self) -> None:
        spec_path = self._spec("run-broken", dataset_paths=["/definitely/not/here.jsonl"])
        result = trainer.JobRunner(spec_path).run()
        self.assertEqual(result["status"], "failed")
        status = json.loads((spec_path.parent / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["state"], "failed")
        self.assertTrue(status["error"]["message"])
        self.assertIn("log_tail", status)

    def test_recovery_marks_dead_jobs_as_interrupted_with_a_resume_point(self) -> None:
        spec_path = self._spec("run-crash")
        trainer.JobRunner(spec_path).run()
        status_path = spec_path.parent / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.update({"state": "running", "pid": 999_999_991})
        status_path.write_text(json.dumps(status), encoding="utf-8")
        findings = trainer.reconcile(self.workspace / "jobs")
        record = next(item for item in findings if item["job_id"] == "run-crash")
        self.assertEqual(record["action"], "marked_interrupted")
        self.assertIsNotNone(record["resumable_from"])

    def test_reconcile_keeps_live_jobs(self) -> None:
        spec_path = self._spec("run-live")
        import os as _os

        status_path = spec_path.parent / "status.json"
        status_path.write_text(json.dumps({
            "job_id": "run-live", "state": "running", "pid": _os.getpid(), "heartbeat": 9_999_999_999,
        }), encoding="utf-8")
        findings = trainer.reconcile(self.workspace / "jobs")
        record = next(item for item in findings if item["job_id"] == "run-live")
        self.assertEqual(record["action"], "reconnected")


class PlanningTests(unittest.TestCase):
    def test_plan_is_labelled_estimated_and_uses_real_hardware(self) -> None:
        report = hardware.detect([str(Path.home())])
        self.assertIn("capabilities", report)
        self.assertIn("cpu", report)
        plan = estimator.plan_run({
            "method": "lora",
            "precision": "fp32",
            "batch_size": 2,
            "sequence_length": 128,
            "dataset_paths": [],
            "hidden_size": 128,
        }, report)
        self.assertEqual(plan["labelled"], "estimated")
        self.assertGreater(plan["memory"]["vram_estimate"], 0)
        self.assertIn(plan["risk"], ("fits", "tight", "will_not_fit", "fits_cpu", "unknown"))
        self.assertIn("steps", plan)

    def test_prepare_reports_missing_dataset_and_model(self) -> None:
        report = trainer.prepare({
            "method": "lora",
            "backend": "tiny",
            "workspace": tempfile.mkdtemp(prefix="zxtrain-prep-"),
            "dataset_paths": [],
        })
        self.assertFalse(report["ok"])
        names = {check["name"] for check in report["checks"]}
        self.assertIn("dataset", names)
        self.assertTrue(any(check["level"] == "error" for check in report["checks"]))

    def test_continuation_report_is_honest_about_restored_state(self) -> None:
        report = trainer.continuation_report(
            {"parent_checkpoint": "/tmp/none", "resume_from": "/tmp/does-not-exist",
             "learning_rate": 1e-4},
            {"config": {"model_type": "zx-tiny"}},
        )
        self.assertIn(report["verdict"], ("compatible", "unsafe", "exact"))
        self.assertTrue(any(not item["ok"] for item in report["checks"]))


class ModelTests(unittest.TestCase):
    def test_gguf_like_and_unreadable_paths_explain_themselves(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="zxtrain-model-"))
        try:
            path = tmp / "toy.gguf"
            path.write_bytes(b"NOTGGUF" + b"\x00" * 32)
            with self.assertRaises(ZxError):
                model_tools.inspect(tmp / "missing")
            report = model_tools.inspect(path)
            self.assertEqual(report["format"]["kind"], "gguf")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_inspect_reads_safetensors_headers_for_exact_parameter_counts(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="zxtrain-st-"))
        try:
            header = json.dumps({
                "weight": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]},
                "__metadata__": {"format": "pt"},
            }).encode("utf-8")
            import struct

            payload = struct.pack("<Q", len(header)) + header + b"\x00" * 64
            (tmp / "model.safetensors").write_bytes(payload)
            (tmp / "config.json").write_text(json.dumps({
                "model_type": "llama", "architectures": ["LlamaForCausalLM"], "vocab_size": 100,
                "hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 4,
                "intermediate_size": 32, "max_position_embeddings": 128, "torch_dtype": "float32",
            }), encoding="utf-8")
            report = model_tools.inspect(tmp)
            self.assertEqual(report["weights"]["parameter_count"], 16)
            self.assertEqual(report["architecture"]["model_type"], "llama")
            self.assertIn("safetensors", report["weights"]["source"])
            self.assertEqual(report["estimated_vram"]["labelled"], "estimated")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class InterfaceTests(unittest.TestCase):
    def test_engine_capabilities_are_internally_consistent(self) -> None:
        payload = m_engine_capabilities({"workspace": tempfile.mkdtemp(prefix="zxtrain-cap-")})
        ids = {backend["id"] for backend in payload["backends"]}
        self.assertIn("tiny", ids)
        tiny_info = next(item for item in payload["backends"] if item["id"] == "tiny")
        self.assertTrue(tiny_info["available"])
        methods = {item["method"]: item for item in payload["methods"]}
        self.assertTrue(methods["scratch"]["available"])
        self.assertIn("hardware", payload)

    def test_cli_call_returns_structured_json(self) -> None:
        import subprocess

        process = subprocess.run(
            [sys.executable, "-m", "zxtrain.cli", "health"],
            input="{}", capture_output=True, text=True, cwd=str(ROOT),
        )
        self.assertEqual(process.returncode, 0, msg=process.stderr)
        payload = json.loads(process.stdout.strip().splitlines()[-1])
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"]["engine"], "zxtrain")

    def test_cli_unknown_command_is_reported_not_crashed(self) -> None:
        stdout = io.StringIO()
        original = sys.stdout
        sys.stdout = stdout
        try:
            code = main(["nope.nothing"])
        finally:
            sys.stdout = original
        self.assertEqual(code, 2)
        payload = json.loads(stdout.getvalue().strip())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "unknown_command")

    def test_every_registered_method_is_callable_signature(self) -> None:
        self.assertGreater(len(METHODS), 25)
        for name, handler in METHODS.items():
            self.assertTrue(callable(handler), name)

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="zxtrain-interface-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_precision_plan_and_real_conversion(self) -> None:
        """The converter must rewrite real safetensors bytes, not fake a report."""
        from array import array
        import struct

        from zxtrain import precision
        from zxtrain.safetensors import read_header

        source = self.tmp / "model"
        source.mkdir()
        values = [index * 0.5 for index in range(64)]
        payload = array("f", values).tobytes()
        header = json.dumps({
            "w": {"dtype": "F32", "shape": [8, 8], "data_offsets": [0, len(payload)]},
            "__metadata__": {"format": "pt"},
        }).encode("utf-8")
        header += b" " * ((-len(header)) % 8)
        (source / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)
        (source / "config.json").write_text(json.dumps({"model_type": "gpt2", "hidden_size": 8}), encoding="utf-8")

        plan = precision.plan(source, "F16", "safetensors")
        self.assertEqual(plan["status"], "Supported")
        self.assertTrue(plan["estimated"])
        self.assertGreater(plan["parameter_count"], 0)

        result = precision.convert(source, self.tmp / "converted", "F16", "safetensors")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["files"][0]["converted_tensors"], 1)
        produced = read_header(self.tmp / "converted" / "model.safetensors")
        self.assertEqual(produced["tensors"]["w"]["dtype"], "F16")
        self.assertEqual(produced["metadata"], {"format": "pt"})
        raw = (self.tmp / "converted" / "model.safetensors").read_bytes()
        roundtrip = struct.unpack("<64e", raw[-128:])
        self.assertAlmostEqual(roundtrip[1], 0.5, places=3)
        self.assertTrue((self.tmp / "converted" / "CONVERSION.md").exists())
        self.assertEqual(json.loads((self.tmp / "converted" / "config.json").read_text())["torch_dtype"], "f16")

        # The source must never be touched, and the converter must refuse to write over it.
        self.assertEqual(read_header(source / "model.safetensors")["tensors"]["w"]["dtype"], "F32")
        with self.assertRaises(ZxError):
            precision.convert(source, source, "F16", "safetensors")
        # Formats whose tool is missing are refused instead of simulated.
        with self.assertRaises(ZxError):
            precision.convert(source, self.tmp / "gguf", "F16", "gguf")

    def test_adapters_describe_and_merge_plan(self) -> None:
        from zxtrain import adapters

        adapter = self.tmp / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA", "base_model_name_or_path": str(self.tmp / "missing-base"),
            "r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"],
        }), encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(b"\x00" * 32)

        described = adapters.describe(adapter)
        self.assertEqual(described["rank"], 16)
        self.assertEqual(described["alpha"], 32)
        self.assertIn("q_proj", described["target_modules"])
        found = adapters.scan([str(self.tmp)])
        self.assertTrue(any(item.get("name") == "adapter" for item in found))
        plan = adapters.merge_plan(adapter)
        self.assertEqual(plan["status"], "Unsupported")
        self.assertTrue(any(not check["status"] == "Supported" for check in plan["checks"]))
        with self.assertRaises(ZxError):
            adapters.describe(self.tmp / "nothing-here")

    def test_tools_report_is_evidence_based(self) -> None:
        from zxtrain import tools

        model = self.tmp / "toolmodel"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
        report = tools.report(str(self.tmp), str(model))
        self.assertFalse(report["tool_calling"]["available"])
        self.assertIn("does not declare", report["tool_calling"]["reason"])
        (model / "tokenizer_config.json").write_text(json.dumps({
            "chat_template": "{% for m in messages %}{{ m }}{% endfor %}{{ tools }}",
        }), encoding="utf-8")
        declared = tools.tool_calling(model)
        self.assertTrue(declared["available"])
        self.assertTrue(declared["evidence"])

    def test_storage_protection_blocks_deletion(self) -> None:
        workspace = Path(tempfile.mkdtemp(prefix="zxtrain-protect-"))
        try:
            model_dir = workspace / "models" / "keepme"
            model_dir.mkdir(parents=True)
            (model_dir / "model.safetensors").write_bytes(b"\x00" * 16)
            storage.set_protected(model_dir, True)
            report = storage.report(workspace)
            self.assertTrue(any(entry["path"] == str(model_dir) for entry in report["protected"]))
            outcome = storage.clean([str(model_dir)], confirm=True)
            self.assertEqual(outcome["removed"], [])
            self.assertTrue(model_dir.exists())
            storage.set_protected(model_dir, False)
            outcome = storage.clean([str(model_dir)], confirm=True)
            self.assertEqual(len(outcome["removed"]), 1)
            self.assertFalse(model_dir.exists())
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def test_storage_report_and_orphans_on_a_fresh_workspace(self) -> None:
        workspace = Path(tempfile.mkdtemp(prefix="zxtrain-storage-"))
        try:
            report = storage.report(workspace)
            self.assertIn("entries", report)
            findings = storage.orphans(workspace)
            self.assertIn("findings", findings)
            with self.assertRaises(ZxError):
                storage.clean([str(workspace)], confirm=False)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
