import { useMemo, useState } from "react";
import { BookOpen, Search } from "lucide-react";
import { api } from "../lib/api";
import { useApp } from "../state/app";
import { useRouter } from "../state/router";
import { Badge, Button, Callout, CodeBlock, Input, Panel, SectionHeader, Table, Td, Th, cx } from "../components/ui";
import { ErrorPanel, useEngine } from "./common";

interface DocEntry {
  id: string;
  title: string;
  body: string;
  keywords: string[];
  category: string;
}

const ENTRIES: DocEntry[] = [
  {
    id: "sequence-length",
    title: "Sequence length",
    category: "Training parameters",
    body: "How many tokens a single training example is packed to. Memory grows roughly linearly with it, and attention cost grows quadratically. Raise it when examples are long conversations; lower it first when you hit memory limits. Very small values fragment long documents into many samples and waste the tokens spent on padding.",
    keywords: ["context", "packing", "tokens", "memory"],
  },
  {
    id: "batch-size",
    title: "Batch size and gradient accumulation",
    category: "Training parameters",
    body: "Batch size is what fits in memory on one forward/backward pass. Gradient accumulation splits a larger effective batch across several passes without extra memory. The effective batch is batch_size × gradient_accumulation; the engine shows it on the preview page. Change accumulation, not batch size, when memory is tight.",
    keywords: ["memory", "vram", "effective batch", "steps"],
  },
  {
    id: "learning-rate",
    title: "Learning rate",
    category: "Training parameters",
    body: "Too high and the loss oscillates or diverges; too low and the run wastes compute. Starting points the Auto Configure uses: 2e-4 for LoRA/QLoRA, 1e-4 for SFT, 2e-5 for full fine-tuning, 5e-5 for continued pretraining. On a resumed run, changing the learning rate breaks exact resume — the engine warns you before it happens.",
    keywords: ["lr", "optimizer", "scheduler"],
  },
  {
    id: "precision",
    title: "Precision (fp32 / fp16 / bf16)",
    category: "Training parameters",
    body: "fp32 is the safest and the slowest. fp16 halves the memory but needs loss scaling tuning. bf16 has the same range as fp32 with fewer mantissa bits and is the usual choice on Ampere or newer GPUs. The Hardware page reports exactly which of these this machine supports; picking one that is unavailable is refused before the run starts.",
    keywords: ["bf16", "fp16", "mixed precision", "tf32"],
  },
  {
    id: "quantization",
    title: "Quantization (QLoRA, int8, int4)",
    category: "Training parameters",
    body: "Quantized training keeps the frozen base weights in 4 or 8 bits and trains small adapter matrices on top. It needs bitsandbytes; the Quantization page reports whether it is installed. Training itself stays in bf16/fp16, so quality loss is usually small, but quantized runs are slower per step.",
    keywords: ["qlora", "bitsandbytes", "4-bit", "8-bit"],
  },
  {
    id: "lora",
    title: "LoRA rank, alpha and dropout",
    category: "Training parameters",
    body: "Rank is the size of the adapter matrices: 8–16 for style and tone, 32–64 when teaching a new domain. Alpha scales the adapter's contribution (alpha/rank is the effective scaling). Dropout regularises small datasets; 0.05 is a sane default and 0 disables it.",
    keywords: ["peft", "adapter", "target modules"],
  },
  {
    id: "gradient-checkpointing",
    title: "Gradient checkpointing",
    category: "Training parameters",
    body: "Recomputes activations during the backward pass instead of storing them. Cuts activation memory a lot — often the difference between fitting and OOM — at roughly 15–25% slower steps. Enable it before shrinking the batch size.",
    keywords: ["activation", "memory", "oom"],
  },
  {
    id: "resume",
    title: "Resume vs continue training",
    category: "Continuation",
    body: "Resume restores the interrupted run exactly: weights, optimizer, scheduler, gradient scaler, RNG and global step, and refuses configuration changes that would make that impossible. Continue training starts a new child run from a checkpoint or model with a fresh optimizer, which is what you want when the dataset or the learning rate changes. The engine shows, per item, what was really restored — never assuming.",
    keywords: ["checkpoint", "interrupted", "lineage", "continue"],
  },
  {
    id: "lineage",
    title: "Lineage and versioning",
    category: "Continuation",
    body: "Every run writes lineage.json naming its parent model or checkpoint, the datasets it used and the method. Child runs stay attached to that parent, so the Experiments page can draw the real tree and Compare can put a base model next to any descendant.",
    keywords: ["parent", "tree", "experiment"],
  },
  {
    id: "estimated",
    title: "Estimated vs measured numbers",
    category: "Concepts",
    body: "Memory and size predictions are labelled Estimated and come from parameter counts, dtype widths and configured batch/sequence sizes. Everything shown for a run that has actually executed — loss, tokens/second, VRAM in use, latency, perplexity — is measured from the process or the machine. The interface never presents an estimate as a measurement.",
    keywords: ["estimate", "prediction", "honesty"],
  },
  {
    id: "backends",
    title: "Backends: tiny and Transformers/PEFT",
    category: "Concepts",
    body: "The tiny backend is pure Python: it trains real small transformers from scratch or continues them, works on any machine with no dependencies, and is ideal for learning the workflow end to end. The Transformers/PEFT backend trains real Hugging Face checkpoints with LoRA/QLoRA/SFT. Availability is detected, not assumed; the Environment page shows exactly which packages are present.",
    keywords: ["torch", "transformers", "peft", "backend"],
  },
  {
    id: "formats",
    title: "Model formats",
    category: "Concepts",
    body: "safetensors is the primary format and is parsed directly, including parameter counts and dtypes straight from the header. PyTorch .bin checkpoints are read through torch. GGUF files are inspected (metadata, quantization type) and can be produced or consumed when a llama.cpp converter is available. Adapters are folders with adapter_config.json.",
    keywords: ["gguf", "bin", "safetensors", "adapter"],
  },
  {
    id: "tokenizer",
    title: "Tokenizer",
    category: "Concepts",
    body: "The engine reads tokenizer.json, tokenizer.model, vocab.json/merges.txt and tokenizer_config.json, and reports special tokens, vocabulary size and the chat template. The Tokenizer Lab encodes real text with the selected tokenizer, so token counts on the Dataset page are measured the same way training will see them. The tiny backend can train its own BPE tokenizer from your data.",
    keywords: ["bpe", "vocab", "tokens", "chatml"],
  },
  {
    id: "oom",
    title: "CUDA out of memory",
    category: "Troubleshooting",
    body: "Reduce per-device batch size, raise gradient accumulation by the same factor, enable gradient checkpointing, shorten the sequence length, or load the base model in 4-bit. The engine classifies the error from the real traceback and lists these options with the values from your configuration.",
    keywords: ["oom", "memory", "error"],
  },
  {
    id: "tokenizer-error",
    title: "Tokenizer or vocabulary mismatch",
    category: "Troubleshooting",
    body: "Happens when a checkpoint is continued with a different tokenizer, or when a dataset is tokenized with a model whose vocabulary does not match. Resuming across a tokenizer change is refused: it would silently scramble the embedding table. Fine-tune the base model with its own tokenizer, or start a new run from the base model instead.",
    keywords: ["mismatch", "vocab", "error", "resume"],
  },
  {
    id: "torch-missing",
    title: "PyTorch is missing or unsupported",
    category: "Troubleshooting",
    body: "The Environment page reports the interpreter version and whether torch is importable. Newer Python releases are not supported by PyTorch yet; the engine tells you the ceiling and offers to create its own environment with a supported interpreter instead of installing into the system Python.",
    keywords: ["torch", "python", "install"],
  },
  {
    id: "api",
    title: "Local API",
    category: "Integrations",
    body: "The Deploy page can serve a loaded model over loopback with an OpenAI-compatible surface: /health, /v1/models, /v1/chat/completions and /v1/completions (streaming supported). Requests are logged to the job folder. Nothing binds to a public interface unless you turn that on in Settings.",
    keywords: ["rest", "openai", "server", "endpoint"],
  },
  {
    id: "cli",
    title: "CLI companion",
    category: "Integrations",
    body: "The GUI and the CLI share one engine. Every engine command is available as `python -m zxtrain.cli <method>` with a JSON payload on stdin, jobs run as `python -m zxtrain.cli run <spec.json>`, and the inference sidecar is `python -m zxtrain.cli serve`.",
    keywords: ["command line", "automation", "headless"],
  },
  {
    id: "plugins",
    title: "Plugins",
    category: "Integrations",
    body: "Place a folder with plugin.json in <workspace>/plugins. Each plugin declares the kinds it extends (loaders, methods, evaluators, exporters, dataset processors). The app lists what it finds and each plugin's declared compatibility instead of assuming it works.",
    keywords: ["extension", "backend", "compatibility"],
  },
  {
    id: "reproducibility",
    title: "Reproducibility bundle",
    category: "Integrations",
    body: "A run can export a bundle containing its configuration, seed, hardware snapshot, package versions, model and dataset digests and the exact engine version. Feed that bundle back to a colleague and the numbers are reproducible from the same inputs.",
    keywords: ["seed", "digest", "bundle", "export"],
  },
];

const CATEGORIES = ["All", "Training parameters", "Continuation", "Concepts", "Troubleshooting", "Integrations"];

export function DocsPage() {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("All");
  const { appInfo } = useApp();
  const router = useRouter();
  const capabilities = useEngine<any>("engine.capabilities", {}, { timeout: 180_000 });

  const entries = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return ENTRIES.filter((entry) => {
      if (category !== "All" && entry.category !== category) return false;
      if (!needle) return true;
      return `${entry.title} ${entry.body} ${entry.keywords.join(" ")}`.toLowerCase().includes(needle);
    });
  }, [query, category]);

  const methods = useMemo(() => {
    const list = capabilities.data?.methods ?? [];
    return list as Array<{ id: string; label?: string; supported: boolean; reason?: string; requirements?: string[] }>;
  }, [capabilities.data]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Documentation</h1>
          <p className="mt-0.5 max-w-3xl text-xs text-ink-2">
            Contextual help for every parameter and workflow in this build. It is bundled locally — no network call is
            made to show it.
          </p>
        </div>
        <Badge tone="muted">
          <BookOpen size={10} /> offline
        </Badge>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-[220px] flex-1">
          <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-3" />
          <Input className="pl-7" placeholder="Search parameters, errors and concepts…" value={query} onChange={(event) => setQuery(event.target.value)} />
        </div>
        <div className="flex flex-wrap gap-1">
          {CATEGORIES.map((entry) => (
            <button
              key={entry}
              onClick={() => setCategory(entry)}
              className={cx(
                "rounded-full border px-2.5 py-1 text-2xs transition-colors",
                category === entry ? "border-accent/40 bg-accent/15 text-ink-0" : "border-line-soft text-ink-2 hover:bg-surface-3",
              )}
            >
              {entry}
            </button>
          ))}
        </div>
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        {entries.map((entry) => (
          <Panel key={entry.id}>
            <SectionHeader title={entry.title} subtitle={entry.category} />
            <p className="text-xs leading-relaxed text-ink-2">{entry.body}</p>
            <div className="mt-2 flex flex-wrap gap-1">
              {entry.keywords.map((keyword) => (
                <Badge key={keyword} tone="muted">
                  {keyword}
                </Badge>
              ))}
            </div>
          </Panel>
        ))}
        {!entries.length ? (
          <Panel>
            <Callout tone="info" title="Nothing matched">
              Try a different word — the search covers parameter names, error codes and workflow terms.
            </Callout>
          </Panel>
        ) : null}
      </div>

      <Panel>
        <SectionHeader
          title="Capability matrix"
          subtitle="Which training methods this machine can actually run right now, and why not when it cannot."
          actions={<Button size="sm" variant="subtle" onClick={() => router.navigate("/hardware")}>Hardware Center</Button>}
        />
        {capabilities.error ? <ErrorPanel error={capabilities.error} onRetry={() => void capabilities.reload()} /> : null}
        {methods.length ? (
          <Table>
            <thead>
              <tr>
                <Th>Method</Th>
                <Th>Status</Th>
                <Th>Why</Th>
              </tr>
            </thead>
            <tbody>
              {methods.map((method) => (
                <tr key={method.id}>
                  <Td>{method.label ?? method.id}</Td>
                  <Td>
                    <Badge tone={method.supported ? "ok" : "warn"}>{method.supported ? "supported" : "not available"}</Badge>
                  </Td>
                  <Td>
                    <span className="text-2xs text-ink-2">{method.reason}</span>
                    {method.requirements?.length ? (
                      <span className="mt-1 block font-mono text-2xs text-ink-3">needs: {method.requirements.join(", ")}</span>
                    ) : null}
                  </Td>
                </tr>
              ))}
            </tbody>
          </Table>
        ) : null}
      </Panel>

      <Panel>
        <SectionHeader title="Command line" subtitle="The GUI and the CLI drive the same engine." />
        <CodeBlock>{`# list every engine command
python -m zxtrain.cli --help

# call one command with a JSON payload on stdin
echo '{"workspace":"D:\\\\Zeqou"}' | python -m zxtrain.cli hardware.detect

# run a job exactly as the app does
python -m zxtrain.cli run <workspace>/jobs/<job-id>/spec.json

# start the inference sidecar used by the Playground
python -m zxtrain.cli serve`}</CodeBlock>
        <div className="mt-2">
          <Button
            size="sm"
            variant="subtle"
            onClick={() => void api.shell.reveal(appInfo?.engineDir ?? ".")}
          >
            Open the engine folder
          </Button>
        </div>
      </Panel>
    </div>
  );
}
