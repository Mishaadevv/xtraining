import { useMemo, useState } from "react";
import { Brain, FileJson, GitBranch, Layers, Play, Rocket } from "lucide-react";
import { api } from "../lib/api";
import { basename, bytes, clock, number } from "../lib/format";
import { useApp } from "../state/app";
import { useRouter } from "../state/router";
import {
  Badge,
  Button,
  Callout,
  CodeBlock,
  CopyButton,
  EmptyState,
  KeyValue,
  Panel,
  SectionHeader,
  Stat,
  Table,
  Tab,
  Tabs,
  Td,
  Th,
} from "../components/ui";
import { ErrorPanel, Loading, NotFound, PathText, useEngine } from "./common";

export function ModelDetailPage({ id }: { id: string }) {
  const { registry, refreshRegistry, reportError, toast } = useApp();
  const router = useRouter();
  const entry = (registry?.models ?? []).find((model) => model.id === id);
  const [tab, setTab] = useState("overview");

  const inspection = useEngine<any>(
    "models.inspect",
    { path: entry?.path ?? "", digest: false },
    { auto: Boolean(entry?.path), timeout: 300_000, deps: [entry?.path] },
  );

  const card = useEngine<any>(
    "models.card",
    { path: entry?.path ?? "", notes: entry?.notes ?? "" },
    { auto: Boolean(entry?.path) && tab === "card", timeout: 120_000, deps: [entry?.path, tab, entry?.notes] },
  );

  const rawConfig = useMemo(() => inspection.data?.config ?? null, [inspection.data]);

  if (!entry) {
    return <NotFound what="This model" onBack={() => router.navigate("/models")} />;
  }

  const report = inspection.data;
  const architecture = report?.architecture ?? {};
  const weights = report?.weights ?? {};

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-2xs uppercase tracking-wider text-ink-3">{report?.format?.kind ?? "model"}</div>
          <h1 className="flex items-center gap-2 text-lg font-semibold tracking-tight">
            {entry.name}
            {report?.format?.is_adapter ? <Badge tone="accent">LoRA adapter</Badge> : null}
          </h1>
          <PathText value={entry.path} />
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            variant="primary"
            icon={<Rocket size={13} />}
            onClick={() => router.navigate(`/training/new?model=${encodeURIComponent(entry.path)}`)}
            title="Start a new run that continues from this model"
          >
            Continue training
          </Button>
          <Button icon={<Play size={13} />} onClick={() => router.navigate(`/playground?model=${encodeURIComponent(entry.path)}`)}>
            Test in playground
          </Button>
          <Button icon={<GitBranch size={13} />} onClick={() => router.navigate(`/evaluation?model=${encodeURIComponent(entry.path)}`)}>
            Evaluate
          </Button>
          <Button onClick={() => router.navigate(`/compare?left=${encodeURIComponent(entry.path)}`)}>Compare</Button>
          <Button onClick={() => api.shell.reveal(entry.path)}>Reveal</Button>
          <Button
            onClick={async () => {
              try {
                const updated = await api.registry.update("models", { ...entry, favorite: !entry.favorite });
                await refreshRegistry();
                toast({ title: updated ? "Updated" : "Updated", body: entry.favorite ? "Removed from favourites." : "Pinned to favourites.", tone: "info" });
              } catch (error) {
                reportError(error, "Could not update the model");
              }
            }}
          >
            {entry.favorite ? "Unpin" : "Favourite"}
          </Button>
        </div>
      </div>

      {inspection.error ? <ErrorPanel error={inspection.error} onRetry={inspection.reload} /> : null}
      {inspection.loading && !report ? <Loading label="Inspecting config, weights and tokenizer…" lines={5} /> : null}

      {report ? (
        <>
          <div className="grid gap-3 md:grid-cols-4">
            <Stat label="Parameters" value={weights.parameter_count ? number(weights.parameter_count) : "unknown"} hint={weights.source ?? ""} />
            <Stat label="Size on disk" value={bytes(report.size_bytes)} hint={`${report.file_count} files`} />
            <Stat label="Architecture" value={String(architecture.model_type ?? "unknown")} hint={`${architecture.num_layers ?? "?"} layers`} />
            <Stat label="Context" value={String(architecture.max_position_embeddings ?? "unknown")} hint={`vocab ${architecture.vocab_size ?? "?"}`} />
          </div>

          <Tabs value={tab} onChange={setTab}>
            <Tab value="overview">Overview</Tab>
            <Tab value="architecture">Architecture</Tab>
            <Tab value="tokenizer">Tokenizer</Tab>
            <Tab value="files">Files</Tab>
            <Tab value="raw">Raw config</Tab>
            <Tab value="card">Model card</Tab>
          </Tabs>

          {tab === "overview" ? (
            <div className="grid gap-3 lg:grid-cols-2">
              <Panel>
                <SectionHeader title="Summary" icon={<Brain size={13} className="text-accent" />} />
                <KeyValue
                  items={[
                    ["Format", report.format?.kind ?? "unknown"],
                    ["Weight source", weights.source ?? "unknown"],
                    ["Tensors", weights.tensor_count ? number(weights.tensor_count) : "unknown"],
                    ["Dtypes", Object.entries(weights.dtypes ?? {}).map(([key, value]) => `${key}: ${number(Number(value))}`).join(", ") || "unknown"],
                    ["Adapter", report.adapter ? "yes" : "no"],
                    ["Inspected", clock(report.inspected_at)],
                  ]}
                />
              </Panel>
              <Panel>
                <SectionHeader title="Estimated memory" subtitle="Derived from the real parameter count" icon={<Layers size={13} className="text-accent" />} />
                {report.estimated_vram ? (
                  <>
                    <KeyValue
                      items={[
                        ["Weights at file dtype", bytes(Number(report.estimated_vram.weights_only))],
                        ["Inference fp16", bytes(Number(report.estimated_vram.fp16_inference))],
                        ["Inference int8", bytes(Number(report.estimated_vram.int8_inference))],
                        ["Inference int4", bytes(Number(report.estimated_vram.int4_inference))],
                        ["LoRA training (fp16)", bytes(Number(report.estimated_vram.lora_training_fp16))],
                        ["Full fine-tune (fp16)", bytes(Number(report.estimated_vram.full_training_fp16))],
                      ]}
                    />
                    <div className="mt-2 text-2xs text-ink-3">{String(report.estimated_vram.formula)}</div>
                  </>
                ) : (
                  <EmptyState title="No weight headers were readable">
                    Parameter counts are unavailable for this folder, so no memory estimate is shown.
                  </EmptyState>
                )}
              </Panel>
            </div>
          ) : null}

          {tab === "architecture" ? (
            <Panel>
              <SectionHeader title="Architecture" subtitle={String(architecture.note ?? "")} />
              <KeyValue
                items={[
                  ["Architectures", (architecture.architectures ?? []).join(", ") || "unknown"],
                  ["Hidden size", String(architecture.hidden_size ?? "unknown")],
                  ["Intermediate size", String(architecture.intermediate_size ?? "unknown")],
                  ["Layers", String(architecture.num_layers ?? "unknown")],
                  ["Attention heads", String(architecture.num_attention_heads ?? "unknown")],
                  ["Key/value heads", String(architecture.num_key_value_heads ?? "unknown")],
                  ["Head dimension", String(architecture.head_dim ?? "unknown")],
                  ["Vocabulary", String(architecture.vocab_size ?? "unknown")],
                  ["Positional encoding", architecture.rope_scaling ? `RoPE (scaled, theta ${architecture.rope_theta})` : `RoPE theta ${architecture.rope_theta ?? "default"}`],
                  ["Activation", String(architecture.hidden_act ?? "unknown")],
                  ["Normalisation", String(architecture.norm ?? "unknown")],
                  ["Tied embeddings", architecture.tie_word_embeddings === undefined ? "unknown" : String(architecture.tie_word_embeddings)],
                  ["Sliding window", architecture.sliding_window ? String(architecture.sliding_window) : "none"],
                  ["Experts", architecture.expert_config ? `${architecture.expert_config.num_experts} experts, ${architecture.expert_config.experts_per_token} per token` : "dense"],
                  ["Declared dtype", String(architecture.torch_dtype ?? "unknown")],
                ]}
              />
            </Panel>
          ) : null}

          {tab === "tokenizer" ? (
            <Panel>
              <SectionHeader title="Tokenizer" subtitle="Special tokens and vocabulary read from the tokenizer files" />
              <KeyValue
                items={[
                  ["Class", String(report.tokenizer?.class ?? "unknown")],
                  ["Vocabulary size", String(report.tokenizer?.vocab_size ?? "unknown")],
                  ["Model max length", String(report.tokenizer?.model_max_length ?? "unknown")],
                  ["Chat template", report.tokenizer?.chat_template ? "present" : "absent"],
                  ["Tokenizer type", String(report.tokenizer?.tokenizer_type ?? "unknown")],
                  ["Merges", String(report.tokenizer?.merges ?? "n/a")],
                ]}
              />
              <div className="mt-3 flex flex-wrap gap-1.5">
                {Object.entries(report.tokenizer?.special_tokens ?? {}).map(([key, value]) => (
                  <Badge key={key} tone="muted">
                    {key}: {String(value)}
                  </Badge>
                ))}
              </div>
              <div className="mt-3">
                <div className="mb-1 text-2xs uppercase tracking-wide text-ink-3">Tokenizer files</div>
                <div className="flex flex-wrap gap-1.5">
                  {(report.tokenizer?.files ?? []).map((file: any) => (
                    <Badge key={file.name} tone="muted">
                      {file.name} · {bytes(file.size)}
                    </Badge>
                  ))}
                  {!(report.tokenizer?.files ?? []).length ? <span className="text-2xs text-ink-3">No tokenizer files found.</span> : null}
                </div>
              </div>
            </Panel>
          ) : null}

          {tab === "files" ? (
            <Panel padded={false}>
              <div className="p-3">
                <SectionHeader title="Files" subtitle={`${report.files.length} entries listed`} />
              </div>
              <Table>
                <thead>
                  <tr>
                    <Th>File</Th>
                    <Th align="right">Size</Th>
                  </tr>
                </thead>
                <tbody>
                  {(report.files ?? []).map((file: any) => (
                    <tr key={file.name}>
                      <Td>
                        <span className="font-mono text-2xs">{file.name}</span>
                      </Td>
                      <Td align="right">{file.size_human}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            </Panel>
          ) : null}

          {tab === "raw" ? (
            <div className="space-y-3">
              <Panel>
                <SectionHeader
                  title="config.json"
                  icon={<FileJson size={13} className="text-accent" />}
                  actions={<CopyButton value={JSON.stringify(rawConfig, null, 2)} label="Copy config" />}
                />
                <CodeBlock max="max-h-[420px]">{JSON.stringify(rawConfig, null, 2)}</CodeBlock>
              </Panel>
              <Panel>
                <SectionHeader title="generation_config.json" actions={<CopyButton value={JSON.stringify(report.generation_config ?? {}, null, 2)} />} />
                <CodeBlock max="max-h-72">{JSON.stringify(report.generation_config ?? {}, null, 2)}</CodeBlock>
              </Panel>
              {report.weights?.gguf_metadata ? (
                <Panel>
                  <SectionHeader title="GGUF metadata" />
                  <CodeBlock max="max-h-72">{JSON.stringify(report.weights.gguf_metadata, null, 2)}</CodeBlock>
                </Panel>
              ) : null}
            </div>
          ) : null}

          {tab === "card" ? (
            <Panel>
              <SectionHeader
                title="Model card"
                subtitle="Generated from this model's real metadata and run configuration"
                actions={
                  <>
                    {card.data?.markdown ? <CopyButton value={card.data.markdown} label="Copy card" /> : null}
                    <Button
                      size="sm"
                      variant="subtle"
                      onClick={async () => {
                        try {
                          await api.call("models.card", { path: entry.path, notes: entry.notes ?? "", write: true });
                          toast({ title: "Model card written", body: `${entry.path}\\README.md`, tone: "ok" });
                        } catch (error) {
                          reportError(error, "Could not write the model card");
                        }
                      }}
                    >
                      Write README.md
                    </Button>
                  </>
                }
              />
              {card.loading ? <Loading lines={6} /> : null}
              {card.error ? <ErrorPanel error={card.error} onRetry={card.reload} /> : null}
              {card.data ? <CodeBlock max="max-h-[520px]">{card.data.markdown}</CodeBlock> : null}
            </Panel>
          ) : null}

          <Panel>
            <SectionHeader title="Continuation and lineage" subtitle="What the engine can do with this model right now" />
            <div className="grid gap-2 md:grid-cols-2 lg:grid-cols-4">
              <Button onClick={() => router.navigate(`/training/new?model=${encodeURIComponent(entry.path)}&method=continued_pretraining`)}>
                Continue pretraining
              </Button>
              <Button onClick={() => router.navigate(`/training/new?model=${encodeURIComponent(entry.path)}&method=lora`)}>
                New LoRA fine-tune
              </Button>
              <Button onClick={() => router.navigate(`/training?model=${encodeURIComponent(entry.path)}`)}>
                Runs using this model
              </Button>
              <Button onClick={() => router.navigate(`/adapters?model=${encodeURIComponent(entry.path)}`)}>
                Adapter workspace
              </Button>
            </div>
            <Callout tone="info" title="Would you like to append the model card into the folder?">
              {basename(entry.path)} has the training metadata stored in {report.format?.is_adapter ? "adapter_config.json" : "zxtrain-run.json"} when
              it was produced by this app; imported models show only what their own files declare.
            </Callout>
          </Panel>
        </>
      ) : null}
    </div>
  );
}
