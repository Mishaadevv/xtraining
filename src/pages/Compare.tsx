import { useMemo, useState } from "react";
import { Eye, EyeOff, FlaskConical, GitCompare, Loader2, Play, Save } from "lucide-react";
import { api } from "../lib/api";
import { basename, bytes, number } from "../lib/format";
import { useApp } from "../state/app";
import {
  Badge,
  Button,
  Callout,
  Field,
  NumberInput,
  Panel,
  SectionHeader,
  Stat,
  TextInput,
  Toggle,
} from "../components/ui";
import { ErrorPanel, ModelPicker, useEngine } from "./common";

interface Side {
  path: string | null;
  label: string;
  status: "idle" | "loading" | "running" | "done" | "error";
  text: string;
  stream: string;
  latency: number | null;
  tokensPerSecond: number | null;
  completionTokens: number | null;
  promptTokens: number | null;
  error: string | null;
}

const emptySide = (label: string): Side => ({
  path: null,
  label,
  status: "idle",
  text: "",
  stream: "",
  latency: null,
  tokensPerSecond: null,
  completionTokens: null,
  promptTokens: null,
  error: null,
});

export function ComparePage() {
  const { registry, toast, reportError, refreshRegistry } = useApp();
  const [left, setLeft] = useState<Side>(emptySide("A"));
  const [right, setRight] = useState<Side>(emptySide("B"));
  const [prompt, setPrompt] = useState("Explain what fine-tuning changes about a model in three sentences.");
  const [systemPrompt, setSystemPrompt] = useState("You are a concise technical assistant.");
  const [blind, setBlind] = useState(false);
  const [revealed, setRevealed] = useState(false);
  const [running, setRunning] = useState(false);
  const [paramsState, setParamsState] = useState({ temperature: 0.7, top_p: 0.95, max_tokens: 192, seed: "" as string | number });
  const backends = useEngine<any>("backends.list", {}, { timeout: 120_000 });

  const models = registry?.models ?? [];
  const description = useMemo(() => {
    const describe = (side: Side) => {
      if (!side.path) return null;
      const entry = models.find((model) => model.path === side.path);
      return {
        name: entry?.name ?? basename(side.path),
        parameters: (entry?.summary as any)?.parameters ?? null,
        architecture: (entry?.summary as any)?.architecture ?? null,
        size: (entry?.summary as any)?.size_bytes ?? null,
        quantization: (entry?.summary as any)?.quantization ?? null,
      };
    };
    return { left: describe(left), right: describe(right) };
  }, [left, right, models]);

  const update = (which: "left" | "right", patch: Partial<Side>) =>
    which === "left" ? setLeft((current) => ({ ...current, ...patch })) : setRight((current) => ({ ...current, ...patch }));

  /** One model at a time: the sidecar keeps a single model resident. */
  const runSide = async (which: "left" | "right", side: Side): Promise<void> => {
    if (!side.path) return;
    update(which, { status: "loading", text: "", stream: "", error: null, latency: null, tokensPerSecond: null });
    try {
      await api.sidecar.load(side.path, undefined);
      update(which, { status: "running" });
      const payload = {
        messages: [
          ...(systemPrompt.trim() ? [{ role: "system", content: systemPrompt.trim() }] : []),
          { role: "user", content: prompt },
        ],
        temperature: paramsState.temperature,
        top_p: paramsState.top_p,
        max_tokens: paramsState.max_tokens,
        seed: paramsState.seed === "" ? null : Number(paramsState.seed),
      };
      const result = await api.sidecar.generate(payload);
      update(which, {
        status: "done",
        text: result.text ?? "",
        stream: "",
        latency: result.latency_seconds ?? null,
        tokensPerSecond: result.tokens_per_second ?? null,
        completionTokens: result.completion_tokens ?? null,
        promptTokens: result.prompt_tokens ?? null,
      });
    } catch (error: any) {
      update(which, {
        status: "error",
        error: error?.structured?.message ?? String(error?.message ?? error),
      });
    }
  };

  const runBoth = async () => {
    if (!left.path || !right.path) {
      toast({ title: "Pick two models", body: "Comparison needs a model on both sides.", tone: "warn" });
      return;
    }
    if (left.path === right.path) {
      toast({
        title: "Both sides point at the same model",
        body: "Comparing a model with itself is a useful sanity check, but the numbers will match.",
        tone: "warn",
      });
    }
    setRunning(true);
    setRevealed(false);
    try {
      await runSide("left", left);
      await runSide("right", right);
      toast({
        title: "Comparison finished",
        body: "Latency and tokens/second were measured on this machine, one model at a time.",
        tone: "ok",
      });
    } finally {
      setRunning(false);
      await api.sidecar.unload().catch(() => undefined);
    }
  };

  const saveComparison = async () => {
    try {
      await api.registry.add("evaluations", {
        id: `${Date.now().toString(36)}`,
        kind: "comparison",
        createdAt: new Date().toISOString(),
        prompt,
        systemPrompt,
        params: paramsState,
        blind,
        sides: [
          { path: left.path, text: left.text, latency: left.latency, tokensPerSecond: left.tokensPerSecond, tokens: left.completionTokens },
          { path: right.path, text: right.text, latency: right.latency, tokensPerSecond: right.tokensPerSecond, tokens: right.completionTokens },
        ],
      });
      await refreshRegistry();
      toast({ title: "Comparison saved", body: "Stored in the workspace registry as an evaluation record.", tone: "ok" });
    } catch (error) {
      reportError(error, "Could not save the comparison");
    }
  };

  const sidePanel = (which: "left" | "right", side: Side) => {
    const info = which === "left" ? description.left : description.right;
    const name = blind && !revealed ? `Model ${side.label}` : info?.name ?? side.path ?? "not selected";
    return (
      <Panel className="flex min-h-[420px] flex-col">
        <SectionHeader
          title={name}
          subtitle={
            blind && !revealed
              ? "hidden for blind comparison"
              : side.path ?? "Choose a model from the library"
          }
          actions={
            <div className="flex items-center gap-2">
              {side.status === "running" || side.status === "loading" ? (
                <Badge tone="info">
                  <Loader2 size={10} className="animate-spin" /> {side.status === "loading" ? "loading" : "generating"}
                </Badge>
              ) : null}
              {side.status === "done" ? <Badge tone="ok">measured</Badge> : null}
              {side.status === "error" ? <Badge tone="danger">failed</Badge> : null}
            </div>
          }
        />

        <div className="mb-3">
          <ModelPicker
            value={side.path}
            onChange={(path) => update(which, { path, status: "idle", text: "", error: null })}
            label={`Model ${side.label}`}
          />
        </div>

        {info && !(blind && !revealed) ? (
          <div className="mb-3 grid gap-2 sm:grid-cols-4">
            <Stat label="Parameters" value={info.parameters ? number(info.parameters) : "—"} />
            <Stat label="Architecture" value={info.architecture ?? "—"} />
            <Stat label="On disk" value={info.size ? bytes(info.size) : "—"} />
            <Stat label="Quantization" value={info.quantization ?? "none"} />
          </div>
        ) : null}

        {side.error ? (
          <div className="mb-2">
            <Callout tone="danger" title="This side failed" hint="The other side still ran; fix this model and re-run just it.">
              {side.error}
            </Callout>
          </div>
        ) : null}

        <div className="min-h-[180px] flex-1 overflow-y-auto rounded-md border border-line-soft bg-surface-2 p-3 text-xs leading-relaxed text-ink-1">
          {side.status === "running" && !side.text ? (
            <span className="text-ink-3">generating…</span>
          ) : (
            <span className="whitespace-pre-wrap">{side.text || (side.status === "idle" ? "No output yet." : "")}</span>
          )}
        </div>

        <div className="mt-3 grid gap-2 sm:grid-cols-4">
          <Stat label="Latency" value={side.latency !== null ? `${side.latency.toFixed(2)} s` : "—"} />
          <Stat label="Tokens / s" value={side.tokensPerSecond !== null ? side.tokensPerSecond.toFixed(1) : "—"} />
          <Stat label="Completion tokens" value={side.completionTokens ?? "—"} />
          <Stat label="Prompt tokens" value={side.promptTokens ?? "—"} />
        </div>

        <div className="mt-2 flex justify-end">
          <Button size="sm" variant="subtle" disabled={!side.path || running} onClick={() => void runSide(which, side)}>
            Re-run this side
          </Button>
        </div>
      </Panel>
    );
  };

  const speedup =
    left.tokensPerSecond && right.tokensPerSecond
      ? right.tokensPerSecond / left.tokensPerSecond
      : null;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Compare</h1>
          <p className="mt-0.5 max-w-3xl text-xs text-ink-2">
            Both models answer the same prompt with the same sampling settings. Because the engine keeps one model
            resident in the inference process, the sides run one after another — every latency and tokens/second number
            is measured, never estimated.
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="subtle" icon={blind ? <EyeOff size={12} /> : <Eye size={12} />} onClick={() => setBlind((value) => !value)}>
            {blind ? "Blind: on" : "Blind: off"}
          </Button>
          {blind && !revealed ? (
            <Button size="sm" variant="subtle" icon={<Eye size={12} />} onClick={() => setRevealed(true)}>
              Reveal
            </Button>
          ) : null}
          <Button size="sm" variant="subtle" icon={<Save size={12} />} disabled={!left.text && !right.text} onClick={() => void saveComparison()}>
            Save comparison
          </Button>
          <Button size="sm" variant="primary" icon={<Play size={12} />} loading={running} onClick={() => void runBoth()}>
            Run both
          </Button>
        </div>
      </div>

      {backends.error ? <ErrorPanel error={backends.error} onRetry={() => void backends.reload()} /> : null}

      <Panel>
        <SectionHeader title="Prompt" subtitle="Applied to both sides unchanged." />
        <div className="space-y-3">
          <Field label="System prompt">
            <TextInput value={systemPrompt} onChange={(event) => setSystemPrompt(event.target.value)} />
          </Field>
          <Field label="User message">
            <textarea
              className="field min-h-[80px] w-full resize-y"
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
            />
          </Field>
          <div className="grid gap-3 sm:grid-cols-4">
            <Field label="Temperature">
              <NumberInput value={paramsState.temperature} step={0.05} min={0} max={2} onChange={(value) => setParamsState({ ...paramsState, temperature: Number(value) })} />
            </Field>
            <Field label="Top-p">
              <NumberInput value={paramsState.top_p} step={0.05} min={0} max={1} onChange={(value) => setParamsState({ ...paramsState, top_p: Number(value) })} />
            </Field>
            <Field label="Max tokens">
              <NumberInput value={paramsState.max_tokens} step={16} min={1} max={4096} onChange={(value) => setParamsState({ ...paramsState, max_tokens: Number(value) })} />
            </Field>
            <Field label="Seed" hint="Empty means a fresh seed per call.">
              <TextInput
                value={String(paramsState.seed)}
                onChange={(event) => setParamsState({ ...paramsState, seed: event.target.value })}
                placeholder="random"
              />
            </Field>
          </div>
          <Toggle
            checked={blind}
            onChange={setBlind}
            label="Blind comparison"
            hint="Hides model names and metadata until you press Reveal, so the judgement is about the output."
          />
        </div>
      </Panel>

      <div className="grid gap-3 lg:grid-cols-2">
        {sidePanel("left", left)}
        {sidePanel("right", right)}
      </div>

      {speedup !== null ? (
        <Panel>
          <SectionHeader title="Measured difference" subtitle="Derived from the two real runs above." />
          <div className="grid gap-3 sm:grid-cols-3">
            <Stat
              label="Speed ratio (B / A)"
              value={`${speedup.toFixed(2)}×`}
              tone={speedup > 1 ? "ok" : "warn"}
              hint={speedup > 1 ? "B produced tokens faster" : "A produced tokens faster"}
            />
            <Stat
              label="Latency difference"
              value={
                left.latency !== null && right.latency !== null
                  ? `${(right.latency - left.latency).toFixed(2)} s`
                  : "—"
              }
            />
            <Stat
              label="Token difference"
              value={
                left.completionTokens !== null && right.completionTokens !== null
                  ? right.completionTokens - left.completionTokens
                  : "—"
              }
            />
          </div>
          <div className="mt-3 flex items-center gap-2 text-2xs text-ink-3">
            <GitCompare size={11} /> Ratios are only meaningful for the same prompt, settings and token budget.
          </div>
        </Panel>
      ) : null}

      {!models.length ? (
        <Callout tone="info" title="No models in the library yet">
          Import a base model or train one, then come back — the comparison runs against anything the engine can load.
        </Callout>
      ) : null}

      <Callout tone="info" title="No invented rankings">
        <span className="flex items-center gap-2">
          <FlaskConical size={12} /> This page never scores an answer for you. It reports the measured cost of producing
          it and leaves the judgement to you — which is what a blind comparison is for.
        </span>
      </Callout>
    </div>
  );
}
