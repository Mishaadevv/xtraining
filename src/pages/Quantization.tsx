import { useMemo, useState } from "react";
import { AlertTriangle, Cpu, Gauge, HardDrive, Package, Play, ShieldCheck, Sparkles } from "lucide-react";
import { api } from "../lib/api";
import { basename, bytes, number } from "../lib/format";
import { useApp } from "../state/app";
import { useRouter } from "../state/router";
import {
  Badge,
  Button,
  Callout,
  Field,
  KeyValue,
  Panel,
  SectionHeader,
  Select,
  Stat,
  Table,
  Td,
  TextInput,
  Th,
} from "../components/ui";
import { ErrorPanel, Loading, ModelPicker, useEngine } from "./common";

const TARGETS = [
  { value: "F16", label: "fp16 — half precision", mode: "native", hint: "Built-in converter, no dependencies." },
  { value: "BF16", label: "bf16 — truncated mantissa", mode: "native", hint: "Same size as fp16, wider exponent range." },
  { value: "F32", label: "fp32 — full precision", mode: "native", hint: "Use this to widen a checkpoint back up." },
  { value: "INT8", label: "int8 — bitsandbytes", mode: "bnb", hint: "Needs bitsandbytes; weights plus scales." },
  { value: "INT4", label: "int4 (NF4) — bitsandbytes", mode: "bnb", hint: "Needs bitsandbytes; smallest usable size." },
];

export function QuantizationPage() {
  const { registry, refreshRegistry, refreshJobs, toast, reportError, settings } = useApp();
  const router = useRouter();
  const [modelPath, setModelPath] = useState<string | null>(null);
  const [target, setTarget] = useState("F16");
  const [destination, setDestination] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<any>(null);

  const selection = TARGETS.find((entry) => entry.value === target) ?? TARGETS[0];
  const plan = useEngine<any>("quantization.plan", { model: modelPath, target_dtype: target, target_format: "safetensors" }, {
    deps: [modelPath, target],
    auto: Boolean(modelPath),
    timeout: 300_000,
  });
  const tools = useEngine<any>("tools.report", {}, { timeout: 120_000 });

  const defaultDestination = useMemo(() => {
    if (!modelPath || !settings?.workspace) return "";
    return `${settings.workspace}\\models\\${basename(modelPath)}-${target.toLowerCase()}`;
  }, [modelPath, settings?.workspace, target]);

  const effectiveDestination = destination || defaultDestination;
  const bnbAvailable = Boolean(tools.data?.packages?.bitsandbytes?.available);

  const runNative = async () => {
    if (!modelPath || !effectiveDestination) return;
    setBusy(true);
    setResult(null);
    try {
      const response = await api.call<any>(
        "quantization.convert",
        {
          model: modelPath,
          destination: effectiveDestination,
          target_dtype: target,
          target_format: "safetensors",
        },
        { timeout: 3_600_000 },
      );
      setResult(response);
      const inspection = await api.call<any>("models.inspect", { path: effectiveDestination });
      await api.registry.add("models", {
        id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
        name: `${basename(modelPath)} (${target.toLowerCase()})`,
        path: effectiveDestination,
        addedAt: new Date().toISOString(),
        kind: inspection?.format?.kind ?? "transformers-safetensors",
        tags: ["converted", target.toLowerCase()],
        summary: {
          parameters: inspection?.weights?.parameter_count ?? null,
          architecture: inspection?.architecture?.model_type ?? null,
          size_bytes: inspection?.size_bytes ?? null,
          quantization: target === "F32" ? "none" : target.toLowerCase(),
        },
      });
      await refreshRegistry();
      toast({
        title: `Converted to ${target}`,
        body: `${response.files?.length ?? 0} file(s) written · ${bytes(response.result_bytes)} measured · source untouched.`,
        tone: "ok",
      });
    } catch (error) {
      reportError(error, "Conversion failed");
    } finally {
      setBusy(false);
    }
  };

  const runBitsandbytes = async () => {
    if (!modelPath || !effectiveDestination) return;
    setBusy(true);
    try {
      const bits = target === "INT4" ? 4 : 8;
      const job = await api.jobs.start({
        kind: "quantize",
        base_model: modelPath,
        output_dir: effectiveDestination,
        bits,
        method: `bnb-${bits}bit`,
      });
      await refreshJobs();
      toast({
        title: `${bits}-bit quantization started`,
        body: `Job ${job.jobId} is running in the engine. Progress and logs are in the Jobs page.`,
        tone: "info",
      });
      router.navigate("/jobs");
    } catch (error) {
      reportError(error, "Quantization could not start");
    } finally {
      setBusy(false);
    }
  };

  const statusTone = (status: string) =>
    status === "Supported" ? "ok" : status === "Supported with limitations" ? "warn" : status === "Experimental" ? "warn" : "danger";

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Quantization</h1>
          <p className="mt-0.5 max-w-3xl text-xs text-ink-2">
            Two genuinely different operations live here. dtype conversion (fp32 → fp16/bf16) is done by the engine's own
            safetensors writer and works with no dependencies. int8/int4 weights need bitsandbytes, and the page says so
            when it is missing instead of pretending.
          </p>
        </div>
        <Button size="sm" variant="subtle" icon={<Gauge size={12} />} onClick={() => router.navigate("/conversion")}>
          Format conversion
        </Button>
      </div>

      <div className="grid gap-3 lg:grid-cols-[400px_1fr]">
        <div className="space-y-3">
          <Panel>
            <SectionHeader title="Source and target" subtitle="Estimates are labelled; measured sizes come from the real files." />
            <div className="space-y-3">
              <ModelPicker value={modelPath} onChange={setModelPath} label="Model to convert" />
              <Field label="Target precision">
                <Select value={target} onChange={(event) => setTarget(event.target.value)}>
                  {TARGETS.map((entry) => (
                    <option key={entry.value} value={entry.value}>
                      {entry.label}
                    </option>
                  ))}
                </Select>
              </Field>
              <div className="text-2xs text-ink-3">{selection.hint}</div>
              <Field label="Output folder" hint="The source model is never modified.">
                <div className="flex gap-2">
                  <TextInput value={effectiveDestination} readOnly placeholder="choose a model first" />
                  <Button
                    size="sm"
                    onClick={async () => {
                      const folder = await api.dialog.pickFolder({ title: "Choose the output folder", defaultPath: settings?.workspace ?? undefined });
                      if (folder) setDestination(folder);
                    }}
                  >
                    Choose
                  </Button>
                </div>
              </Field>

              {selection.mode === "native" ? (
                <Button variant="primary" icon={<Play size={12} />} loading={busy} disabled={!modelPath} onClick={() => void runNative()}>
                  Convert precision
                </Button>
              ) : (
                <div className="space-y-2">
                  <Button variant="primary" icon={<Play size={12} />} loading={busy} disabled={!modelPath || !bnbAvailable} onClick={() => void runBitsandbytes()}>
                    Run {target.toLowerCase()} quantization
                  </Button>
                  {!bnbAvailable ? (
                    <div className="text-2xs text-warn">
                      bitsandbytes is not installed. Install the ML runtime from the Environment page, or pick fp16/bf16.
                    </div>
                  ) : null}
                </div>
              )}
            </div>
          </Panel>

          <Panel>
            <SectionHeader title="Tools on this machine" subtitle="Checked right now, not assumed." />
            {tools.error ? <ErrorPanel error={tools.error} onRetry={() => void tools.reload()} /> : null}
            {tools.data ? (
              <div className="space-y-2">
                {Object.entries(tools.data.packages ?? {}).map(([name, info]: [string, any]) => (
                  <div key={name} className="flex items-start gap-2 rounded border border-line-soft bg-surface-2 px-2 py-1.5">
                    <Package size={11} className={info.available ? "mt-0.5 text-ok" : "mt-0.5 text-ink-3"} />
                    <span className="min-w-0 flex-1">
                      <span className="flex items-center gap-2">
                        <span className="text-xs">{name}</span>
                        <Badge tone={info.available ? "ok" : "muted"}>{info.available ? "available" : "missing"}</Badge>
                      </span>
                      <span className="mt-0.5 block text-2xs text-ink-3">{info.used_for}</span>
                      {!info.available && info.install ? (
                        <span className="mt-0.5 block font-mono text-2xs text-ink-3">{info.install}</span>
                      ) : null}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <Loading lines={3} />
            )}
          </Panel>
        </div>

        <div className="space-y-3">
          {!modelPath ? (
            <Panel>
              <SectionHeader title="Pick a model" subtitle="Choose a safetensors checkpoint from the library to see the plan." />
              {registry?.models?.length ? (
                <Table>
                  <thead>
                    <tr>
                      <Th>Model</Th>
                      <Th>Format</Th>
                      <Th align="right">Parameters</Th>
                      <Th align="right">On disk</Th>
                    </tr>
                  </thead>
                  <tbody>
                    {registry.models.slice(0, 12).map((entry) => (
                      <tr key={entry.id}>
                        <Td>
                          <button className="text-left hover:text-accent" onClick={() => setModelPath(entry.path)}>
                            {entry.name}
                          </button>
                        </Td>
                        <Td>{entry.kind ?? "—"}</Td>
                        <Td align="right">{entry.summary?.parameters ? number(Number(entry.summary.parameters)) : "—"}</Td>
                        <Td align="right">{entry.summary?.size_bytes ? bytes(Number(entry.summary.size_bytes)) : "—"}</Td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              ) : (
                <Callout tone="info" title="No models in the library">
                  Import a checkpoint first, or train one — then quantize the result.
                </Callout>
              )}
            </Panel>
          ) : null}

          {plan.error ? <ErrorPanel error={plan.error} onRetry={() => void plan.reload()} /> : null}
          {plan.loading && !plan.data ? <Loading label="Inspecting the weights…" lines={4} /> : null}

          {plan.data ? (
            <>
              <Panel>
                <SectionHeader
                  title="Plan"
                  subtitle={`${basename(plan.data.source)} → ${plan.data.target_dtype} (${plan.data.target_format})`}
                  actions={<Badge tone={statusTone(plan.data.status)}>{plan.data.status}</Badge>}
                />
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                  <Stat label="Source precision" value={Object.keys(plan.data.source_dtype_bytes ?? {}).join(", ") || "—"} />
                  <Stat label="Source size" value={plan.data.source_human} />
                  <Stat label="Estimated output" value={plan.data.estimated_human} estimated hint={`ratio ${plan.data.ratio ?? "—"}`} />
                  <Stat label="Parameters" value={plan.data.parameter_count ? number(plan.data.parameter_count) : "—"} />
                </div>

                <div className="mt-3 space-y-2">
                  {plan.data.checks?.map((check: any) => (
                    <div key={check.name} className="rounded border border-line-soft bg-surface-2 px-2.5 py-2">
                      <div className="flex items-center gap-2">
                        <Badge tone={statusTone(check.status)}>{check.status}</Badge>
                        <span className="text-xs">{check.name}</span>
                      </div>
                      <div className="mt-1 text-2xs text-ink-2">{check.message}</div>
                      {check.hint ? <div className="mt-0.5 text-2xs text-ink-3">→ {check.hint}</div> : null}
                    </div>
                  ))}
                </div>

                <div className="mt-3">
                  <KeyValue
                    items={[
                      ["Output format", plan.data.format_support?.tool ?? "—"],
                      ["Format status", plan.data.format_support?.status ?? "—"],
                      ["Metadata preserved", plan.data.metadata_preserved ? "yes (safetensors __metadata__)" : "no"],
                      ["Source modified", "never — conversion writes a new folder"],
                    ]}
                    columns={1}
                  />
                </div>
                {plan.data.notes?.length ? (
                  <div className="mt-3 space-y-1">
                    {plan.data.notes.map((note: string) => (
                      <div key={note} className="flex items-start gap-2 text-2xs text-ink-3">
                        <AlertTriangle size={10} className="mt-0.5" />
                        {note}
                      </div>
                    ))}
                  </div>
                ) : null}
              </Panel>

              {result ? (
                <Panel>
                  <SectionHeader
                    title="Result"
                    subtitle="Measured after the conversion finished."
                    actions={<Badge tone="ok">completed</Badge>}
                  />
                  <div className="grid gap-3 sm:grid-cols-3">
                    <Stat label="Result size (measured)" value={bytes(result.result_bytes)} tone="ok" />
                    <Stat label="Files written" value={result.files?.length ?? 0} />
                    <Stat label="Elapsed" value={`${result.elapsed_seconds}s`} />
                  </div>
                  <div className="mt-3">
                    <Table>
                      <thead>
                        <tr>
                          <Th>File</Th>
                          <Th align="right">Source</Th>
                          <Th align="right">Result</Th>
                          <Th align="right">Tensors rewritten</Th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.files?.map((file: any) => (
                          <tr key={file.file}>
                            <Td>
                              <span className="font-mono text-2xs">{basename(file.file)}</span>
                            </Td>
                            <Td align="right">{bytes(file.source_bytes)}</Td>
                            <Td align="right">{bytes(file.bytes)}</Td>
                            <Td align="right">{file.converted_tensors ?? "—"}</Td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  </div>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <Button size="sm" variant="subtle" icon={<HardDrive size={11} />} onClick={() => void api.shell.reveal(result.destination)}>
                      Open output folder
                    </Button>
                    <Button size="sm" variant="subtle" onClick={() => router.navigate(`/models/${encodeURIComponent(basename(result.destination))}`)}>
                      Inspect in Models
                    </Button>
                    <Button size="sm" variant="primary" onClick={() => router.navigate(`/playground?model=${encodeURIComponent(result.destination)}`)}>
                      Test in Playground
                    </Button>
                  </div>
                  {result.copied?.length ? (
                    <div className="mt-3 text-2xs text-ink-3">Copied alongside the weights: {result.copied.join(", ")}</div>
                  ) : null}
                </Panel>
              ) : null}

              <Panel>
                <SectionHeader title="What the numbers mean" subtitle="How this page avoids inventing anything." />
                <div className="space-y-2 text-xs leading-relaxed text-ink-2">
                  <div className="flex items-start gap-2">
                    <ShieldCheck size={12} className="mt-0.5 text-ok" />
                    Source size, parameter counts and dtypes come from the safetensors headers — no model is loaded into
                    memory to read them.
                  </div>
                  <div className="flex items-start gap-2">
                    <Gauge size={12} className="mt-0.5 text-warn" />
                    The estimated output size assumes every float tensor moves to the target dtype. The measured size
                    after a conversion is shown separately.
                  </div>
                  <div className="flex items-start gap-2">
                    <Cpu size={12} className="mt-0.5 text-info" />
                    Speed effects are not predicted here: run the Playground or Compare page against both versions and the
                    engine will report real tokens per second.
                  </div>
                  <div className="flex items-start gap-2">
                    <Sparkles size={12} className="mt-0.5 text-accent" />
                    bf16 truncates the low mantissa bits — that is what bf16 is. Quality differences are yours to measure,
                    not ours to claim.
                  </div>
                </div>
              </Panel>
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
}
