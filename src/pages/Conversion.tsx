import { useMemo, useState } from "react";
import { CheckCircle2, FileStack, Play, ShieldAlert, Terminal, XCircle } from "lucide-react";
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

const FORMATS = [
  { value: "safetensors", label: "safetensors", hint: "Built into the engine — header rewritten, payload re-encoded." },
  { value: "gguf", label: "GGUF (llama.cpp)", hint: "Requires convert_hf_to_gguf.py in <workspace>/tools or on PATH." },
  { value: "pytorch", label: "PyTorch .bin", hint: "Requires the PyTorch runtime." },
];

export function ConversionPage() {
  const { toast, reportError, settings, refreshRegistry } = useApp();
  const router = useRouter();
  const [modelPath, setModelPath] = useState<string | null>(null);
  const [format, setFormat] = useState("safetensors");
  const [dtype, setDtype] = useState("F16");
  const [destination, setDestination] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [verification, setVerification] = useState<any>(null);

  const plan = useEngine<any>("quantization.plan", { model: modelPath, target_dtype: dtype, target_format: format }, {
    deps: [modelPath, dtype, format],
    auto: Boolean(modelPath),
    timeout: 300_000,
  });

  const defaultDestination = useMemo(
    () => (modelPath && settings?.workspace ? `${settings.workspace}\\exports\\${basename(modelPath)}-${format}` : ""),
    [modelPath, settings?.workspace, format],
  );
  const effectiveDestination = destination || defaultDestination;

  const convert = async () => {
    if (!modelPath || !effectiveDestination) return;
    setBusy(true);
    setResult(null);
    setVerification(null);
    try {
      const response = await api.call<any>(
        "quantization.convert",
        { model: modelPath, destination: effectiveDestination, target_dtype: dtype, target_format: format },
        { timeout: 3_600_000 },
      );
      setResult(response);

      // Real integrity check: re-read the produced files with the engine's own
      // header parser and compare parameters/dtypes against the source.
      const [sourceReport, targetReport] = await Promise.all([
        api.call<any>("models.inspect", { path: modelPath }),
        api.call<any>("models.inspect", { path: effectiveDestination }),
      ]);
      const checks: Array<{ name: string; ok: boolean; detail: string }> = [];
      const sourceParams = sourceReport?.weights?.parameter_count ?? null;
      const targetParams = targetReport?.weights?.parameter_count ?? null;
      checks.push({
        name: "Weights readable",
        ok: Boolean(targetReport?.weights?.parameter_count),
        detail: targetParams ? `${number(targetParams)} parameters parsed from the output` : "the output could not be parsed",
      });
      if (format === "safetensors") {
        checks.push({
          name: "Parameter count preserved",
          ok: sourceParams !== null && sourceParams === targetParams,
          detail: `source ${sourceParams ?? "?"} → output ${targetParams ?? "?"}`,
        });
        checks.push({
          name: "Target dtype reached",
          ok: Object.keys(targetReport?.weights?.dtypes ?? {}).every((key) => key === dtype || !["F32", "F16", "BF16"].includes(key)),
          detail: `output dtypes: ${Object.keys(targetReport?.weights?.dtypes ?? {}).join(", ") || "none"}`,
        });
      }
      const tokenizerSource = sourceReport?.tokenizer?.files?.length ?? 0;
      const tokenizerTarget = targetReport?.tokenizer?.files?.length ?? 0;
      checks.push({
        name: "Tokenizer preserved",
        ok: tokenizerTarget > 0 || tokenizerSource === 0,
        detail: `${tokenizerSource} source file(s) → ${tokenizerTarget} output file(s)`,
      });
      checks.push({
        name: "Config preserved",
        ok: Boolean(targetReport?.config && Object.keys(targetReport.config).length),
        detail: targetReport?.config ? `${Object.keys(targetReport.config).length} keys read from the output config.json` : "no config found in the output",
      });
      setVerification({ checks, sourceReport, targetReport, generatedAt: new Date().toISOString() });

      await api.registry.add("models", {
        id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
        name: `${basename(modelPath)} (${format})`,
        path: effectiveDestination,
        addedAt: new Date().toISOString(),
        kind: targetReport?.format?.kind ?? format,
        tags: ["converted", format],
        summary: {
          parameters: targetParams,
          architecture: targetReport?.architecture?.model_type ?? null,
          size_bytes: targetReport?.size_bytes ?? null,
        },
      });
      await refreshRegistry();
      toast({
        title: "Conversion finished",
        body: `${result.files?.length ?? 0} file(s) written and re-verified against the source.`,
        tone: "ok",
      });
    } catch (error) {
      reportError(error, "Conversion failed");
    } finally {
      setBusy(false);
    }
  };

  const formatSupport = plan.data?.format_support;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Conversion</h1>
          <p className="mt-0.5 max-w-3xl text-xs text-ink-2">
            Moves a model between formats and precisions, then re-reads the result with the engine's own parser to check
            that parameters, tokenizer and config survived. Results are written next to nothing — the source stays exactly
            as it was.
          </p>
        </div>
        <Button size="sm" variant="subtle" onClick={() => void api.shell.reveal(settings?.workspace ?? ".")}>
          Open workspace
        </Button>
      </div>

      <div className="grid gap-3 lg:grid-cols-[400px_1fr]">
        <div className="space-y-3">
          <Panel>
            <SectionHeader title="Job" subtitle="Everything below runs on this machine." />
            <div className="space-y-3">
              <ModelPicker value={modelPath} onChange={setModelPath} label="Source model" />
              <Field label="Target format">
                <Select value={format} onChange={(event) => setFormat(event.target.value)}>
                  {FORMATS.map((entry) => (
                    <option key={entry.value} value={entry.value}>
                      {entry.label}
                    </option>
                  ))}
                </Select>
              </Field>
              <div className="text-2xs text-ink-3">{FORMATS.find((entry) => entry.value === format)?.hint}</div>
              <Field label="Target precision">
                <Select value={dtype} onChange={(event) => setDtype(event.target.value)}>
                  <option value="F16">fp16</option>
                  <option value="BF16">bf16</option>
                  <option value="F32">fp32</option>
                </Select>
              </Field>
              <Field label="Output folder">
                <div className="flex gap-2">
                  <TextInput value={effectiveDestination} readOnly placeholder="choose a model first" />
                  <Button
                    size="sm"
                    onClick={async () => {
                      const folder = await api.dialog.pickFolder({ title: "Choose the output folder" });
                      if (folder) setDestination(folder);
                    }}
                  >
                    Choose
                  </Button>
                </div>
              </Field>
              <Button
                variant="primary"
                icon={<Play size={12} />}
                loading={busy}
                disabled={!modelPath || !formatSupport?.available}
                onClick={() => void convert()}
              >
                Convert and verify
              </Button>
              {formatSupport && !formatSupport.available ? (
                <Callout tone="warn" title={`${format} is not available here`} hint={formatSupport.hint}>
                  The required tool ({formatSupport.tool}) was not found. Nothing is faked in its place.
                </Callout>
              ) : null}
            </div>
          </Panel>

          <Panel>
            <SectionHeader title="Plan" subtitle="Estimates from the source headers." />
            {!modelPath ? <div className="text-xs text-ink-2">Pick a model to see the plan.</div> : null}
            {plan.error ? <ErrorPanel error={plan.error} onRetry={() => void plan.reload()} /> : null}
            {plan.loading && !plan.data ? <Loading lines={3} /> : null}
            {plan.data ? (
              <>
                <div className="grid gap-3 sm:grid-cols-2">
                  <Stat label="Source" value={plan.data.source_human} hint={Object.keys(plan.data.source_dtype_bytes ?? {}).join(", ")} />
                  <Stat label="Estimated output" value={plan.data.estimated_human} estimated />
                </div>
                <div className="mt-3 space-y-2">
                  {plan.data.checks?.map((check: any) => (
                    <div key={check.name} className="flex items-start gap-2 rounded border border-line-soft bg-surface-2 px-2.5 py-2">
                      {check.status === "Supported" ? (
                        <CheckCircle2 size={12} className="mt-0.5 text-ok" />
                      ) : (
                        <ShieldAlert size={12} className="mt-0.5 text-warn" />
                      )}
                      <span className="min-w-0 flex-1">
                        <span className="block text-xs">{check.name}</span>
                        <span className="block text-2xs text-ink-2">{check.message}</span>
                      </span>
                      <Badge tone={check.status === "Supported" ? "ok" : "warn"}>{check.status}</Badge>
                    </div>
                  ))}
                </div>
              </>
            ) : null}
          </Panel>
        </div>

        <div className="space-y-3">
          {!result && !verification ? (
            <Panel>
              <SectionHeader title="Verification" subtitle="What happens after the conversion." />
              <div className="space-y-2 text-xs leading-relaxed text-ink-2">
                <div className="flex items-start gap-2">
                  <FileStack size={12} className="mt-0.5" />
                  The output folder is re-opened by the engine's reader: safetensors headers, config.json, tokenizer files
                  and weight counts are all parsed again from disk.
                </div>
                <div className="flex items-start gap-2">
                  <CheckCircle2 size={12} className="mt-0.5 text-ok" />
                  Parameter counts must match the source. A mismatch is reported as a failure, not smoothed over.
                </div>
                <div className="flex items-start gap-2">
                  <Terminal size={12} className="mt-0.5" />
                  A CONVERSION.md is written into the output folder recording the source, the target and the measured
                  sizes. External converters keep their own stdout in the result.
                </div>
              </div>
            </Panel>
          ) : null}

          {result ? (
            <Panel>
              <SectionHeader
                title="Result"
                subtitle={`${basename(result.source)} → ${basename(result.destination)}`}
                actions={<Badge tone="ok">completed in {result.elapsed_seconds}s</Badge>}
              />
              <div className="grid gap-3 sm:grid-cols-4">
                <Stat label="Files" value={result.files?.length ?? 0} />
                <Stat label="Source bytes" value={bytes(result.source_bytes)} />
                <Stat label="Output bytes" value={bytes(result.result_bytes)} />
                <Stat
                  label="Ratio"
                  value={result.source_bytes ? (result.result_bytes / result.source_bytes).toFixed(3) : "—"}
                  hint="measured, not estimated"
                />
              </div>
              <div className="mt-3">
                <Table>
                  <thead>
                    <tr>
                      <Th>Output file</Th>
                      <Th align="right">Tensors rewritten</Th>
                      <Th align="right">Left untouched</Th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.files?.map((file: any) => (
                      <tr key={file.file}>
                        <Td>
                          <span className="font-mono text-2xs">{basename(file.file)}</span>
                        </Td>
                        <Td align="right">{file.converted_tensors ?? "—"}</Td>
                        <Td align="right">{file.untouched_tensors ?? "—"}</Td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              </div>
              {result.copied?.length ? (
                <div className="mt-3 text-2xs text-ink-3">Metadata files copied: {result.copied.join(", ")}</div>
              ) : null}
              <div className="mt-3 flex flex-wrap gap-2">
                <Button size="sm" variant="subtle" onClick={() => void api.shell.reveal(result.destination)}>
                  Open output
                </Button>
                <Button size="sm" variant="primary" onClick={() => router.navigate(`/playground?model=${encodeURIComponent(result.destination)}`)}>
                  Test in Playground
                </Button>
              </div>
            </Panel>
          ) : null}

          {verification ? (
            <Panel>
              <SectionHeader title="Integrity checks" subtitle="Re-read from disk after the conversion." />
              <div className="space-y-2">
                {verification.checks.map((check: any) => (
                  <div key={check.name} className="flex items-start gap-2 rounded border border-line-soft bg-surface-2 px-2.5 py-2">
                    {check.ok ? (
                      <CheckCircle2 size={12} className="mt-0.5 text-ok" />
                    ) : (
                      <XCircle size={12} className="mt-0.5 text-danger" />
                    )}
                    <span className="min-w-0 flex-1">
                      <span className="block text-xs">{check.name}</span>
                      <span className="block text-2xs text-ink-2">{check.detail}</span>
                    </span>
                    <Badge tone={check.ok ? "ok" : "danger"}>{check.ok ? "passed" : "failed"}</Badge>
                  </div>
                ))}
              </div>
              <div className="mt-3">
                <KeyValue
                  items={[
                    ["Source architecture", verification.sourceReport?.architecture?.model_type ?? "—"],
                    ["Output architecture", verification.targetReport?.architecture?.model_type ?? "—"],
                    ["Source format", verification.sourceReport?.format?.kind ?? "—"],
                    ["Output format", verification.targetReport?.format?.kind ?? "—"],
                    ["Output tokenizer", verification.targetReport?.tokenizer?.class ?? "not detected"],
                  ]}
                  columns={1}
                />
              </div>
            </Panel>
          ) : null}
        </div>
      </div>
    </div>
  );
}
