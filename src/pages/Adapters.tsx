import { useMemo, useState } from "react";
import { CheckCircle2, Layers, Merge, Play, RefreshCw, ShieldAlert, Trash2, Upload } from "lucide-react";
import { api } from "../lib/api";
import { basename, number } from "../lib/format";
import { useApp } from "../state/app";
import { useRouter } from "../state/router";
import {
  Badge,
  Button,
  Callout,
  EmptyState,
  Field,
  KeyValue,
  Panel,
  SectionHeader,
  Stat,
  Table,
  Td,
  TextInput,
  Th,
  Toggle,
} from "../components/ui";
import { ErrorPanel, Loading, useEngine } from "./common";

export function AdaptersPage() {
  const { registry, settings, refreshRegistry, refreshJobs, toast, reportError } = useApp();
  const router = useRouter();
  const [selected, setSelected] = useState<string | null>(null);
  const [destination, setDestination] = useState("");
  const [mergeBase, setMergeBase] = useState("");
  const [keepSource, setKeepSource] = useState(true);
  const [busy, setBusy] = useState(false);
  const [mergeResult, setMergeResult] = useState<any>(null);
  const [exportJson, setExportJson] = useState(false);

  const scan = useEngine<any>("adapters.scan", { max_depth: 3 }, { timeout: 300_000 });
  const adapters: any[] = scan.data?.adapters ?? [];
  const current = adapters.find((entry) => entry.path === selected) ?? adapters[0] ?? null;

  const plan = useEngine<any>(
    "adapters.plan",
    { path: current?.path, base_model: mergeBase || undefined, destination: destination || undefined },
    { deps: [current?.path, mergeBase], auto: Boolean(current?.path), timeout: 300_000 },
  );

  const baseCandidates = useMemo(() => {
    const models = registry?.models ?? [];
    return models.filter((entry) => entry.kind !== "peft-adapter");
  }, [registry]);

  const defaultDestination = useMemo(
    () => (current && settings?.workspace ? `${settings.workspace}\\models\\${basename(current.path)}-merged` : ""),
    [current, settings?.workspace],
  );
  const effectiveDestination = destination || defaultDestination;

  const merge = async () => {
    if (!current?.path || !effectiveDestination) return;
    setBusy(true);
    setMergeResult(null);
    try {
      const job = await api.jobs.start({
        kind: "merge",
        base_model: current.path,
        merge_base: mergeBase || current.base_model || null,
        output_dir: effectiveDestination,
      });
      await refreshJobs();
      toast({
        title: "Merge started",
        body: `Job ${job.jobId} is writing the merged model to ${effectiveDestination}.`,
        tone: "info",
      });
      setMergeResult({ jobId: job.jobId, destination: effectiveDestination });
    } catch (error) {
      reportError(error, "Merging could not start");
    } finally {
      setBusy(false);
    }
  };

  const exportAdapter = async () => {
    if (!current?.path || !settings?.workspace) return;
    setBusy(true);
    try {
      const target = `${settings.workspace}\\exports\\${basename(current.path)}`;
      const result = await api.call<any>("models.import", { source: current.path, destination: target, copy: true });
      let cardNote = "";
      if (exportJson) {
        await api.call<any>("models.card", { path: result.imported, write: true });
        cardNote = " A model card was written into the exported folder.";
      }
      toast({
        title: "Adapter copied into exports",
        body: `${result.imported}.${cardNote}`,
        tone: "ok",
      });
      await api.shell.reveal(result.imported);
    } catch (error) {
      reportError(error, "Export failed");
    } finally {
      setBusy(false);
    }
  };

  const forget = async (path: string) => {
    if (!window.confirm(`Remove ${path} from the adapter list?\n\nOnly the listing entry is removed — the folder on disk is untouched.`)) return;
    try {
      const entries = registry?.models ?? [];
      const match = entries.find((entry) => entry.path === path);
      if (match) {
        await api.registry.remove("models", match.id);
        await refreshRegistry();
      }
      setSelected(null);
      await scan.reload();
      toast({ title: "Removed from the list", body: "Files were left alone.", tone: "info" });
    } catch (error) {
      reportError(error, "Could not update the list");
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Adapters</h1>
          <p className="mt-0.5 max-w-3xl text-xs text-ink-2">
            LoRA/PEFT adapters found in the workspace, described exactly as their{" "}
            <span className="font-mono text-2xs">adapter_config.json</span> declares them. Merging writes a new model
            folder and leaves both the adapter and the base model as they were.
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="subtle" icon={<RefreshCw size={12} />} loading={scan.loading} onClick={() => void scan.reload()}>
            Rescan
          </Button>
          <Button size="sm" variant="subtle" onClick={() => router.navigate("/training/new?method=lora")}>
            Train a new adapter
          </Button>
        </div>
      </div>

      {scan.error ? <ErrorPanel error={scan.error} onRetry={() => void scan.reload()} /> : null}
      {scan.loading && !adapters.length ? <Loading label="Looking for adapter_config.json…" lines={3} /> : null}

      <div className="grid gap-3 lg:grid-cols-[360px_1fr]">
        <Panel padded={false}>
          <div className="border-b border-line-soft p-2.5 text-2xs uppercase tracking-wide text-ink-3">
            {adapters.length} adapter{adapters.length === 1 ? "" : "s"} in {scan.data?.roots?.join(", ") || "the workspace"}
          </div>
          <div className="max-h-[520px] space-y-1 overflow-y-auto p-2">
            {!adapters.length && !scan.loading ? (
              <EmptyState
                title="No adapters found"
                hint="Train a LoRA/QLoRA run, or copy an adapter folder into the workspace and rescan."
              />
            ) : null}
            {adapters.map((entry) => (
              <button
                key={entry.path}
                onClick={() => setSelected(entry.path)}
                className={`w-full rounded-md border px-2.5 py-2 text-left transition-colors ${
                  current?.path === entry.path ? "border-accent/40 bg-surface-3" : "border-line-soft bg-surface-2 hover:bg-surface-3"
                }`}
              >
                <div className="flex items-center gap-2">
                  <Layers size={12} className="text-ink-3" />
                  <span className="min-w-0 flex-1 truncate text-xs">{entry.name}</span>
                  {entry.error ? <Badge tone="danger">unreadable</Badge> : <Badge tone="muted">r{entry.rank ?? "?"}</Badge>}
                </div>
                <div className="mt-1 truncate font-mono text-2xs text-ink-3">{entry.path}</div>
                {!entry.error ? (
                  <div className="mt-1 text-2xs text-ink-3">
                    {entry.size_human} · {entry.trainable_parameters ? `${number(entry.trainable_parameters)} adapter tensors` : "size unknown"}
                  </div>
                ) : (
                  <div className="mt-1 text-2xs text-danger">{entry.error.message}</div>
                )}
              </button>
            ))}
          </div>
        </Panel>

        {!current ? (
          <Panel>
            <EmptyState title="Select an adapter" hint="Its configuration, compatibility and merge options appear here." />
          </Panel>
        ) : (
          <div className="space-y-3">
            <Panel>
              <SectionHeader
                title={current.name}
                subtitle={current.path}
                actions={
                  <div className="flex gap-2">
                    <Button size="sm" variant="subtle" icon={<Play size={11} />} onClick={() => router.navigate(`/playground?model=${encodeURIComponent(current.path)}`)}>
                      Test in Playground
                    </Button>
                    <Button size="sm" variant="subtle" icon={<Upload size={11} />} loading={busy} onClick={() => void exportAdapter()}>
                      Copy to exports
                    </Button>
                    <Button size="sm" variant="subtle" icon={<Trash2 size={11} />} onClick={() => void forget(current.path)}>
                      Forget
                    </Button>
                  </div>
                }
              />
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <Stat label="Rank (r)" value={current.rank ?? "—"} />
                <Stat label="Alpha" value={current.alpha ?? "—"} hint={current.scaling ? `scaling ${current.scaling}` : undefined} />
                <Stat label="Dropout" value={current.dropout ?? "—"} />
                <Stat label="Type" value={current.peft_type ?? "—"} hint={current.task_type ?? undefined} />
              </div>
              <div className="mt-3">
                <KeyValue
                  items={[
                    ["Base model (declared)", current.base_model ?? "not recorded in adapter_config.json"],
                    ["Base model on disk", current.base_model_local ? "found" : "not found at the declared path"],
                    ["Target modules", (current.target_modules ?? []).join(", ") || "default"],
                    ["Modules to save", (current.modules_to_save ?? []).join(", ") || "none"],
                    ["Bias", current.bias ?? "none"],
                    ["rsLoRA", current.use_rslora ? "yes" : "no"],
                    ["Architecture", current.architecture ?? "—"],
                    ["Weights", `${(current.weight_files ?? []).length} file(s) · ${current.size_human}`],
                  ]}
                  columns={1}
                />
              </div>
            </Panel>

            <Panel>
              <SectionHeader title="Merge plan" subtitle="Compatibility is checked before anything is written." actions={<Badge tone={plan.data?.status === "Supported" ? "ok" : "warn"}>{plan.data?.status ?? "checking…"}</Badge>} />
              {plan.error ? <ErrorPanel error={plan.error} onRetry={() => void plan.reload()} /> : null}
              {plan.loading && !plan.data ? <Loading lines={2} /> : null}
              <div className="space-y-3">
                <div className="grid gap-3 sm:grid-cols-2">
                  <Field label="Base model for the merge" hint={current.base_model ? "Pre-filled from the adapter config." : "The adapter does not name a base model — choose one."}>
                    <select
                      className="field"
                      value={mergeBase}
                      onChange={(event) => setMergeBase(event.target.value)}
                    >
                      <option value="">{current.base_model ? `declared: ${basename(current.base_model)}` : "Select a base model…"}</option>
                      {baseCandidates.map((entry) => (
                        <option key={entry.id} value={entry.path}>
                          {entry.name}
                        </option>
                      ))}
                    </select>
                  </Field>
                  <Field label="Output folder" hint="A new folder; nothing existing is overwritten.">
                    <div className="flex gap-2">
                      <TextInput value={effectiveDestination} readOnly placeholder="set a workspace folder first" />
                      <Button
                        size="sm"
                        onClick={async () => {
                          const folder = await api.dialog.pickFolder({ title: "Choose the merged model folder" });
                          if (folder) setDestination(folder);
                        }}
                      >
                        Choose
                      </Button>
                    </div>
                  </Field>
                </div>

                {plan.data?.checks?.length ? (
                  <div className="space-y-2">
                    {plan.data.checks.map((check: any) => (
                      <div key={check.name} className="flex items-start gap-2 rounded border border-line-soft bg-surface-2 px-2.5 py-2">
                        {check.status === "Supported" ? (
                          <CheckCircle2 size={12} className="mt-0.5 text-ok" />
                        ) : (
                          <ShieldAlert size={12} className="mt-0.5 text-warn" />
                        )}
                        <span className="min-w-0 flex-1">
                          <span className="block text-xs">{check.name}</span>
                          <span className="block text-2xs text-ink-2">{check.message}</span>
                          {check.hint ? <span className="block text-2xs text-ink-3">→ {check.hint}</span> : null}
                        </span>
                        <Badge tone={check.status === "Supported" ? "ok" : "warn"}>{check.status}</Badge>
                      </div>
                    ))}
                  </div>
                ) : null}

                <Toggle
                  checked={keepSource}
                  onChange={setKeepSource}
                  label="Keep the adapter and base model as they are"
                  hint="Always on in this build: merging never modifies its inputs, so an adapter can be merged again with different settings."
                />

                <Button
                  variant="primary"
                  icon={<Merge size={12} />}
                  loading={busy}
                  disabled={!plan.data?.backend_available || plan.data?.status === "Unsupported"}
                  onClick={() => void merge()}
                >
                  Merge adapter into a new model
                </Button>
                {!plan.data?.backend_available ? (
                  <div className="text-2xs text-warn">
                    Merging needs the PyTorch runtime (transformers + peft + torch). Install it from the Environment page.
                  </div>
                ) : null}

                {mergeResult ? (
                  <Callout tone="info" title={`Merge job ${mergeResult.jobId} running`}>
                    The merged model is written to {mergeResult.destination}. Follow its progress in the Jobs page; when it
                    finishes, the model can be tested, converted or quantized like any other.
                  </Callout>
                ) : null}
              </div>
            </Panel>

            <Panel>
              <SectionHeader title="Adapter configuration (raw)" subtitle="Exactly what the file contains — no interpretation." />
              <Table>
                <thead>
                  <tr>
                    <Th>Key</Th>
                    <Th>Value</Th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(current.raw_config ?? {}).map(([key, value]) => (
                    <tr key={key}>
                      <Td>
                        <span className="font-mono text-2xs">{key}</span>
                      </Td>
                      <Td>
                        <span className="font-mono text-2xs">{typeof value === "object" ? JSON.stringify(value) : String(value)}</span>
                      </Td>
                    </tr>
                  ))}
                </tbody>
              </Table>
              <div className="mt-3">
                <Toggle
                  checked={exportJson}
                  onChange={setExportJson}
                  label="Also write a model card when copying to exports"
                  hint="The card records base model, method and configuration from this adapter's own files."
                />
              </div>
            </Panel>
          </div>
        )}
      </div>

      <Callout tone="info" title="Why adapters are separate">
        An adapter is a small set of trained matrices that must be paired with its base model. The engine keeps them
        separate so you can train several adapters over one base model, compare them, and merge only the ones that earn it.
      </Callout>
    </div>
  );
}
