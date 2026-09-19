import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  Brain,
  CheckCircle2,
  Database,
  Gauge,
  Play,
  Rocket,
  Settings2,
  Sparkles,
  Sliders,
  Wand2,
} from "lucide-react";

import { PageBody, PageHeader } from "@/components/layout/Shell";
import {
  Badge,
  Button,
  Dot,
  EmptyState,
  Field,
  Input,
  KeyValue,
  Note,
  Panel,
  PanelHeader,
  ProgressBar,
  Segmented,
  Select,
  Stat,
  Switch,
  Textarea,
} from "@/components/ui/primitives";
import type { Method, Precision, Quantization, TrainingConfig } from "@/lib/types";
import { cn, formatBytes, formatCount } from "@/lib/utils";
import { useStore } from "@/state/store";
import {
  addHfDataset,
  appStore,
  navigate,
  pickAndImportDatasets,
  refreshWizardEstimate,
  resetWizard,
  runAutoConfig,
  setWizard,
  startTraining,
  updateWizardConfig,
  validateDataset,
  wizardSelectDataset,
  wizardSelectMethod,
  wizardSelectModel,
} from "@/state/appStore";

const STEPS = [
  { id: 0, label: "Model", icon: <Brain className="h-3.5 w-3.5" /> },
  { id: 1, label: "Dataset", icon: <Database className="h-3.5 w-3.5" /> },
  { id: 2, label: "Method", icon: <Sliders className="h-3.5 w-3.5" /> },
  { id: 3, label: "Settings", icon: <Settings2 className="h-3.5 w-3.5" /> },
  { id: 4, label: "Check", icon: <Gauge className="h-3.5 w-3.5" /> },
  { id: 5, label: "Train", icon: <Rocket className="h-3.5 w-3.5" /> },
];

const METHOD_BLURB: Record<Method, { label: string; summary: string; detail: string }> = {
  lora: {
    label: "LoRA",
    summary: "Train small adapters, keep the base model frozen",
    detail: "The default choice. Produces a few megabytes instead of gigabytes, trains fastest, and rarely destabilises the base model.",
  },
  qlora: {
    label: "QLoRA",
    summary: "LoRA over a 4-bit quantized base model",
    detail: "Fits models that would otherwise run out of VRAM. Needs bitsandbytes and an NVIDIA GPU. Costs some speed per step.",
  },
  sft: {
    label: "SFT",
    summary: "Instruction tuning over chat-formatted data",
    detail: "Uses the same adapter engine but applies the model's chat template to message-style datasets, teaching assistant behaviour.",
  },
  full: {
    label: "Full fine-tune",
    summary: "Update every weight of the base model",
    detail: "Highest capacity and highest cost. Optimizer state alone needs roughly two extra fp32 copies of the model.",
  },
  scratch: {
    label: "From scratch",
    summary: "Train a fresh model from random weights — no base model",
    detail: "Builds a small transformer and trains it on your dataset only. Nothing is downloaded. Needs much more data than fine-tuning to become good; best for tiny domain models and experiments.",
  },
};

function StepNav({ current, onSelect }: { current: number; onSelect: (step: number) => void }) {
  return (
    <div className="mb-4 flex flex-wrap items-center gap-1.5">
      {STEPS.map((step, index) => {
        const active = current === step.id;
        const done = current > step.id;
        return (
          <div key={step.id} className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => onSelect(step.id)}
              className={cn(
                "flex items-center gap-1.5 rounded-[9px] border px-2.5 py-1.5 text-[12px] font-medium transition-colors",
                active
                  ? "border-[var(--border)] bg-[var(--panel-2)] text-[var(--text)]"
                  : done
                    ? "border-transparent text-[var(--text-2)] hover:bg-[var(--hover)]"
                    : "border-transparent text-[var(--text-3)] hover:bg-[var(--hover)] hover:text-[var(--text-2)]",
              )}
            >
              <span className={cn(active ? "text-[var(--acc)]" : done ? "text-[var(--green)]" : "text-[var(--text-3)]")}>
                {done ? <CheckCircle2 className="h-3.5 w-3.5" /> : step.icon}
              </span>
              {step.label}
            </button>
            {index < STEPS.length - 1 ? <span className="text-[var(--text-3)]">›</span> : null}
          </div>
        );
      })}
    </div>
  );
}

function MethodCard({
  method,
  active,
  disabled,
  disabledReason,
  onSelect,
}: {
  method: Method;
  active: boolean;
  disabled?: boolean;
  disabledReason?: string;
  onSelect: () => void;
}) {
  const info = METHOD_BLURB[method];
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onSelect}
      className={cn(
        "rounded-[12px] border p-3 text-left transition-all duration-200",
        active
          ? "border-[var(--acc)] bg-[var(--acc-soft)]"
          : "border-[var(--border-soft)] bg-[var(--panel)] hover:border-[var(--border)] hover:bg-[var(--hover)]",
        disabled && "cursor-not-allowed opacity-50 hover:border-[var(--border-soft)] hover:bg-[var(--panel)]",
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-[13px] font-semibold">{info.label}</span>
        {active ? <Badge tone="accent">selected</Badge> : null}
      </div>
      <p className="mt-1 text-[12px] leading-[18px] text-[var(--text-2)]">{info.summary}</p>
      <p className="mt-1.5 text-[11.5px] leading-[17px] text-[var(--text-3)]">
        {disabled && disabledReason ? disabledReason : info.detail}
      </p>
    </button>
  );
}

export function NewTrainingPage() {
  const { wizard, env, datasets, busy, models, settings } = useStore(appStore);
  const autoConfigure = settings?.autoConfigure ?? true;
  const [hfId, setHfId] = useState("");
  const [hfSplit, setHfSplit] = useState("train");

  const addHubDataset = async () => {
    const entry = await addHfDataset(hfId, hfSplit);
    if (entry) {
      setHfId("");
      await wizardSelectDataset(entry.id);
    }
  };
  const methodAvailable = (method: Method): { ok: boolean; reason?: string } => {
    // From scratch runs on its own backend, which needs no peft.
    const backend = env.backends?.find((entry) => entry.name === (method === "scratch" ? "scratch" : "hf-peft"));
    if (backend && !backend.available) {
      return { ok: false, reason: `Needs: ${backend.missing.join(", ") || "the ML runtime"}. Install it in Settings → Environment.` };
    }
    if (method === "qlora" && !env.hardware?.cuda_ready) {
      return { ok: false, reason: "QLoRA needs an NVIDIA GPU with CUDA. This machine has no usable CUDA device." };
    }
    return { ok: true };
  };

  const config = wizard.config;
  const datasetEntry = datasets.find((dataset) => dataset.id === wizard.datasetId) ?? null;
  const datasetPath = datasetEntry?.path ?? null;
  const cudaReady = Boolean(env.hardware?.cuda_ready);
  const trainingReady = Boolean(env.dependencies?.training_ready);

  useEffect(() => {
    if (wizard.step === 3 && !config && (wizard.baseModel || wizard.datasetId)) {
      void runAutoConfig();
    }
  }, [wizard.step, config, wizard.baseModel, wizard.datasetId]);

  const blockers = useMemo(() => {
    const list: { title: string; detail: string; tone: "warn" | "bad" }[] = [];
    if (!trainingReady) {
      list.push({
        title: "The ML runtime is not installed",
        detail: "PyTorch and friends are missing, so no training can run yet. Settings → Environment generates the exact install command.",
        tone: "bad",
      });
    }
    // From scratch is the one method with no base model.
    if (!wizard.baseModel && wizard.method !== "scratch") {
      list.push({ title: "No base model selected", detail: "Pick a model in step 1.", tone: "bad" });
    }
    if (!datasetPath) {
      list.push({ title: "No dataset selected", detail: "Pick a dataset in step 2.", tone: "bad" });
    }
    if (wizard.datasetReport && wizard.datasetReport.status === "errors") {
      list.push({
        title: "The dataset has errors",
        detail: wizard.datasetReport.issues.find((issue) => issue.severity === "error")?.message ?? "",
        tone: "bad",
      });
    }
    const method = methodAvailable(wizard.method);
    if (!method.ok) {
      list.push({ title: `${METHOD_BLURB[wizard.method].label} is unavailable`, detail: method.reason ?? "", tone: "bad" });
    }
    if (wizard.estimate?.verdict === "exceeds") {
      list.push({
        title: "Estimated to exceed available VRAM",
        detail: wizard.estimate.suggestions?.[0] ?? "Lower the batch size or enable QLoRA.",
        tone: "bad",
      });
    } else if (wizard.estimate?.verdict === "tight") {
      list.push({
        title: "Memory will be tight",
        detail: wizard.estimate.suggestions?.[0] ?? "Consider lowering the batch size.",
        tone: "warn",
      });
    }
    if (!cudaReady && wizard.baseModel && trainingReady) {
      list.push({
        title: "Training on CPU",
        detail: env.hardware?.cuda_blockers?.[0] ?? "No CUDA device was found.",
        tone: "warn",
      });
    }
    return list;
  }, [trainingReady, wizard.baseModel, wizard.datasetId, wizard.datasetReport, wizard.method, wizard.estimate, cudaReady, datasetPath, env.hardware, env.backends]);

  const fatal = blockers.some((blocker) => blocker.tone === "bad");

  const step = wizard.step;
  const setStep = (next: number) => setWizard({ step: Math.max(0, Math.min(STEPS.length - 1, next)) });

  return (
    <>
      <PageHeader
        icon={<Play className="h-4 w-4" />}
        title="New training"
        subtitle="Model → Dataset → Method → Settings → Check → Train"
        actions={
          <>
            <Button size="sm" variant="ghost" onClick={() => resetWizard({ step: step, projectId: wizard.projectId, projectName: wizard.projectName })}>
              Reset choices
            </Button>
            <Button
              size="sm"
              variant="quiet"
              icon={<Wand2 className="h-3.5 w-3.5" />}
              loading={busy.autoConfig}
              disabled={!wizard.baseModel && !wizard.datasetId}
              onClick={() => void runAutoConfig()}
            >
              Auto-configure
            </Button>
          </>
        }
      />

      <PageBody wide>
        <StepNav current={step} onSelect={setStep} />

        {/* ---------------------------------------------------------- step 0 */}
        {step === 0 ? (
          <Panel>
            <PanelHeader
              icon={<Brain className="h-4 w-4" />}
              title="Choose a base model"
              description="Use a Hugging Face repository id or a local folder. Weights are only fetched when the run starts."
            />
            {models.length === 0 ? (
              <EmptyState
                icon={<Brain className="h-6 w-6" />}
                title="The model library is empty"
                description="Add a model from the Models screen, or type a repository id below."
                action={
                  <Button size="sm" onClick={() => navigate("models")}>
                    Open Models
                  </Button>
                }
              />
            ) : (
              <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
                {models
                  .filter((model) => !model.trained)
                  .map((model) => {
                    const active = wizard.baseModel === model.source;
                    return (
                      <button
                        key={model.id}
                        type="button"
                        onClick={() => void wizardSelectModel(model.source, model.id)}
                        className={cn(
                          "rounded-[12px] border p-3 text-left transition-colors",
                          active
                            ? "border-[var(--acc)] bg-[var(--acc-soft)]"
                            : "border-[var(--border-soft)] hover:border-[var(--border)] hover:bg-[var(--hover)]",
                        )}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="truncate text-[12.5px] font-medium">{model.name}</span>
                          {active ? <Badge tone="accent">selected</Badge> : null}
                        </div>
                        <p className="zq-mono mt-0.5 truncate text-[10.5px] text-[var(--text-3)]">{model.source}</p>
                        <p className="mt-1 text-[11px] text-[var(--text-3)]">
                          {model.params ? `${formatCount(model.params)} params` : "unknown size"} ·{" "}
                          {model.architecture ?? model.kind}
                        </p>
                      </button>
                    );
                  })}
              </div>
            )}

            {models.some((model) => model.trained) ? (
              <div className="mt-4">
                <p className="mb-2 flex items-center gap-1.5 text-[12px] font-medium text-[var(--text-2)]">
                  <Sparkles className="h-3.5 w-3.5" style={{ color: "var(--acc)" }} />
                  Continue training your own models
                </p>
                <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
                  {models
                    .filter((model) => model.trained && model.path)
                    .map((model) => {
                      const active = wizard.baseModel === model.path;
                      return (
                        <button
                          key={model.id}
                          type="button"
                          onClick={() => void wizardSelectModel(model.path as string, null)}
                          className={cn(
                            "rounded-[12px] border p-3 text-left transition-colors",
                            active
                              ? "border-[var(--acc)] bg-[var(--acc-soft)]"
                              : "border-[var(--border-soft)] hover:border-[var(--border)] hover:bg-[var(--hover)]",
                          )}
                        >
                          <div className="flex items-center justify-between gap-2">
                            <span className="truncate text-[12.5px] font-medium">{model.name}</span>
                            {active ? <Badge tone="accent">selected</Badge> : null}
                          </div>
                          <p className="mt-1 text-[11px] text-[var(--text-3)]">
                            {model.method?.toUpperCase() ?? "trained"} · loss {model.finalLoss?.toFixed(3) ?? "—"}
                            {model.adapter ? " · LoRA adapter — it is merged into the weights and a new one is trained on top" : " · full model"}
                          </p>
                        </button>
                      );
                    })}
                </div>
              </div>
            ) : null}

            <div className="mt-4 border-t border-[var(--border-soft)] pt-3">
              <Field
                label="Or enter a model source"
                hint="Hugging Face id (owner/name), an absolute path to a folder with config.json, or a trained model folder — a LoRA-adapter folder is merged into its base weights and training continues from there."
              >
                <div className="flex gap-2">
                  <Input
                    value={wizard.baseModel}
                    placeholder="Qwen/Qwen2.5-0.5B-Instruct"
                    onChange={(event) => setWizard({ baseModel: event.target.value })}
                    onBlur={() => {
                      if (wizard.baseModel) void wizardSelectModel(wizard.baseModel, wizard.modelEntryId);
                    }}
                  />
                  <Button
                    variant="secondary"
                    loading={busy.inspectModel}
                    disabled={!wizard.baseModel.trim()}
                    onClick={() => void wizardSelectModel(wizard.baseModel, wizard.modelEntryId)}
                  >
                    Inspect
                  </Button>
                  <Button variant="ghost" onClick={() => navigate("models")}>
                    Add to library
                  </Button>
                </div>
              </Field>

              {wizard.modelInfo ? (
                <div className="mt-3 grid grid-cols-2 gap-x-6 sm:grid-cols-4">
                  <KeyValue label="Parameters" value={wizard.modelInfo.params ? formatCount(Number(wizard.modelInfo.params)) : "unknown"} />
                  <KeyValue label="Architecture" value={(wizard.modelInfo.architecture as string) ?? (wizard.modelInfo.fields as any)?.architectures?.[0] ?? "—"} />
                  <KeyValue
                    label="Max context"
                    value={wizard.modelInfo.max_position_embeddings ? formatCount(Number(wizard.modelInfo.max_position_embeddings)) : "—"}
                  />
                  <KeyValue label="Stored dtype" value={((wizard.modelInfo.fields as any)?.torch_dtype as string) ?? "—"} />
                </div>
              ) : null}
              {wizard.autoIssue ? (
                <div className="mt-3">
                  <Note tone="warn" title="Could not inspect this model">
                    {wizard.autoIssue.message}
                  </Note>
                </div>
              ) : null}
            </div>
          </Panel>
        ) : null}

        {/* ---------------------------------------------------------- step 1 */}
        {step === 1 ? (
          <Panel>
            <PanelHeader
              icon={<Database className="h-4 w-4" />}
              title="Choose a dataset"
              description="JSON, JSONL, CSV, TXT, Parquet, a folder of shards, or a dataset from the Hugging Face Hub."
              actions={
                <Button size="sm" variant="quiet" loading={busy.importDataset} onClick={() => void pickAndImportDatasets()}>
                  Import
                </Button>
              }
            />
            {datasets.length === 0 ? (
              <EmptyState
                icon={<Database className="h-6 w-6" />}
                title="No datasets imported yet"
                description="Import a file and it will be validated immediately: field mapping, duplicates, empty rows and over-length samples."
                action={
                  <Button size="sm" variant="primary" loading={busy.importDataset} onClick={() => void pickAndImportDatasets()}>
                    Import a dataset
                  </Button>
                }
              />
            ) : (
              <div className="space-y-2">
                {datasets.map((dataset) => {
                  const active = wizard.datasetId === dataset.id;
                  const report = active ? wizard.datasetReport ?? dataset.report : dataset.report;
                  return (
                    <button
                      key={dataset.id}
                      type="button"
                      onClick={() => void wizardSelectDataset(dataset.id)}
                      className={cn(
                        "w-full rounded-[12px] border p-3 text-left transition-colors",
                        active
                          ? "border-[var(--acc)] bg-[var(--acc-soft)]"
                          : "border-[var(--border-soft)] hover:border-[var(--border)] hover:bg-[var(--hover)]",
                      )}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate text-[12.5px] font-medium">{dataset.name}</span>
                        <span className="flex shrink-0 items-center gap-2">
                          <Badge
                            tone={
                              dataset.status === "ok"
                                ? "good"
                                : dataset.status === "errors"
                                  ? "bad"
                                  : dataset.status === "warnings"
                                    ? "warn"
                                    : "neutral"
                            }
                          >
                            <Dot
                              tone={
                                dataset.status === "ok"
                                  ? "good"
                                  : dataset.status === "errors"
                                    ? "bad"
                                    : dataset.status === "warnings"
                                      ? "warn"
                                      : "neutral"
                              }
                            />
                            {dataset.status}
                          </Badge>
                          {active ? <Badge tone="accent">selected</Badge> : null}
                        </span>
                      </div>
                      <p className="zq-mono mt-0.5 truncate text-[10.5px] text-[var(--text-3)]">{dataset.path}</p>
                      <p className="mt-1 text-[11.5px] text-[var(--text-2)]">
                        {formatCount(report?.stats?.usable ?? dataset.usable)} usable of{" "}
                        {formatCount(report?.stats?.records ?? dataset.records)} records ·{" "}
                        {formatBytes(dataset.sizeBytes)} · {report?.mapping?.kind ?? dataset.mapping?.kind ?? "unmapped"}
                      </p>
                      {report?.issues?.length ? (
                        <p className="mt-1 truncate text-[11px] text-[var(--text-3)]">
                          {report.issues.length} finding{report.issues.length === 1 ? "" : "s"}:{" "}
                          {report.issues[0].message}
                        </p>
                      ) : null}
                    </button>
                  );
                })}
              </div>
            )}

            {datasetEntry ? (
              <div className="mt-3 border-t border-[var(--border-soft)] pt-3">
                <div className="flex flex-wrap items-center gap-2">
                  <Button
                    size="sm"
                    variant="quiet"
                    loading={busy[`dataset:${datasetEntry.id}`]}
                    onClick={() => void validateDataset(datasetEntry.id)}
                  >
                    Validate now
                  </Button>
                  <Button size="sm" variant="ghost" onClick={() => navigate("datasets")}>
                    Open dataset details
                  </Button>
                </div>
              </div>
            ) : null}

            <div className="mt-3 border-t border-[var(--border-soft)] pt-3">
              <Field
                label="Or add a dataset from the Hugging Face Hub"
                aside={<span className="text-[11px] text-[var(--text-3)]">rows are fetched when you validate or train</span>}
                hint="A Hub id looks like owner/name. Change the split only if the rows you want are not in `train`."
              >
                <div className="flex items-center gap-2">
                  <Input
                    value={hfId}
                    onChange={(event) => setHfId(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && hfId.trim()) void addHubDataset();
                    }}
                    placeholder="tatsu-lab/alpaca"
                    spellCheck={false}
                  />
                  <Input
                    className="w-[104px] shrink-0"
                    value={hfSplit}
                    onChange={(event) => setHfSplit(event.target.value)}
                    placeholder="train"
                    spellCheck={false}
                    aria-label="Split"
                  />
                  <Button
                    size="sm"
                    variant="primary"
                    loading={busy.importDataset}
                    disabled={!hfId.trim()}
                    onClick={() => void addHubDataset()}
                  >
                    Add
                  </Button>
                </div>
              </Field>
            </div>
          </Panel>
        ) : null}

        {/* ---------------------------------------------------------- step 2 */}
        {step === 2 ? (
          <Panel>
            <PanelHeader
              icon={<Sliders className="h-4 w-4" />}
              title="Choose a training method"
              description="Unavailable methods explain exactly what is missing instead of failing later."
            />
            <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
              {(["lora", "qlora", "sft", "full", "scratch"] as Method[]).map((method) => {
                const availability = methodAvailable(method);
                return (
                  <MethodCard
                    key={method}
                    method={method}
                    active={wizard.method === method}
                    disabled={!availability.ok}
                    disabledReason={availability.reason}
                    onSelect={() => void wizardSelectMethod(method)}
                  />
                );
              })}
            </div>
            {!trainingReady ? (
              <div className="mt-3">
                <Note tone="bad" title="No training backend is available yet">
                  {env.dependencies?.missing_core?.length
                    ? `Missing packages: ${env.dependencies.missing_core.join(", ")}.`
                    : "The Python backend could not be reached."}{" "}
                  Open Settings → Environment to install the ML runtime.
                </Note>
              </div>
            ) : null}
          </Panel>
        ) : null}

        {/* ---------------------------------------------------------- step 3 */}
        {step === 3 ? (
          <div className="space-y-4">
            <Panel>
              <PanelHeader
                icon={<Settings2 className="h-4 w-4" />}
                title="Parameters"
                description={
                  autoConfigure
                    ? "Simple mode exposes the settings that matter most; everything else is derived from your hardware."
                    : "Automatic configuration is off in Settings → Advanced, so these are the documented defaults rather than values chosen for this machine."
                }
                actions={
                  <Segmented
                    size="sm"
                    value={wizard.simple ? "simple" : "advanced"}
                    onChange={(value) => setWizard({ simple: value === "simple" })}
                    options={[
                      { value: "simple", label: "Simple" },
                      { value: "advanced", label: "Advanced" },
                    ]}
                  />
                }
              />

              {!config ? (
                <div className="space-y-2">
                  {[0, 1, 2].map((index) => (
                    <div key={index} className="zq-skeleton h-10" />
                  ))}
                  <p className="text-center text-[12px] text-[var(--text-3)]">
                    Choosing safe parameters for your hardware…
                  </p>
                </div>
              ) : (
                <ConfigForm
                  config={config}
                  simple={wizard.simple}
                  reasons={wizard.autoReasons}
                  onChange={(patch) => {
                    updateWizardConfig(patch);
                    void refreshWizardEstimate();
                  }}
                />
              )}
            </Panel>

            {config && !autoConfigure ? (
              <Note tone="warn" title="These values were not tuned to your hardware">
                Automatic configuration is disabled, so this is the documented starting point. Set the
                context length and batch size to something your hardware can hold — or press
                Auto-configure above to derive them from this machine.
              </Note>
            ) : null}

            {config ? (
              <Panel>
                <PanelHeader
                  icon={<Sparkles className="h-4 w-4" />}
                  title="Why these values"
                  description={
                    autoConfigure
                      ? "Every automatic decision, with its reason. Nothing is chosen silently."
                      : "Where these defaults come from — nothing here was measured on this machine."
                  }
                />
                {wizard.autoReasons.length === 0 ? (
                  <p className="text-[12.5px] text-[var(--text-3)]">
                    No automatic reasoning is available yet — select a model or dataset first.
                  </p>
                ) : (
                  <div className="space-y-1.5">
                    {wizard.autoReasons.map((reason, index) => (
                      <div key={index} className="flex items-start gap-2.5">
                        <Badge tone="accent">{reason.field}</Badge>
                        <p className="min-w-0 flex-1 text-[12px] leading-[18px] text-[var(--text-2)]">
                          {reason.reason}{" "}
                          <span className="zq-mono text-[var(--text-3)]">= {String(reason.value)}</span>
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </Panel>
            ) : null}
          </div>
        ) : null}

        {/* ---------------------------------------------------------- step 4 */}
        {step === 4 ? (
          <div className="space-y-4">
            <Panel>
              <PanelHeader
                icon={<Gauge className="h-4 w-4" />}
                title="Pre-flight check"
                description="Everything that would stop this run, before a single byte of the model is loaded."
                actions={
                  <Button size="sm" variant="quiet" loading={busy.autoConfig} onClick={() => void runAutoConfig()}>
                    Re-check
                  </Button>
                }
              />
              {blockers.length === 0 ? (
                <Note tone="good" title="Ready to train">
                  The dataset is valid, the method is available on this hardware, and the estimated
                  memory fits.
                </Note>
              ) : (
                <div className="space-y-2">
                  {blockers.map((blocker, index) => (
                    <Note key={index} tone={blocker.tone} title={blocker.title}>
                      {blocker.detail}
                    </Note>
                  ))}
                </div>
              )}
            </Panel>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <Panel>
                <PanelHeader icon={<Gauge className="h-4 w-4" />} title="Memory estimate" />
                {wizard.estimate?.available ? (
                  <>
                    <div className="mb-3 flex items-baseline gap-2">
                      <span className="zq-mono text-[22px] font-semibold">
                        {((wizard.estimate.estimated_total_mb ?? 0) / 1024).toFixed(2)} GB
                      </span>
                      <span className="text-[12px] text-[var(--text-3)]">
                        of {wizard.estimate.available_vram_mb ? `${(wizard.estimate.available_vram_mb / 1024).toFixed(1)} GB VRAM` : "unknown capacity"}
                      </span>
                    </div>
                    <ProgressBar
                      value={wizard.estimate.available_vram_mb ? ((wizard.estimate.estimated_total_mb ?? 0) / wizard.estimate.available_vram_mb) * 100 : 0}
                      tone={
                        wizard.estimate.verdict === "exceeds"
                          ? "bad"
                          : wizard.estimate.verdict === "tight"
                            ? "warn"
                            : "good"
                      }
                    />
                    <div className="mt-3 space-y-0">
                      <KeyValue label="Weights" value={`${((wizard.estimate.weights_mb ?? 0) / 1024).toFixed(2)} GB`} />
                      <KeyValue label="Optimizer state" value={`${((wizard.estimate.optimizer_mb ?? 0) / 1024).toFixed(2)} GB`} />
                      <KeyValue label="Gradients" value={`${((wizard.estimate.gradients_mb ?? 0) / 1024).toFixed(2)} GB`} />
                      <KeyValue label="Activations" value={`${((wizard.estimate.activations_mb ?? 0) / 1024).toFixed(2)} GB`} />
                      <KeyValue label="Runtime overhead" value={`${((wizard.estimate.overhead_mb ?? 0) / 1024).toFixed(2)} GB`} />
                      <KeyValue
                        label="Trainable parameters"
                        value={formatCount(wizard.estimate.trainable_params ?? null)}
                      />
                    </div>
                    <p className="mt-2 text-[11px] leading-[17px] text-[var(--text-3)]">
                      Approximate — attention kernels, the MLP ratio and fragmentation all shift the real
                      figure. Plausible range {( (wizard.estimate.range_low_mb ?? 0) / 1024).toFixed(1)}–
                      {((wizard.estimate.range_high_mb ?? 0) / 1024).toFixed(1)} GB.
                    </p>
                    {wizard.estimate.suggestions?.length ? (
                      <div className="mt-3">
                        <p className="mb-1 text-[11px] font-medium uppercase tracking-[0.06em] text-[var(--text-3)]">
                          Suggestions
                        </p>
                        <ul className="space-y-1">
                          {wizard.estimate.suggestions.map((suggestion, index) => (
                            <li key={index} className="flex items-start gap-2 text-[11.5px] leading-[17px] text-[var(--text-2)]">
                              <span className="mt-[6px] h-1 w-1 shrink-0 rounded-full bg-[var(--text-3)]" />
                              {suggestion}
                            </li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                  </>
                ) : (
                  <p className="text-[12.5px] text-[var(--text-3)]">
                    {wizard.estimate?.reason ?? "Select a model to estimate memory use."}
                  </p>
                )}
              </Panel>

              <Panel>
                <PanelHeader icon={<Database className="h-4 w-4" />} title="Dataset & configuration" />
                <div className="space-y-0">
                  <KeyValue label="Dataset" value={datasetEntry?.name ?? "—"} />
                  <KeyValue label="Records" value={formatCount(wizard.datasetReport?.stats?.records ?? null)} />
                  <KeyValue label="Usable samples" value={formatCount(wizard.datasetReport?.stats?.usable ?? null)} />
                  <KeyValue label="Mapping" value={wizard.datasetReport?.mapping?.kind ?? "—"} />
                  <KeyValue label="Epochs" value={config?.epochs ?? "—"} />
                  <KeyValue
                    label="Effective batch"
                    value={config ? config.batch_size * config.gradient_accumulation : "—"}
                  />
                  <KeyValue label="Context length" value={config?.context_length ?? "—"} />
                  <KeyValue label="Precision" value={config?.precision ?? "—"} />
                  <KeyValue label="Device" value={env.hardware?.training_device ?? "—"} />
                </div>
                {config ? (
                  <p className="zq-mono mt-3 rounded-[10px] border border-[var(--border-soft)] bg-[var(--code-bg)] p-2.5 text-[11px] leading-[17px] text-[var(--text-2)]">
                    {config.method} · r={config.lora_r} alpha={config.lora_alpha} · lr={config.learning_rate} ·{" "}
                    {config.gradient_accumulation > 1 ? `ga=${config.gradient_accumulation} · ` : ""}
                    {config.gradient_checkpointing ? "grad-ckpt on" : "grad-ckpt off"}
                  </p>
                ) : null}
              </Panel>
            </div>

            {wizard.datasetReport?.issues?.length ? (
              <Panel>
                <PanelHeader
                  icon={<AlertTriangle className="h-4 w-4" />}
                  title="Dataset findings"
                  description="Warnings do not block training; errors do."
                />
                <div className="space-y-2">
                  {wizard.datasetReport.issues.map((issue, index) => (
                    <div key={index} className="flex items-start gap-2.5">
                      <Badge tone={issue.severity === "error" ? "bad" : "warn"}>{issue.severity}</Badge>
                      <div className="min-w-0 flex-1">
                        <p className="text-[12.5px] leading-[18px]">{issue.message}</p>
                        {issue.hint ? (
                          <p className="text-[11.5px] leading-[17px] text-[var(--text-2)]">{issue.hint}</p>
                        ) : null}
                      </div>
                    </div>
                  ))}
                </div>
              </Panel>
            ) : null}
          </div>
        ) : null}

        {/* ---------------------------------------------------------- step 5 */}
        {step === 5 ? (
          <div className="space-y-4">
            <Panel>
              <PanelHeader
                icon={<Rocket className="h-4 w-4" />}
                title="Name the run and start"
                description="The name is also the folder name for checkpoints, logs and the saved model."
              />
              <Field label="Run name" hint="Saved under the app data folder, alongside history and GPU telemetry.">
                <Input
                  value={wizard.projectName}
                  placeholder="my-model-finetune"
                  onChange={(event) => setWizard({ projectName: event.target.value })}
                />
              </Field>

              <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
                <Stat label="Method" value={config?.method ?? wizard.method} />
                <Stat label="Epochs" value={config?.epochs ?? "—"} />
                <Stat label="Steps / epoch" value={config ? Math.max(1, Math.ceil((wizard.datasetReport?.stats?.usable ?? 0) / (config.batch_size * config.gradient_accumulation))) : "—"} />
                <Stat label="Eta" value="after start" />
              </div>

              <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-[var(--border-soft)] pt-4">
                <Button
                  variant="primary"
                  size="lg"
                  icon={<Play className="h-4 w-4" />}
                  loading={busy.startTraining}
                  disabled={fatal || !config}
                  onClick={() => void startTraining()}
                >
                  Start training
                </Button>
                {fatal ? (
                  <span className="text-[12px] text-[var(--red)]">Resolve the blocking issues in step 4 first.</span>
                ) : (
                  <span className="text-[12px] text-[var(--text-3)]">
                    Training runs in a separate process and can be paused or stopped at any time.
                  </span>
                )}
              </div>
            </Panel>

            {blockers.length ? (
              <div className="space-y-2">
                {blockers.map((blocker, index) => (
                  <Note key={index} tone={blocker.tone} title={blocker.title}>
                    {blocker.detail}
                  </Note>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}

        <div className="mt-4 flex items-center justify-between">
          <Button
            variant="ghost"
            icon={<ArrowLeft className="h-3.5 w-3.5" />}
            disabled={step === 0}
            onClick={() => setStep(step - 1)}
          >
            Back
          </Button>
          <div className="flex items-center gap-2">
            {step === 4 && fatal ? (
              <Badge tone="bad">
                <Dot tone="bad" />
                {blockers.filter((blocker) => blocker.tone === "bad").length} blocking
              </Badge>
            ) : null}
            <Button
              variant="secondary"
              disabled={step === STEPS.length - 1}
              onClick={() => setStep(step + 1)}
            >
              Continue
              <ArrowRight className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      </PageBody>
    </>
  );
}

function ConfigForm({
  config,
  simple,
  reasons,
  onChange,
}: {
  config: TrainingConfig;
  simple: boolean;
  reasons: { field: string; reason: string }[];
  onChange: (patch: Partial<TrainingConfig>) => void;
}) {
  const reasonFor = (field: string) => reasons.find((entry) => entry.field === field)?.reason;
  const auto = (field: string) => reasonFor(field) ?? "Derived from your hardware and dataset.";

  const numberFields: {
    key: keyof TrainingConfig;
    label: string;
    hint?: string;
    step?: number;
    min?: number;
    max?: number;
    simple?: boolean;
  }[] = [
    { key: "epochs", label: "Epochs", hint: auto("epochs"), min: 1, max: 100, simple: true },
    { key: "batch_size", label: "Batch size (per device)", hint: auto("batch_size"), min: 1, max: 256, simple: true },
    { key: "gradient_accumulation", label: "Gradient accumulation", hint: auto("gradient_accumulation"), min: 1, max: 512 },
    { key: "learning_rate", label: "Learning rate", hint: auto("learning_rate"), step: 1e-5, min: 1e-7, max: 1, simple: true },
    { key: "context_length", label: "Context length (tokens)", hint: auto("context_length"), min: 64, max: 32768, simple: true },
    { key: "warmup_ratio", label: "Warmup ratio", hint: "Fraction of steps spent ramping the learning rate.", step: 0.005, min: 0, max: 1 },
    { key: "weight_decay", label: "Weight decay", hint: "L2 regularisation applied to the adapters.", step: 0.005, min: 0, max: 1 },
    { key: "save_steps", label: "Checkpoint every N steps", hint: "0 lets the app choose a sensible interval.", min: 0, max: 100000 },
    { key: "save_total_limit", label: "Keep last N checkpoints", hint: "Older checkpoints are deleted to protect disk space.", min: 1, max: 100 },
    { key: "eval_split", label: "Evaluation split", hint: "Fraction held back to measure overfitting.", step: 0.01, min: 0, max: 0.5 },
    { key: "max_samples", label: "Max samples", hint: "0 uses the whole dataset. Useful for a quick smoke test.", min: 0, max: 10_000_000 },
    { key: "seed", label: "Seed", hint: "Makes shuffling reproducible.", min: 0, max: 1_000_000 },
  ];

  const adapter = config.method !== "full" && config.method !== "scratch";

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-x-4 gap-y-3 sm:grid-cols-2 lg:grid-cols-3">
        {numberFields
          .filter((field) => (simple ? field.simple : true))
          .map((field) => (
            <Field key={String(field.key)} label={field.label} hint={field.hint}>
              <Input
                type="number"
                value={String(config[field.key] ?? "")}
                step={field.step ?? 1}
                min={field.min}
                max={field.max}
                onChange={(event) => {
                  const raw = event.target.value;
                  const parsed = field.step && field.step < 1 ? Number.parseFloat(raw) : Number.parseInt(raw, 10);
                  onChange({ [field.key]: Number.isFinite(parsed) ? parsed : 0 } as Partial<TrainingConfig>);
                }}
              />
            </Field>
          ))}
      </div>

      {config.method === "scratch" ? (
        <div className="grid grid-cols-1 gap-x-4 gap-y-3 border-t border-[var(--border-soft)] pt-4 sm:grid-cols-2 lg:grid-cols-4">
          <Field label="Model size" hint="Named presets; every value below can be overridden.">
            <Select
              value={config.scratch_size ?? "tiny"}
              onChange={(event) => onChange({ scratch_size: event.target.value })}
            >
              <option value="micro">micro (~3M params)</option>
              <option value="tiny">tiny (~10M params)</option>
              <option value="small">small (~25M params)</option>
            </Select>
          </Field>
          <Field label="Layers" hint="Depth of the transformer.">
            <Input
              type="number"
              min={1}
              max={24}
              value={String(config.scratch_layers ?? 4)}
              onChange={(event) => onChange({ scratch_layers: Number.parseInt(event.target.value, 10) || 4 })}
            />
          </Field>
          <Field label="Hidden size" hint="Width of the model; must be divisible by the heads.">
            <Input
              type="number"
              min={32}
              max={2048}
              step={32}
              value={String(config.scratch_hidden ?? 256)}
              onChange={(event) => onChange({ scratch_hidden: Number.parseInt(event.target.value, 10) || 256 })}
            />
          </Field>
          <Field label="Attention heads" hint="Rounded down to divide the hidden size evenly.">
            <Input
              type="number"
              min={1}
              max={16}
              value={String(config.scratch_heads ?? 4)}
              onChange={(event) => onChange({ scratch_heads: Number.parseInt(event.target.value, 10) || 4 })}
            />
          </Field>
          {!simple ? (
            <Field label="FFN width" hint="Inner size of the feed-forward blocks.">
              <Input
                type="number"
                min={64}
                max={8192}
                step={64}
                value={String(config.scratch_ffn ?? 1024)}
                onChange={(event) => onChange({ scratch_ffn: Number.parseInt(event.target.value, 10) || 1024 })}
              />
            </Field>
          ) : null}
          {!simple ? (
            <Field label="Vocabulary" hint="Ceiling for the character-level vocabulary learned from the dataset.">
              <Input
                type="number"
                min={1000}
                max={100000}
                step={1000}
                value={String(config.scratch_vocab ?? 16000)}
                onChange={(event) => onChange({ scratch_vocab: Number.parseInt(event.target.value, 10) || 16000 })}
              />
            </Field>
          ) : null}
        </div>
      ) : null}

      {adapter ? (
        <div className="grid grid-cols-1 gap-x-4 gap-y-3 border-t border-[var(--border-soft)] pt-4 sm:grid-cols-2 lg:grid-cols-4">
          <Field label="LoRA rank (r)" hint={auto("lora_r")}>
            <Input
              type="number"
              min={1}
              max={512}
              value={String(config.lora_r)}
              onChange={(event) => onChange({ lora_r: Number.parseInt(event.target.value, 10) || 1 })}
            />
          </Field>
          <Field label="LoRA alpha" hint={auto("lora_alpha")}>
            <Input
              type="number"
              min={1}
              max={1024}
              value={String(config.lora_alpha)}
              onChange={(event) => onChange({ lora_alpha: Number.parseInt(event.target.value, 10) || 1 })}
            />
          </Field>
          {!simple ? (
            <Field label="LoRA dropout" hint="Regularisation on the adapter updates.">
              <Input
                type="number"
                min={0}
                max={0.9}
                step={0.01}
                value={String(config.lora_dropout)}
                onChange={(event) => onChange({ lora_dropout: Number.parseFloat(event.target.value) || 0 })}
              />
            </Field>
          ) : null}
          {!simple ? (
            <Field label="Target modules" hint="'auto' detects the attention projections from the real module names.">
              <Input
                value={typeof config.lora_target_modules === "string" ? config.lora_target_modules : config.lora_target_modules.join(", ")}
                onChange={(event) => onChange({ lora_target_modules: event.target.value })}
              />
            </Field>
          ) : null}
        </div>
      ) : null}

      {!simple ? (
        <div className="grid grid-cols-1 gap-x-4 gap-y-3 border-t border-[var(--border-soft)] pt-4 sm:grid-cols-2 lg:grid-cols-4">
          <Field label="Precision" hint={auto("precision")}>
            <Select
              value={config.precision}
              onChange={(event) => onChange({ precision: event.target.value as Precision })}
            >
              <option value="auto">auto</option>
              <option value="bf16">bfloat16</option>
              <option value="fp16">float16</option>
              <option value="fp32">float32</option>
            </Select>
          </Field>
          <Field label="Quantization" hint="4-bit or 8-bit needs an NVIDIA GPU and bitsandbytes.">
            <Select
              value={config.quantization}
              onChange={(event) => onChange({ quantization: event.target.value as Quantization })}
            >
              <option value="none">none</option>
              <option value="4bit">4-bit (NF4)</option>
              <option value="8bit">8-bit</option>
            </Select>
          </Field>
          <Field label="LR scheduler">
            <Select value={config.lr_scheduler} onChange={(event) => onChange({ lr_scheduler: event.target.value })}>
              {["cosine", "linear", "constant", "constant_with_warmup", "cosine_with_restarts", "polynomial"].map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Optimizer" hint="'auto' picks a fused or paged optimizer for the device.">
            <Select value={config.optimizer} onChange={(event) => onChange({ optimizer: event.target.value })}>
              {["auto", "adamw_torch", "adamw_torch_fused", "paged_adamw_8bit", "adafactor"].map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </Select>
          </Field>
          <div className="sm:col-span-2 lg:col-span-4">
            <Switch
              checked={config.gradient_checkpointing}
              onChange={(value) => onChange({ gradient_checkpointing: value })}
              label="Gradient checkpointing"
              description={auto("gradient_checkpointing")}
            />
          </div>
        </div>
      ) : null}

      <div className="border-t border-[var(--border-soft)] pt-3">
        <Field
          label="Extra notes (optional)"
          hint="Stored with the run so you remember what this configuration was for."
        >
          <Textarea rows={2} placeholder="What is this run trying to achieve?" />
        </Field>
      </div>
    </div>
  );
}
