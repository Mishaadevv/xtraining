import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ArrowRight, Rocket, Wand2 } from "lucide-react";
import { api } from "../lib/api";
import { basename, bytes, number, riskLabel } from "../lib/format";
import { useApp } from "../state/app";
import { useRouter } from "../state/router";
import {
  Badge,
  Button,
  Callout,
  CodeBlock,
  Field,
  HelpTip,
  KeyValue,
  NumberInput,
  Panel,
  SectionHeader,
  Select,
  Stat,
  Table,
  Td,
  TextInput,
  Th,
  Toggle,
  cx,
} from "../components/ui";
import { DatasetPicker, ErrorPanel, Loading, ModelPicker, useEngine } from "./common";

const METHOD_LABELS: Record<string, string> = {
  scratch: "Train from scratch",
  continued_pretraining: "Continued pretraining",
  continued_training: "Continued training",
  full_finetune: "Full fine-tuning",
  sft: "Supervised fine-tuning (SFT)",
  lora: "LoRA",
  qlora: "QLoRA (4-bit)",
  adapter: "Adapter training",
};

const STEPS = ["Model", "Data", "Method", "Configuration", "Preview"];

export function NewTrainingPage() {
  const { registry, settings, toast, reportError, backends, refreshJobs } = useApp();
  const router = useRouter();
  const params = useMemo(() => new URLSearchParams(window.location.hash.split("?")[1] ?? ""), []);

  const [step, setStep] = useState(0);
  const [baseModel, setBaseModel] = useState<string | null>(params.get("model") ?? null);
  const [resumeFrom, setResumeFrom] = useState<string | null>(params.get("resume") ?? null);
  const [parentJob, setParentJob] = useState<string | null>(params.get("parent") ?? null);
  const [datasets, setDatasets] = useState<string[]>(params.get("dataset") ? [params.get("dataset") as string] : []);
  const [template, setTemplate] = useState("chatml");
  const [mapping, setMapping] = useState<Record<string, string>>({});
  const [method, setMethod] = useState(params.get("method") ?? "lora");
  const [backend, setBackend] = useState<string | null>(settings?.defaultBackend ?? null);
  const [device, setDevice] = useState("auto");
  const [busy, setBusy] = useState(false);
  const [autoReasons, setAutoReasons] = useState<string[]>([]);
  const [advanced, setAdvanced] = useState(false);

  const [config, setConfig] = useState({
    precision: "fp32",
    quantization: "none",
    sequence_length: 128,
    batch_size: 2,
    gradient_accumulation: 4,
    epochs: 1,
    max_steps: 0,
    learning_rate: 3e-4,
    lr_scheduler: "cosine",
    warmup_steps: 0,
    weight_decay: 0.0,
    optimizer: "adamw",
    max_grad_norm: 1.0,
    seed: 42,
    save_every: 100,
    logging_every: 10,
    eval_every: 50,
    checkpoint_limit: 3,
    gradient_checkpointing: false,
    eval_ratio: 0.05,
    lora_rank: 8,
    lora_alpha: 16,
    lora_dropout: 0.05,
    target_modules: "",
    workers: 0,
    hidden_size: 32,
    context_length: 8,
    vocab_size: 320,
  });

  const firstDataset = datasets[0] ?? null;
  const datasetReport = useEngine<any>(
    "datasets.inspect",
    { path: firstDataset ?? "", sample_size: 600 },
    { auto: Boolean(firstDataset), timeout: 600_000, deps: [firstDataset] },
  );

  useEffect(() => {
    if (datasetReport.data?.detected_mapping) {
      setMapping((current) => (Object.keys(current).length ? current : datasetReport.data.detected_mapping));
    }
  }, [datasetReport.data]);

  const modelReport = useEngine<any>(
    "models.inspect",
    { path: baseModel ?? "" },
    { auto: Boolean(baseModel), timeout: 300_000, deps: [baseModel] },
  );

  const request = useMemo(
    () => ({
      backend,
      method,
      device,
      base_model: baseModel,
      parent_checkpoint: baseModel,
      resume_from: resumeFrom,
      dataset_paths: datasets,
      dataset_weights: datasets.map(() => 1),
      mapping,
      template,
      tokenizer_path: null,
      ...config,
      target_modules: config.target_modules
        ? config.target_modules.split(",").map((item) => item.trim()).filter(Boolean)
        : [],
    }),
    [backend, method, device, baseModel, resumeFrom, datasets, mapping, template, config],
  );

  const prepare = useEngine<any>("training.prepare", request, { auto: step === 4, timeout: 300_000, deps: [step, JSON.stringify(request)] });

  const runAutoConfigure = async () => {
    setBusy(true);
    try {
      const result = await api.call<any>("training.auto_configure", {
        method,
        parameters: modelReport.data?.weights?.parameter_count ?? 0,
        hidden_size: modelReport.data?.architecture?.hidden_size ?? config.hidden_size,
        num_hidden_layers: modelReport.data?.architecture?.num_layers ?? 3,
        sequence_length: config.sequence_length,
        steps_per_epoch: datasetReport.data?.record_count ? Math.floor(datasetReport.data.record_count / 8) : 0,
        has_eval_data: datasets.length > 0,
      });
      const suggested = result.suggested ?? {};
      setConfig((current) => ({
        ...current,
        ...Object.fromEntries(Object.entries(suggested).filter(([key]) => key in current)),
      }));
      setAutoReasons(result.reasons ?? []);
      toast({ title: "Configuration derived from this machine", body: `${(result.reasons ?? []).length} decisions explained below.`, tone: "ok" });
    } catch (error) {
      reportError(error, "Auto configuration failed");
    } finally {
      setBusy(false);
    }
  };

  const start = async () => {
    setBusy(true);
    try {
      const job = await api.jobs.start({
        kind: "train",
        ...request,
        parent_job_id: parentJob,
      });
      toast({
        title: "Training started",
        body: `${job.jobId} is running. Progress streams into the monitor.`,
        tone: "ok",
      });
      await refreshJobs();
      router.navigate(`/training/${encodeURIComponent(job.jobId)}`);
    } catch (error) {
      reportError(error, "The run could not be started");
    } finally {
      setBusy(false);
    }
  };

  const continuation = prepare.data?.continuation ?? null;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">New training run</h1>
          <p className="mt-0.5 max-w-3xl text-xs text-ink-2">
            Five steps: choose the weights, choose the data, choose the method, configure it, then read the
            pre-flight report before anything starts. Nothing runs until you press Start.
          </p>
        </div>
        <div className="flex items-center gap-1">
          {STEPS.map((label, index) => (
            <button
              key={label}
              onClick={() => setStep(index)}
              className={cx(
                "rounded-md border px-2 py-1 text-2xs transition-colors",
                index === step ? "border-accent/50 bg-accent/12 text-ink-0" : "border-line-soft text-ink-2 hover:text-ink-0",
              )}
            >
              {index + 1}. {label}
            </button>
          ))}
        </div>
      </div>

      {step === 0 ? (
        <div className="grid gap-3 lg:grid-cols-2">
          <Panel>
            <SectionHeader title="Base weights" subtitle="Continue from a model in the library, or start from random weights" />
            <div className="space-y-3">
              <ModelPicker value={baseModel} onChange={setBaseModel} allowScratch filter={(entry) => Boolean(entry.path)} />
              {modelReport.data ? (
                <KeyValue
                  items={[
                    ["Format", modelReport.data.format?.kind ?? "unknown"],
                    ["Parameters", modelReport.data.weights?.parameter_count ? number(modelReport.data.weights.parameter_count) : "unknown"],
                    ["Architecture", String(modelReport.data.architecture?.model_type ?? "unknown")],
                    ["Context", String(modelReport.data.architecture?.max_position_embeddings ?? "unknown")],
                    ["Adapter", modelReport.data.adapter ? "yes" : "no"],
                  ]}
                />
              ) : null}
              <Field label="Checkpoint to resume exactly" hint="Leave empty for a new run. A tiny-backend checkpoint restores optimizer and RNG state.">
                <TextInput value={resumeFrom ?? ""} onChange={(event) => setResumeFrom(event.target.value || null)} placeholder="…\\checkpoint-100" />
              </Field>
              <Field label="Parent run id" hint="Recorded for lineage so the result is traceable to its parent.">
                <TextInput value={parentJob ?? ""} onChange={(event) => setParentJob(event.target.value || null)} placeholder="train-20260922-120000-ab12" />
              </Field>
              <Callout tone="info" title="Training from scratch">
                Only the pure-Python tiny backend can train from scratch here. It builds a new context-window
                language model and a new tokenizer; leave the base model empty to use it.
              </Callout>
            </div>
          </Panel>
          <Panel>
            <SectionHeader title="What will be restored" subtitle="Continuation is explicit: the engine reports what it can actually restore" />
            {continuation ? (
              <div className="space-y-2">
                <Badge tone={continuation.verdict === "exact" ? "ok" : continuation.verdict === "unsafe" ? "danger" : "warn"}>
                  {continuation.verdict}
                </Badge>
                <div className="text-xs text-ink-1">{continuation.verdict_text}</div>
                <Table>
                  <thead>
                    <tr>
                      <Th>Item</Th>
                      <Th align="right">Restored</Th>
                    </tr>
                  </thead>
                  <tbody>
                    {continuation.restored.map((item: any) => (
                      <tr key={item.item}>
                        <Td>{item.item}</Td>
                        <Td align="right">
                          <Badge tone={item.restored ? "ok" : "muted"}>{item.restored ? "yes" : "no"}</Badge>
                        </Td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              </div>
            ) : (
              <div className="text-2xs leading-relaxed text-ink-2">
                Pick a checkpoint in the field on the left and this panel updates with the real state files the
                engine found: weights, optimizer, dataset state, scheduler and RNG.
              </div>
            )}
          </Panel>
        </div>
      ) : null}

      {step === 1 ? (
        <div className="grid gap-3 lg:grid-cols-2">
          <Panel>
            <SectionHeader title="Datasets" subtitle="Mix several datasets in one run; weights are applied by the backend" />
            <div className="space-y-3">
              <DatasetPicker multiple multipleValues={datasets} onMultipleChange={setDatasets} />
              <Field label="Chat / completion template" hint="How multi-turn records are flattened into training text.">
                <Select value={template} onChange={(event) => setTemplate(event.target.value)}>
                  <option value="chatml">ChatML (&lt;|role|&gt;)</option>
                  <option value="plain">role: content</option>
                  <option value="markdown">### Role:</option>
                </Select>
              </Field>
              {datasetReport.error ? <ErrorPanel error={datasetReport.error} onRetry={datasetReport.reload} /> : null}
              {datasetReport.loading ? <Loading label="Validating the dataset…" lines={3} /> : null}
              {datasetReport.data ? (
                <>
                  <div className="grid gap-2 sm:grid-cols-3">
                    <Stat label="Records" value={number(datasetReport.data.record_count)} />
                    <Stat label="Duplicates" value={number(datasetReport.data.duplicates)} tone={datasetReport.data.duplicates ? "warn" : "muted"} />
                    <Stat label="Empty" value={number(datasetReport.data.empty_records)} tone={datasetReport.data.empty_records ? "warn" : "muted"} />
                  </div>
                  <div className="space-y-2">
                    <div className="text-2xs uppercase tracking-wide text-ink-3">Field mapping</div>
                    {Object.entries(datasetReport.data.mapping_suggestion?.mapping ?? {}).map(([role, field]) => (
                      <div key={role} className="flex items-center gap-2">
                        <Badge tone="accent">{role}</Badge>
                        <Select
                          className="flex-1"
                          value={mapping[role] ?? String(field)}
                          onChange={(event) => setMapping({ ...mapping, [role]: event.target.value })}
                        >
                          {datasetReport.data.field_names.map((name: string) => (
                            <option key={name} value={name}>
                              {name}
                            </option>
                          ))}
                        </Select>
                      </div>
                    ))}
                    {!Object.keys(datasetReport.data.mapping_suggestion?.mapping ?? {}).length ? (
                      <Callout tone="warn" title="No field mapping was detected">
                        The engine found no prompt/response/text columns. Set the mapping on the dataset page
                        first — training on an empty mapping would produce zero training text.
                      </Callout>
                    ) : null}
                  </div>
                </>
              ) : null}
            </div>
          </Panel>
          <Panel>
            <SectionHeader title="Sample training text" subtitle="The exact text the trainer will read" />
            <div className="space-y-2">
              {(datasetReport.data?.preview ?? []).slice(0, 3).map((row: any) => (
                <div key={row.index} className="rounded-md border border-line-soft bg-surface-2 p-2">
                  <div className="mb-1 flex items-center justify-between">
                    <Badge tone="muted">#{row.index}</Badge>
                    <span className="text-2xs text-ink-3">{row.length} characters</span>
                  </div>
                  <CodeBlock max="max-h-32">
                    {JSON.stringify(row.normalised, null, 2).slice(0, 700)}
                  </CodeBlock>
                </div>
              ))}
              {!datasetReport.data ? <div className="text-2xs text-ink-3">Select a dataset to preview it.</div> : null}
            </div>
          </Panel>
        </div>
      ) : null}

      {step === 2 ? (
        <div className="grid gap-3 lg:grid-cols-2">
          <Panel>
            <SectionHeader title="Training method" subtitle="Methods that this machine cannot run are listed with the reason" />
            <div className="space-y-2">
              {Object.keys(METHOD_LABELS).map((name) => {
                const available = backends.some((item) => item.available && item.methods.includes(name));
                const reason = backends
                  .filter((item) => item.methods.includes(name) && !item.available)
                  .map((item) => `${item.name}: ${item.reason}`)
                  .join(" · ");
                return (
                  <button
                    key={name}
                    disabled={!available}
                    onClick={() => setMethod(name)}
                    className={cx(
                      "w-full rounded-md border px-3 py-2 text-left transition-colors",
                      method === name ? "border-accent/50 bg-accent/12" : "border-line-soft hover:bg-surface-2",
                      !available && "cursor-not-allowed opacity-50",
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs">{METHOD_LABELS[name]}</span>
                      <Badge tone={available ? "ok" : "muted"}>{available ? "available" : "unavailable"}</Badge>
                    </div>
                    {!available && reason ? <div className="mt-1 text-2xs text-ink-3">{reason}</div> : null}
                  </button>
                );
              })}
            </div>
          </Panel>
          <Panel>
            <SectionHeader title="Backend and device" subtitle="Only installed backends can be selected" />
            <div className="space-y-3">
              {backends.map((item) => (
                <button
                  key={item.id}
                  disabled={!item.available}
                  onClick={() => setBackend(item.id)}
                  className={cx(
                    "w-full rounded-md border px-3 py-2 text-left transition-colors",
                    backend === item.id ? "border-accent/50 bg-accent/12" : "border-line-soft hover:bg-surface-2",
                    !item.available && "cursor-not-allowed opacity-50",
                  )}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs">{item.name}</span>
                    <Badge tone={item.available ? "ok" : "warn"}>{item.available ? "ready" : "not installed"}</Badge>
                  </div>
                  <div className="mt-1 text-2xs leading-relaxed text-ink-2">{item.available ? item.description : item.reason}</div>
                  {item.speed_note ? <div className="mt-1 text-2xs text-ink-3">{item.speed_note}</div> : null}
                </button>
              ))}
              <Field label="Device">
                <Select value={device} onChange={(event) => setDevice(event.target.value)}>
                  <option value="auto">Automatic (GPU if available, else CPU)</option>
                  <option value="cpu">CPU only</option>
                  {backends.length ? <option value="cuda:0">GPU 0</option> : null}
                </Select>
              </Field>
              <Callout tone="info" title="Method and backend must agree">
                The pre-flight check on the last step re-verifies that the chosen backend really implements the
                chosen method before the job is created.
              </Callout>
            </div>
          </Panel>
        </div>
      ) : null}

      {step === 3 ? (
        <div className="space-y-3">
          <Panel>
            <SectionHeader
              title="Configuration"
              subtitle="Auto Configure derives values from the detected hardware and the selected model"
              actions={
                <Button variant="primary" size="sm" icon={<Wand2 size={12} />} loading={busy} onClick={() => void runAutoConfigure()}>
                  Auto Configure
                </Button>
              }
            />
            <div className="grid gap-3 md:grid-cols-3 xl:grid-cols-4">
              <Field label="Precision" help="fp32 always works; fp16/bf16 need a CUDA device and change both memory use and stability.">
                <Select value={config.precision} onChange={(event) => setConfig({ ...config, precision: event.target.value })}>
                  {["fp32", "fp16", "bf16"].map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label="Quantisation" help="4-bit/8-bit base weights cut memory a lot, at some quality cost, and need bitsandbytes plus a GPU.">
                <Select value={config.quantization} onChange={(event) => setConfig({ ...config, quantization: event.target.value })}>
                  {["none", "int8", "int4"].map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label="Sequence length" help="More context costs activation memory quadratically in attention and linearly in the MLP.">
                <NumberInput value={config.sequence_length} step={16} min={16} onChange={(value) => setConfig({ ...config, sequence_length: Number(value) || 16 })} />
              </Field>
              <Field label="Batch size" help="Per-device batch size. Reduce this first when memory runs out.">
                <NumberInput value={config.batch_size} min={1} onChange={(value) => setConfig({ ...config, batch_size: Number(value) || 1 })} />
              </Field>
              <Field label="Gradient accumulation" help="Accumulates gradients over several micro-batches to reach a larger effective batch without more memory.">
                <NumberInput value={config.gradient_accumulation} min={1} onChange={(value) => setConfig({ ...config, gradient_accumulation: Number(value) || 1 })} />
              </Field>
              <Field label="Epochs" help="Passes over the dataset. Ignored when a step budget is set.">
                <NumberInput value={config.epochs} step={0.5} min={0.1} onChange={(value) => setConfig({ ...config, epochs: Number(value) || 1 })} />
              </Field>
              <Field label="Max steps" help="0 means 'derive from epochs'. When resuming, this is the number of additional steps.">
                <NumberInput value={config.max_steps} min={0} onChange={(value) => setConfig({ ...config, max_steps: Number(value) || 0 })} />
              </Field>
              <Field label="Learning rate" help="Too high and loss diverges; too low and nothing happens. LoRA tolerates ~2e-4, full fine-tuning usually needs <=2e-5.">
                <TextInput
                  value={String(config.learning_rate)}
                  onChange={(event) => setConfig({ ...config, learning_rate: Number(event.target.value) || 0 })}
                />
              </Field>
              <Field label="LR schedule">
                <Select value={config.lr_scheduler} onChange={(event) => setConfig({ ...config, lr_scheduler: event.target.value })}>
                  {["cosine", "linear", "constant", "constant_with_warmup"].map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label="Seed" help="Recorded for reproducibility; the engine passes it to the backend's random generators.">
                <NumberInput value={config.seed} onChange={(value) => setConfig({ ...config, seed: Number(value) || 0 })} />
              </Field>
              {method === "lora" || method === "qlora" || method === "adapter" ? (
                <>
                  <Field label="LoRA rank" help="Higher rank means more trainable parameters and more memory.">
                    <NumberInput value={config.lora_rank} min={1} onChange={(value) => setConfig({ ...config, lora_rank: Number(value) || 1 })} />
                  </Field>
                  <Field label="LoRA alpha" help="Scaling factor, conventionally twice the rank.">
                    <NumberInput value={config.lora_alpha} min={1} onChange={(value) => setConfig({ ...config, lora_alpha: Number(value) || 1 })} />
                  </Field>
                  <Field label="Target modules" help="Comma separated. Leave empty to use the preset for the detected architecture.">
                    <TextInput value={config.target_modules} onChange={(event) => setConfig({ ...config, target_modules: event.target.value })} placeholder="q_proj, v_proj" />
                  </Field>
                </>
              ) : null}
              {backend === "tiny" || !backend ? (
                <>
                  <Field label="Hidden size (tiny backend)">
                    <NumberInput value={config.hidden_size} min={4} onChange={(value) => setConfig({ ...config, hidden_size: Number(value) || 4 })} />
                  </Field>
                  <Field label="Context window (tiny backend)">
                    <NumberInput value={config.context_length} min={1} onChange={(value) => setConfig({ ...config, context_length: Number(value) || 1 })} />
                  </Field>
                  <Field label="Tokenizer vocabulary">
                    <NumberInput value={config.vocab_size} step={64} min={260} onChange={(value) => setConfig({ ...config, vocab_size: Number(value) || 260 })} />
                  </Field>
                </>
              ) : null}
            </div>

            <button className="mt-3 text-2xs text-accent hover:underline" onClick={() => setAdvanced((value) => !value)}>
              {advanced ? "Hide advanced options" : "Show advanced options"}
            </button>
            {advanced ? (
              <div className="mt-3 grid gap-3 md:grid-cols-3 xl:grid-cols-4">
                <Field label="Optimizer">
                  <Select value={config.optimizer} onChange={(event) => setConfig({ ...config, optimizer: event.target.value })}>
                    <option value="adamw">adamw</option>
                    <option value="adamw_torch">adamw_torch</option>
                    <option value="adafactor">adafactor</option>
                    <option value="sgd">sgd</option>
                  </Select>
                </Field>
                <Field label="Warmup steps" help="Ramps the learning rate up; stabilises early training.">
                  <NumberInput value={config.warmup_steps} min={0} onChange={(value) => setConfig({ ...config, warmup_steps: Number(value) || 0 })} />
                </Field>
                <Field label="Weight decay" help="L2 regularisation on the non-bias parameters.">
                  <TextInput value={String(config.weight_decay)} onChange={(event) => setConfig({ ...config, weight_decay: Number(event.target.value) || 0 })} />
                </Field>
                <Field label="Max gradient norm" help="Clips exploding gradients; 1.0 is a safe default.">
                  <TextInput value={String(config.max_grad_norm)} onChange={(event) => setConfig({ ...config, max_grad_norm: Number(event.target.value) || 0 })} />
                </Field>
                <Field label="Log every N steps">
                  <NumberInput value={config.logging_every} min={1} onChange={(value) => setConfig({ ...config, logging_every: Number(value) || 1 })} />
                </Field>
                <Field label="Checkpoint every N steps">
                  <NumberInput value={config.save_every} min={1} onChange={(value) => setConfig({ ...config, save_every: Number(value) || 1 })} />
                </Field>
                <Field label="Evaluate every N steps" help="0 disables evaluation. Evaluation costs time but catches divergence.">
                  <NumberInput value={config.eval_every} min={0} onChange={(value) => setConfig({ ...config, eval_every: Number(value) || 0 })} />
                </Field>
                <Field label="Keep last N checkpoints" help="Older checkpoints are pruned by the retention policy. Protected checkpoints are never deleted.">
                  <NumberInput value={config.checkpoint_limit} min={1} onChange={(value) => setConfig({ ...config, checkpoint_limit: Number(value) || 1 })} />
                </Field>
                <Field label="Evaluation split" help="Share of the data held out for evaluation.">
                  <NumberInput value={config.eval_ratio} step={0.01} min={0} max={0.5} onChange={(value) => setConfig({ ...config, eval_ratio: Number(value) || 0 })} />
                </Field>
                <Field label="Data loader workers" help="More workers speed up data loading on CPU-bound setups.">
                  <NumberInput value={config.workers} min={0} onChange={(value) => setConfig({ ...config, workers: Number(value) || 0 })} />
                </Field>
                <div className="md:col-span-2">
                  <Toggle
                    checked={config.gradient_checkpointing}
                    onChange={(value) => setConfig({ ...config, gradient_checkpointing: value })}
                    label="Gradient checkpointing"
                    hint="Recomputes activations instead of storing them: slower, but a large memory saving."
                  />
                </div>
              </div>
            ) : null}

            {autoReasons.length ? (
              <div className="mt-3 space-y-1">
                <div className="text-2xs uppercase tracking-wide text-ink-3">Why these values</div>
                {autoReasons.map((reason) => (
                  <div key={reason} className="rounded-md border border-line-soft bg-surface-2 px-2 py-1.5 text-2xs text-ink-1">
                    {reason}
                  </div>
                ))}
              </div>
            ) : null}
          </Panel>
        </div>
      ) : null}

      {step === 4 ? (
        <div className="space-y-3">
          {prepare.error ? <ErrorPanel error={prepare.error} onRetry={prepare.reload} /> : null}
          {prepare.loading ? <Loading label="Running the pre-flight checks…" lines={5} /> : null}
          {prepare.data ? (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone={prepare.data.ok ? "ok" : "danger"}>{prepare.data.ok ? "Ready to start" : `${prepare.data.errors.length} blocking problem(s)`}</Badge>
                {prepare.data.plan ? <Badge tone={riskLabel(prepare.data.plan.risk).tone as any}>{riskLabel(prepare.data.plan.risk).label}</Badge> : null}
                <span className="text-2xs text-ink-3">
                  {prepare.data.model?.parameters ? `${number(prepare.data.model.parameters)} parameters` : "parameter count unknown"}
                </span>
              </div>

              {prepare.data.plan ? (
                <div className="grid gap-3 lg:grid-cols-4">
                  <Stat label="Trainable parameters" value={number(prepare.data.plan.parameters.trainable)} hint={prepare.data.plan.parameters.source} estimated />
                  <Stat label="Estimated VRAM" value={bytes(prepare.data.plan.memory.vram_estimate)} estimated />
                  <Stat label="Estimated RAM" value={bytes(prepare.data.plan.memory.ram_estimate)} estimated />
                  <Stat label="Steps" value={number(prepare.data.plan.steps.total)} hint={`effective batch ${prepare.data.plan.steps.effective_batch_size}`} estimated />
                </div>
              ) : null}

              <div className="grid gap-3 lg:grid-cols-2">
                <Panel>
                  <SectionHeader title="Pre-flight checks" subtitle="Each check runs against the real files and the real environment" />
                  <div className="space-y-1.5">
                    {prepare.data.checks.map((check: any) => (
                      <div
                        key={`${check.name}-${check.message}`}
                        className={cx(
                          "rounded-md border px-2.5 py-1.5",
                          check.ok
                            ? "border-line-soft bg-surface-2"
                            : check.level === "error"
                              ? "border-danger/30 bg-danger/8"
                              : "border-warn/30 bg-warn/8",
                        )}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-mono text-2xs text-ink-2">{check.name}</span>
                          <Badge tone={check.ok ? "ok" : check.level === "error" ? "danger" : "warn"}>
                            {check.ok ? "pass" : check.level}
                          </Badge>
                        </div>
                        <div className="mt-0.5 text-2xs text-ink-1">{check.message}</div>
                        {check.hint ? <div className="mt-0.5 text-2xs text-ink-3">Fix: {check.hint}</div> : null}
                      </div>
                    ))}
                  </div>
                </Panel>

                <Panel>
                  <SectionHeader title="Plan detail" subtitle="All values labelled estimated were computed, not measured" />
                  <KeyValue
                    items={[
                      ["Method", prepare.data.plan.method],
                      ["Backend", prepare.data.backend?.name ?? "auto"],
                      ["Precision", prepare.data.plan.precision],
                      ["Quantisation", prepare.data.plan.quantization],
                      ["Weights memory", bytes(prepare.data.plan.memory.weights)],
                      ["Gradients memory", bytes(prepare.data.plan.memory.gradients)],
                      ["Optimizer memory", bytes(prepare.data.plan.memory.optimizer)],
                      ["Activations memory", bytes(prepare.data.plan.memory.activations)],
                      ["Checkpoint size", bytes(prepare.data.plan.disk.checkpoint_size_estimate)],
                      ["Workspace need", bytes(prepare.data.plan.disk.workspace_needed)],
                      ["Dataset tokens", number(prepare.data.plan.data.tokens)],
                      ["Token basis", prepare.data.plan.data.basis],
                    ]}
                  />
                  {continuation ? (
                    <div className="mt-3">
                      <div className="mb-1 flex items-center gap-2 text-2xs uppercase tracking-wide text-ink-3">
                        Continuation <HelpTip content={continuation.verdict_text} />
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {continuation.restored.map((item: any) => (
                          <Badge key={item.item} tone={item.restored ? "ok" : "muted"}>
                            {item.item}: {item.restored ? "restored" : "no"}
                          </Badge>
                        ))}
                      </div>
                      {continuation.changed.length ? (
                        <div className="mt-2 space-y-1">
                          {continuation.changed.map((change: any) => (
                            <div key={change.field} className="text-2xs text-ink-2">
                              {change.field}: {String(change.from)} → {String(change.to)} ({change.impact})
                            </div>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  ) : null}
                </Panel>
              </div>

              {prepare.data.plan.warnings?.length ? (
                <Callout tone="warn" title="Warnings">
                  <ul className="space-y-1">
                    {prepare.data.plan.warnings.map((warning: string) => (
                      <li key={warning}>· {warning}</li>
                    ))}
                  </ul>
                </Callout>
              ) : null}

              <Panel>
                <SectionHeader title="Final configuration" subtitle="This exact object is written to spec.json and executed" actions={<Button size="sm" variant="subtle" onClick={() => void navigator.clipboard.writeText(JSON.stringify(request, null, 2))}>Copy</Button>} />
                <CodeBlock max="max-h-72">{JSON.stringify(request, null, 2)}</CodeBlock>
              </Panel>
            </>
          ) : null}
        </div>
      ) : null}

      <div className="flex items-center justify-between border-t border-line-soft pt-3">
        <Button icon={<ArrowLeft size={13} />} disabled={step === 0} onClick={() => setStep((value) => Math.max(0, value - 1))}>
          Back
        </Button>
        <div className="flex items-center gap-2">
          <span className="text-2xs text-ink-3">
            {baseModel ? basename(baseModel) : "from scratch"} · {datasets.length} dataset(s) · {METHOD_LABELS[method] ?? method}
          </span>
          {step < STEPS.length - 1 ? (
            <Button variant="primary" icon={<ArrowRight size={13} />} onClick={() => setStep((value) => Math.min(STEPS.length - 1, value + 1))}>
              Continue
            </Button>
          ) : (
            <Button
              variant="primary"
              icon={<Rocket size={13} />}
              loading={busy}
              disabled={!prepare.data?.ok}
              onClick={() => void start()}
              title={prepare.data?.ok ? "Start the run" : "Fix the blocking checks first"}
            >
              Start training
            </Button>
          )}
        </div>
      </div>

      <Panel>
        <SectionHeader title="Presets" subtitle="Filled in from the current method; every value stays editable" />
        <div className="flex flex-wrap gap-2">
          {[
            { label: "Low VRAM LoRA", patch: { method: "lora", precision: "fp16", batch_size: 1, gradient_accumulation: 8, sequence_length: 128, lora_rank: 8, gradient_checkpointing: true } },
            { label: "QLoRA", patch: { method: "qlora", quantization: "int4", batch_size: 1, gradient_accumulation: 8, lora_rank: 16 } },
            { label: "Full fine-tune", patch: { method: "full_finetune", precision: "bf16", batch_size: 1, gradient_accumulation: 16, learning_rate: 2e-5 } },
            { label: "Continued pretraining", patch: { method: "continued_pretraining", precision: "bf16", batch_size: 2, learning_rate: 5e-5 } },
            { label: "Chat SFT", patch: { method: "sft", batch_size: 2, learning_rate: 1e-4, template: "chatml" } },
            { label: "Tiny CPU run", patch: { method: "scratch", backend: "tiny", context_length: 8, hidden_size: 32, vocab_size: 320, max_steps: 200 } },
          ].map((preset) => (
            <Button
              key={preset.label}
              size="sm"
              variant="subtle"
              onClick={() => {
                const { method: presetMethod, backend: presetBackend, ...rest } = preset.patch as any;
                if (presetMethod) setMethod(presetMethod);
                if (presetBackend) setBackend(presetBackend);
                setConfig((current) => ({ ...current, ...rest }));
                void api.registry.add("presets", { id: `${Date.now()}`, name: preset.label, patch: preset.patch, createdAt: new Date().toISOString() });
                toast({ title: `${preset.label} applied`, body: "Values can still be edited before starting.", tone: "info" });
              }}
            >
              {preset.label}
            </Button>
          ))}
        </div>
        <div className="mt-2 text-2xs text-ink-3">
          {(registry?.presets ?? []).length} preset(s) saved in this workspace's registry.
        </div>
      </Panel>
    </div>
  );
}
