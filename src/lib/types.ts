/**
 * Types shared by the renderer.
 *
 * These mirror the payloads the Python backend and the Electron main process
 * actually produce — snake_case for anything that crosses the Python boundary
 * (training config, dataset reports, VRAM estimates) and camelCase for
 * Electron-native records (runs, library entries, settings).
 */

/* ------------------------------------------------------------------ errors */

export interface BackendError {
  code?: string;
  message: string;
  hint?: string;
  traceback?: string;
}

export interface CallResult {
  ok: boolean;
  error?: BackendError;
  [key: string]: unknown;
}

/* -------------------------------------------------------------- training */

export type Method = "lora" | "qlora" | "sft" | "full" | "scratch";
export type Quantization = "none" | "4bit" | "8bit";
export type Precision = "auto" | "bf16" | "fp16" | "fp32";

export interface DatasetMapping {
  kind: "pair" | "chat" | "text" | "unknown";
  instruction_field?: string | null;
  input_field?: string | null;
  output_field?: string | null;
  text_field?: string | null;
  messages_field?: string | null;
  messages_field_alternatives?: string[];
  template?: string | null;
  auto?: boolean;
  fields?: string[];
}

export interface DatasetSelection {
  /** Null for a Hub dataset, which is referenced by `hf_id` instead. */
  path: string | null;
  hf_id?: string | null;
  name?: string;
  format: string;
  mapping: DatasetMapping | null;
  split?: string;
}

export interface TrainingConfig {
  method: Method;
  base_model: string;
  output_name: string;
  epochs: number;
  batch_size: number;
  gradient_accumulation: number;
  learning_rate: number;
  lr_scheduler: string;
  warmup_ratio: number;
  weight_decay: number;
  max_grad_norm: number;
  context_length: number;
  lora_r: number;
  lora_alpha: number;
  lora_dropout: number;
  lora_target_modules: string | string[];
  quantization: Quantization;
  precision: Precision;
  gradient_checkpointing: boolean;
  optimizer: string;
  save_steps: number;
  save_total_limit: number;
  eval_split: number;
  logging_steps: number;
  max_samples: number;
  seed: number;
  /** auto: CUDA when present; cuda: force GPU; cpu: force CPU; both = cuda-if-present (alias of auto for users). */
  device: "auto" | "cpu" | "cuda" | "both";
  resume_from_checkpoint: string | null;
  dataset: DatasetSelection;
  trust_remote_code?: boolean;
  /** From-scratch architecture (method === "scratch"). */
  scratch_size?: string;
  scratch_layers?: number;
  scratch_hidden?: number;
  scratch_heads?: number;
  scratch_vocab?: number;
  scratch_ffn?: number;
}

export interface AutoReason {
  field: string;
  value: string | number | boolean;
  reason: string;
}

/* --------------------------------------------------------------- datasets */

export interface ValidationIssue {
  severity: "error" | "warning" | "info";
  code: string;
  message: string;
  hint?: string;
  count?: number;
  samples?: number[];
  breakdown?: Record<string, number>;
  file?: string;
  fields?: string[];
}

export interface DatasetReport {
  ok: boolean;
  status: "ok" | "warnings" | "errors";
  error: BackendError | null;
  issues: ValidationIssue[];
  dataset: {
    path: string;
    name: string;
    format: string;
    source?: string;
    split?: string | null;
    bytes: number | null;
    records: number;
    files: string[];
    file_errors: { file: string; message: string; code: string }[];
    mixed_shapes?: { fields: string[]; files: string[] }[];
    container_key?: string | null;
    truncated?: boolean;
    /** Compression codec when the file was read through gzip/bz2/xz. */
    compression?: string | null;
    /** The table or sheet that was read, for sqlite and spreadsheets. */
    items?: string | null;
  };
  stats: {
    records?: number;
    usable?: number;
    unusable?: number;
    empty?: number;
    duplicates?: number;
    over_length?: number;
    very_short?: number;
    avg_chars?: number;
    max_chars?: number;
    est_tokens_total?: number;
    est_tokens_avg?: number;
    roles?: Record<string, number>;
    empty_reasons?: Record<string, number>;
    field_coverage?: Record<string, number>;
  };
  preview: string[];
  mapping: DatasetMapping;
  sampled: boolean;
  context_length?: number;
}

export interface DatasetEntry {
  id: string;
  name: string;
  /** A local path, or the Hub dataset id when `format` is "hf". */
  path: string;
  hfId?: string;
  split?: string;
  format: string;
  isDirectory: boolean;
  sizeBytes: number | null;
  addedAt: number;
  validatedAt?: number | null;
  records: number;
  usable: number;
  status: "ok" | "warnings" | "errors" | "unvalidated";
  mapping: DatasetMapping | null;
  issues: ValidationIssue[];
  report: DatasetReport | null;
  /** Where the entry came from: the scanned folder, a manual import, or the Hub. */
  origin?: "folder" | "import" | "hub";
  /** The folder this entry was found in, when it came from a scan. */
  folder?: string | null;
  /** Scanned datasets the user removed from the list stay hidden, not deleted. */
  hidden?: boolean;
}

/** One dataset type the backend can read (reported by `dataset-formats`). */
export interface DatasetFormatInfo {
  id: string;
  label: string;
  requires: string | null;
  available: boolean;
  hint: string;
  extensions: string[];
}

export interface DatasetFormats {
  ok?: boolean;
  error?: BackendError;
  formats: DatasetFormatInfo[];
  extensions: string[];
  compression: string[];
  export_formats: string[];
  sources?: string[];
  hub_available?: boolean;
}

/* ----------------------------------------------------------------- models */

export interface ModelIssue {
  severity: "error" | "warning" | "info";
  code: string;
  message: string;
  hint?: string;
}

export interface ModelEntry {
  id: string;
  name: string;
  source: string;
  kind: "local" | "huggingface" | "trained";
  path: string | null;
  cached?: boolean;
  params: number | null;
  paramsExact?: boolean;
  sizeBytes: number | null;
  architecture: string | null;
  maxPositionEmbeddings: number | null;
  torchDtype: string | null;
  trainable: boolean;
  trained: boolean;
  adapter: boolean;
  /** The folder itself is a PEFT adapter (it can be fine-tuned further). */
  isAdapterFolder?: boolean;
  /** Base model recorded inside adapter_config.json (for adapter folders). */
  adapterBase?: string | null;
  method?: string;
  baseModel?: string;
  runId?: string;
  finalLoss?: number | null;
  steps?: number;
  status?: string;
  totalParams?: number | null;
  issues: ModelIssue[];
  addedAt: number;
  createdAt?: number;
}

/** What the backend reported about a model folder, before exporting it. */
export interface ModelExportInfo {
  path: string;
  name: string;
  files: string[];
  weight_files: string[];
  is_adapter: boolean;
  adapter_config: Record<string, unknown> | null;
  base_model: string | null;
  checkpoints: string[];
  size_bytes: number;
  modes: {
    copy: { ready: boolean };
    /** A packed export is the same content as one .zip file. */
    pack?: { ready: boolean };
    merge: { ready: boolean; applicable: boolean; missing: string[] };
  };
  blockers: { code: string; message: string; hint: string }[];
}

export interface ModelInspectInfo {
  name?: string;
  source?: string;
  kind?: string;
  cached?: boolean;
  cached_path?: string | null;
  params?: number | null;
  params_exact?: boolean;
  size_bytes?: number | null;
  max_position_embeddings?: number | null;
  trainable?: boolean;
  adapter?: boolean;
  adapter_base?: string | null;
  fields?: Record<string, unknown> & { architectures?: string[] };
  issues?: ModelIssue[];
}

/* ---------------------------------------------------------------- runs */

export type RunStatus =
  | "starting"
  | "running"
  | "completed"
  | "stopped"
  | "paused"
  | "failed";

export interface CheckpointEntry {
  path: string;
  step: number;
  sizeBytes?: number | null;
  loss?: number | null;
  epoch?: number | null;
  hasOptimizer?: boolean;
  createdAt?: number;
}

export interface GpuSummary {
  samples: number;
  peakUtilization: number;
  averageUtilization: number;
  peakVramMb: number;
  peakTemperatureC: number;
  device: string;
}

export interface RunRecord {
  id: string;
  projectId: string | null;
  name: string;
  method: Method;
  baseModel: string;
  datasetName: string | null;
  datasetPath: string | null;
  status: RunStatus;
  phase: string;
  progress: number;
  step: number;
  totalSteps: number;
  loss: number | null;
  evalLoss: number | null;
  learningRate: number | null;
  epoch: number;
  elapsedSeconds: number;
  etaSeconds: number | null;
  samplesPerSecond: number | null;
  secondsPerStep: number | null;
  gpuMemoryAllocatedMb: number | null;
  gpuMemoryReservedMb: number | null;
  startedAt: number;
  finishedAt: number | null;
  runDir: string;
  outputDir: string | null;
  lastCheckpoint: string | null;
  finalLoss: number | null;
  config: TrainingConfig | null;
  history: {
    step?: number[];
    loss?: number[];
    eval_loss?: number[];
    learning_rate?: number[];
    grad_norm?: number[];
    epoch?: number[];
  } | null;
  datasetReport: {
    records?: number;
    usable?: number;
    mapping?: DatasetMapping;
    source?: { name?: string; format?: string; path?: string };
  } | null;
  error: BackendError | null;
  checkpoints: CheckpointEntry[];
  gpuSummary?: GpuSummary | null;
  runSizeBytes?: number | null;
  trainableParams?: number | null;
  totalParams?: number | null;
  targetModules?: string[];
  device?: string;
  precision?: string;
  optimizer?: string;
  message?: string;
  pendingAction?: "stopping" | "pausing" | null;
  resumedFrom?: string | null;
}

export interface Project {
  id: string;
  name: string;
  description: string;
  createdAt: number;
  updatedAt: number;
  method: Method | null;
  baseModel: string | null;
  datasetName: string | null;
  datasetPath: string | null;
  config: Partial<TrainingConfig> | null;
  runIds: string[];
  runCount: number;
  lastStatus: string | null;
  lastRunAt: number | null;
  bestLoss: number | null;
}

/* ------------------------------------------------------------- estimates */

export interface VramEstimate {
  available: boolean;
  reason?: string;
  params?: number | null;
  params_exact?: boolean;
  trainable_params?: number | null;
  weights_mb?: number | null;
  gradients_mb?: number | null;
  optimizer_mb?: number | null;
  activations_mb?: number | null;
  overhead_mb?: number | null;
  estimated_total_mb?: number | null;
  range_low_mb?: number | null;
  range_high_mb?: number | null;
  available_vram_mb?: number | null;
  headroom_mb?: number | null;
  verdict: "fits" | "tight" | "exceeds" | "unknown";
  suggestions?: string[];
}

export interface InstallPlan {
  command: string;
  argv: string[];
  packages: string[];
  cuda_tag: string | null;
  note?: string;
}

/* ------------------------------------------------------------- hardware */

export interface GpuDevice {
  index?: number;
  name?: string;
  memory_total_mb?: number | null;
  memory_used_mb?: number | null;
  memory_free_mb?: number | null;
  utilization_gpu?: number | null;
  utilization_memory?: number | null;
  temperature_c?: number | null;
  driver_version?: string | null;
  compute_capability?: string | null;
}

/** One nvidia-smi reading, sampled by the main process. */
export interface GpuSample {
  available: boolean;
  reason?: string | null;
  gpu?: GpuDevice | null;
  gpus: GpuDevice[];
  sampled_at?: number;
}

export interface CudaInfo {
  torch_installed: boolean;
  torch_version?: string | null;
  cuda_build_version?: string | null;
  cudnn_version?: string | null;
  available: boolean;
  device_count?: number;
  bf16_supported?: boolean;
  devices: {
    index: number;
    name?: string;
    total_memory_mb?: number;
    compute_capability?: string;
    error?: string;
  }[];
  reason?: string;
}

export interface HardwareSnapshot {
  os: { system?: string; release?: string; hostname?: string };
  python: { version: string; executable: string };
  cpu: { model: string; logical_cores: number | null; architecture: string };
  memory: { total_mb: number | null };
  gpu: { available: boolean; reason?: string; binary?: string; gpus: GpuDevice[] };
  cuda: CudaInfo;
  cuda_toolkit: string | null;
  nvidia_ready: boolean;
  cuda_ready: boolean;
  training_device: "cuda" | "cpu";
  cuda_blockers: string[];
}

/* ----------------------------------------------------------- environment */

export interface DependencyPackage {
  installed: boolean;
  version: string | null;
  pip: string;
  group: string;
  purpose: string;
}

export interface DependencyInfo {
  python: { version: string; executable: string; implementation: string; platform: string };
  packages: Record<string, DependencyPackage>;
  missing_core: string[];
  capabilities: Record<string, { ready: boolean; requires: string[]; missing: string[] }>;
  training_ready: boolean;
}

export interface BackendCapability {
  name: string;
  label: string;
  methods: Method[];
  requires: string[];
  available: boolean;
  missing: string[];
}

export interface PythonHealth {
  available: boolean;
  version?: string;
  executable?: string;
  label?: string;
  reason?: string;
  packageDir?: string;
  backendPresent?: boolean;
}

/** A single interpreter candidate found on this machine. */
export interface InterpreterCandidate {
  command: string;
  args?: string[];
  label: string;
  available: boolean;
  reason?: string;
  info?: { executable: string; version: string; implementation: string };
}

export interface EnvSnapshot {
  loading: boolean;
  system: {
    platform: string;
    release: string;
    hostname: string;
    cpu: { model: string; logical_cores: number };
    memory: { total_mb: number; free_mb: number };
    uptime_seconds: number;
    electron: string;
    node: string;
  } | null;
  smi: GpuSample | null;
  hardware: HardwareSnapshot | null;
  python: PythonHealth | null;
  dependencies: DependencyInfo | null;
  backends: BackendCapability[] | null;
  installPlan: InstallPlan | null;
  error: BackendError | null;
  refreshedAt: number | null;
}

/* -------------------------------------------------------------- settings */

export interface Settings {
  version: number;
  interpreterPath: string | null;
  cudaWheelTag: string;
  hfCacheDir: string | null;
  /** The folder scanned for datasets; null means the app's own folder. */
  datasetsDir: string | null;
  theme: "dark" | "light";
  /** Off => the wizard uses the documented defaults and says so. */
  autoConfigure: boolean;
  /** The initial Simple/Advanced state of a new wizard. */
  simpleMode: boolean;
  advanced: {
    trustRemoteCode: boolean;
  };
  lastProjectId: string | null;
}

export interface TokenStorageInfo {
  present: boolean;
  encrypted: boolean;
  backend?: string;
  warning?: string;
}

/* ------------------------------------------------------------------ misc */

export interface AppInfo {
  name: string;
  version: string;
  platform: string;
  electron: string;
  node: string;
  chrome: string;
  userData: string;
  runsDir: string;
  modelsDir: string;
  datasetsDir: string;
  hfCacheDir: string;
  backendDir: string;
  encryptionAvailable: boolean;
}

export interface Toast {
  id: string;
  title: string;
  message?: string;
  tone: "info" | "good" | "warn" | "bad";
}

export interface LogEntry {
  key: string;
  stream: "event" | "stderr";
  level: "info" | "warn" | "error";
  message: string;
  at: number;
  event?: string;
}
