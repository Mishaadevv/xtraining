/** Types shared by the renderer, mirroring the Python backend payloads. */

export interface BackendError {
  code?: string;
  message: string;
  hint?: string;
  traceback?: string;
}

export interface Result {
  ok: boolean;
  error?: BackendError;
  [key: string]: unknown;
}

export type Method = "lora" | "qlora" | "sft" | "full" | "scratch";
export type Quantization = "none" | "4bit" | "8bit";
export type Precision = "auto" | "bf16" | "fp16" | "fp32";

export interface DatasetSelection {
  /** Null for a Hub dataset, which is referenced by `hf_id` instead. */
  path: string | null;
  name?: string;
  format: string;
  mapping: DatasetMapping | null;
  hf_id?: string | null;
  split?: string;
}

export interface DatasetMapping {
  kind: "pair" | "chat" | "text" | "unknown";
  instruction_field?: string | null;
  input_field?: string | null;
  output_field?: string | null;
  text_field?: string | null;
  messages_field?: string | null;
  messages_field_alternatives?: string[];
  template?: string;
  auto?: boolean;
  fields?: string[];
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
  device: "auto" | "cpu" | "cuda";
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
  error?: BackendError | null;
  issues: ValidationIssue[];
  mapping: DatasetMapping;
  dataset: {
    path: string;
    name: string;
    format: string;
    bytes?: number;
    records: number;
    files?: string[];
    file_errors?: { file: string; message: string; code: string }[];
    mixed_shapes?: { fields: string[]; files: string[] }[];
    container_key?: string | null;
    truncated?: boolean;
  };
  stats: {
    records?: number;
    usable?: number;
    empty?: number;
    empty_reasons?: Record<string, number>;
    short?: number;
    duplicates?: number;
    over_length?: number;
    avg_chars?: number;
    max_chars?: number;
    max_chars_index?: number;
    est_tokens_total?: number;
    est_tokens_avg?: number;
    roles?: Record<string, number>;
    field_coverage?: Record<string, number>;
  };
  preview: string[];
  sampled: boolean;
  context_length?: number;
}

export interface VramEstimate {
  available: boolean;
  reason?: string;
  params?: number;
  params_exact?: boolean;
  trainable_params?: number;
  weights_mb?: number;
  gradients_mb?: number;
  optimizer_mb?: number;
  activations_mb?: number;
  overhead_mb?: number;
  estimated_total_mb?: number;
  range_low_mb?: number;
  range_high_mb?: number;
  available_vram_mb?: number | null;
  headroom_mb?: number | null;
  verdict: "fits" | "tight" | "exceeds" | "unknown";
  suggestions?: string[];
  assumptions?: Record<string, unknown>;
}

export interface GpuDevice {
  index: number;
  name: string;
  memory_total_mb?: number | null;
  memory_used_mb?: number | null;
  memory_free_mb?: number | null;
  utilization_gpu?: number | null;
  temperature_c?: number | null;
  driver_version?: string | null;
  compute_capability?: string | null;
}

export interface GpuSample {
  available: boolean;
  reason?: string | null;
  gpu?: GpuDevice | null;
  gpus?: GpuDevice[];
  sampled_at: number;
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
  os: Record<string, string>;
  python: { version: string; executable: string };
  cpu: { model: string; logical_cores: number | null; architecture: string };
  memory: { total_mb: number | null };
  gpu: { available: boolean; reason?: string; gpus: GpuDevice[]; binary?: string };
  cuda: CudaInfo;
  cuda_toolkit: string | null;
  nvidia_ready: boolean;
  cuda_ready: boolean;
  training_device: "cuda" | "cpu";
  cuda_blockers: string[];
}

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

export interface InstallPlan {
  command: string;
  argv: string[];
  packages: string[];
  cuda_tag: string | null;
  note: string;
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
  command?: string;
  args?: string[];
  reason?: string;
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
  python: (PythonHealth & { packageDir?: string; backendPresent?: boolean }) | null;
  dependencies: DependencyInfo | null;
  backends: BackendCapability[] | null;
  installPlan: InstallPlan | null;
  error: BackendError | null;
  refreshedAt: number | null;
}

export interface Settings {
  version: number;
  interpreterPath: string | null;
  cudaWheelTag: string;
  hfCacheDir: string | null;
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

export interface DatasetEntry {
  id: string;
  name: string;
  /** A local path, or the Hub dataset id when `format` is "hf". */
  path: string;
  /** Set for Hugging Face Hub datasets. */
  hfId?: string;
  split?: string;
  format: string;
  isDirectory: boolean;
  sizeBytes: number | null;
  addedAt: number;
  validatedAt?: number;
  records: number;
  usable: number;
  status: "ok" | "warnings" | "errors" | "unvalidated";
  mapping: DatasetMapping | null;
  issues: ValidationIssue[];
  report: DatasetReport | null;
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
  method?: string;
  baseModel?: string;
  runId?: string;
  finalLoss?: number | null;
  steps?: number;
  status?: string;
  issues: ValidationIssue[];
  addedAt: number;
  createdAt?: number;
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

export type RunStatus =
  | "starting"
  | "running"
  | "completed"
  | "stopped"
  | "paused"
  | "failed";

export interface CheckpointEntry {
  name?: string;
  path: string;
  step: number;
  sizeBytes?: number;
  loss?: number | null;
  epoch?: number | null;
  hasOptimizer?: boolean;
  createdAt?: number;
}

export interface RunRecord {
  id: string;
  projectId: string | null;
  name: string;
  method: Method;
  baseModel: string;
  datasetName: string;
  datasetPath: string;
  status: RunStatus;
  phase: string;
  progress: number;
  step: number;
  totalSteps: number;
  loss: number | null;
  evalLoss: number | null;
  learningRate: number | null;
  gradNorm?: number | null;
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
  config: TrainingConfig;
  history: { loss?: number[]; eval_loss?: number[]; lr?: number[]; grad_norm?: number[]; epoch?: number[]; step?: number[] } | null;
  datasetReport: {
    source?: { name?: string; format?: string; records?: number; path?: string };
    mapping?: DatasetMapping;
    records?: number;
    samples?: number;
  } | null;
  error: BackendError | null;
  checkpoints: CheckpointEntry[];
  gpuSummary?: {
    samples: number;
    peakUtilization: number;
    averageUtilization: number;
    peakVramMb: number;
    peakTemperatureC: number;
    device: string;
  } | null;
  runSizeBytes?: number;
  trainableParams?: number;
  totalParams?: number;
  targetModules?: string[];
  device?: string;
  precision?: string;
  optimizer?: string;
  message?: string;
  pendingAction?: string;
  resumedFrom?: string | null;
}

export interface TrainingEventArgs {
  runId: string;
  event: string;
  detail: Record<string, unknown>;
}

export interface LogEntry {
  key: string;
  stream: "event" | "stderr" | "stdout";
  level: "info" | "warn" | "error";
  message: string;
  at: number;
  event?: string;
  detail?: Record<string, unknown>;
}

export interface Toast {
  id: string;
  title: string;
  message?: string;
  tone: "info" | "good" | "warn" | "bad";
}
