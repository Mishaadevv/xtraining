/** Shared shapes exchanged with the engine and the Electron main process. */

export type Result<T> = { ok: true; data: T } | { ok: false; error: EngineError };

export interface EngineError {
  code: string;
  message: string;
  hint?: string;
  detail?: string;
  context?: Record<string, unknown>;
}

export interface HardwareReport {
  os: { name: string; release: string; version: string; machine: string; python: string; python_executable: string };
  cpu: {
    model: string | null;
    physical_cores: number | null;
    logical_cores: number | null;
    architecture: string;
    flags: string[];
    avx2: boolean;
    avx512: boolean;
    notes: string[];
  };
  memory: { total: number | null; available: number | null };
  gpus: GpuDevice[];
  disks: Array<{ path: string; total: number; used: number; free: number; percent: number | null }>;
  torch: {
    installed: boolean;
    version: string | null;
    cuda_build: string | null;
    cuda_available: boolean;
    device_count: number;
    mps_available: boolean;
    notes: string[];
  };
  capabilities: Capabilities;
  notes: string[];
}

export interface GpuDevice {
  index: number;
  name: string;
  memory_total: number | null;
  memory_used_mb?: number | null;
  memory_free_mb?: number | null;
  utilization_gpu?: number | null;
  temperature_c?: number | null;
  power_draw_w?: number | null;
  driver_version?: string | null;
  compute_capability?: string | null;
  source?: string;
}

export interface Capabilities {
  cuda: boolean;
  cuda_compute_capability: number | null;
  mps: boolean;
  cpu_training: boolean;
  distribution: Record<string, boolean | number>;
  attention: Record<string, boolean>;
  quantization_backends: Record<string, boolean>;
  memory_optimizations: Record<string, boolean>;
  precisions: Array<{ name: string; available: boolean; reason: string }>;
  libraries: Record<string, { installed: boolean; version: string | null }>;
  cpu_instructions: Record<string, boolean>;
}

export interface LiveSample {
  timestamp: number;
  ram: { total: number | null; available: number | null; used: number | null; percent: number | null; note?: string | null };
  cpu: { percent: number | null; logical_cores: number | null };
  gpus: GpuDevice[];
  process: { rss: number | null; pid: number };
}

export interface Backend {
  id: string;
  name: string;
  description: string;
  available: boolean;
  reason: string;
  requires: string[];
  methods: string[];
  supports_gpu: boolean;
  supports_resume: boolean;
  supports_pause: boolean;
  supports_streaming_inference: boolean;
  supports_continue: boolean;
  supports_from_scratch: boolean;
  supports_adapters: boolean;
  supports_preference_training: boolean;
  speed_note: string;
}

export interface MethodAvailability {
  method: string;
  backends: string[];
  available: boolean;
  reasons: string[];
}

export interface EnvironmentReport {
  engine_python: { executable: string; version: string; prefix: string; in_venv: boolean };
  interpreters: Array<{
    label: string;
    executable: string;
    version: string;
    inVenv?: boolean;
    torch?: boolean;
    transformers?: boolean;
    zxtrain?: boolean;
    major?: number;
    minor?: number;
    torch_compatible?: boolean;
    torch_note?: string;
    score?: number;
    note?: string;
  }>;
  workspace_venv: { path: string; python: string; exists: boolean };
  packages: Record<string, { installed: boolean; version: string | null }>;
  ready: Record<string, boolean>;
  recommendations: string[];
  install_plan: { venv: string; steps: string[]; packages: string[]; index_url: string; notes: string[] };
  reported_at: string;
}

export interface DatasetField {
  name: string;
  type: string;
  types: Record<string, number>;
  missing: number;
  missing_percent: number | null;
}

export interface DatasetReport {
  path: string;
  name: string;
  kind: string;
  size_bytes: number;
  size_human: string;
  file_count: number;
  record_count: number;
  sampled: number;
  fields: DatasetField[];
  field_names: string[];
  duplicates: number;
  duplicate_percent: number | null;
  near_duplicates: number | null;
  empty_records: number;
  length: {
    average: number | null;
    max: number | null;
    min: number | null;
    p50?: number | null;
    p90?: number | null;
    p99?: number | null;
    histogram: Array<{ from: number; to: number; count: number }>;
  };
  token_estimate: { labelled: string; total: number | null; average: number | null; note?: string };
  media_kind: string | null;
  detected_mapping: Record<string, string>;
  mapping_suggestion: {
    mapping: Record<string, string>;
    confidence: Record<string, string>;
    fields: string[];
    unmapped: string[];
  };
  preview: Array<{
    index: number;
    record: Record<string, unknown>;
    normalised: Record<string, unknown>;
    length: number;
  }>;
  inspected_at: string;
  empty?: boolean;
}

export interface ModelReport {
  path: string;
  name: string;
  format: { kind: string; weight_files: Record<string, string[]>; is_adapter: boolean };
  size_bytes: number;
  file_count: number;
  architecture: Record<string, any>;
  config: Record<string, any>;
  generation_config: Record<string, any>;
  tokenizer: Record<string, any>;
  weights: {
    parameter_count: number | null;
    weight_bytes: number;
    dtypes: Record<string, number>;
    tensor_count?: number;
    source: string | null;
    errors?: string[];
    gguf?: Record<string, unknown>;
  };
  adapter: Record<string, unknown> | null;
  adapter_details: Record<string, any> | null;
  files: Array<{ name: string; size: number; size_human: string }>;
  estimated_vram: Record<string, number | string> | null;
  inspected_at: string;
}

export interface RunPlan {
  labelled: string;
  method: string;
  precision: string;
  quantization: string;
  parameters: { total: number; trainable: number; frozen: number; source: string };
  memory: {
    weights: number;
    gradients: number;
    optimizer: number;
    activations: number;
    overhead_factor: number;
    vram_estimate: number;
    ram_estimate: number;
    gpu_memory_available: number | null;
    ram_available: number | null;
  };
  steps: { per_epoch: number; total: number; effective_batch_size: number; tokens_per_step: number };
  data: {
    records: number;
    tokens: number;
    average_tokens_per_record: number;
    exact: boolean;
    basis: string;
    per_dataset: Array<Record<string, unknown>>;
  };
  disk: { checkpoint_size_estimate: number; planned_checkpoints: number; workspace_needed: number };
  risk: string;
  notes: string[];
  warnings: string[];
  formulas: Record<string, string>;
}

export interface PreparedRun {
  ok: boolean;
  errors: string[];
  warnings: string[];
  checks: Array<{ name: string; ok: boolean; message: string; level: string; hint: string }>;
  plan: RunPlan;
  datasets: Array<Record<string, unknown>>;
  model: { path: string | null; format: string | null; parameters: number | null; architecture: string | null } | null;
  continuation: ContinuationReport | null;
  backend: Backend | null;
  hardware_summary: { gpus: string[]; ram: number | null; cpu: string | null };
  prepared_at: string;
}

export interface ContinuationReport {
  verdict: string;
  verdict_text: string;
  restored: Array<{ item: string; restored: boolean; detail?: string }>;
  changed: Array<{ field: string; from: unknown; to: unknown; impact: string }>;
  checks: Array<{ name: string; ok: boolean; message: string; level?: string; hint?: string }>;
}

export interface JobEvent {
  type: string;
  [key: string]: unknown;
}

export interface JobSummary {
  job_id: string;
  epoch?: number | null;
  job_dir?: string;
  state: string;
  kind?: string;
  backend?: string;
  method?: string;
  step?: number | null;
  total_steps?: number | null;
  loss?: number | null;
  eval_loss?: number | null;
  learning_rate?: number | null;
  tokens_per_second?: number | null;
  eta_seconds?: number | null;
  elapsed_seconds?: number | null;
  message?: string | null;
  started_at?: string;
  updated_at?: string;
  created?: string;
  checkpoints?: Array<{ name: string; path: string; step: number | null; kind: string }>;
  result?: any;
  error?: EngineError | null;
  model?: string | null;
  datasets?: string[];
  metrics?: any[];
  config?: Record<string, any>;
  metrics_state?: Record<string, number | null>;
  attached?: boolean;
}

export interface Project {
  id: string;
  name: string;
  description?: string;
  createdAt: string;
  updatedAt?: string;
  tags?: string[];
  color?: string;
}

export interface RegistryEntry {
  id: string;
  name: string;
  path: string;
  addedAt: string;
  kind?: string;
  project?: string | null;
  tags?: string[];
  favorite?: boolean;
  notes?: string;
  summary?: Record<string, unknown>;
}

export interface Registry {
  projects: Project[];
  models: RegistryEntry[];
  datasets: RegistryEntry[];
  presets: Array<Record<string, any>>;
  evaluations: Array<Record<string, any>>;
  servers: Array<Record<string, any>>;
  conversations: Array<Record<string, any>>;
  workflows: Array<Record<string, any>>;
  notes: Array<Record<string, any>>;
}

/** The app's own update state, as the main process sees it. */
export interface UpdateState {
  currentVersion: string;
  supported: boolean;
  reason: string;
  mode: "development" | "installed" | "portable" | "appimage";
  status: "idle" | "checking" | "available" | "downloading" | "downloaded" | "up-to-date";
  channel: {
    source: "override" | "release" | "configured";
    provider: string | null;
    owner: string | null;
    repo: string | null;
    url: string | null;
    label: string;
    configFile: string;
  };
  signatureVerification: boolean;
  available: {
    version: string;
    releaseName: string | null;
    releaseNotes: string | null;
    releaseDate: string | null;
    size: number | null;
  } | null;
  progress: { percent: number; transferred: number; total: number; bytesPerSecond: number } | null;
  error: EngineError | null;
  /** A failed check nobody asked for: reported quietly, never as a failure. */
  backgroundError: EngineError | null;
  downloadedFile: string | null;
  lastCheckedAt: string | null;
  logFile: string | null;
}

export interface Settings {
  version: number;
  workspace: string | null;
  pythonPath: string | null;
  theme: "system" | "light" | "dark";
  defaultBackend: string | null;
  defaultMethod: string;
  defaultDevice: string;
  notifications: { jobFinished: boolean; jobFailed: boolean; serverStarted: boolean };
  updateChannel: string | null;
  checkForUpdatesOnStart: boolean;
  offlineMode: boolean;
  logLevel: string;
  checkpointPolicy: { keepLast: number; protectBest: boolean };
  window?: { width: number; height: number };
}

export interface TrainingConfig {
  method: string;
  backend: string | null;
  device: string;
  precision: string;
  quantization: string;
  sequence_length: number;
  batch_size: number;
  gradient_accumulation: number;
  epochs: number;
  max_steps: number;
  learning_rate: number;
  lr_scheduler: string;
  warmup_steps: number;
  weight_decay: number;
  optimizer: string;
  max_grad_norm: number;
  seed: number;
  save_every: number;
  logging_every: number;
  eval_every: number;
  checkpoint_limit: number;
  gradient_checkpointing: boolean;
  eval_ratio: number;
  lora_rank: number;
  lora_alpha: number;
  lora_dropout: number;
  target_modules: string[];
  template: string;
  hidden_size: number;
  context_length: number;
  vocab_size: number;
  workers: number;
}
