import { Store } from "./store";
import { bridge, isDesktop, type ModelExportInfo } from "@/lib/bridge";
import { basename } from "@/lib/utils";
import type {
  AutoReason,
  BackendError,
  DatasetEntry,
  DatasetReport,
  EnvSnapshot,
  HardwareSnapshot,
  LogEntry,
  Method,
  ModelEntry,
  Project,
  RunRecord,
  Settings,
  Toast,
  TrainingConfig,
  ValidationIssue,
  VramEstimate,
} from "@/lib/types";

export type Page =
  | "projects"
  | "new"
  | "training"
  | "models"
  | "datasets"
  | "playground"
  | "hardware"
  | "settings";

export interface SeriesPoint {
  step: number;
  loss: number | null;
  lr: number | null;
  epoch: number | null;
  at: number;
}

export interface GpuPoint {
  at: number;
  utilization: number | null;
  vramUsedMb: number | null;
  vramTotalMb: number | null;
  temperature: number | null;
}

export interface WizardState {
  step: number;
  method: Method;
  baseModel: string;
  modelEntryId: string | null;
  modelInfo: Record<string, unknown> | null;
  datasetId: string | null;
  datasetReport: DatasetReport | null;
  projectId: string | null;
  projectName: string;
  simple: boolean;
  config: TrainingConfig | null;
  autoReasons: AutoReason[];
  autoIssue: BackendError | null;
  estimate: VramEstimate | null;
  issues: ValidationIssue[];
  running: boolean;
  error: BackendError | null;
  blockers: BackendError[];
  lastUpdated: number | null;
}

export interface PlaygroundState {
  modelDir: string | null;
  modelName: string | null;
  loaded: boolean;
  loading: boolean;
  generating: boolean;
  streamed: string;
  output: string;
  error: BackendError | null;
  history: { prompt: string; output: string; seconds: number | null; tokensPerSecond: number | null }[];
  params: { maxNewTokens: number; temperature: number; topP: number; repetitionPenalty: number };
}

export interface AppState {
  booted: boolean;
  bootError: BackendError | null;
  page: Page;
  env: EnvSnapshot;
  appInfo: Record<string, unknown> | null;
  settings: Settings | null;
  projects: Project[];
  datasets: DatasetEntry[];
  models: ModelEntry[];
  runs: RunRecord[];
  selectedRunId: string | null;
  selectedRun: RunRecord | null;
  series: SeriesPoint[];
  gpuSeries: GpuPoint[];
  liveGpu: GpuPoint | null;
  logs: LogEntry[];
  logsLoading: boolean;
  wizard: WizardState;
  playground: PlaygroundState;
  toasts: Toast[];
  busy: Record<string, boolean>;
}

const initialWizard: WizardState = {
  step: 0,
  method: "lora",
  baseModel: "",
  modelEntryId: null,
  modelInfo: null,
  datasetId: null,
  datasetReport: null,
  projectId: null,
  projectName: "",
  simple: true,
  config: null,
  autoReasons: [],
  autoIssue: null,
  estimate: null,
  issues: [],
  running: false,
  error: null,
  blockers: [],
  lastUpdated: null,
};

const initialState: AppState = {
  booted: false,
  bootError: null,
  page: "projects",
  env: {
    loading: false,
    system: null,
    smi: null,
    hardware: null,
    python: null,
    dependencies: null,
    backends: null,
    installPlan: null,
    error: null,
    refreshedAt: null,
  },
  appInfo: null,
  settings: null,
  projects: [],
  datasets: [],
  models: [],
  runs: [],
  selectedRunId: null,
  selectedRun: null,
  series: [],
  gpuSeries: [],
  liveGpu: null,
  logs: [],
  logsLoading: false,
  wizard: initialWizard,
  playground: {
    modelDir: null,
    modelName: null,
    loaded: false,
    loading: false,
    generating: false,
    streamed: "",
    output: "",
    error: null,
    history: [],
    params: { maxNewTokens: 256, temperature: 0.7, topP: 0.9, repetitionPenalty: 1.1 },
  },
  toasts: [],
  busy: {},
};

export const appStore = new Store<AppState>(initialState);

const MAX_LOGS = 1500;
const MAX_SERIES = 2000;
const MAX_GPU = 1200;

let toastCounter = 0;
let logCounter = 0;

/* ------------------------------------------------------------------ toasts */

export function pushToast(toast: Omit<Toast, "id">): void {
  toastCounter += 1;
  const id = `toast_${toastCounter}`;
  appStore.set((state) => ({ toasts: [...state.toasts, { ...toast, id }] }));
  setTimeout(() => dismissToast(id), toast.tone === "bad" ? 12000 : 6000);
}

export function dismissToast(id: string): void {
  appStore.set((state) => ({ toasts: state.toasts.filter((entry) => entry.id !== id) }));
}

export function toastError(error: BackendError | undefined | null, title = "Something went wrong"): void {
  if (!error) return;
  pushToast({
    title,
    message: [error.message, error.hint].filter(Boolean).join(" — "),
    tone: "bad",
  });
}

function setBusy(key: string, value: boolean): void {
  appStore.set((state) => ({ busy: { ...state.busy, [key]: value } }));
}

/* --------------------------------------------------------------- logging */

function classifyLog(entry: LogEntry): LogEntry {
  const detail = entry.detail || {};
  const level = (detail.level as string) || "info";
  return {
    ...entry,
    level: level === "warn" || level === "error" ? (level as LogEntry["level"]) : "info",
  };
}

export function appendLog(entry: Omit<LogEntry, "key" | "at"> & { at?: number }): void {
  logCounter += 1;
  const full = classifyLog({
    key: `log_${logCounter}`,
    stream: entry.stream,
    level: entry.level,
    message: entry.message,
    at: entry.at ?? Date.now(),
    event: entry.event,
    detail: entry.detail,
  });
  appStore.set((state) => ({ logs: [...state.logs, full].slice(-MAX_LOGS) }));
}

export function clearLogs(): void {
  appStore.set({ logs: [] });
}

/* -------------------------------------------------------------- navigation */

export function navigate(page: Page): void {
  appStore.set({ page });
}

export function setWizard(
  patch: Partial<WizardState> | ((wizard: WizardState) => Partial<WizardState>),
): void {
  appStore.set((state) => ({
    wizard: {
      ...state.wizard,
      ...(typeof patch === "function" ? patch(state.wizard) : patch),
    },
  }));
}

export function resetWizard(preset: Partial<WizardState> = {}): void {
  // Settings → Advanced decides whether new runs open in Simple or Advanced, and
  // turning automatic configuration off means the user wants the full form. The
  // wizard's own toggle still overrides this for the current run.
  const settings = appStore.get().settings;
  const simple = settings?.autoConfigure === false ? false : settings?.simpleMode ?? initialWizard.simple;
  appStore.set({ wizard: { ...initialWizard, simple, ...preset } });
}

/* -------------------------------------------------------------- bootstrap */

export async function bootstrap(): Promise<void> {
  appStore.set({ env: { ...appStore.get().env, loading: true } });

  if (!isDesktop) {
    appStore.set({
      booted: true,
      bootError: {
        code: "desktop_only",
        message: "The desktop bridge is not available.",
        hint: "This window is rendering the interface without the Electron shell. "
          + "Launch ZeqouXTraining with `npm run dev` to train models.",
      },
      env: { ...appStore.get().env, loading: false },
    });
    return;
  }

  const [info, settings] = await Promise.all([bridge.app.info(), bridge.settings.get()]);
  appStore.set({
    appInfo: info.ok ? (info as Record<string, unknown>) : null,
    settings: settings.settings ?? null,
  });
  applyTheme(settings.settings?.theme ?? "dark");

  await Promise.all([refreshEnv(), refreshProjects(), refreshDatasets(), refreshModels(), refreshRuns()]);
  appStore.set({ booted: true });

  const runs = appStore.get().runs;
  const live = runs.find((run) => run.status === "running" || run.status === "starting");
  if (live) {
    appStore.set({ page: "training", selectedRunId: live.id, selectedRun: live });
    await loadRunDetails(live.id);
  }
}

export function applyTheme(theme: "dark" | "light"): void {
  document.documentElement.dataset.theme = theme;
  document.documentElement.classList.toggle("dark", theme === "dark");
}

/* -------------------------------------------------------------------- env */

export async function refreshEnv(force = false): Promise<void> {
  if (!isDesktop) return;
  appStore.set((state) => ({ env: { ...state.env, loading: true } }));
  const result = await bridge.env.detect({ force });
  if (!result.ok) {
    appStore.set((state) => ({
      env: { ...state.env, loading: false, error: (result.error as BackendError) ?? null },
    }));
    return;
  }
  appStore.set((state) => ({
    env: {
      loading: false,
      system: (result.system as EnvSnapshot["system"]) ?? state.env.system,
      smi: (result.smi as EnvSnapshot["smi"]) ?? state.env.smi,
      hardware: (result.hardware as HardwareSnapshot | null) ?? state.env.hardware,
      python: (result.python as EnvSnapshot["python"]) ?? state.env.python,
      dependencies: (result.dependencies as EnvSnapshot["dependencies"]) ?? state.env.dependencies,
      backends: (result.backends as EnvSnapshot["backends"]) ?? state.env.backends,
      installPlan: (result.installPlan as EnvSnapshot["installPlan"]) ?? state.env.installPlan,
      error: (result.backendError as BackendError) ?? null,
      refreshedAt: Date.now(),
    },
  }));
}

export async function refreshHardware(): Promise<void> {
  if (!isDesktop) return;
  const result = await bridge.hardware.detect();
  if (!result.ok) return;
  appStore.set((state) => ({ env: { ...state.env, hardware: result.hardware ?? null } }));
}

export async function discoverInterpreters(force = false): Promise<Record<string, unknown> | null> {
  if (!isDesktop) return null;
  const result = await bridge.env.interpreters({ force });
  return result.ok ? (result as Record<string, unknown>) : null;
}

export async function selectInterpreter(executablePath: string | null): Promise<void> {
  const result = await bridge.env.setInterpreter(executablePath);
  if (!result.ok) {
    toastError(result.error as BackendError, "Could not select the interpreter");
    return;
  }
  await refreshEnv(true);
  pushToast({ title: "Interpreter updated", tone: "good" });
}

/* ---------------------------------------------------------- settings side */

export async function updateSettings(patch: Partial<Settings>): Promise<void> {
  const result = await bridge.settings.set(patch);
  if (!result.ok) {
    toastError(result.error as BackendError, "Could not save settings");
    return;
  }
  appStore.set({ settings: result.settings ?? null });
  if (patch.theme) applyTheme(patch.theme);
  // Toggling Simple mode should be visible immediately on a wizard that has
  // not been configured yet, not only on the next new run.
  if (typeof patch.simpleMode === "boolean" && !appStore.get().wizard.config) {
    setWizard({ simple: patch.simpleMode });
  }
  // Turning automatic configuration off on an unconfigured wizard drops straight
  // into Advanced, so the controls it governs are actually visible.
  if (patch.autoConfigure === false && !appStore.get().wizard.config) {
    setWizard({ simple: false });
  }
}

export async function saveHfToken(token: string | null): Promise<void> {
  const result = await bridge.settings.setToken(token);
  if (!result.ok) {
    toastError(result.error as BackendError, "Could not store the token");
    return;
  }
  pushToast({
    title: token ? "Token stored" : "Token removed",
    message: token
      ? "Encrypted with the operating system keychain."
      : undefined,
    tone: "good",
  });
}

/* -------------------------------------------------------------- registries */

export async function refreshProjects(): Promise<void> {
  if (!isDesktop) return;
  const result = await bridge.projects.list();
  if (result.ok && result.projects) appStore.set({ projects: result.projects });
}

export async function refreshDatasets(): Promise<void> {
  if (!isDesktop) return;
  const result = await bridge.datasets.list();
  if (result.ok && Array.isArray(result.datasets)) {
    appStore.set({ datasets: result.datasets as DatasetEntry[] });
  }
}

export async function refreshModels(): Promise<void> {
  if (!isDesktop) return;
  const result = await bridge.models.list();
  if (result.ok && Array.isArray(result.models)) {
    appStore.set({ models: result.models as ModelEntry[] });
  }
}

export async function refreshRuns(): Promise<void> {
  if (!isDesktop) return;
  const result = await bridge.training.runs(200);
  if (result.ok && result.runs) {
    appStore.set({ runs: result.runs });
    const selected = appStore.get().selectedRunId;
    if (selected) {
      const current = result.runs.find((run) => run.id === selected);
      if (current) appStore.set({ selectedRun: current });
    }
  }
}

/* --------------------------------------------------------------- datasets */

export async function pickAndImportDatasets(): Promise<void> {
  const picked = await bridge.dialogs.pickDataset();
  const paths = (picked.paths as string[]) || [];
  if (!paths.length) return;
  await importDatasetPaths(paths);
}

export async function importDatasetPaths(paths: string[]): Promise<void> {
  setBusy("importDataset", true);
  try {
    const result = await bridge.datasets.import(paths);
    if (!result.ok) {
      toastError(result.error as BackendError, "Import failed");
      return;
    }
    const imported = (result.imported as DatasetEntry[]) || [];
    const failed = (result.failed as { path: string; error: BackendError }[]) || [];
    await refreshDatasets();
    if (imported.length) {
      pushToast({
        title: imported.length === 1 ? "Dataset imported" : `${imported.length} datasets imported`,
        message: "Validating now…",
        tone: "good",
      });
      for (const dataset of imported) {
        // eslint-disable-next-line no-await-in-loop - sequential keeps the report order predictable
        await validateDataset(dataset.id);
      }
    }
    for (const failure of failed) {
      toastError(failure.error, `Could not import ${basename(failure.path)}`);
    }
  } finally {
    setBusy("importDataset", false);
  }
}

/** Ask the user for a folder. Returns null when they cancel. */
export async function pickDirectory(title: string): Promise<string | null> {
  if (!isDesktop) return null;
  const result = await bridge.dialogs.pickDirectory(title);
  const paths = (result.paths as string[]) || [];
  return paths[0] ?? null;
}

/** Read what a trained model folder holds before offering to export it. */
export async function inspectModelExport(modelId: string): Promise<ModelExportInfo | null> {
  const result = await bridge.models.exportInfo({ modelId });
  if (!result.ok || !result.info) {
    toastError((result.error as BackendError) ?? null, "Could not inspect the model folder");
    return null;
  }
  return result.info;
}

/**
 * Export a model folder. Copy mode always works; merging the adapter into the
 * base model needs the ML runtime, and the backend refuses honestly without it.
 */
export async function exportModel(
  modelId: string,
  outputDir: string,
  merge: boolean,
): Promise<{ outputDir: string; mode: string; files: string[] } | null> {
  setBusy("exportModel", true);
  try {
    const result = await bridge.models.export({ modelId, outputDir, merge });
    if (!result.ok) {
      toastError((result.error as BackendError) ?? null, "Export failed");
      return null;
    }
    const output = String(result.output_dir ?? outputDir);
    pushToast({
      title: merge ? "Merged model exported" : "Export finished",
      message: output,
      tone: "good",
    });
    return {
      outputDir: output,
      mode: String(result.mode ?? "copy"),
      files: (result.files as string[]) ?? [],
    };
  } finally {
    setBusy("exportModel", false);
  }
}

/**
 * Register a Hugging Face dataset by id. The rows are fetched by the backend
 * during validation or at training start, so this is instant.
 */
export async function addHfDataset(source: string, split = "train"): Promise<DatasetEntry | null> {
  setBusy("importDataset", true);
  try {
    const result = await bridge.datasets.addHf({ id: source, split });
    if (!result.ok) {
      toastError(result.error as BackendError, "Could not add the dataset");
      return null;
    }
    await refreshDatasets();
    const entry = (result.dataset as DatasetEntry | undefined) ?? null;
    if (entry) {
      pushToast({ title: "Hub dataset added", message: "Validating now…", tone: "good" });
      await validateDataset(entry.id);
    }
    return entry;
  } finally {
    setBusy("importDataset", false);
  }
}

export async function validateDataset(id: string, contextLength?: number): Promise<DatasetReport | null> {
  setBusy(`dataset:${id}`, true);
  try {
    const result = await bridge.datasets.validate({ id, contextLength });
    if (!result.ok || !result.report) {
      toastError((result.error as BackendError) ?? null, "Validation failed");
      return null;
    }
    await refreshDatasets();
    const report = result.report;
    if (report.status === "errors") {
      pushToast({
        title: "Dataset has problems",
        message: report.issues.find((issue) => issue.severity === "error")?.message,
        tone: "warn",
      });
    }
    return report;
  } finally {
    setBusy(`dataset:${id}`, false);
  }
}

export async function previewDataset(id: string, limit = 5): Promise<string[] | null> {
  const result = await bridge.datasets.preview({ id, limit });
  if (!result.ok) {
    toastError(result.error as BackendError, "Preview failed");
    return null;
  }
  return (result.samples as string[]) || [];
}

export async function removeDataset(id: string): Promise<void> {
  const result = await bridge.datasets.remove(id);
  if (!result.ok) {
    toastError(result.error as BackendError, "Could not remove the dataset");
    return;
  }
  await refreshDatasets();
  pushToast({ title: "Dataset removed", tone: "info" });
}

/* ----------------------------------------------------------------- models */

export async function pickAndAddLocalModel(): Promise<void> {
  const picked = await bridge.dialogs.pickModelFolder();
  const paths = (picked.paths as string[]) || [];
  if (!paths.length) return;
  await addModel(paths[0]);
}

export async function addModel(source: string): Promise<ModelEntry | null> {
  const trimmed = source.trim();
  if (!trimmed) return null;
  setBusy("addModel", true);
  try {
    const result = await bridge.models.add(trimmed);
    if (!result.ok || !result.model) {
      toastError((result.error as BackendError) ?? null, "Could not add the model");
      return null;
    }
    await refreshModels();
    const issues = result.model.issues || [];
    const blocking = issues.filter((issue) => issue.severity === "error");
    pushToast({
      title: "Model added",
      message: blocking.length ? blocking[0].message : result.model.name,
      tone: blocking.length ? "warn" : "good",
    });
    return result.model;
  } finally {
    setBusy("addModel", false);
  }
}

export async function removeModel(id: string): Promise<void> {
  const result = await bridge.models.remove(id);
  if (!result.ok) {
    toastError(result.error as BackendError, "Could not remove the model");
    return;
  }
  await refreshModels();
  pushToast({ title: "Model removed", tone: "info" });
}

/* ----------------------------------------------------------------- wizard */

export async function wizardSelectModel(source: string, entryId: string | null): Promise<void> {
  setWizard({ baseModel: source, modelEntryId: entryId, modelInfo: null, autoReasons: [], estimate: null });
  if (!isDesktop || !source) return;

  setBusy("inspectModel", true);
  try {
    const result = await bridge.models.inspect(source);
    if (result.ok && result.info) {
      setWizard({ modelInfo: result.info as Record<string, unknown> });
    } else if (result.error) {
      setWizard({ autoIssue: result.error as BackendError });
    }
  } finally {
    setBusy("inspectModel", false);
  }
  await runAutoConfig();
}

export async function wizardSelectDataset(id: string): Promise<void> {
  const entry = appStore.get().datasets.find((dataset) => dataset.id === id);
  setWizard({ datasetId: id, datasetReport: entry?.report ?? null });
  if (!entry) return;

  setBusy("wizardDataset", true);
  try {
    const report = await validateDataset(id, appStore.get().wizard.config?.context_length ?? 512);
    setWizard({ datasetReport: report });
  } finally {
    setBusy("wizardDataset", false);
  }
  await runAutoConfig();
}

export async function wizardSelectMethod(method: Method): Promise<void> {
  setWizard({ method });
  await runAutoConfig();
}

export async function runAutoConfig(): Promise<void> {
  if (!isDesktop) return;
  const wizard = appStore.get().wizard;
  if (!wizard.baseModel && !wizard.datasetId) return;

  setBusy("autoConfig", true);
  try {
    const datasetPath = resolveDatasetPath(wizard.datasetId);
    const result = await bridge.training.autoConfig({
      baseModel: wizard.baseModel || undefined,
      datasetPath: datasetPath || undefined,
      contextLength: wizard.config?.context_length ?? 512,
      method: wizard.method,
      // Off means: give me the documented defaults, not choices made for me.
      baseline: appStore.get().settings?.autoConfigure === false,
    });

    if (!result.ok) {
      setWizard({ autoIssue: (result.error as BackendError) ?? null });
      return;
    }

    const config = result.config as TrainingConfig | undefined;
    const reasons = (result.reasons as AutoReason[]) || [];
    const issues = (result.issues as ValidationIssue[]) || [];
    const estimate = (result.estimate as VramEstimate) || null;
    const hardware = (result.hardware as HardwareSnapshot) || null;
    const modelInfo = (result.model as Record<string, unknown>) || null;
    const datasetReport = (result.dataset_report as DatasetReport) || null;
    const modelError = result.model_error as string | undefined;

    setWizard((current) => ({
      ...current,
      config: config ? { ...config, method: current.method, base_model: current.baseModel || config.base_model } : current.config,
      autoReasons: reasons,
      issues,
      estimate,
      modelInfo: modelInfo || current.modelInfo,
      datasetReport: datasetReport || current.datasetReport,
      autoIssue: modelError ? { code: "model_inspect", message: modelError } : null,
      baseModel: current.baseModel || config?.base_model || "",
      projectName: current.projectName || defaultProjectName(current.baseModel, datasetPath),
      lastUpdated: Date.now(),
    }));

    if (hardware) {
      appStore.set((state) => ({ env: { ...state.env, hardware } }));
    }
  } finally {
    setBusy("autoConfig", false);
  }
}

function defaultProjectName(model: string, datasetPath: string | null): string {
  const modelName = basename(model) || "model";
  const dataName = datasetPath ? basename(datasetPath).replace(/\.[^.]+$/, "") : "dataset";
  return `${modelName} · ${dataName}`;
}

export function resolveDatasetPath(datasetId: string | null): string | null {
  if (!datasetId) return null;
  const entry = appStore.get().datasets.find((dataset) => dataset.id === datasetId);
  return entry ? entry.path : null;
}

export function updateWizardConfig(patch: Partial<TrainingConfig>): void {
  const current = appStore.get().wizard.config;
  if (!current) return;
  setWizard({ config: { ...current, ...patch } });
}

export async function refreshWizardEstimate(): Promise<void> {
  if (!isDesktop) return;
  const wizard = appStore.get().wizard;
  if (!wizard.config) return;
  const devices = appStore.get().env.hardware?.cuda?.devices ?? [];
  const vram = devices[0]?.total_memory_mb ?? appStore.get().env.smi?.gpu?.memory_total_mb ?? undefined;

  const result = await bridge.training.estimate({
    config: wizard.config,
    modelInfo: wizard.modelInfo || undefined,
    availableVramMb: vram || undefined,
  });
  if (result.ok && result.estimate) {
    setWizard({ estimate: result.estimate as VramEstimate });
  }
}

/* ------------------------------------------------------------- start run */

export async function startTraining(): Promise<boolean> {
  const wizard = appStore.get().wizard;
  if (!wizard.config) {
    pushToast({ title: "Nothing to start", message: "Configure the run first.", tone: "warn" });
    return false;
  }

  const datasetPath = resolveDatasetPath(wizard.datasetId);
  if (!datasetPath) {
    pushToast({ title: "Select a dataset", tone: "warn" });
    return false;
  }
  if (!wizard.config.base_model) {
    pushToast({ title: "Select a base model", tone: "warn" });
    return false;
  }

  const name = (wizard.projectName || "").trim() || defaultProjectName(wizard.config.base_model, datasetPath);
  const datasetEntry = appStore.get().datasets.find((dataset) => dataset.id === wizard.datasetId) ?? null;
  const isHubDataset = Boolean(datasetEntry && datasetEntry.format === "hf");

  setBusy("startTraining", true);
  setWizard({ running: true, error: null });
  try {
    // The project has to exist before the run starts, so the job manager can
    // record the run against it from the very first event.
    let projectId = wizard.projectId;
    if (!projectId) {
      const project = await bridge.projects.create({
        name,
        method: wizard.config.method,
        baseModel: wizard.config.base_model,
        datasetName: datasetEntry?.name ?? null,
        datasetPath,
        config: wizard.config,
      });
      if (project.ok && project.project) projectId = project.project.id;
    }

    const result = await bridge.training.start({
      name,
      projectId,
      config: {
        ...wizard.config,
        output_name: name,
        dataset: {
          // A Hub dataset is identified by id and downloaded by the backend;
          // a local one is read in place.
          path: isHubDataset ? null : datasetPath,
          hf_id: isHubDataset ? datasetEntry?.hfId ?? datasetEntry?.path ?? null : null,
          name: datasetEntry?.name,
          format: datasetEntry?.format ?? "auto",
          mapping: wizard.datasetReport?.mapping ?? datasetEntry?.mapping ?? null,
          split: datasetEntry?.split ?? "train",
        },
      },
      modelInfo: wizard.modelInfo,
      datasetName: datasetEntry?.name,
      datasetReport: wizard.datasetReport,
    });

    if (!result.ok || !result.runId) {
      const error = (result.error as BackendError) ?? { message: "The run could not be started." };
      setWizard({ running: false, error });
      toastError(error, "Could not start training");
      return false;
    }

    await refreshProjects();
    await refreshRuns();
    appStore.set({ page: "training", selectedRunId: result.runId, selectedRun: result.record ?? null });
    resetSeriesFromRun(result.record ?? null);
    await loadRunDetails(result.runId);
    clearLogs();
    pushToast({ title: "Training started", message: name, tone: "good" });
    return true;
  } finally {
    setBusy("startTraining", false);
    setWizard({ running: false });
  }
}

/* ----------------------------------------------------------- run control */

export async function selectRun(runId: string): Promise<void> {
  const run = appStore.get().runs.find((entry) => entry.id === runId) ?? null;
  appStore.set({ selectedRunId: runId, selectedRun: run });
  resetSeriesFromRun(run);
  await loadRunDetails(runId);
}

export async function loadRunDetails(runId: string): Promise<void> {
  if (!isDesktop) return;
  const [runResult, logResult] = await Promise.all([
    bridge.training.run(runId),
    bridge.training.log(runId, 600),
  ]);

  if (runResult.ok && runResult.run) {
    appStore.set({ selectedRun: runResult.run });
    resetSeriesFromRun(runResult.run);
  }

  if (logResult.ok) {
    const events = (logResult.events as { event: string; detail: Record<string, unknown> }[]) || [];
    const stderr = (logResult.stderr as string[]) || [];
    const entries: LogEntry[] = [];
    events.forEach((event, index) => {
      logCounter += 1;
      entries.push({
        key: `file_${logCounter}_${index}`,
        stream: "event",
        level: event.event === "error" ? "error" : "info",
        message: formatEventLog(event.event, event.detail),
        at: Date.now(),
        event: event.event,
        detail: event.detail,
      });
    });
    for (const line of stderr) {
      logCounter += 1;
      entries.push({
        key: `err_${logCounter}`,
        stream: "stderr",
        level: /error|traceback|exception/i.test(line) ? "error" : "info",
        message: line,
        at: Date.now(),
      });
    }
    appStore.set({ logs: entries.slice(-MAX_LOGS) });
  }
}

export function formatEventLog(event: string, detail: Record<string, unknown>): string {
  const message = detail.message as string | undefined;
  const level = (detail.level as string) || "info";
  if (event === "log") return `[${level}] ${message ?? ""}`;
  if (message) return `${event}: ${message}`;
  if (event === "training-progress") {
    const parts = [`step ${detail.step ?? "?"}/${detail.total_steps ?? "?"}`];
    if (detail.loss != null) parts.push(`loss ${Number(detail.loss).toFixed(4)}`);
    if (detail.learning_rate != null) parts.push(`lr ${Number(detail.learning_rate).toExponential(2)}`);
    if (detail.epoch != null) parts.push(`epoch ${detail.epoch}`);
    return `training-progress: ${parts.join(" · ")}`;
  }
  return `${event}: ${JSON.stringify(detail)}`;
}

function resetSeriesFromRun(run: RunRecord | null): void {
  if (!run || !run.history) {
    appStore.set({ series: [] });
    return;
  }
  const history = run.history;
  const steps = history.step || [];
  const losses = history.loss || [];
  const rates = history.lr || [];
  const epochs = history.epoch || [];
  const points: SeriesPoint[] = steps.map((step, index) => ({
    step,
    loss: losses[index] ?? null,
    lr: rates[index] ?? null,
    epoch: epochs[index] ?? null,
    at: run.startedAt,
  }));
  appStore.set({ series: points.slice(-MAX_SERIES) });
}

export async function pauseRun(runId: string): Promise<void> {
  const result = await bridge.training.pause(runId);
  if (!result.ok) {
    toastError((result.error as BackendError) ?? { message: String(result.error) }, "Could not pause");
    return;
  }
  pushToast({
    title: "Pausing",
    message: "The trainer will checkpoint at the next step boundary and stop.",
    tone: "info",
  });
}

export async function stopRun(runId: string): Promise<void> {
  const result = await bridge.training.stop(runId);
  if (!result.ok) {
    toastError((result.error as BackendError) ?? { message: String(result.error) }, "Could not stop");
    return;
  }
  pushToast({ title: "Stopping", message: "Checkpoints are saved before the run exits.", tone: "info" });
}

export async function resumeRun(runId: string): Promise<void> {
  setBusy("resume", true);
  try {
    const result = await bridge.training.resume(runId);
    if (!result.ok || !result.runId) {
      toastError((result.error as BackendError) ?? { message: "Resume failed." }, "Could not resume");
      return;
    }
    await refreshRuns();
    appStore.set({ selectedRunId: result.runId, page: "training" });
    await loadRunDetails(result.runId);
    pushToast({ title: "Resuming from checkpoint", tone: "good" });
  } finally {
    setBusy("resume", false);
  }
}

export async function deleteRun(runId: string): Promise<void> {
  const result = await bridge.training.deleteRun(runId);
  if (!result.ok) {
    toastError((result.error as BackendError) ?? { message: "Delete failed." }, "Could not delete the run");
    return;
  }
  await refreshRuns();
  if (appStore.get().selectedRunId === runId) {
    appStore.set({ selectedRunId: null, selectedRun: null, series: [], logs: [] });
  }
  pushToast({ title: "Run deleted", tone: "info" });
}

export async function removeProject(id: string): Promise<void> {
  const result = await bridge.projects.remove(id);
  if (!result.ok) {
    toastError((result.error as BackendError) ?? { message: "Delete failed." }, "Could not delete the project");
    return;
  }
  await refreshProjects();
  pushToast({ title: "Project deleted", message: "Its runs remain in history.", tone: "info" });
}

export async function openPath(target: string | null | undefined): Promise<void> {
  if (!target) return;
  const result = await bridge.shell.openPath(target);
  if (!result.ok) toastError(result.error as BackendError, "Could not open the folder");
}

export async function revealPath(target: string | null | undefined): Promise<void> {
  if (!target) return;
  await bridge.shell.showItem(target);
}

/** Open a real URL in the user's browser. Nothing is loaded inside the app. */
export async function openExternal(url: string): Promise<void> {
  if (!url) return;
  const result = await bridge.shell.openExternal(url);
  if (!result.ok) toastError(result.error as BackendError, "Could not open the link");
}

/* -------------------------------------------------------------- playground */

export async function loadPlaygroundModel(modelDir: string, name: string): Promise<boolean> {
  appStore.set((state) => ({
    playground: { ...state.playground, loading: true, error: null, loaded: false, output: "", streamed: "" },
  }));
  const result = await bridge.inference.load(modelDir);
  if (!result.ok) {
    appStore.set((state) => ({
      playground: {
        ...state.playground,
        loading: false,
        error: (result.error as BackendError) ?? { message: "Load failed." },
        modelDir,
        modelName: name,
      },
    }));
    return false;
  }
  appStore.set((state) => ({
    playground: { ...state.playground, loading: false, loaded: true, modelDir, modelName: name, error: null },
  }));
  pushToast({ title: "Model loaded", message: name, tone: "good" });
  return true;
}

export async function unloadPlaygroundModel(): Promise<void> {
  await bridge.inference.unload();
  appStore.set((state) => ({
    playground: { ...state.playground, loaded: false, loading: false, streamed: "", output: "" },
  }));
}

export async function generateInPlayground(prompt: string, system: string): Promise<void> {
  const playground = appStore.get().playground;
  if (!playground.loaded) {
    pushToast({ title: "No model loaded", tone: "warn" });
    return;
  }
  if (!prompt.trim()) return;

  appStore.set((state) => ({
    playground: { ...state.playground, generating: true, streamed: "", output: "", error: null },
  }));

  const result = await bridge.inference.generate({
    prompt,
    system: system || undefined,
    max_new_tokens: playground.params.maxNewTokens,
    temperature: playground.params.temperature,
    top_p: playground.params.topP,
    repetition_penalty: playground.params.repetitionPenalty,
    stream: true,
  });

  if (!result.ok) {
    appStore.set((state) => ({
      playground: {
        ...state.playground,
        generating: false,
        error: (result.error as BackendError) ?? { message: "Generation failed." },
      },
    }));
    toastError(result.error as BackendError, "Generation failed");
    return;
  }

  const text = String(result.text ?? "");
  const seconds = (result.seconds as number) ?? null;
  const tps = (result.tokens_per_second as number) ?? null;
  appStore.set((state) => ({
    playground: {
      ...state.playground,
      generating: false,
      output: text,
      streamed: "",
      history: [{ prompt, output: text, seconds, tokensPerSecond: tps }, ...state.playground.history].slice(0, 50),
    },
  }));
}

export function setPlaygroundParams(patch: Partial<PlaygroundState["params"]>): void {
  appStore.set((state) => ({ playground: { ...state.playground, params: { ...state.playground.params, ...patch } } }));
}

export function clearPlaygroundHistory(): void {
  appStore.set((state) => ({
    playground: { ...state.playground, history: [], output: "", streamed: "" },
  }));
}

/* ------------------------------------------------------------- live events */

export function subscribeToEvents(): () => void {
  if (!isDesktop) return () => {};

  const unsubscribers = [
    bridge.on("zeqou:training:event", (payload: any) => {
      const { detail, event } = payload as { runId: string; event: string; detail: Record<string, unknown> };
      if (event === "training-progress") {
        const point: SeriesPoint = {
          step: Number(detail.step ?? 0),
          loss: detail.loss == null ? null : Number(detail.loss),
          lr: detail.learning_rate == null ? null : Number(detail.learning_rate),
          epoch: detail.epoch == null ? null : Number(detail.epoch),
          at: Date.now(),
        };
        appStore.set((state) => ({
          series: [...state.series, point]
            .filter((entry, index, list) => index === 0 || entry.step !== list[index - 1].step)
            .slice(-MAX_SERIES),
        }));
        return;
      }
      if (event === "training-status" && detail.phase === "warning") {
        appendLog({
          stream: "event",
          level: "warn",
          message: String(detail.message ?? ""),
          event,
          detail,
        });
      }
    }),

    bridge.on("zeqou:training:log", (payload: any) => {
      const { line } = payload as { line: string };
      if (!line) return;
      appendLog({
        stream: "stderr",
        level: /error|traceback|exception/i.test(line) ? "error" : "info",
        message: line,
      });
    }),

    bridge.on("zeqou:training:state", (payload: any) => {
      const { record } = payload as { record: RunRecord };
      const state = appStore.get();
      if (record && state.selectedRunId === record.id) {
        appStore.set({ selectedRun: record });
      }
      if (!record) return;
      const runs = state.runs.map((run) => (run.id === record.id ? record : run));
      if (!runs.some((run) => run.id === record.id)) runs.unshift(record);
      appStore.set({ runs });
    }),

    bridge.on("zeqou:training:finished", (payload: any) => {
      const { record } = payload as { record: RunRecord };
      if (!record) return;
      const tone = record.status === "failed" ? "bad" : "good";
      pushToast({
        title:
          record.status === "completed"
            ? "Training completed"
            : record.status === "failed"
              ? "Training failed"
              : record.status === "paused"
                ? "Training paused"
                : "Training stopped",
        message: record.error?.message ?? `${record.name} · final loss ${record.finalLoss?.toFixed(4) ?? "—"}`,
        tone,
      });
      void (async () => {
        await refreshRuns();
        await refreshModels();
        await refreshProjects();
        appStore.set({ page: "training" });
      })();
    }),

    bridge.on("zeqou:gpu", (payload: any) => {
      const { sample } = payload as { sample: { available: boolean; gpu?: Record<string, number | string> } };
      if (!sample || !sample.available || !sample.gpu) {
        appStore.set({ liveGpu: null });
        return;
      }
      const point: GpuPoint = {
        at: Date.now(),
        utilization: (sample.gpu.utilization_gpu as number) ?? null,
        vramUsedMb: (sample.gpu.memory_used_mb as number) ?? null,
        vramTotalMb: (sample.gpu.memory_total_mb as number) ?? null,
        temperature: (sample.gpu.temperature_c as number) ?? null,
      };
      appStore.set((state) => ({
        liveGpu: point,
        gpuSeries: [...state.gpuSeries, point].slice(-MAX_GPU),
      }));
    }),

    bridge.on("zeqou:datasets:changed", () => void refreshDatasets()),
    bridge.on("zeqou:models:changed", () => void refreshModels()),
    bridge.on("zeqou:projects:changed", () => void refreshProjects()),
    bridge.on("zeqou:inference:token", (payload: any) => {
      const { token } = payload as { token: string };
      appStore.set((state) => ({
        playground: { ...state.playground, streamed: state.playground.streamed + token, output: state.playground.streamed + token },
      }));
    }),
    bridge.on("zeqou:inference:state", (payload: any) => {
      const loaded = (payload as { loaded: unknown }).loaded;
      appStore.set((state) => ({ playground: { ...state.playground, loaded: Boolean(loaded) } }));
    }),
    bridge.on("zeqou:inference:log", (payload: any) => {
      const { line, level } = payload as { line: string; level: string };
      if (line) appendLog({ stream: "stderr", level: level === "error" ? "error" : "info", message: `[inference] ${line}` });
    }),
  ];

  return () => unsubscribers.forEach((unsubscribe) => unsubscribe());
}

export function getState(): AppState {
  return appStore.get();
}
