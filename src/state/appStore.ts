/**
 * The application store.
 *
 * One Store instance, one set of async actions. The renderer never talks to
 * the bridge from a component except through these actions, so every mutation
 * has exactly one place to look at.
 */
import { bridge, isDesktop } from "@/lib/bridge";
import { basename, formatCount } from "@/lib/utils";
import { Store } from "@/state/store";
import type {
  AppInfo,
  AutoReason,
  BackendError,
  GpuSample,
  InterpreterCandidate,
  DatasetEntry,
  DatasetFormats,
  DatasetReport,
  DatasetSelection,
  EnvSnapshot,
  LogEntry,
  Method,
  ModelEntry,
  ModelExportInfo,
  ModelInspectInfo,
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
  simple: boolean;
  method: Method;
  baseModel: string;
  baseModelEntryId: string | null;
  modelInfo: ModelInspectInfo | null;
  datasetId: string | null;
  datasetReport: DatasetReport | null;
  projectId: string | null;
  runName: string;
  config: TrainingConfig | null;
  reasons: AutoReason[];
  issues: ValidationIssue[];
  estimate: VramEstimate | null;
  note: BackendError | null;
  starting: boolean;
  lastAutoAt: number | null;
}

export interface PlaygroundState {
  modelDir: string | null;
  modelName: string | null;
  mode: string | null;
  loading: boolean;
  loaded: boolean;
  generating: boolean;
  streamed: string;
  thinkingStreamed: string;
  output: string;
  thinking: string;
  error: BackendError | null;
  history: {
    prompt: string;
    output: string;
    thinking: string;
    seconds: number | null;
    tokensPerSecond: number | null;
  }[];
  params: {
    maxNewTokens: number;
    temperature: number;
    topP: number;
    repetitionPenalty: number;
    thinking: boolean;
  };
}

export interface RuntimeInstallState {
  running: boolean;
  phase: "idle" | "running" | "done" | "failed";
  lines: string[];
  exitCode: number | null;
  error: string | null;
  venv: string | null;
}

export interface AppState {
  booted: boolean;
  bootError: BackendError | null;
  page: Page;
  env: EnvSnapshot;
  appInfo: AppInfo | null;
  settings: Settings | null;
  projects: Project[];
  datasets: DatasetEntry[];
  /** Scanned datasets the user removed from the list (restorable, not deleted). */
  hiddenDatasets: DatasetEntry[];
  /** The folder the app scans for datasets. */
  datasetFolder: string | null;
  /** Which dataset types this install can read, straight from the backend. */
  datasetFormats: DatasetFormats | null;
  models: ModelEntry[];
  runs: RunRecord[];
  selectedRunId: string | null;
  selectedRun: RunRecord | null;
  series: SeriesPoint[];
  gpuSeries: GpuPoint[];
  liveGpu: GpuPoint | null;
  logs: LogEntry[];
  wizard: WizardState;
  playground: PlaygroundState;
  toasts: Toast[];
  busy: Record<string, boolean>;
  runtimeInstall: RuntimeInstallState;
}

/* ------------------------------------------------------------------ shape */

const initialEnv: EnvSnapshot = {
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
};

const initialWizard: WizardState = {
  step: 0,
  simple: true,
  method: "lora",
  baseModel: "",
  baseModelEntryId: null,
  modelInfo: null,
  datasetId: null,
  datasetReport: null,
  projectId: null,
  runName: "",
  config: null,
  reasons: [],
  issues: [],
  estimate: null,
  note: null,
  starting: false,
  lastAutoAt: null,
};

const initialPlayground: PlaygroundState = {
  modelDir: null,
  modelName: null,
  mode: null,
  loading: false,
  loaded: false,
  generating: false,
  streamed: "",
  thinkingStreamed: "",
  output: "",
  thinking: "",
  error: null,
  history: [],
  params: { maxNewTokens: 256, temperature: 0.7, topP: 0.9, repetitionPenalty: 1.1, thinking: false },
};

const initialState: AppState = {
  booted: false,
  bootError: null,
  page: "projects",
  env: initialEnv,
  appInfo: null,
  settings: null,
  projects: [],
  datasets: [],
  hiddenDatasets: [],
  datasetFolder: null,
  datasetFormats: null,
  models: [],
  runs: [],
  selectedRunId: null,
  selectedRun: null,
  series: [],
  gpuSeries: [],
  liveGpu: null,
  logs: [],
  wizard: initialWizard,
  playground: initialPlayground,
  toasts: [],
  busy: {},
  runtimeInstall: { running: false, phase: "idle", lines: [], exitCode: null, error: null, venv: null },
};

export const appStore = new Store<AppState>(initialState);

const MAX_LOG_LINES = 1200;
const MAX_SERIES_POINTS = 4000;
const MAX_GPU_POINTS = 900;

let toastCounter = 0;
let logCounter = 0;

/* ----------------------------------------------------------------- toasts */

export function pushToast(toast: Omit<Toast, "id">): void {
  toastCounter += 1;
  const id = `toast_${toastCounter}`;
  appStore.set((state) => ({ toasts: [...state.toasts, { ...toast, id }] }));
  const timeout = setTimeout(() => dismissToast(id), toast.tone === "bad" ? 12000 : 6000);
  // A background timer must never hold the process open on shutdown.
  (timeout as unknown as { unref?: () => void }).unref?.();
}

export function dismissToast(id: string): void {
  appStore.set((state) => ({ toasts: state.toasts.filter((toast) => toast.id !== id) }));
}

export function toastError(error: BackendError | null | undefined, title: string): void {
  if (!error) return;
  pushToast({
    title,
    message: [error.message, error.hint].filter(Boolean).join(" — "),
    tone: "bad",
  });
}

/* ------------------------------------------------------------------- logs */

export function appendLog(entry: Omit<LogEntry, "key" | "at"> & { at?: number }): void {
  logCounter += 1;
  const full: LogEntry = {
    key: `log_${logCounter}`,
    at: entry.at ?? Date.now(),
    stream: entry.stream,
    level: entry.level,
    message: entry.message,
    event: entry.event,
  };
  appStore.set((state) => ({ logs: [...state.logs, full].slice(-MAX_LOG_LINES) }));
}

export function clearLogs(): void {
  appStore.set({ logs: [] });
}

/** One readable line per protocol event, for the log panel. */
export function formatEventLog(event: string, detail: Record<string, unknown>): string {
  const message = detail.message as string | undefined;
  const level = (detail.level as string) || "info";
  if (event === "log") return `[${level}] ${message ?? ""}`;
  if (event === "training-progress") {
    const parts = [`step ${detail.step ?? "?"}/${detail.total_steps ?? "?"}`];
    if (detail.loss != null) parts.push(`loss ${Number(detail.loss).toFixed(4)}`);
    if (detail.learning_rate != null) parts.push(`lr ${Number(detail.learning_rate).toExponential(2)}`);
    if (detail.epoch != null) parts.push(`epoch ${Number(detail.epoch).toFixed(2)}`);
    return parts.join(" · ");
  }
  if (message) return `${event}: ${message}`;
  return event;
}

function setBusy(key: string, value: boolean): void {
  appStore.set((state) => ({ busy: { ...state.busy, [key]: value } }));
}

/* ------------------------------------------------------------- navigation */

export function navigate(page: Page): void {
  appStore.set({ page });
}

export function patchWizard(patch: Partial<WizardState>): void {
  appStore.set((state) => ({ wizard: { ...state.wizard, ...patch } }));
}

/** Alias kept for call sites that patch more than one wizard field at once. */
export const setWizard = patchWizard;

export function resetWizard(preset: Partial<WizardState> = {}): void {
  // Settings decide the initial Simple/Advanced state; switching automatic
  // configuration off means the user wants the full form instead.
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
        message: "The desktop bridge is not connected.",
        hint: "This window is rendering the interface without the Electron shell. "
          + "Launch ZeqouXTraining with `npm run dev` to train models.",
      },
      env: { ...appStore.get().env, loading: false },
    });
    return;
  }

  const [info, settings] = await Promise.all([bridge.app.info(), bridge.settings.get()]);
  appStore.set({
    appInfo: info.ok ? info.info ?? null : null,
    settings: settings.settings ?? null,
  });
  if (settings.settings?.theme) applyTheme(settings.settings.theme);

  await Promise.all([refreshEnv(), refreshProjects(), refreshDatasets(), refreshModels(), refreshRuns()]);
  appStore.set({ booted: true });
  // Index the dataset folder in the background: a dataset the user drops in
  // should simply be there, without an import step.
  void scanDatasets({ quiet: true });
  void refreshDatasetFormats();

  // Re-attach to a run that is still going (app restarted mid-training).
  const live = appStore.get().runs.find((run) => run.status === "running" || run.status === "starting");
  if (live) {
    appStore.set({ page: "training", selectedRunId: live.id, selectedRun: live });
    await loadRunDetails(live.id);
  }
}

export function applyTheme(theme: "dark" | "light"): void {
  document.documentElement.dataset.theme = theme;
}

/* ------------------------------------------------------------ auto refresh */

// Push channels only cover a live run. Everything else would sit stale until
// the user pressed Refresh, so the lists and the environment snapshot refresh
// themselves on a quiet cadence instead.
const RUNS_REFRESH_MS = 4000;
const LIBRARY_REFRESH_MS = 15000;
const ENV_REFRESH_MS = 60000;

export function startAutoRefresh(): () => void {
  if (!isDesktop) return () => {};

  const runsTimer = setInterval(() => void refreshRuns(), RUNS_REFRESH_MS);
  const libraryTimer = setInterval(() => {
    void refreshDatasets();
    void refreshModels();
    void refreshProjects();
  }, LIBRARY_REFRESH_MS);
  const envTimer = setInterval(() => void refreshEnv(), ENV_REFRESH_MS);

  const stop = () => {
    clearInterval(runsTimer);
    clearInterval(libraryTimer);
    clearInterval(envTimer);
  };
  for (const timer of [runsTimer, libraryTimer, envTimer]) {
    (timer as unknown as { unref?: () => void }).unref?.();
  }
  return stop;
}

/* -------------------------------------------------------------------- env */

export async function refreshEnv(force = false): Promise<void> {
  if (!isDesktop) return;
  appStore.set((state) => ({ env: { ...state.env, loading: true } }));

  const result = await bridge.env.detect({ force });
  if (!result.ok) {
    appStore.set((state) => ({
      env: {
        ...state.env,
        loading: false,
        error: result.error ?? { message: "Environment detection failed." },
      },
    }));
    return;
  }

  appStore.set((state) => ({
    env: {
      loading: false,
      system: (result.system as EnvSnapshot["system"]) ?? state.env.system,
      smi: (result.smi as GpuSample) ?? state.env.smi,
      hardware: (result.hardware as EnvSnapshot["hardware"]) ?? state.env.hardware,
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
  if (!result.ok || !result.hardware) return;
  appStore.set((state) => ({ env: { ...state.env, hardware: result.hardware ?? null } }));
}

export async function discoverInterpreters(force = false): Promise<InterpreterCandidate[] | null> {
  if (!isDesktop) return null;
  const result = await bridge.env.interpreters({ force });
  if (!result.ok || !result.interpreters) return null;
  return result.interpreters;
}

export async function selectInterpreter(executablePath: string): Promise<void> {
  const result = await bridge.env.setInterpreter(executablePath);
  if (!result.ok) {
    toastError(result.error, "Could not select the interpreter");
    return;
  }
  await refreshEnv(true);
  pushToast({ title: "Interpreter updated", tone: "good" });
}

/* --------------------------------------------------------------- settings */

export async function updateSettings(patch: Partial<Settings>): Promise<void> {
  const result = await bridge.settings.set(patch);
  if (!result.ok) {
    toastError(result.error, "Could not save the setting");
    return;
  }
  appStore.set({ settings: result.settings ?? null });
  if (patch.theme) applyTheme(patch.theme);
}

export async function saveHfToken(token: string | null): Promise<boolean> {
  const result = await bridge.settings.setToken(token);
  if (!result.ok) {
    toastError(result.error, "Could not store the token");
    return false;
  }
  pushToast({
    title: token ? "Token stored" : "Token removed",
    message: token ? "Encrypted with the operating system keychain." : undefined,
    tone: "good",
  });
  return true;
}

export async function resetSettings(): Promise<void> {
  const result = await bridge.settings.reset();
  if (!result.ok) {
    toastError(result.error, "Could not reset settings");
    return;
  }
  appStore.set({ settings: result.settings ?? null });
  if (result.settings?.theme) applyTheme(result.settings.theme);
  await refreshEnv(true);
  pushToast({ title: "Settings restored to defaults", tone: "good" });
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
  if (!result.ok) return;
  appStore.set({
    datasets: result.datasets ?? [],
    hiddenDatasets: result.hidden ?? [],
    datasetFolder: result.folder ?? null,
  });
}

/** What the backend can read — shown in the Datasets screen. */
export async function refreshDatasetFormats(): Promise<void> {
  if (!isDesktop) return;
  const result = await bridge.datasets.formats();
  if (result.ok && result.formats) appStore.set({ datasetFormats: result as DatasetFormats });
}

/**
 * Index the dataset folder. This is what makes "I put my files in the folder"
 * work: nothing has to be imported by hand when the folder is scanned.
 */
export async function scanDatasets(options: { quiet?: boolean } = {}): Promise<void> {
  setBusy("scanDatasets", true);
  try {
    const result = await bridge.datasets.scan();
    if (!result.ok) {
      toastError(result.error, "Could not scan the dataset folder");
      return;
    }
    await refreshDatasets();
    if (options.quiet) return;
    const added = result.added ?? 0;
    pushToast({
      title: added ? `Found ${added} dataset${added === 1 ? "" : "s"}` : "Folder scanned",
      message: added
        ? `${result.folder} → ${result.found} file${result.found === 1 ? "" : "s"} indexed.`
        : `Nothing new in ${result.folder}.`,
      tone: "good",
    });
  } finally {
    setBusy("scanDatasets", false);
  }
}

/** Point the scanner at another folder (null restores the app's own folder). */
export async function setDatasetsFolder(folder: string | null): Promise<void> {
  setBusy("scanDatasets", true);
  try {
    const result = await bridge.datasets.setFolder(folder);
    if (!result.ok) {
      toastError(result.error, "Could not change the dataset folder");
      return;
    }
    if (result.settings) appStore.set({ settings: result.settings });
    await refreshDatasets();
    pushToast({
      title: "Dataset folder updated",
      message: result.folder,
      tone: "good",
    });
  } finally {
    setBusy("scanDatasets", false);
  }
}

/** Pick a folder with the native dialog and start scanning it. */
export async function chooseDatasetsFolder(): Promise<void> {
  const picked = await pickDirectory("Choose the folder the app should scan for datasets");
  if (!picked) return;
  await setDatasetsFolder(picked);
}

/** Bring back a scanned dataset that was removed from the list. */
export async function restoreDataset(id: string): Promise<void> {
  const result = await bridge.datasets.restore(id);
  if (!result.ok) {
    toastError(result.error, "Could not restore the dataset");
    return;
  }
  await refreshDatasets();
  pushToast({ title: "Dataset restored", tone: "info" });
}

/**
 * Write a dataset — file, folder of shards or Hub id — as ONE file.
 *
 * The destination is chosen in a native save dialog, and its extension decides
 * the format unless `format` says otherwise. Nothing else is written.
 */
export async function exportDatasetToFile(
  id: string,
  options: { format?: string; raw?: boolean } = {},
): Promise<string | null> {
  const dataset = appStore.get().datasets.find((entry) => entry.id === id) ?? null;
  const format = options.format ?? "jsonl";
  const base = (dataset?.name ?? "dataset").replace(/\.[A-Za-z0-9]+$/, "");

  const picked = await bridge.dialogs.saveDataset({
    format,
    name: base,
    title: "Export the dataset as one file",
  });
  const outputPath = ((picked.paths as string[] | undefined) ?? [])[0];
  if (!outputPath) return null;

  setBusy(`dataset-export:${id}`, true);
  try {
    const result = await bridge.datasets.export({
      id,
      outputPath,
      raw: Boolean(options.raw),
    });
    if (!result.ok) {
      toastError(result.error, "Export failed");
      return null;
    }
    const written = result.output ?? outputPath;
    pushToast({
      title: "Dataset exported",
      message: `${formatCount(result.records ?? 0)} records → ${written}`,
      tone: "good",
    });
    await revealPath(written);
    return written;
  } finally {
    setBusy(`dataset-export:${id}`, false);
  }
}

export async function refreshModels(): Promise<void> {
  if (!isDesktop) return;
  const result = await bridge.models.list();
  if (result.ok && result.models) appStore.set({ models: result.models });
}

export async function refreshRuns(): Promise<void> {
  if (!isDesktop) return;
  const result = await bridge.training.runs(200);
  if (!result.ok || !result.runs) return;
  appStore.set({ runs: result.runs });
  const selectedId = appStore.get().selectedRunId;
  if (selectedId) {
    const selected = result.runs.find((run) => run.id === selectedId);
    if (selected) appStore.set({ selectedRun: selected });
  }
}

/* --------------------------------------------------------------- datasets */

export async function pickAndImportDatasets(): Promise<void> {
  const picked = await bridge.dialogs.pickDataset();
  const paths = (picked.paths as string[] | undefined) ?? [];
  if (!paths.length) return;
  await importDatasetPaths(paths);
}

export async function importDatasetPaths(paths: string[]): Promise<void> {
  setBusy("importDataset", true);
  try {
    const result = await bridge.datasets.import(paths);
    if (!result.ok) {
      toastError(result.error, "Import failed");
      return;
    }
    await refreshDatasets();

    const imported = result.imported ?? [];
    const failed = result.failed ?? [];
    if (imported.length) {
      pushToast({
        title: imported.length === 1 ? "Dataset imported" : `${imported.length} datasets imported`,
        message: "Validating now…",
        tone: "good",
      });
      for (const dataset of imported) {
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

export async function pickDirectory(title: string): Promise<string | null> {
  if (!isDesktop) return null;
  const result = await bridge.dialogs.pickDirectory(title);
  const paths = (result.paths as string[] | undefined) ?? [];
  return paths[0] ?? null;
}

export async function addHfDataset(id: string, split = "train"): Promise<DatasetEntry | null> {
  const trimmed = id.trim();
  if (!trimmed) return null;
  setBusy("importDataset", true);
  try {
    const result = await bridge.datasets.addHf({ id: trimmed, split });
    if (!result.ok) {
      toastError(result.error, "Could not add the Hub dataset");
      return null;
    }
    await refreshDatasets();
    const entry = result.dataset ?? null;
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
    const wizard = appStore.get().wizard;
    const length = contextLength ?? wizard.config?.context_length ?? 512;
    const result = await bridge.datasets.validate({ id, contextLength: length });
    if (!result.ok) {
      toastError(result.error, "Validation failed");
      return null;
    }
    const report = result.report ?? null;
    await refreshDatasets();
    if (report?.status === "errors") {
      const first = report.issues.find((issue) => issue.severity === "error");
      pushToast({ title: "Dataset has problems", message: first?.message, tone: "warn" });
    }
    return report;
  } finally {
    setBusy(`dataset:${id}`, false);
  }
}

export async function previewDataset(id: string, limit = 5): Promise<string[]> {
  const result = await bridge.datasets.preview({ id, limit });
  if (!result.ok) {
    toastError(result.error, "Preview failed");
    return [];
  }
  return result.samples ?? [];
}

export async function removeDataset(id: string): Promise<void> {
  const result = await bridge.datasets.remove(id);
  if (!result.ok) {
    toastError(result.error, "Could not remove the dataset");
    return;
  }
  await refreshDatasets();
  const wizard = appStore.get().wizard;
  if (wizard.datasetId === id) patchWizard({ datasetId: null, datasetReport: null });
  pushToast({ title: "Dataset removed", tone: "info" });
}

/**
 * Validate every dataset in the library, one at a time: a Python process per
 * file in parallel would only fight over the same cores, and the backend is the
 * single source of truth for every report.
 */
export async function validateAllDatasets(): Promise<void> {
  const datasets = appStore.get().datasets;
  if (!datasets.length) return;
  setBusy("validateAll", true);
  try {
    let clean = 0;
    for (const dataset of datasets) {
      // eslint-disable-next-line no-await-in-loop - one backend process at a time
      const report = await validateDataset(dataset.id);
      if (report && report.status !== "errors") clean += 1;
    }
    pushToast({
      title: "Validation finished",
      message: `${clean} of ${datasets.length} datasets look good.`,
      tone: clean === datasets.length ? "good" : "warn",
    });
  } finally {
    setBusy("validateAll", false);
  }
}

/* ----------------------------------------------------------------- models */

export async function pickAndAddLocalModel(): Promise<void> {
  const picked = await bridge.dialogs.pickModelFolder();
  const paths = (picked.paths as string[] | undefined) ?? [];
  if (!paths.length) return;
  await addModel(paths[0]);
}

export async function addModel(source: string): Promise<ModelEntry | null> {
  const trimmed = source.trim();
  if (!trimmed) return null;
  setBusy("addModel", true);
  try {
    const result = await bridge.models.add(trimmed);
    if (!result.ok) {
      toastError(result.error, "Could not add the model");
      return null;
    }
    await refreshModels();
    const model = result.model ?? null;
    if (model) {
      const blocking = model.issues.find((issue) => issue.severity === "error");
      pushToast({
        title: "Model added",
        message: blocking ? blocking.message : model.name,
        tone: blocking ? "warn" : "good",
      });
    }
    return model;
  } finally {
    setBusy("addModel", false);
  }
}

export async function removeModel(id: string): Promise<void> {
  const result = await bridge.models.remove(id);
  if (!result.ok) {
    toastError(result.error, "Could not remove the model");
    return;
  }
  await refreshModels();
  pushToast({ title: "Model removed", tone: "info" });
}

export async function inspectModelExport(modelId: string): Promise<ModelExportInfo | null> {
  const result = await bridge.models.exportInfo({ modelId });
  if (!result.ok || !result.info) {
    toastError(result.error, "Could not inspect the model folder");
    return null;
  }
  return result.info;
}

/**
 * Export a trained model.
 *
 * Two shapes are offered: a folder of loose files (what every loader expects)
 * or one single .zip file (`pack`) for handing the artefact over as one thing.
 */
export async function exportModel(
  modelId: string,
  outputDir: string,
  options: { merge?: boolean; pack?: boolean } = {},
): Promise<{ outputDir: string; mode: string; files: string[]; archive: boolean } | null> {
  const merge = Boolean(options.merge);
  const pack = Boolean(options.pack);
  setBusy("exportModel", true);
  try {
    const result = await bridge.models.export({ modelId, outputDir, merge, pack });
    if (!result.ok) {
      toastError(result.error, "Export failed");
      return null;
    }
    const output = result.output_dir ?? outputDir;
    pushToast({
      title: merge ? "Merged model exported" : pack ? "Exported as one file" : "Export finished",
      message: output,
      tone: "good",
    });
    return {
      outputDir: output,
      mode: result.mode ?? "copy",
      files: result.files ?? [],
      archive: Boolean(result.archive),
    };
  } finally {
    setBusy("exportModel", false);
  }
}

/** Pick the destination file for a packed (single .zip) model export. */
export async function pickModelPackPath(name: string): Promise<string | null> {
  const result = await bridge.models.packPath({ name });
  const paths = (result.paths as string[] | undefined) ?? [];
  return paths[0] ?? null;
}

export async function removeProject(id: string): Promise<void> {
  const result = await bridge.projects.remove(id);
  if (!result.ok) {
    toastError(result.error, "Could not delete the project");
    return;
  }
  await refreshProjects();
  pushToast({ title: "Project deleted", message: "Its runs remain in history.", tone: "info" });
}

/* ----------------------------------------------------------------- paths */

export async function openPath(target: string | null | undefined): Promise<void> {
  if (!target) return;
  const result = await bridge.shell.openPath(target);
  if (!result.ok) toastError(result.error, "Could not open the folder");
}

export async function revealPath(target: string | null | undefined): Promise<void> {
  if (!target) return;
  const result = await bridge.shell.showItem(target);
  if (!result.ok) toastError(result.error, "Could not reveal the file");
}

export async function openExternal(url: string): Promise<void> {
  if (!url) return;
  const result = await bridge.shell.openExternal(url);
  if (!result.ok) toastError(result.error, "Could not open the link");
}

/* -------------------------------------------------------------- wizard */

export async function wizardSelectModel(source: string, entryId: string | null = null): Promise<void> {
  patchWizard({
    baseModel: source,
    baseModelEntryId: entryId,
    modelInfo: null,
    estimate: null,
    note: null,
  });
  if (!isDesktop || !source.trim()) return;

  setBusy("inspectModel", true);
  try {
    const result = await bridge.models.inspect(source.trim());
    if (result.ok && result.info) {
      patchWizard({ modelInfo: result.info });
    } else if (result.error) {
      patchWizard({ note: result.error });
    }
  } finally {
    setBusy("inspectModel", false);
  }
  await runAutoConfig();
}

export async function wizardSelectDataset(id: string, options: { validate?: boolean } = {}): Promise<void> {
  const entry = appStore.get().datasets.find((dataset) => dataset.id === id) ?? null;
  patchWizard({ datasetId: id, datasetReport: entry?.report ?? null });
  if (!entry) return;
  // A quiet selection records the choice without re-spawning the Python backend.
  if (options.validate === false) return;

  setBusy("wizardDataset", true);
  try {
    const report = await validateDataset(id, appStore.get().wizard.config?.context_length ?? 512);
    patchWizard({ datasetReport: report });
  } finally {
    setBusy("wizardDataset", false);
  }
  await runAutoConfig();
}

export async function wizardSelectMethod(method: Method): Promise<void> {
  patchWizard({ method });
  await runAutoConfig();
}

export async function runAutoConfig(): Promise<void> {
  if (!isDesktop) return;
  const wizard = appStore.get().wizard;
  if (!wizard.baseModel.trim() && !wizard.datasetId) return;

  setBusy("autoConfig", true);
  try {
    // A fresh environment first: the Check step must reflect the interpreter
    // that will actually run training, not a stale snapshot.
    await refreshEnv(true);

    const current = appStore.get().wizard;
    const dataset = current.datasetId
      ? appStore.get().datasets.find((entry) => entry.id === current.datasetId) ?? null
      : null;
    // Hub datasets are identified by id on the backend; auto-config reads only
    // local files, so a Hub selection simply contributes no dataset path.
    const datasetPath = dataset && dataset.format !== "hf" ? dataset.path : null;

    const result = await bridge.training.autoConfig({
      baseModel: current.baseModel.trim() || undefined,
      datasetPath: datasetPath ?? undefined,
      contextLength: current.config?.context_length ?? 512,
      method: current.method,
      baseline: appStore.get().settings?.autoConfigure === false,
    });

    if (!result.ok) {
      patchWizard({ note: result.error ?? { message: "Automatic configuration failed." } });
      return;
    }

    const config = (result.config as TrainingConfig | undefined) ?? null;
    const reasons = (result.reasons as AutoReason[] | undefined) ?? [];
    const issues = (result.issues as ValidationIssue[] | undefined) ?? [];
    const estimate = (result.estimate as VramEstimate | undefined) ?? null;
    const hardware = result.hardware as EnvSnapshot["hardware"] | undefined;
    const modelInfo = (result.model as ModelInspectInfo | undefined) ?? null;
    const datasetReport = (result.dataset_report as DatasetReport | undefined) ?? null;
    const modelError = result.model_error as string | undefined;

    patchWizard({
      config: config
        ? {
            ...config,
            method: current.method,
            base_model: current.baseModel.trim() || config.base_model,
          }
        : current.config,
      reasons,
      issues,
      estimate,
      modelInfo: modelInfo ?? current.modelInfo,
      datasetReport: datasetReport ?? current.datasetReport,
      baseModel: current.baseModel.trim() || config?.base_model || "",
      runName: current.runName || defaultRunName(current.baseModel, datasetPath),
      note: modelError ? { code: "model_inspect", message: modelError } : current.note,
      lastAutoAt: Date.now(),
    });

    if (hardware) {
      appStore.set((state) => ({ env: { ...state.env, hardware } }));
    }
  } finally {
    setBusy("autoConfig", false);
  }
}

function defaultRunName(model: string, datasetPath: string | null): string {
  const modelPart = basename(model) || "model";
  const dataPart = datasetPath ? basename(datasetPath).replace(/\.[^.]+$/, "") : "dataset";
  return `${modelPart}-${dataPart}`.toLowerCase().replace(/[^a-z0-9-]+/g, "-").replace(/^-+|-+$/g, "");
}

export function updateWizardConfig(patch: Partial<TrainingConfig>): void {
  const current = appStore.get().wizard.config;
  if (!current) return;
  patchWizard({ config: { ...current, ...patch } });
  scheduleEstimateRefresh();
}

let estimateTimer: ReturnType<typeof setTimeout> | null = null;

/** Debounced VRAM re-estimate while the user edits the form. */
function scheduleEstimateRefresh(): void {
  if (estimateTimer) clearTimeout(estimateTimer);
  estimateTimer = setTimeout(() => {
    estimateTimer = null;
    void refreshWizardEstimate();
  }, 400);
  (estimateTimer as unknown as { unref?: () => void }).unref?.();
}

export async function refreshWizardEstimate(): Promise<void> {
  if (!isDesktop) return;
  const wizard = appStore.get().wizard;
  if (!wizard.config) return;
  // CPU runs are bounded by system RAM, CUDA runs by VRAM — send the right
  // budget so the estimate means something for the device actually selected.
  const hardware = appStore.get().env.hardware;
  const wantsCpu = wizard.config.device === "cpu";
  const devices = wantsCpu ? [] : hardware?.cuda.devices ?? [];
  const vram = devices[0]?.total_memory_mb ?? appStore.get().env.smi?.gpu?.memory_total_mb ?? undefined;
  const ram = hardware?.memory?.total_mb ?? undefined;

  const result = await bridge.training.estimate({
    config: wizard.config,
    modelInfo: wizard.modelInfo ?? undefined,
    availableVramMb: vram,
    availableRamMb: wantsCpu ? ram : undefined,
  });
  if (result.ok && result.estimate) {
    patchWizard({ estimate: result.estimate as unknown as VramEstimate });
  }
}

/* ------------------------------------------------------------ start run */

export async function startTraining(): Promise<boolean> {
  const wizard = appStore.get().wizard;
  if (!wizard.config) {
    pushToast({ title: "Nothing to start", message: "Wait for the configuration to be prepared.", tone: "warn" });
    return false;
  }

  // The app ships no datasets, so a run needs a real choice — never a silent
  // stand-in. The library is exactly what the user imported or scanned.
  const library = appStore.get().datasets;
  const dataset = wizard.datasetId
    ? library.find((entry) => entry.id === wizard.datasetId) ?? null
    : null;
  if (!dataset) {
    pushToast({
      title: "Select a dataset",
      message: library.length
        ? "Pick one in the Dataset step."
        : "The library is empty — scan the dataset folder or import a file.",
      tone: "warn",
    });
    return false;
  }
  if (!wizard.config.base_model && wizard.config.method !== "scratch") {
    pushToast({ title: "Select a base model", tone: "warn" });
    return false;
  }

  const name = (wizard.runName || "").trim()
    || defaultRunName(wizard.config.base_model || (wizard.config.method === "scratch" ? "scratch" : "model"), dataset.path);
  const isHub = dataset.format === "hf";
  const selection: DatasetSelection = {
    // A Hub dataset is referenced by id and downloaded by the backend; a local
    // one is read in place, so importing costs no disk space.
    path: isHub ? null : dataset.path,
    hf_id: isHub ? dataset.hfId ?? dataset.path : null,
    name: dataset.name,
    format: dataset.format,
    mapping: wizard.datasetReport?.mapping ?? dataset.mapping ?? null,
    split: isHub ? dataset.split ?? "train" : "train",
  };

  setBusy("start", true);
  patchWizard({ starting: true, note: null });
  try {
    // The project must exist before the run starts, so the job manager can
    // record the run against it from the very first event.
    let projectId = wizard.projectId;
    if (!projectId) {
      const created = await bridge.projects.create({
        name,
        method: wizard.config.method,
        baseModel: wizard.config.base_model,
        datasetName: dataset.name,
        datasetPath: dataset.path,
        config: wizard.config,
      });
      if (created.ok && created.project) projectId = created.project.id;
    }

    const result = await bridge.training.start({
      name,
      projectId,
      config: { ...wizard.config, output_name: name, dataset: selection },
      modelInfo: wizard.modelInfo ?? undefined,
      datasetReport: wizard.datasetReport ?? undefined,
    });

    if (!result.ok || !result.runId) {
      const error = result.error ?? { message: "The run could not be started." };
      patchWizard({ starting: false, note: error });
      toastError(error, "Could not start training");
      return false;
    }

    await Promise.all([refreshProjects(), refreshRuns()]);
    appStore.set({ page: "training", selectedRunId: result.runId, selectedRun: result.record ?? null });
    resetSeriesFromRun(result.record ?? null);
    clearLogs();
    await loadRunDetails(result.runId);
    pushToast({ title: "Training started", message: name, tone: "good" });
    return true;
  } finally {
    setBusy("start", false);
    patchWizard({ starting: false });
  }
}

/* ----------------------------------------------------------- run control */

function resetSeriesFromRun(run: RunRecord | null): void {
  const history = run?.history;
  if (!history || !history.step?.length) {
    appStore.set({ series: [] });
    return;
  }
  const points: SeriesPoint[] = history.step.map((step, index) => ({
    step,
    loss: history.loss?.[index] ?? null,
    lr: history.learning_rate?.[index] ?? null,
    epoch: history.epoch?.[index] ?? null,
  }));
  appStore.set({ series: points.slice(-MAX_SERIES_POINTS) });
}

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
    bridge.training.log(runId, 800),
  ]);

  if (runResult.ok && runResult.run) {
    appStore.set({ selectedRun: runResult.run });
    resetSeriesFromRun(runResult.run);
  }

  if (logResult.ok) {
    const entries: LogEntry[] = [];
    for (const event of logResult.events ?? []) {
      appendLog({
        stream: "event",
        level: event.event === "error" ? "error" : "info",
        message: formatEventLog(event.event, event.detail),
        event: event.event,
      });
    }
    for (const line of logResult.stderr ?? []) {
      entries.push({
        key: `err_${logCounter}`, // filled by appendLog normally
        at: Date.now(),
        stream: "stderr",
        level: /error|traceback|exception/i.test(line) ? "error" : "info",
        message: line,
      });
    }
    if (entries.length) {
      appStore.set((state) => ({ logs: [...state.logs, ...entries].slice(-MAX_LOG_LINES) }));
    }
  }
}

export async function pauseRun(runId: string): Promise<void> {
  const result = await bridge.training.pause(runId);
  if (!result.ok) {
    toastError(result.error, "Could not pause the run");
    return;
  }
  pushToast({ title: "Pausing", message: "The trainer finishes the current step, saves a checkpoint and stops.", tone: "info" });
}

export async function stopRun(runId: string): Promise<void> {
  const result = await bridge.training.stop(runId);
  if (!result.ok) {
    toastError(result.error, "Could not stop the run");
    return;
  }
  pushToast({ title: "Stopping", message: "Checkpoints are saved before the process exits.", tone: "info" });
}

export async function resumeRun(runId: string): Promise<boolean> {
  setBusy("resume", true);
  try {
    const result = await bridge.training.resume(runId);
    if (!result.ok || !result.runId) {
      toastError(result.error, "Could not resume the run");
      return false;
    }
    await refreshRuns();
    appStore.set({ page: "training", selectedRunId: result.runId, selectedRun: result.record ?? null });
    await loadRunDetails(result.runId);
    pushToast({ title: "Resuming from the last checkpoint", tone: "good" });
    return true;
  } finally {
    setBusy("resume", false);
  }
}

export async function deleteRun(runId: string): Promise<void> {
  const result = await bridge.training.deleteRun(runId);
  if (!result.ok) {
    toastError(result.error, "Could not delete the run");
    return;
  }
  await refreshRuns();
  if (appStore.get().selectedRunId === runId) {
    appStore.set({ selectedRunId: null, selectedRun: null, series: [], logs: [] });
  }
  pushToast({ title: "Run deleted", tone: "info" });
}

/* ------------------------------------------------------------- playground */

export async function loadPlaygroundModel(modelDir: string, name: string): Promise<boolean> {
  if (!isDesktop) return false;
  appStore.set((state) => ({
    playground: {
      ...state.playground,
      loading: true,
      loaded: false,
      error: null,
      streamed: "",
      thinkingStreamed: "",
      output: "",
      thinking: "",
    },
  }));

  const result = await bridge.inference.load(modelDir);
  if (!result.ok) {
    const error = result.error ?? { message: "The model could not be loaded." };
    appStore.set((state) => ({
      playground: {
        ...state.playground,
        loading: false,
        loaded: false,
        modelDir,
        modelName: name,
        error,
      },
    }));
    toastError(error, "Could not load the model");
    return false;
  }

  appStore.set((state) => ({
    playground: {
      ...state.playground,
      loading: false,
      loaded: true,
      modelDir,
      modelName: name,
      error: null,
    },
  }));
  pushToast({ title: "Model loaded", message: name, tone: "good" });
  return true;
}

export async function unloadPlaygroundModel(): Promise<void> {
  if (isDesktop) await bridge.inference.unload();
  appStore.set((state) => ({
    playground: {
      ...state.playground,
      loaded: false,
      loading: false,
      streamed: "",
      thinkingStreamed: "",
      output: "",
      thinking: "",
    },
  }));
}

export async function generateInPlayground(prompt: string, system: string): Promise<void> {
  const playground = appStore.get().playground;
  if (!playground.loaded || !playground.modelDir) {
    pushToast({ title: "Load a model first", tone: "warn" });
    return;
  }
  if (!prompt.trim() || playground.generating) return;

  appStore.set((state) => ({
    playground: {
      ...state.playground,
      generating: true,
      streamed: "",
      thinkingStreamed: "",
      output: "",
      thinking: "",
      error: null,
    },
  }));

  const result = await bridge.inference.generate({
    model_dir: playground.modelDir,
    prompt,
    system: system.trim() || undefined,
    thinking: playground.params.thinking,
    max_new_tokens: playground.params.maxNewTokens,
    temperature: playground.params.temperature,
    top_p: playground.params.topP,
    repetition_penalty: playground.params.repetitionPenalty,
  });

  if (!result.ok) {
    const error = result.error ?? { message: "Generation failed." };
    appStore.set((state) => ({
      playground: { ...state.playground, generating: false, error },
    }));
    toastError(error, "Generation failed");
    return;
  }

  const text = result.result?.text ?? "";
  const thinkingText = result.result?.thinking ?? "";
  const seconds = result.result?.seconds ?? null;
  const tokensPerSecond = result.result?.tokens_per_second ?? null;
  appStore.set((state) => ({
    playground: {
      ...state.playground,
      generating: false,
      output: text,
      thinking: thinkingText,
      streamed: "",
      thinkingStreamed: "",
      history: [
        { prompt, output: text, thinking: thinkingText, seconds, tokensPerSecond },
        ...state.playground.history,
      ].slice(0, 40),
    },
  }));
}

export function setPlaygroundParams(patch: Partial<PlaygroundState["params"]>): void {
  appStore.set((state) => ({
    playground: { ...state.playground, params: { ...state.playground.params, ...patch } },
  }));
}

export function clearPlaygroundHistory(): void {
  appStore.set((state) => ({
    playground: { ...state.playground, history: [], output: "", streamed: "", thinking: "", thinkingStreamed: "" },
  }));
}

/* -------------------------------------------------------- runtime install */

export async function installRuntime(): Promise<boolean> {
  if (!isDesktop) return false;
  if (appStore.get().runtimeInstall.running) return false;

  appStore.set({
    runtimeInstall: { running: true, phase: "running", lines: [], exitCode: null, error: null, venv: null },
  });
  setBusy("installRuntime", true);
  try {
    const result = await bridge.env.installRuntime();
    if (!result.ok) {
      const message = result.error?.message ?? "The installation failed.";
      appStore.set((state) => ({
        runtimeInstall: { ...state.runtimeInstall, running: false, phase: "failed", error: message },
      }));
      toastError(result.error, "Could not install the ML runtime");
      return false;
    }
    // The zeqou:runtime:install channel delivers the final phase; nothing to
    // do here except report that the installer accepted the request.
    return true;
  } finally {
    setBusy("installRuntime", false);
  }
}

/* ------------------------------------------------------------ live events */

export function subscribeToEvents(): () => void {
  if (!isDesktop) return () => {};

  const unsubs = [
    bridge.on("zeqou:training:event", (payload) => {
      const { event, detail } = payload as { runId: string; event: string; detail: Record<string, unknown> };

      if (event === "training-progress") {
        const point: SeriesPoint = {
          step: Number(detail.step ?? 0),
          loss: detail.loss == null ? null : Number(detail.loss),
          lr: detail.learning_rate == null ? null : Number(detail.learning_rate),
          epoch: detail.epoch == null ? null : Number(detail.epoch),
        };
        appStore.set((state) => ({
          // Repeated steps can arrive after a resume; keep the newest.
          series: [...state.series.filter((entry) => entry.step !== point.step), point]
            .sort((a, b) => a.step - b.step)
            .slice(-MAX_SERIES_POINTS),
        }));
        return;
      }

      if (event === "dataset-progress") return; // surfaced through the wizard spinner instead
      appendLog({
        stream: "event",
        level: event === "error" ? "error" : detail.level === "warn" ? "warn" : "info",
        message: formatEventLog(event, detail),
        event,
      });
    }),

    bridge.on("zeqou:training:log", (payload) => {
      const { line } = payload as { runId: string; line: string };
      if (!line) return;
      appendLog({
        stream: "stderr",
        level: /error|traceback|exception/i.test(line) ? "error" : "info",
        message: line,
      });
    }),

    bridge.on("zeqou:training:state", (payload) => {
      const { record } = payload as { runId: string; record: RunRecord };
      if (!record) return;
      const state = appStore.get();
      const runs = state.runs.some((run) => run.id === record.id)
        ? state.runs.map((run) => (run.id === record.id ? { ...run, ...record } : run))
        : [record, ...state.runs];
      const patch: Partial<AppState> = { runs };
      if (state.selectedRunId === record.id) {
        patch.selectedRun = state.selectedRun ? { ...state.selectedRun, ...record } : record;
      }
      appStore.set(patch);
    }),

    bridge.on("zeqou:training:finished", (payload) => {
      const { record } = payload as { runId: string; record: RunRecord };
      if (!record) return;
      const titles: Record<string, string> = {
        completed: "Training completed",
        failed: "Training failed",
        stopped: "Training stopped",
        paused: "Training paused",
      };
      pushToast({
        title: titles[record.status] ?? "Training finished",
        message: record.error?.message
          ?? `${record.name} · final loss ${record.finalLoss != null ? record.finalLoss.toFixed(4) : "—"}`,
        tone: record.status === "failed" ? "bad" : "good",
      });
      void (async () => {
        await Promise.all([refreshRuns(), refreshModels(), refreshProjects()]);
        appStore.set({ page: "training" });
      })();
    }),

    bridge.on("zeqou:gpu", (payload) => {
      const { sample } = payload as { runId: string | null; sample: GpuSample };
      if (!sample?.available || !sample.gpu) {
        appStore.set({ liveGpu: null });
        return;
      }
      const point: GpuPoint = {
        at: Date.now(),
        utilization: sample.gpu.utilization_gpu ?? null,
        vramUsedMb: sample.gpu.memory_used_mb ?? null,
        vramTotalMb: sample.gpu.memory_total_mb ?? null,
        temperature: sample.gpu.temperature_c ?? null,
      };
      appStore.set((state) => ({
        liveGpu: point,
        gpuSeries: [...state.gpuSeries, point].slice(-MAX_GPU_POINTS),
      }));
    }),

    bridge.on("zeqou:datasets:changed", () => void refreshDatasets()),
    bridge.on("zeqou:models:changed", () => void refreshModels()),
    bridge.on("zeqou:projects:changed", () => void refreshProjects()),

    bridge.on("zeqou:settings:changed", (payload) => {
      const { settings } = payload as { settings: Settings };
      if (!settings) return;
      appStore.set({ settings });
      applyTheme(settings.theme);
    }),

    bridge.on("zeqou:runtime:install", (payload) => {
      const { phase, line, exitCode, message, venv } = payload as {
        phase: "output" | "done" | "failed";
        line?: string;
        exitCode?: number;
        message?: string;
        venv?: string;
      };
      appStore.set((state) => {
        const install = state.runtimeInstall;
        if (phase === "output") {
          return {
            runtimeInstall: {
              ...install,
              running: true,
              phase: "running",
              lines: [...install.lines, String(line ?? "")].slice(-500),
            },
          };
        }
        if (phase === "done") {
          return {
            runtimeInstall: { ...install, running: false, phase: "done", exitCode: exitCode ?? 0, venv: venv ?? null },
          };
        }
        return {
          runtimeInstall: {
            ...install,
            running: false,
            phase: "failed",
            exitCode: exitCode ?? null,
            error: message ?? "The installation failed.",
          },
        };
      });

      if (phase === "done") {
        pushToast({ title: "ML runtime installed", message: "The app environment is ready.", tone: "good" });
        void refreshEnv(true);
      }
      if (phase === "failed") {
        pushToast({ title: "Installation failed", message: message, tone: "bad" });
      }
    }),

    bridge.on("zeqou:inference:token", (payload) => {
      const { token } = payload as { requestId: string; token: string };
      if (!token) return;
      appStore.set((state) => ({
        playground: {
          ...state.playground,
          streamed: state.playground.streamed + token,
          output: state.playground.streamed + token,
        },
      }));
    }),

    bridge.on("zeqou:inference:thinking", (payload) => {
      const { token } = payload as { requestId: string; token: string };
      if (!token) return;
      appStore.set((state) => ({
        playground: {
          ...state.playground,
          thinkingStreamed: state.playground.thinkingStreamed + token,
          thinking: state.playground.thinkingStreamed + token,
        },
      }));
    }),

    bridge.on("zeqou:inference:state", (payload) => {
      const { loaded } = payload as { loaded: { model_dir: string; mode: string } | null };
      appStore.set((state) => ({
        playground: {
          ...state.playground,
          loaded: Boolean(loaded),
          mode: loaded?.mode ?? null,
        },
      }));
    }),

    bridge.on("zeqou:inference:log", (payload) => {
      const { line, level } = payload as { line: string; level: string };
      if (!line) return;
      appendLog({
        stream: "stderr",
        level: level === "error" ? "error" : "info",
        message: `[inference] ${line}`,
      });
    }),
  ];

  return () => unsubs.forEach((unsubscribe) => unsubscribe());
}

export function getState(): AppState {
  return appStore.get();
}
