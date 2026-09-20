/**
 * Access to the desktop bridge.
 *
 * When the renderer runs in a plain browser (Vite dev server, preview pane),
 * every privileged call resolves with an explicit `desktop_only` error instead
 * of pretending to work — the interface stays inspectable while the UI stays
 * honest about what it can do.
 */
import type {
  AppInfo,
  BackendError,
  BackendCapability,
  CallResult,
  DatasetEntry,
  DatasetReport,
  EnvSnapshot,
  GpuSample,
  HardwareSnapshot,
  InstallPlan,
  InterpreterCandidate,
  ModelEntry,
  ModelExportInfo,
  ModelInspectInfo,
  Project,
  RunRecord,
  Settings,
  TokenStorageInfo,
} from "./types";

/** The surface preload.js exposes on `window.zeqou`, before unwrapping. */
interface RawBridge {
  platform: string;
  channels: string[];
  on(channel: string, handler: (payload: unknown) => void): () => void;
  app: { info(): Promise<Record<string, unknown>> };
  shell: {
    openPath(target: string): Promise<Record<string, unknown>>;
    openExternal(url: string): Promise<Record<string, unknown>>;
    showItem(target: string): Promise<Record<string, unknown>>;
  };
  dialogs: {
    pickDataset(): Promise<Record<string, unknown>>;
    pickModelFolder(): Promise<Record<string, unknown>>;
    pickPython(): Promise<Record<string, unknown>>;
    pickDirectory(title?: string): Promise<Record<string, unknown>>;
  };
  settings: {
    get(): Promise<Record<string, unknown>>;
    set(patch: Partial<Settings>): Promise<Record<string, unknown>>;
    reset(): Promise<Record<string, unknown>>;
    setToken(token: string | null): Promise<Record<string, unknown>>;
    tokenStatus(): Promise<Record<string, unknown>>;
  };
  env: {
    detect(options?: { force?: boolean }): Promise<Record<string, unknown>>;
    interpreters(options?: { force?: boolean }): Promise<Record<string, unknown>>;
    setInterpreter(executablePath: string | null): Promise<Record<string, unknown>>;
    installPlan(cudaTag?: string): Promise<Record<string, unknown>>;
    installRuntime(): Promise<Record<string, unknown>>;
    installStatus(): Promise<Record<string, unknown>>;
    backends(): Promise<Record<string, unknown>>;
  };
  hardware: {
    detect(): Promise<Record<string, unknown>>;
    sample(): Promise<Record<string, unknown>>;
  };
  datasets: {
    list(): Promise<Record<string, unknown>>;
    import(paths: string | string[]): Promise<Record<string, unknown>>;
    addHf(payload: { id: string; split?: string }): Promise<Record<string, unknown>>;
    validate(payload: Record<string, unknown>): Promise<Record<string, unknown>>;
    preview(payload: Record<string, unknown>): Promise<Record<string, unknown>>;
    remove(id: string): Promise<Record<string, unknown>>;
  };
  models: {
    list(): Promise<Record<string, unknown>>;
    add(source: string): Promise<Record<string, unknown>>;
    inspect(source: string): Promise<Record<string, unknown>>;
    exportInfo(payload: { modelId: string }): Promise<Record<string, unknown>>;
    export(payload: { modelId: string; outputDir: string; merge?: boolean }): Promise<Record<string, unknown>>;
    remove(id: string): Promise<Record<string, unknown>>;
  };
  projects: {
    list(): Promise<Record<string, unknown>>;
    get(id: string): Promise<Record<string, unknown>>;
    create(patch: Record<string, unknown>): Promise<Record<string, unknown>>;
    update(id: string, patch: Record<string, unknown>): Promise<Record<string, unknown>>;
    remove(id: string): Promise<Record<string, unknown>>;
  };
  training: {
    autoConfig(payload: Record<string, unknown>): Promise<Record<string, unknown>>;
    estimate(payload: Record<string, unknown>): Promise<Record<string, unknown>>;
    check(payload: Record<string, unknown>): Promise<Record<string, unknown>>;
    start(payload: Record<string, unknown>): Promise<Record<string, unknown>>;
    progress(runId: string): Promise<Record<string, unknown>>;
    stop(runId: string): Promise<Record<string, unknown>>;
    pause(runId: string): Promise<Record<string, unknown>>;
    resume(runId: string): Promise<Record<string, unknown>>;
    runs(limit?: number): Promise<Record<string, unknown>>;
    run(runId: string): Promise<Record<string, unknown>>;
    deleteRun(runId: string): Promise<Record<string, unknown>>;
    log(runId: string, lines?: number): Promise<Record<string, unknown>>;
    checkpoints(runDir: string): Promise<Record<string, unknown>>;
  };
  inference: {
    load(modelDir: string): Promise<Record<string, unknown>>;
    generate(payload: Record<string, unknown>): Promise<Record<string, unknown>>;
    unload(): Promise<Record<string, unknown>>;
    status(): Promise<Record<string, unknown>>;
  };
}

declare global {
  interface Window {
    zeqou?: RawBridge;
  }
}

export const raw: RawBridge | undefined =
  typeof window !== "undefined" ? window.zeqou : undefined;

export const isDesktop = Boolean(raw);

const DESKTOP_HINT = "Launch ZeqouXTraining with `npm run dev` to use this.";

function fail(message: string, code = "bridge"): CallResult {
  return { ok: false, error: { code, message, hint: DESKTOP_HINT } };
}

function desktopOnly(feature: string): CallResult {
  return {
    ok: false,
    error: {
      code: "desktop_only",
      message: `${feature} is available in the desktop app only.`,
      hint: DESKTOP_HINT,
    },
  };
}

/** Uniform wrapper: every handler resolves with a CallResult, never throws. */
async function call(promise: Promise<Record<string, unknown>>): Promise<CallResult> {
  try {
    const result = (await promise) as unknown as CallResult;
    if (!result || typeof result !== "object") return fail("The desktop bridge returned no data.");
    return result;
  } catch (error) {
    return fail(error instanceof Error ? error.message : String(error));
  }
}

/** Helpers for extracting typed fields from a CallResult. */
export function field<T>(result: CallResult, key: string): T | null {
  const value = result[key];
  return value === undefined ? null : (value as T);
}

export const bridge = {
  isDesktop,

  on(channel: string, handler: (payload: never) => void): () => void {
    if (!raw) return () => {};
    return raw.on(channel, handler as (payload: unknown) => void);
  },

  app: {
    info: (): Promise<CallResult & { info?: AppInfo }> =>
      raw ? call(raw.app.info()) : Promise.resolve(desktopOnly("App information")),
  },

  shell: {
    openPath: (target: string) =>
      raw ? call(raw.shell.openPath(target)) : Promise.resolve(desktopOnly("Opening paths")),
    openExternal: (url: string) =>
      raw ? call(raw.shell.openExternal(url)) : Promise.resolve(desktopOnly("Opening links")),
    showItem: (target: string) =>
      raw ? call(raw.shell.showItem(target)) : Promise.resolve(desktopOnly("Revealing files")),
  },

  dialogs: {
    pickDataset: () =>
      raw ? call(raw.dialogs.pickDataset()) : Promise.resolve(desktopOnly("File pickers")),
    pickModelFolder: () =>
      raw ? call(raw.dialogs.pickModelFolder()) : Promise.resolve(desktopOnly("Folder pickers")),
    pickPython: () =>
      raw ? call(raw.dialogs.pickPython()) : Promise.resolve(desktopOnly("File pickers")),
    pickDirectory: (title?: string) =>
      raw ? call(raw.dialogs.pickDirectory(title)) : Promise.resolve(desktopOnly("Folder pickers")),
  },

  settings: {
    get: (): Promise<CallResult & { settings?: Settings }> =>
      raw ? call(raw.settings.get()) : Promise.resolve(desktopOnly("Settings")),
    set: (patch: Partial<Settings>): Promise<CallResult & { settings?: Settings }> =>
      raw ? call(raw.settings.set(patch)) : Promise.resolve(desktopOnly("Settings")),
    reset: (): Promise<CallResult & { settings?: Settings }> =>
      raw ? call(raw.settings.reset()) : Promise.resolve(desktopOnly("Settings")),
    setToken: (token: string | null): Promise<CallResult & { storage?: TokenStorageInfo }> =>
      raw ? call(raw.settings.setToken(token)) : Promise.resolve(desktopOnly("Storing tokens")),
    tokenStatus: (): Promise<CallResult & { storage?: TokenStorageInfo; encryptionAvailable?: boolean }> =>
      raw ? call(raw.settings.tokenStatus()) : Promise.resolve(desktopOnly("Token status")),
  },

  env: {
    detect: (options?: { force?: boolean }): Promise<CallResult & Partial<EnvSnapshot>> =>
      raw ? call(raw.env.detect(options)) : Promise.resolve(desktopOnly("Environment detection")),
    interpreters: (options?: { force?: boolean }): Promise<CallResult & { interpreters?: InterpreterCandidate[]; active?: Record<string, unknown> }> =>
      raw ? call(raw.env.interpreters(options)) : Promise.resolve(desktopOnly("Interpreter discovery")),
    setInterpreter: (path: string | null): Promise<CallResult & { interpreterPath?: string | null }> =>
      raw ? call(raw.env.setInterpreter(path)) : Promise.resolve(desktopOnly("Interpreter selection")),
    installPlan: (cudaTag?: string): Promise<CallResult & { plan?: InstallPlan }> =>
      raw ? call(raw.env.installPlan(cudaTag)) : Promise.resolve(desktopOnly("Install plans")),
    installRuntime: (): Promise<CallResult & { exitCode?: number; venv?: string; interpreter?: string }> =>
      raw ? call(raw.env.installRuntime()) : Promise.resolve(desktopOnly("Runtime installation")),
    installStatus: (): Promise<CallResult & { running?: boolean; venvExists?: boolean; venvPython?: string }> =>
      raw ? call(raw.env.installStatus()) : Promise.resolve(desktopOnly("Runtime installation")),
    backends: (): Promise<CallResult & { backends?: BackendCapability[] }> =>
      raw ? call(raw.env.backends()) : Promise.resolve(desktopOnly("Backend detection")),
  },

  hardware: {
    detect: (): Promise<CallResult & { hardware?: HardwareSnapshot | null }> =>
      raw ? call(raw.hardware.detect()) : Promise.resolve(desktopOnly("Hardware detection")),
    sample: (): Promise<CallResult & { sample?: GpuSample }> =>
      raw ? call(raw.hardware.sample()) : Promise.resolve(desktopOnly("GPU sampling")),
  },

  datasets: {
    list: (): Promise<CallResult & { datasets?: DatasetEntry[] }> =>
      raw ? call(raw.datasets.list()) : Promise.resolve(desktopOnly("Datasets")),
    import: (paths: string | string[]): Promise<CallResult & { imported?: DatasetEntry[]; failed?: { path: string; error: BackendError }[] }> =>
      raw ? call(raw.datasets.import(paths)) : Promise.resolve(desktopOnly("Importing datasets")),
    addHf: (payload: { id: string; split?: string }): Promise<CallResult & { dataset?: DatasetEntry }> =>
      raw ? call(raw.datasets.addHf(payload)) : Promise.resolve(desktopOnly("Adding Hub datasets")),
    validate: (payload: Record<string, unknown>): Promise<CallResult & { report?: DatasetReport }> =>
      raw ? call(raw.datasets.validate(payload)) : Promise.resolve(desktopOnly("Validating datasets")),
    preview: (payload: Record<string, unknown>): Promise<CallResult & { samples?: string[] }> =>
      raw ? call(raw.datasets.preview(payload)) : Promise.resolve(desktopOnly("Dataset preview")),
    remove: (id: string): Promise<CallResult> =>
      raw ? call(raw.datasets.remove(id)) : Promise.resolve(desktopOnly("Removing datasets")),
  },

  models: {
    list: (): Promise<CallResult & { models?: ModelEntry[] }> =>
      raw ? call(raw.models.list()) : Promise.resolve(desktopOnly("Models")),
    add: (source: string): Promise<CallResult & { model?: ModelEntry; info?: ModelInspectInfo }> =>
      raw ? call(raw.models.add(source)) : Promise.resolve(desktopOnly("Adding models")),
    inspect: (source: string): Promise<CallResult & { info?: ModelInspectInfo }> =>
      raw ? call(raw.models.inspect(source)) : Promise.resolve(desktopOnly("Inspecting models")),
    exportInfo: (payload: { modelId: string }): Promise<CallResult & { info?: ModelExportInfo }> =>
      raw ? call(raw.models.exportInfo(payload)) : Promise.resolve(desktopOnly("Exporting models")),
    export: (payload: { modelId: string; outputDir: string; merge?: boolean }): Promise<CallResult & { output_dir?: string; files?: string[]; mode?: string }> =>
      raw ? call(raw.models.export(payload)) : Promise.resolve(desktopOnly("Exporting models")),
    remove: (id: string): Promise<CallResult> =>
      raw ? call(raw.models.remove(id)) : Promise.resolve(desktopOnly("Removing models")),
  },

  projects: {
    list: (): Promise<CallResult & { projects?: Project[] }> =>
      raw ? call(raw.projects.list()) : Promise.resolve(desktopOnly("Projects")),
    get: (id: string): Promise<CallResult & { project?: Project }> =>
      raw ? call(raw.projects.get(id)) : Promise.resolve(desktopOnly("Projects")),
    create: (patch: Record<string, unknown>): Promise<CallResult & { project?: Project }> =>
      raw ? call(raw.projects.create(patch)) : Promise.resolve(desktopOnly("Projects")),
    update: (id: string, patch: Record<string, unknown>): Promise<CallResult & { project?: Project }> =>
      raw ? call(raw.projects.update(id, patch)) : Promise.resolve(desktopOnly("Projects")),
    remove: (id: string): Promise<CallResult> =>
      raw ? call(raw.projects.remove(id)) : Promise.resolve(desktopOnly("Projects")),
  },

  training: {
    autoConfig: (payload: Record<string, unknown>): Promise<CallResult> =>
      raw ? call(raw.training.autoConfig(payload)) : Promise.resolve(desktopOnly("Automatic configuration")),
    estimate: (payload: Record<string, unknown>): Promise<CallResult & { estimate?: Record<string, unknown> }> =>
      raw ? call(raw.training.estimate(payload)) : Promise.resolve(desktopOnly("VRAM estimation")),
    check: (payload: Record<string, unknown>): Promise<CallResult> =>
      raw ? call(raw.training.check(payload)) : Promise.resolve(desktopOnly("Pre-flight checks")),
    start: (payload: Record<string, unknown>): Promise<CallResult & { runId?: string; record?: RunRecord }> =>
      raw ? call(raw.training.start(payload)) : Promise.resolve(desktopOnly("Starting training")),
    progress: (runId: string): Promise<CallResult & { record?: RunRecord | null; live?: Record<string, unknown> | null }> =>
      raw ? call(raw.training.progress(runId)) : Promise.resolve(desktopOnly("Run progress")),
    stop: (runId: string): Promise<CallResult> =>
      raw ? call(raw.training.stop(runId)) : Promise.resolve(desktopOnly("Stopping runs")),
    pause: (runId: string): Promise<CallResult> =>
      raw ? call(raw.training.pause(runId)) : Promise.resolve(desktopOnly("Pausing runs")),
    resume: (runId: string): Promise<CallResult & { runId?: string; record?: RunRecord }> =>
      raw ? call(raw.training.resume(runId)) : Promise.resolve(desktopOnly("Resuming runs")),
    runs: (limit?: number): Promise<CallResult & { runs?: RunRecord[] }> =>
      raw ? call(raw.training.runs(limit)) : Promise.resolve(desktopOnly("Run history")),
    run: (runId: string): Promise<CallResult & { run?: RunRecord }> =>
      raw ? call(raw.training.run(runId)) : Promise.resolve(desktopOnly("Run history")),
    deleteRun: (runId: string): Promise<CallResult> =>
      raw ? call(raw.training.deleteRun(runId)) : Promise.resolve(desktopOnly("Deleting runs")),
    log: (runId: string, lines?: number): Promise<CallResult & { events?: { event: string; detail: Record<string, unknown> }[]; stderr?: string[] }> =>
      raw ? call(raw.training.log(runId, lines)) : Promise.resolve(desktopOnly("Logs")),
    checkpoints: (runDir: string): Promise<CallResult & { checkpoints?: RunRecord["checkpoints"] }> =>
      raw ? call(raw.training.checkpoints(runDir)) : Promise.resolve(desktopOnly("Checkpoints")),
  },

  inference: {
    load: (modelDir: string): Promise<CallResult> =>
      raw ? call(raw.inference.load(modelDir)) : Promise.resolve(desktopOnly("Loading models")),
    generate: (payload: Record<string, unknown>): Promise<
      CallResult & {
        result?: {
          text?: string;
          thinking?: string;
          thinking_tokens?: number;
          seconds?: number;
          tokens_per_second?: number;
          prompt_tokens?: number;
          output_tokens?: number;
        };
      }
    > =>
      raw ? call(raw.inference.generate(payload)) : Promise.resolve(desktopOnly("Running inference")),
    unload: (): Promise<CallResult> =>
      raw ? call(raw.inference.unload()) : Promise.resolve(desktopOnly("Unloading models")),
    status: (): Promise<CallResult & { status?: { running?: boolean; loaded?: unknown } }> =>
      raw ? call(raw.inference.status()) : Promise.resolve(desktopOnly("Inference status")),
  },
};
