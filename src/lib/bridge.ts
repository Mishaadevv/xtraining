/**
 * Access to the desktop bridge.
 *
 * When the renderer is opened in a plain browser (Vite dev server, or the
 * preview pane) every privileged call resolves with an explicit
 * `desktop_only` error instead of silently pretending to work. That keeps the
 * UI honest and still lets the interface be developed and inspected.
 */
import type {
  BackendError,
  DatasetReport,
  EnvSnapshot,
  HardwareSnapshot,
  InstallPlan,
  ModelEntry,
  Project,
  RunRecord,
  Settings,
  ValidationIssue,
} from "./types";

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
    setInterpreter(path: string | null): Promise<Record<string, unknown>>;
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
    autoConfig(payload: { baseline?: boolean } & Record<string, unknown>): Promise<Record<string, unknown>>;
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

function desktopOnly(what: string): Promise<never> {
  return Promise.resolve({
    ok: false,
    error: {
      code: "desktop_only",
      message: `${what} is available in the desktop app only.`,
      hint: DESKTOP_HINT,
    },
  }) as never;
}

/** Uniform result shape used by every store action. */
export interface CallResult {
  ok: boolean;
  error?: BackendError;
  [key: string]: unknown;
}

function fail(message: string, code = "bridge"): CallResult {
  return { ok: false, error: { code, message, hint: DESKTOP_HINT } };
}

async function call<T extends CallResult>(fn: () => Promise<Record<string, unknown>>): Promise<T> {
  try {
    const result = (await fn()) as unknown as T;
    if (!result || typeof result !== "object") return fail("The desktop bridge returned no data.") as T;
    return result;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return fail(message) as T;
  }
}

/* ------------------------------------------------------------------ typed API */

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
    merge: { ready: boolean; applicable: boolean; missing: string[] };
  };
  blockers: { code: string; message: string; hint: string }[];
}

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
  hfCacheDir: string;
  backendDir: string;
  encryptionAvailable: boolean;
}

export const bridge = {
  isDesktop,

  on(channel: string, handler: (payload: any) => void): () => void {
    if (!raw) return () => {};
    return raw.on(channel, handler);
  },

  app: {
    info: (): Promise<CallResult & { [k: string]: unknown }> =>
      raw ? call(() => raw.app.info()) : (desktopOnly("App information") as never),
  },

  shell: {
    openPath: (target: string) => (raw ? call(() => raw.shell.openPath(target)) : desktopOnly("Opening paths")),
    openExternal: (url: string) =>
      raw ? call(() => raw.shell.openExternal(url)) : desktopOnly("Opening links"),
    showItem: (target: string) =>
      raw ? call(() => raw.shell.showItem(target)) : desktopOnly("Revealing files"),
  },

  dialogs: {
    pickDataset: () => (raw ? call(() => raw.dialogs.pickDataset()) : desktopOnly("File pickers")),
    pickModelFolder: () =>
      raw ? call(() => raw.dialogs.pickModelFolder()) : desktopOnly("Folder pickers"),
    pickPython: () => (raw ? call(() => raw.dialogs.pickPython()) : desktopOnly("File pickers")),
    pickDirectory: (title?: string) =>
      raw ? call(() => raw.dialogs.pickDirectory(title)) : desktopOnly("Folder pickers"),
  },

  settings: {
    get: (): Promise<CallResult & { settings?: Settings }> =>
      raw ? call(() => raw.settings.get()) : (desktopOnly("Settings") as never),
    set: (patch: Partial<Settings>): Promise<CallResult & { settings?: Settings }> =>
      raw ? call(() => raw.settings.set(patch)) : (desktopOnly("Settings") as never),
    reset: (): Promise<CallResult & { settings?: Settings }> =>
      raw ? call(() => raw.settings.reset()) : (desktopOnly("Settings") as never),
    setToken: (token: string | null) =>
      raw ? call(() => raw.settings.setToken(token)) : desktopOnly("Storing tokens"),
    tokenStatus: () =>
      raw ? call(() => raw.settings.tokenStatus()) : desktopOnly("Token status"),
  },

  env: {
    detect: (options?: { force?: boolean }): Promise<
      CallResult & { [K in keyof EnvSnapshot]?: EnvSnapshot[K] } & { error?: BackendError }
    > => (raw ? call(() => raw.env.detect(options)) : (desktopOnly("Environment detection") as never)),
    interpreters: (options?: { force?: boolean }) =>
      raw ? call(() => raw.env.interpreters(options)) : desktopOnly("Interpreter discovery"),
    setInterpreter: (path: string | null) =>
      raw ? call(() => raw.env.setInterpreter(path)) : desktopOnly("Interpreter selection"),
    installPlan: (cudaTag?: string): Promise<CallResult & { plan?: InstallPlan }> =>
      raw ? call(() => raw.env.installPlan(cudaTag)) : (desktopOnly("Install plans") as never),
    installRuntime: () => (raw ? call(() => raw.env.installRuntime()) : desktopOnly("Runtime installation")),
    installStatus: () => (raw ? call(() => raw.env.installStatus()) : desktopOnly("Runtime installation")),
    backends: () => (raw ? call(() => raw.env.backends()) : desktopOnly("Backend detection")),
  },

  hardware: {
    detect: (): Promise<CallResult & { hardware?: HardwareSnapshot | null }> =>
      raw ? call(() => raw.hardware.detect()) : (desktopOnly("Hardware detection") as never),
    sample: () => (raw ? call(() => raw.hardware.sample()) : desktopOnly("GPU sampling")),
  },

  datasets: {
    list: () => (raw ? call(() => raw.datasets.list()) : desktopOnly("Datasets")),
    import: (paths: string | string[]) =>
      raw ? call(() => raw.datasets.import(paths)) : desktopOnly("Importing datasets"),
    addHf: (payload: { id: string; split?: string }): Promise<CallResult & { dataset?: any }> =>
      raw ? call(() => raw.datasets.addHf(payload)) : (desktopOnly("Adding Hub datasets") as never),
    validate: (payload: Record<string, unknown>): Promise<
      CallResult & { report?: DatasetReport; error?: BackendError }
    > => (raw ? call(() => raw.datasets.validate(payload)) : (desktopOnly("Validating datasets") as never)),
    preview: (payload: Record<string, unknown>) =>
      raw ? call(() => raw.datasets.preview(payload)) : desktopOnly("Dataset preview"),
    remove: (id: string) => (raw ? call(() => raw.datasets.remove(id)) : desktopOnly("Removing datasets")),
  },

  models: {
    list: () => (raw ? call(() => raw.models.list()) : desktopOnly("Models")),
    add: (source: string): Promise<CallResult & { model?: ModelEntry; info?: any }> =>
      raw ? call(() => raw.models.add(source)) : (desktopOnly("Adding models") as never),
    inspect: (source: string) =>
      raw ? call(() => raw.models.inspect(source)) : desktopOnly("Inspecting models"),
    exportInfo: (payload: { modelId: string }): Promise<CallResult & { info?: ModelExportInfo }> =>
      raw ? call(() => raw.models.exportInfo(payload)) : (desktopOnly("Exporting models") as never),
    export: (payload: { modelId: string; outputDir: string; merge?: boolean }): Promise<CallResult & { output_dir?: string }> =>
      raw ? call(() => raw.models.export(payload)) : (desktopOnly("Exporting models") as never),
    remove: (id: string) => (raw ? call(() => raw.models.remove(id)) : desktopOnly("Removing models")),
  },

  projects: {
    list: (): Promise<CallResult & { projects?: Project[] }> =>
      raw ? call(() => raw.projects.list()) : (desktopOnly("Projects") as never),
    get: (id: string) => (raw ? call(() => raw.projects.get(id)) : desktopOnly("Projects")),
    create: (patch: Record<string, unknown>): Promise<CallResult & { project?: Project }> =>
      raw ? call(() => raw.projects.create(patch)) : (desktopOnly("Projects") as never),
    update: (id: string, patch: Record<string, unknown>) =>
      raw ? call(() => raw.projects.update(id, patch)) : desktopOnly("Projects"),
    remove: (id: string) => (raw ? call(() => raw.projects.remove(id)) : desktopOnly("Projects")),
  },

  training: {
    autoConfig: (payload: Record<string, unknown>) =>
      raw ? call(() => raw.training.autoConfig(payload)) : desktopOnly("Automatic configuration"),
    estimate: (payload: Record<string, unknown>) =>
      raw ? call(() => raw.training.estimate(payload)) : desktopOnly("VRAM estimation"),
    check: (payload: Record<string, unknown>) =>
      raw ? call(() => raw.training.check(payload)) : desktopOnly("Pre-flight checks"),
    start: (payload: Record<string, unknown>): Promise<CallResult & { runId?: string; record?: RunRecord }> =>
      raw ? call(() => raw.training.start(payload)) : (desktopOnly("Starting training") as never),
    progress: (runId: string) => (raw ? call(() => raw.training.progress(runId)) : desktopOnly("Run progress")),
    stop: (runId: string) => (raw ? call(() => raw.training.stop(runId)) : desktopOnly("Stopping runs")),
    pause: (runId: string) => (raw ? call(() => raw.training.pause(runId)) : desktopOnly("Pausing runs")),
    resume: (runId: string): Promise<CallResult & { runId?: string }> =>
      raw ? call(() => raw.training.resume(runId)) : (desktopOnly("Resuming runs") as never),
    runs: (limit?: number): Promise<CallResult & { runs?: RunRecord[] }> =>
      raw ? call(() => raw.training.runs(limit)) : (desktopOnly("Run history") as never),
    run: (runId: string): Promise<CallResult & { run?: RunRecord }> =>
      raw ? call(() => raw.training.run(runId)) : (desktopOnly("Run history") as never),
    deleteRun: (runId: string) => (raw ? call(() => raw.training.deleteRun(runId)) : desktopOnly("Deleting runs")),
    log: (runId: string, lines?: number) =>
      raw ? call(() => raw.training.log(runId, lines)) : desktopOnly("Logs"),
    checkpoints: (runDir: string) =>
      raw ? call(() => raw.training.checkpoints(runDir)) : desktopOnly("Checkpoints"),
  },

  inference: {
    load: (modelDir: string) => (raw ? call(() => raw.inference.load(modelDir)) : desktopOnly("Loading models")),
    generate: (payload: Record<string, unknown>) =>
      raw ? call(() => raw.inference.generate(payload)) : desktopOnly("Running inference"),
    unload: () => (raw ? call(() => raw.inference.unload()) : desktopOnly("Unloading models")),
    status: () => (raw ? call(() => raw.inference.status()) : desktopOnly("Inference status")),
  },
};

export type Issue = ValidationIssue;
