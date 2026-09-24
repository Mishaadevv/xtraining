/**
 * Typed client for the desktop bridge.
 *
 * When the renderer is opened in a plain browser (`npm run dev:web`) there is no
 * bridge, and this module says so plainly instead of silently returning empty
 * data — nothing in the app should ever look like it worked when it did not.
 */
import type { EngineError, Result, UpdateState } from "./types";

type Bridge = {
  app: { info: () => Promise<Result<any>> };
  settings: { get: () => Promise<Result<any>>; set: (patch: any) => Promise<Result<any>> };
  registry: {
    get: () => Promise<Result<any>>;
    set: (patch: any) => Promise<Result<any>>;
    add: (collection: string, item: any) => Promise<Result<any>>;
    update: (collection: string, item: any) => Promise<Result<any>>;
    remove: (collection: string, id: string) => Promise<Result<any>>;
  };
  engine: {
    call: (method: string, payload?: any, options?: any) => Promise<Result<any>>;
    interpreter: () => Promise<Result<any>>;
    probe: (executable: string) => Promise<Result<any>>;
    interpreters: (force?: boolean) => Promise<Result<any>>;
  };
  jobs: {
    list: () => Promise<Result<any>>;
    start: (spec: any) => Promise<Result<any>>;
    control: (jobId: string, patch: any) => Promise<Result<any>>;
    terminate: (jobId: string, force?: boolean) => Promise<Result<any>>;
    status: (jobId: string) => Promise<Result<any>>;
    logs: (jobId: string, lines?: number) => Promise<Result<string>>;
    events: (jobId: string, limit?: number) => Promise<Result<any[]>>;
    metrics: (jobId: string, limit?: number) => Promise<Result<any[]>>;
    onEvent: (handler: (payload: { jobId: string; event: any }) => void) => () => void;
    onReconciled: (handler: (findings: any[]) => void) => () => void;
  };
  sidecar: {
    load: (model: string, backend?: string) => Promise<Result<any>>;
    unload: () => Promise<Result<any>>;
    status: () => Promise<Result<any>>;
    stop: () => Promise<Result<any>>;
    generate: (payload: any) => Promise<Result<any>>;
    evaluate: (payload: any) => Promise<Result<any>>;
    onEvent: (handler: (event: any) => void) => () => void;
    onStream: (handler: (event: any) => void) => () => void;
  };
  updates: {
    state: () => Promise<Result<any>>;
    check: () => Promise<Result<any>>;
    download: () => Promise<Result<any>>;
    install: () => Promise<Result<any>>;
    onState: (handler: (state: any) => void) => () => void;
  };
  dialog: {
    pickFolder: (options?: any) => Promise<Result<string | null>>;
    pickFiles: (options?: any) => Promise<Result<string[]>>;
    saveFile: (options?: any) => Promise<Result<string | null>>;
  };
  shell: { reveal: (target: string) => Promise<Result<any>>; openExternal: (url: string) => Promise<Result<any>> };
  fs: { readText: (file: string, limit?: number) => Promise<Result<string>> };
  notify: (title: string, body: string) => Promise<Result<any>>;
  platform: string;
};

declare global {
  interface Window {
    zx?: Bridge;
  }
}

export const bridgeAvailable = typeof window !== "undefined" && Boolean(window.zx);

const missing: { ok: false; error: EngineError } = {
  ok: false,
  error: {
    code: "bridge_unavailable",
    message: "The desktop bridge is not available in this window.",
    hint: "Start the app with `npm run dev` (Electron) instead of opening the renderer directly in a browser.",
  },
};

function bridge(): Bridge {
  if (!window.zx) throw new Error("bridge_unavailable");
  return window.zx;
}

/** Unwrap a `Result` into data or throw an Error carrying the structured payload. */
export async function unwrap<T>(promise: Promise<Result<T>>): Promise<T> {
  let result: Result<T>;
  try {
    result = await promise;
  } catch (error) {
    throw Object.assign(new Error(String((error as Error)?.message ?? error)), {
      structured: {
        code: "bridge_error",
        message: "The desktop main process did not respond.",
        hint: "Restart the app if this keeps happening.",
      },
    });
  }
  if (!result || result.ok !== true) {
    const error = (result as any)?.error ?? missing.error;
    const wrapped = new Error(error.message ?? "Unknown engine error");
    (wrapped as any).structured = error;
    throw wrapped;
  }
  return result.data;
}

export async function tryCall<T>(promise: Promise<Result<T>>): Promise<{ data: T | null; error: any | null }> {
  try {
    return { data: await unwrap(promise), error: null };
  } catch (error) {
    return { data: null, error: (error as any).structured ?? { code: "unknown", message: String((error as Error).message) } };
  }
}

export const api = {
  available: bridgeAvailable,
  appInfo: () => unwrap(bridge().app.info()),
  settings: {
    get: () => unwrap(bridge().settings.get()),
    set: (patch: Record<string, unknown>) => unwrap(bridge().settings.set(patch)),
  },
  registry: {
    get: () => unwrap(bridge().registry.get()),
    set: (patch: Record<string, unknown>) => unwrap(bridge().registry.set(patch)),
    add: (collection: string, item: Record<string, unknown>) => unwrap(bridge().registry.add(collection, item)),
    update: (collection: string, item: Record<string, unknown>) => unwrap(bridge().registry.update(collection, item)),
    remove: (collection: string, id: string) => unwrap(bridge().registry.remove(collection, id)),
  },
  /** Engine command, with workspace injected by the main process. */
  call: <T = any>(method: string, payload: Record<string, unknown> = {}, options: Record<string, unknown> = {}) =>
    unwrap<T>(bridge().engine.call(method, payload, options)),
  callSafe: <T = any>(method: string, payload: Record<string, unknown> = {}, options: Record<string, unknown> = {}) =>
    tryCall<T>(bridge().engine.call(method, payload, options)),
  interpreter: () => unwrap(bridge().engine.interpreter()),
  interpreters: (force = false) => unwrap(bridge().engine.interpreters(force)),
  probeInterpreter: (executable: string) => unwrap(bridge().engine.probe(executable)),
  jobs: {
    list: () => unwrap(bridge().jobs.list()),
    start: (spec: Record<string, unknown>) => unwrap(bridge().jobs.start(spec)),
    control: (jobId: string, patch: Record<string, unknown>) => unwrap(bridge().jobs.control(jobId, patch)),
    terminate: (jobId: string, force = false) => unwrap(bridge().jobs.terminate(jobId, force)),
    status: (jobId: string) => unwrap(bridge().jobs.status(jobId)),
    logs: (jobId: string, lines = 400) => unwrap<string>(bridge().jobs.logs(jobId, lines)),
    events: (jobId: string, limit = 300) => unwrap<any[]>(bridge().jobs.events(jobId, limit)),
    metrics: (jobId: string, limit = 4000) => unwrap<any[]>(bridge().jobs.metrics(jobId, limit)),
    onEvent: (handler: (payload: { jobId: string; event: any }) => void) => window.zx?.jobs.onEvent(handler) ?? (() => {}),
    onReconciled: (handler: (findings: any[]) => void) => window.zx?.jobs.onReconciled(handler) ?? (() => {}),
  },
  sidecar: {
    load: (model: string, backend?: string) => unwrap(bridge().sidecar.load(model, backend)),
    unload: () => unwrap(bridge().sidecar.unload()),
    status: () => unwrap(bridge().sidecar.status()),
    generate: (payload: Record<string, unknown>) => unwrap(bridge().sidecar.generate(payload)),
    evaluate: (payload: Record<string, unknown>) => unwrap(bridge().sidecar.evaluate(payload)),
    onStream: (handler: (event: any) => void) => window.zx?.sidecar.onStream(handler) ?? (() => {}),
  },
  updates: {
    state: () => unwrap<UpdateState>(bridge().updates.state()),
    check: () => unwrap<UpdateState>(bridge().updates.check()),
    download: () => unwrap<UpdateState>(bridge().updates.download()),
    install: () => unwrap<{ installing: boolean; version: string | null }>(bridge().updates.install()),
    onState: (handler: (state: UpdateState) => void) => window.zx?.updates.onState(handler) ?? (() => {}),
  },
  dialog: {
    pickFolder: (options?: Record<string, unknown>) => unwrap<string | null>(bridge().dialog.pickFolder(options)),
    pickFiles: (options?: Record<string, unknown>) => unwrap<string[]>(bridge().dialog.pickFiles(options)),
    saveFile: (options?: Record<string, unknown>) => unwrap<string | null>(bridge().dialog.saveFile(options)),
  },
  shell: {
    reveal: (target: string) => unwrap(bridge().shell.reveal(target)),
    openExternal: (url: string) => unwrap(bridge().shell.openExternal(url)),
  },
  readText: (file: string, limit?: number) => unwrap<string>(bridge().fs.readText(file, limit)),
  notify: (title: string, body: string) => unwrap(bridge().notify(title, body)),
};

export type { Bridge };
