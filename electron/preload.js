/**
 * Preload bridge.
 *
 * Deliberately a thin, explicit surface: the renderer never sees ipcRenderer,
 * node builtins or an arbitrary `invoke(channel)`. Every capability is named,
 * which keeps the attack surface small and makes the API self-documenting.
 */
const { contextBridge, ipcRenderer } = require("electron");

// Push channels the renderer can subscribe to.
const CHANNELS = [
  "zeqou:training:event",
  "zeqou:training:state",
  "zeqou:training:log",
  "zeqou:training:finished",
  "zeqou:gpu",
  "zeqou:runtime:install",
  "zeqou:datasets:changed",
  "zeqou:models:changed",
  "zeqou:projects:changed",
  "zeqou:settings:changed",
  "zeqou:inference:token",
  "zeqou:inference:thinking",
  "zeqou:inference:state",
  "zeqou:inference:log",
];

const bus = new Map();
for (const channel of CHANNELS) {
  bus.set(channel, new Set());
  ipcRenderer.on(channel, (_event, payload) => {
    for (const handler of bus.get(channel)) {
      try {
        handler(payload);
      } catch (error) {
        // A broken listener must not take the other subscribers down with it.
        console.error(`[zeqou] listener for ${channel} threw`, error);
      }
    }
  });
}

contextBridge.exposeInMainWorld("zeqou", {
  platform: process.platform,
  channels: [...CHANNELS],

  on(channel, handler) {
    if (!bus.has(channel)) throw new Error(`Unknown channel: ${channel}`);
    bus.get(channel).add(handler);
    return () => bus.get(channel).delete(handler);
  },

  app: {
    info: () => ipcRenderer.invoke("zeqou:app:info"),
  },

  shell: {
    openPath: (target) => ipcRenderer.invoke("zeqou:shell:openPath", target),
    openExternal: (url) => ipcRenderer.invoke("zeqou:shell:openExternal", url),
    showItem: (target) => ipcRenderer.invoke("zeqou:shell:showItem", target),
  },

  dialogs: {
    pickDataset: () => ipcRenderer.invoke("zeqou:dialog:dataset"),
    pickModelFolder: () => ipcRenderer.invoke("zeqou:dialog:modelFolder"),
    pickPython: () => ipcRenderer.invoke("zeqou:dialog:python"),
    pickDirectory: (title) => ipcRenderer.invoke("zeqou:dialog:directory", title),
  },

  settings: {
    get: () => ipcRenderer.invoke("zeqou:settings:get"),
    set: (patch) => ipcRenderer.invoke("zeqou:settings:set", patch),
    reset: () => ipcRenderer.invoke("zeqou:settings:reset"),
    setToken: (token) => ipcRenderer.invoke("zeqou:settings:token", { token }),
    tokenStatus: () => ipcRenderer.invoke("zeqou:settings:tokenStatus"),
  },

  env: {
    detect: (options) => ipcRenderer.invoke("zeqou:env:detect", options || {}),
    interpreters: (options) => ipcRenderer.invoke("zeqou:env:interpreters", options || {}),
    setInterpreter: (executablePath) => ipcRenderer.invoke("zeqou:env:setInterpreter", executablePath),
    installPlan: (cudaTag) => ipcRenderer.invoke("zeqou:env:installPlan", cudaTag),
    installRuntime: () => ipcRenderer.invoke("zeqou:env:installRuntime"),
    installStatus: () => ipcRenderer.invoke("zeqou:env:installStatus"),
    backends: () => ipcRenderer.invoke("zeqou:env:backends"),
  },

  hardware: {
    detect: () => ipcRenderer.invoke("zeqou:hardware:detect"),
    sample: () => ipcRenderer.invoke("zeqou:hardware:sample"),
  },

  datasets: {
    list: () => ipcRenderer.invoke("zeqou:datasets:list"),
    import: (paths) => ipcRenderer.invoke("zeqou:datasets:import", paths),
    addHf: (payload) => ipcRenderer.invoke("zeqou:datasets:addHf", payload),
    validate: (payload) => ipcRenderer.invoke("zeqou:datasets:validate", payload),
    preview: (payload) => ipcRenderer.invoke("zeqou:datasets:preview", payload),
    remove: (id) => ipcRenderer.invoke("zeqou:datasets:remove", id),
  },

  models: {
    list: () => ipcRenderer.invoke("zeqou:models:list"),
    add: (source) => ipcRenderer.invoke("zeqou:models:add", { source }),
    inspect: (source) => ipcRenderer.invoke("zeqou:models:inspect", { source }),
    exportInfo: (payload) => ipcRenderer.invoke("zeqou:models:exportInfo", payload),
    export: (payload) => ipcRenderer.invoke("zeqou:models:export", payload),
    remove: (id) => ipcRenderer.invoke("zeqou:models:remove", id),
  },

  projects: {
    list: () => ipcRenderer.invoke("zeqou:projects:list"),
    get: (id) => ipcRenderer.invoke("zeqou:projects:get", id),
    create: (patch) => ipcRenderer.invoke("zeqou:projects:create", patch),
    update: (id, patch) => ipcRenderer.invoke("zeqou:projects:update", { id, patch }),
    remove: (id) => ipcRenderer.invoke("zeqou:projects:remove", id),
  },

  training: {
    autoConfig: (payload) => ipcRenderer.invoke("zeqou:training:autoConfig", payload),
    estimate: (payload) => ipcRenderer.invoke("zeqou:training:estimate", payload),
    check: (payload) => ipcRenderer.invoke("zeqou:training:check", payload),
    start: (payload) => ipcRenderer.invoke("zeqou:training:start", payload),
    progress: (runId) => ipcRenderer.invoke("zeqou:training:progress", runId),
    stop: (runId) => ipcRenderer.invoke("zeqou:training:stop", runId),
    pause: (runId) => ipcRenderer.invoke("zeqou:training:pause", runId),
    resume: (runId) => ipcRenderer.invoke("zeqou:training:resume", runId),
    runs: (limit) => ipcRenderer.invoke("zeqou:training:runs", limit),
    run: (runId) => ipcRenderer.invoke("zeqou:training:run", runId),
    deleteRun: (runId) => ipcRenderer.invoke("zeqou:training:deleteRun", runId),
    log: (runId, lines) => ipcRenderer.invoke("zeqou:training:log", { runId, lines }),
    checkpoints: (runDir) => ipcRenderer.invoke("zeqou:training:checkpoints", runDir),
  },

  inference: {
    load: (modelDir) => ipcRenderer.invoke("zeqou:inference:load", { modelDir }),
    generate: (payload) => ipcRenderer.invoke("zeqou:inference:generate", payload),
    unload: () => ipcRenderer.invoke("zeqou:inference:unload"),
    status: () => ipcRenderer.invoke("zeqou:inference:status"),
  },
});
