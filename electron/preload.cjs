/**
 * The renderer's only door to the outside world.
 *
 * Everything is namespaced under `window.zx`, every call returns
 * `{ ok, data }` or `{ ok: false, error }`, and no engine, filesystem or Node
 * primitive is exposed directly.
 */
const { contextBridge, ipcRenderer } = require("electron");

const invoke = (channel, ...args) => ipcRenderer.invoke(channel, ...args);

function subscribe(channel, handler) {
  const listener = (_event, payload) => handler(payload);
  ipcRenderer.on(channel, listener);
  return () => ipcRenderer.removeListener(channel, listener);
}

contextBridge.exposeInMainWorld("zx", {
  app: {
    info: () => invoke("app:info"),
  },
  settings: {
    get: () => invoke("settings:get"),
    set: (patch) => invoke("settings:set", patch),
  },
  registry: {
    get: () => invoke("registry:get"),
    set: (patch) => invoke("registry:set", patch),
    add: (collection, item) => invoke("registry:mutate", collection, "add", item),
    update: (collection, item) => invoke("registry:mutate", collection, "update", item),
    remove: (collection, id) => invoke("registry:mutate", collection, "remove", { id }),
  },
  engine: {
    call: (method, payload, options) => invoke("engine:call", method, payload, options),
    interpreter: () => invoke("engine:interpreter"),
    probe: (executable) => invoke("engine:probe", executable),
    interpreters: (force) => invoke("interpreters:list", force),
  },
  jobs: {
    list: () => invoke("jobs:list"),
    start: (spec) => invoke("jobs:start", spec),
    control: (jobId, patch) => invoke("jobs:control", jobId, patch),
    terminate: (jobId, force) => invoke("jobs:terminate", jobId, force),
    status: (jobId) => invoke("jobs:status", jobId),
    logs: (jobId, lines) => invoke("jobs:logs", jobId, lines),
    events: (jobId, limit) => invoke("jobs:events", jobId, limit),
    metrics: (jobId, limit) => invoke("jobs:metrics", jobId, limit),
    onEvent: (handler) => subscribe("jobs:event", handler),
    onReconciled: (handler) => subscribe("jobs:reconciled", handler),
  },
  sidecar: {
    load: (model, backend) => invoke("sidecar:load", model, backend),
    unload: () => invoke("sidecar:unload"),
    status: () => invoke("sidecar:status"),
    stop: () => invoke("sidecar:stop"),
    generate: (payload) => invoke("sidecar:generate", payload),
    evaluate: (payload) => invoke("sidecar:evaluate", payload),
    onEvent: (handler) => subscribe("sidecar:event", handler),
    onStream: (handler) => subscribe("sidecar:stream", handler),
  },
  updates: {
    state: () => invoke("updates:state"),
    check: () => invoke("updates:check"),
    download: () => invoke("updates:download"),
    install: () => invoke("updates:install"),
    onState: (handler) => subscribe("updates:state", handler),
  },
  dialog: {
    pickFolder: (options) => invoke("dialog:pick-folder", options),
    pickFiles: (options) => invoke("dialog:pick-files", options),
    saveFile: (options) => invoke("dialog:save-file", options),
  },
  shell: {
    reveal: (target) => invoke("shell:reveal", target),
    openExternal: (url) => invoke("shell:open-external", url),
  },
  fs: {
    readText: (file, limit) => invoke("fs:read-text", file, limit),
  },
  notify: (title, body) => invoke("notify", title, body),
  platform: process.platform,
});
