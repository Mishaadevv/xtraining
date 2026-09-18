/**
 * Inference sessions.
 *
 * The Python inference runtime is a long-lived subprocess that holds one model
 * in memory, so a conversation does not reload weights for every message. This
 * module owns that process and correlates request/response pairs by id.
 */
const python = require("./python");

let session = null; // { handle, loaded, modelDir, pending: Map }
let emitToRenderer = () => {};
let counter = 0;

function attach(sender) {
  emitToRenderer = sender;
}

function send(channel, payload) {
  try {
    emitToRenderer(channel, payload);
  } catch {
    /* window may be gone */
  }
}

function ensureSession() {
  if (session && session.handle) return session;

  const pending = new Map();
  const state = { handle: null, loaded: null, pending, lastTokens: [] };

  const handle = python.stream(["infer"], {
    onEvent: (parsed) => {
      const detail = parsed.detail || {};
      const requestId = detail.request_id;

      if (parsed.event === "inference-token") {
        send("zeqou:inference:token", { requestId, token: detail.token });
        return;
      }

      if (parsed.event === "inference-error") {
        const waiter = pending.get(requestId);
        if (waiter) {
          pending.delete(requestId);
          waiter.resolve({ ok: false, error: {
            code: detail.code,
            message: detail.message,
            hint: detail.hint,
            traceback: detail.traceback,
          } });
        }
        return;
      }

      if (parsed.event === "inference-result") {
        const waiter = pending.get(requestId);
        if (waiter) {
          pending.delete(requestId);
          waiter.resolve({ ok: true, result: detail });
        }
        if (detail.op === "load") {
          state.loaded = { modelDir: detail.model_dir, mode: detail.mode, params: detail.params };
          send("zeqou:inference:state", { loaded: state.loaded });
        }
        if (detail.op === "unload") {
          state.loaded = null;
          send("zeqou:inference:state", { loaded: null });
        }
      }

      if (parsed.event === "log") {
        send("zeqou:inference:log", { line: detail.message, level: detail.level });
      }
    },
    onStderr: (line) => send("zeqou:inference:log", { line, level: "info" }),
  });

  // A dead runtime must not leave callers hanging.
  handle.done.then((exit) => {
    for (const [, waiter] of pending.entries()) {
      waiter.resolve({
        ok: false,
        error: {
          code: "runtime_exited",
          message: `The inference runtime exited (code ${exit.code}).`,
          hint: "It will be restarted on the next request.",
        },
      });
    }
    if (session && session.handle === handle) session = null;
  });

  state.handle = handle;
  session = state;
  return session;
}

function request(payload, { timeoutMs = 30 * 60 * 1000 } = {}) {
  const current = ensureSession();
  counter += 1;
  const requestId = `req_${counter}`;
  const enriched = { ...payload, id: requestId };

  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      current.pending.delete(requestId);
      resolve({
        ok: false,
        error: {
          code: "timeout",
          message: "The inference runtime did not answer in time.",
          hint: "Large models can take a while to load; check the log panel.",
        },
      });
    }, timeoutMs);

    current.pending.set(requestId, {
      resolve: (value) => {
        clearTimeout(timer);
        resolve(value);
      },
    });

    const written = current.handle.write(enriched);
    if (!written) {
      current.pending.delete(requestId);
      clearTimeout(timer);
      resolve({
        ok: false,
        error: { code: "write_failed", message: "Could not reach the inference runtime." },
      });
    }
  });
}

async function load(modelDir) {
  if (!modelDir) {
    return { ok: false, error: { code: "no_model", message: "No model directory was given." } };
  }
  return request({ type: "load", model_dir: modelDir });
}

async function generate(options) {
  return request({ type: "generate", ...options });
}

async function unload() {
  if (!session) return { ok: true };
  return request({ type: "unload" });
}

function status() {
  if (!session) return { running: false, loaded: null };
  return { running: true, loaded: session.loaded };
}

function dispose() {
  if (!session) return;
  try {
    session.handle.write({ type: "quit" });
  } catch {
    /* ignore */
  }
  const current = session;
  session = null;
  setTimeout(() => current.handle.kill(), 1500).unref?.();
}

module.exports = { attach, load, generate, unload, status, dispose };
