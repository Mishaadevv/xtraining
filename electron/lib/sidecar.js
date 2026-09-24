/**
 * The inference sidecar.
 *
 * Keeping a model resident in its own process is what makes the Playground and
 * local API serving usable: the model is loaded once, generation streams back
 * token by token, and the UI thread never blocks. If the sidecar dies, its exit
 * is reported with the real stderr output instead of a silent failure.
 */
import { engineSpawn } from "./python.js";

export class Sidecar {
  constructor({ executable, onEvent, label = "playground" }) {
    this.executable = executable;
    this.onEvent = onEvent;
    this.label = label;
    this.child = null;
    this.buffer = "";
    this.pending = new Map();
    this.nextId = 1;
    this.loaded = null;
    this.state = "stopped";
    this.lastError = null;
  }

  setExecutable(executable) {
    if (this.executable !== executable) this.stop();
    this.executable = executable;
  }

  start() {
    if (this.child && !this.child.killed) return this.info();
    this.state = "starting";
    this.child = engineSpawn(this.executable, ["-m", "zxtrain.cli", "serve"], {
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.child.stdout.setEncoding("utf8");
    this.child.stdout.on("data", (chunk) => this.#consume(chunk));
    this.child.stderr.setEncoding("utf8");
    this.child.stderr.on("data", (chunk) => {
      this.lastError = chunk.toString().slice(-4000);
    });
    this.child.on("close", (code) => {
      this.state = "stopped";
      this.child = null;
      for (const [, pending] of this.pending) {
        pending.reject(new Error(this.lastError || `Inference process exited with code ${code}`));
      }
      this.pending.clear();
      this.onEvent?.({ type: "sidecar", state: "stopped", code, error: this.lastError });
    });
    this.state = "running";
    return this.info();
  }

  #consume(chunk) {
    this.buffer += chunk;
    const lines = this.buffer.split("\n");
    this.buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      let payload;
      try {
        payload = JSON.parse(trimmed);
      } catch {
        this.onEvent?.({ type: "raw", message: trimmed });
        continue;
      }
      if (payload.ready) {
        this.state = "running";
        this.onEvent?.({ type: "sidecar", state: "ready", pid: payload.pid });
        continue;
      }
      const id = payload.id;
      if (payload.event) {
        this.onEvent?.({ type: "event", id, event: payload.event });
        const pending = this.pending.get(id);
        if (pending) pending.onEvent?.(payload.event);
        continue;
      }
      const pending = this.pending.get(id);
      if (!pending) continue;
      this.pending.delete(id);
      if (payload.error) {
        const error = new Error(payload.error.message ?? "Inference failed.");
        error.structured = payload.error;
        pending.reject(error);
      } else {
        if (payload.result?.model) this.loaded = payload.result.model;
        pending.resolve(payload.result);
      }
    }
  }

  request(op, payload = {}, onEvent = null) {
    this.start();
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onEvent });
      const timer = setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error("The inference process did not answer in time."));
        }
      }, payload.timeout ?? 600_000);
      const clear = () => clearTimeout(timer);
      const original = this.pending.get(id);
      this.pending.set(id, {
        onEvent,
        resolve: (value) => {
          clear();
          original.resolve(value);
        },
        reject: (error) => {
          clear();
          original.reject(error);
        },
      });
      try {
        this.child?.stdin.write(`${JSON.stringify({ id, op, ...payload })}\n`);
      } catch (error) {
        this.pending.delete(id);
        clear();
        reject(error);
      }
    });
  }

  load(model, backend) {
    return this.request("load", { model, backend });
  }

  generate(payload, onEvent) {
    return this.request("generate", payload, onEvent);
  }

  evaluate(payload, onEvent) {
    return this.request("evaluate", { timeout: 3_600_000, ...payload }, onEvent);
  }

  unload() {
    return this.request("unload", {});
  }

  stop() {
    try {
      this.child?.kill();
    } catch {
      /* already gone */
    }
    this.child = null;
    this.state = "stopped";
  }

  info() {
    return {
      state: this.state,
      running: Boolean(this.child && !this.child.killed),
      model: this.loaded,
      pending: this.pending.size,
      label: this.label,
    };
  }
}
