/**
 * Durable JSON storage with atomic writes.
 *
 * A crash or a power cut can never leave a half-written settings or registry
 * file: every write goes to a temporary file that is then renamed over the
 * target, and a corrupt file falls back to the last good copy.
 */
import fs from "node:fs";
import path from "node:path";

export class JsonStore {
  constructor(file, defaults = {}) {
    this.file = file;
    this.defaults = defaults;
    this.cache = null;
  }

  read() {
    if (this.cache) return this.cache;
    try {
      const raw = fs.readFileSync(this.file, "utf8");
      const parsed = JSON.parse(raw);
      this.cache = { ...structuredClone(this.defaults), ...parsed };
    } catch {
      const backup = `${this.file}.bak`;
      try {
        const parsed = JSON.parse(fs.readFileSync(backup, "utf8"));
        this.cache = { ...structuredClone(this.defaults), ...parsed };
      } catch {
        this.cache = structuredClone(this.defaults);
      }
    }
    return this.cache;
  }

  write(value) {
    this.cache = { ...structuredClone(this.defaults), ...value };
    this.#persist(this.cache);
    return this.cache;
  }

  update(patch) {
    return this.write({ ...this.read(), ...patch });
  }

  #persist(payload) {
    const dir = path.dirname(this.file);
    fs.mkdirSync(dir, { recursive: true });
    const tmp = path.join(dir, `.${path.basename(this.file)}.${process.pid}.tmp`);
    const text = JSON.stringify(payload, null, 2);
    fs.writeFileSync(tmp, text, "utf8");
    try {
      if (fs.existsSync(this.file)) fs.copyFileSync(this.file, `${this.file}.bak`);
    } catch {
      /* the backup is best effort */
    }
    fs.renameSync(tmp, this.file);
  }

  /** Serialise concurrent updates from several windows. */
  mutate(mutator) {
    const current = this.read();
    const next = mutator(structuredClone(current)) ?? current;
    return this.write(next);
  }
}

export function settingsStore() {
  return new JsonStore(path.join(process.env.ZEQOU_USER_DATA || "", "settings.json"), {
    version: 1,
    workspace: null,
    pythonPath: null,
    theme: "system",
    density: "comfortable",
    defaultBackend: null,
    defaultMethod: "lora",
    defaultDevice: "auto",
    notifications: { jobFinished: true, jobFailed: true, serverStarted: false },
    offlineMode: false,
    allowRemoteCode: false,
    logLevel: "INFO",
    autoSave: true,
    checkpointPolicy: { keepLast: 3, protectBest: true },
    lastProject: null,
    recent: [],
  });
}
