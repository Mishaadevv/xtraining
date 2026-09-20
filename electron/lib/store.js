/**
 * Persistence primitives.
 *
 * `JsonStore` writes atomically (temp file + rename) so a crash mid-write can
 * never leave a truncated index behind — losing the project list would be far
 * worse than losing the last write.
 *
 * Secrets (Hugging Face token) use Electron's safeStorage so they are encrypted
 * with the OS keychain instead of sitting in plain text in the settings file.
 */
const fs = require("node:fs");
const path = require("node:path");
const { safeStorage } = require("electron");
const { files, ensureDirs } = require("./paths");

// Every key here is read somewhere; nothing is stored "just in case", because a
// setting that has no effect is worse than no setting at all.
const SETTINGS_DEFAULTS = {
  version: 1,
  interpreterPath: null,
  cudaWheelTag: "cu124",
  hfCacheDir: null,
  theme: "dark",
  // Off => the wizard uses the documented defaults and says so.
  autoConfigure: true,
  // The initial Simple/Advanced state of a new wizard.
  simpleMode: true,
  advanced: {
    trustRemoteCode: false,
  },
  lastProjectId: null,
};

class JsonStore {
  constructor(filePath, defaults) {
    this.filePath = filePath;
    this.defaults = defaults;
    this.data = this.#read();
  }

  #read() {
    try {
      const raw = fs.readFileSync(this.filePath, "utf8");
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === "object") {
        return Array.isArray(this.defaults) ? parsed : { ...this.defaults, ...parsed };
      }
    } catch {
      /* missing or unreadable: fall through to defaults */
    }
    return Array.isArray(this.defaults) ? [...this.defaults] : { ...this.defaults };
  }

  get() {
    return this.data;
  }

  replace(next) {
    this.data = next;
    this.#write();
    return this.data;
  }

  /** Shallow merge for objects; full replace otherwise. */
  merge(patch) {
    if (Array.isArray(this.data) || Array.isArray(patch)) {
      return this.replace(patch);
    }
    this.data = { ...this.data, ...patch };
    this.#write();
    return this.data;
  }

  #write() {
    ensureDirs();
    const dir = path.dirname(this.filePath);
    fs.mkdirSync(dir, { recursive: true });
    const temp = `${this.filePath}.${process.pid}.tmp`;
    fs.writeFileSync(temp, JSON.stringify(this.data, null, 2), "utf8");
    fs.renameSync(temp, this.filePath);
  }
}

let settingsStore = null;
let datasetsStore = null;
let modelsStore = null;
let projectsStore = null;
let runsStore = null;

function settings() {
  if (!settingsStore) {
    settingsStore = new JsonStore(files.settings(), SETTINGS_DEFAULTS);
    // Pick up newly added default keys after an upgrade.
    for (const [key, value] of Object.entries(SETTINGS_DEFAULTS)) {
      if (!(key in settingsStore.get())) settingsStore.merge({ [key]: value });
    }
  }
  return settingsStore;
}

function datasets() {
  if (!datasetsStore) datasetsStore = new JsonStore(files.datasetsIndex(), []);
  return datasetsStore;
}

function models() {
  if (!modelsStore) modelsStore = new JsonStore(files.modelsIndex(), []);
  return modelsStore;
}

function projects() {
  if (!projectsStore) projectsStore = new JsonStore(files.projectsIndex(), []);
  return projectsStore;
}

function runs() {
  if (!runsStore) runsStore = new JsonStore(files.runsIndex(), []);
  return runsStore;
}

/* ---------------------------------------------------------------- secrets */

function readSecrets() {
  try {
    return JSON.parse(fs.readFileSync(files.secrets(), "utf8"));
  } catch {
    return {};
  }
}

function writeSecrets(data) {
  ensureDirs();
  const temp = `${files.secrets()}.${process.pid}.tmp`;
  fs.writeFileSync(temp, JSON.stringify(data, null, 2), "utf8");
  fs.renameSync(temp, files.secrets());
}

function encryptionAvailable() {
  try {
    return safeStorage.isEncryptionAvailable();
  } catch {
    return false;
  }
}

function setSecret(key, value) {
  const data = readSecrets();
  if (!value) {
    delete data[key];
  } else if (encryptionAvailable()) {
    data[key] = { enc: safeStorage.encryptString(String(value)).toString("base64") };
  } else {
    // Be explicit rather than pretending: the token is stored unencrypted.
    data[key] = { plain: String(value), insecure: true };
  }
  writeSecrets(data);
  return true;
}

function getSecret(key) {
  const entry = readSecrets()[key];
  if (!entry) return null;
  if (entry.enc) {
    try {
      return safeStorage.decryptString(Buffer.from(entry.enc, "base64"));
    } catch {
      return null;
    }
  }
  return entry.plain || null;
}

function hasSecret(key) {
  const entry = readSecrets()[key];
  return Boolean(entry && (entry.enc || entry.plain));
}

function secretStorageInfo(key) {
  const entry = readSecrets()[key];
  if (!entry) return { present: false, encrypted: false };
  return { present: true, encrypted: Boolean(entry.enc), insecure: Boolean(entry.insecure) };
}

module.exports = {
  JsonStore,
  SETTINGS_DEFAULTS,
  settings,
  datasets,
  models,
  projects,
  runs,
  setSecret,
  getSecret,
  hasSecret,
  secretStorageInfo,
  encryptionAvailable,
};
