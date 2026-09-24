/**
 * Application updates.
 *
 * `electron-updater` knows how to talk to a release channel; this module decides
 * what that means for *this* install and never reports more than it knows:
 *
 *  - a portable build can find a new version but cannot install it in place,
 *    because there is no installation to replace;
 *  - an unpacked build is a real install as far as Electron is concerned, so it
 *    checks and downloads normally;
 *  - running from source cannot update at all, and says so instead of failing.
 *
 * Updates are downloaded, never installed behind the user's back: the download is
 * explicit and `quitAndInstall` runs only when the user asks for it. The channel
 * comes from `app-update.yml` (written by the builder) unless a settings override
 * or `ZEQOUX_UPDATE_URL` points somewhere else — which is also how a local channel
 * is tested. `ZEQOUX_UPDATE_SKIP_SIGNATURE=1` disables signature verification for
 * development builds signed with a self-signed certificate; whenever it is on, the
 * UI is told so it can say it out loud.
 */
import { app } from "electron";
import electronUpdater from "electron-updater";
import fs from "node:fs";
import path from "node:path";
import { appVersion } from "./paths.js";

// electron-updater is CommonJS: the named export is not always visible to ESM.
const { autoUpdater } = electronUpdater;

const CHECK_DELAY_MS = 15_000;

/** `app-update.yml` is a flat file written by electron-builder. Only the keys
 *  this module needs are read, so a full YAML parser would be overkill. */
function readUpdateConfig() {
  const file = path.join(process.resourcesPath ?? "", "app-update.yml");
  try {
    const text = fs.readFileSync(file, "utf8");
    const values = {};
    for (const line of text.split(/\r?\n/)) {
      const match = line.match(/^([A-Za-z][\w-]*):\s*(.+)$/);
      if (match) values[match[1]] = match[2].trim();
    }
    return {
      provider: values.provider ?? null,
      url: values.url ?? null,
      owner: values.owner ?? null,
      repo: values.repo ?? null,
      channel: values.channel ?? null,
      file,
    };
  } catch {
    return { provider: null, url: null, owner: null, repo: null, channel: null, file };
  }
}

function describeChannel(config) {
  if (config.provider === "github" && config.owner && config.repo) {
    return `github:${config.owner}/${config.repo}`;
  }
  if (config.url) return config.url;
  return config.provider ?? "not configured";
}

function fileLogger(file) {
  const write = (level, args) => {
    if (!file) return;
    const line = `${new Date().toISOString()} ${level.toUpperCase()} ${args
      .map((value) => (value instanceof Error ? value.stack ?? value.message : typeof value === "string" ? value : JSON.stringify(value)))
      .join(" ")}\n`;
    try {
      fs.mkdirSync(path.dirname(file), { recursive: true });
      fs.appendFileSync(file, line, "utf8");
    } catch {
      /* the log is a convenience, never a failure */
    }
  };
  return {
    info: (...args) => write("info", args),
    warn: (...args) => write("warn", args),
    error: (...args) => write("error", args),
    debug: (...args) => write("debug", args),
  };
}

export class Updates {
  constructor({ onState = () => {}, workspace = null } = {}) {
    this.onState = onState;
    this.workspace = workspace;
    this.logFile = workspace ? path.join(workspace, "logs", "updater.log") : null;
    this.channelOverride = null;
    this.signatureVerification = process.env.ZEQOUX_UPDATE_SKIP_SIGNATURE !== "1";
    this.mode = this.#detectMode();
    this.status = "idle";
    this.available = null;
    this.progress = null;
    this.error = null;
    this.backgroundError = null;
    this.automatic = false;
    this.lastCheckedAt = null;
    this.downloadedFile = null;
    this.checkedOnStart = false;

    autoUpdater.autoDownload = false;
    autoUpdater.autoInstallOnAppQuit = true;
    autoUpdater.logger = fileLogger(this.logFile);
    if (!this.signatureVerification) {
      // Deliberately loud: accepting an update whose publisher cannot be verified
      // is a development affordance for self-signed builds, nothing else.
      autoUpdater.verifyUpdateCodeSignature = async () => {
        autoUpdater.logger?.warn("update signature verification is disabled (ZEQOUX_UPDATE_SKIP_SIGNATURE=1)");
        return null;
      };
    }

    autoUpdater.on("checking-for-update", () => this.#set({ status: "checking", error: null }));
    autoUpdater.on("update-available", (info) => {
      this.#set({
        status: "available",
        error: null,
        backgroundError: null,
        lastCheckedAt: new Date().toISOString(),
        available: {
          version: String(info?.version ?? ""),
          releaseName: info?.releaseName ?? null,
          releaseNotes: typeof info?.releaseNotes === "string" ? info.releaseNotes : null,
          releaseDate: info?.releaseDate ?? null,
          size: info?.files?.[0]?.size ?? null,
        },
      });
    });
    autoUpdater.on("update-not-available", () => {
      this.#set({
        status: "up-to-date",
        error: null,
        backgroundError: null,
        lastCheckedAt: new Date().toISOString(),
        available: null,
        progress: null,
      });
    });
    autoUpdater.on("download-progress", (progress) => {
      this.#set({
        status: "downloading",
        progress: {
          percent: Number(progress?.percent ?? 0),
          transferred: Number(progress?.transferred ?? 0),
          total: Number(progress?.total ?? 0),
          bytesPerSecond: Number(progress?.bytesPerSecond ?? 0),
        },
      });
    });
    autoUpdater.on("update-downloaded", (info) => {
      this.downloadedFile = info?.downloadedFile ?? null;
      this.#set({ status: "downloaded", progress: null, error: null });
    });
    autoUpdater.on("error", (error) => this.#fail(error, this.status === "downloading"));
  }

  #detectMode() {
    if (!app.isPackaged) return "development";
    if (process.platform === "win32") {
      // electron-builder's portable target unpacks to a temp directory and sets
      // these, so there is no installation to replace.
      if (process.env.PORTABLE_EXECUTABLE_DIR || process.env.PORTABLE_EXECUTABLE_FILE) return "portable";
      return "installed";
    }
    if (process.platform === "linux") return process.env.APPIMAGE ? "appimage" : "installed";
    return "installed";
  }

  get supported() {
    return this.mode === "installed" || this.mode === "appimage";
  }

  get reason() {
    if (this.mode === "development") return "Updates apply to an installed build, not to the sources.";
    if (this.mode === "portable") return "This is the portable build: it can find a new version, but the new build is installed by downloading it.";
    if (process.platform === "linux" && this.mode !== "appimage") {
      return "Self-updating needs the AppImage build; .deb and .rpm installs are updated by the package manager.";
    }
    return "";
  }

  setWorkspace(workspace) {
    this.workspace = workspace;
    this.logFile = workspace ? path.join(workspace, "logs", "updater.log") : null;
    autoUpdater.logger = fileLogger(this.logFile);
  }

  configure({ channelUrl = null, checkOnStart = false } = {}) {
    const override = channelUrl || process.env.ZEQOUX_UPDATE_URL || null;
    this.channelOverride = override ? String(override).replace(/\/+$/, "") : null;
    if (this.channelOverride !== (this.appliedOverride ?? null)) {
      if (this.channelOverride) {
        try {
          autoUpdater.setFeedURL({ provider: "generic", url: this.channelOverride });
          this.appliedOverride = this.channelOverride;
          this.available = null;
          this.status = "idle";
          this.error = null;
          autoUpdater.logger?.info(`update channel overridden to ${this.channelOverride}`);
        } catch (error) {
          this.appliedOverride = null;
          this.#fail(error, false);
        }
      } else {
        // Back to whatever the build shipped with, so clearing the field in
        // Settings really means “use the release channel again”.
        const config = readUpdateConfig();
        this.appliedOverride = null;
        try {
          if (config.provider === "github" && config.owner && config.repo) {
            autoUpdater.setFeedURL({ provider: "github", owner: config.owner, repo: config.repo, channel: config.channel ?? "latest" });
          }
        } catch (error) {
          this.#fail(error, false);
        }
      }
    }
    this.checkOnStart = Boolean(checkOnStart);
    if (checkOnStart && !this.checkedOnStart && this.supported) {
      this.checkedOnStart = true;
      const timer = setTimeout(() => {
        this.check({ manual: false }).catch(() => {});
      }, CHECK_DELAY_MS);
      timer.unref?.();
    }
    return this.state();
  }

  state() {
    const config = readUpdateConfig();
    const github = config.provider === "github" && config.owner && config.repo;
    return {
      currentVersion: appVersion(),
      supported: this.supported,
      reason: this.reason,
      mode: this.mode,
      status: this.status,
      channel: {
        // The channel the updater will actually talk to, and where that came from.
        source: this.channelOverride ? "override" : github ? "release" : "configured",
        provider: this.channelOverride ? "generic" : config.provider,
        owner: github ? config.owner : null,
        repo: github ? config.repo : null,
        url: this.channelOverride ?? config.url ?? (github ? `https://github.com/${config.owner}/${config.repo}/releases` : null),
        label: this.channelOverride ?? describeChannel(config),
        configFile: config.file,
      },
      signatureVerification: this.signatureVerification,
      available: this.available,
      progress: this.progress,
      error: this.error,
      backgroundError: this.backgroundError,
      downloadedFile: this.downloadedFile,
      lastCheckedAt: this.lastCheckedAt,
      logFile: this.logFile,
    };
  }

  #set(patch) {
    Object.assign(this, patch);
    this.onState?.(this.state());
  }

  #fail(error, duringDownload) {
    const message = String(error?.message ?? error);
    const structured = {
      code: "update_failed",
      message,
      hint: /latest\.yml|404|ENOTFOUND|ETIMEDOUT|ECONNREFUSED|getaddrinfo|status code/i.test(message)
        ? "The release channel could not be reached. Check the channel address and the connection."
        : "",
    };
    if (this.automatic && !duringDownload) {
      // Reported quietly: nobody asked for this check.
      this.backgroundError = structured;
      autoUpdater.logger?.warn(`automatic update check failed: ${message}`);
    } else {
      this.error = structured;
    }
    // A failed download leaves a pending update behind, not a failed check.
    if (duringDownload) this.status = "available";
    else if (this.status === "checking") this.status = this.available ? "available" : "idle";
    this.onState?.(this.state());
  }

  async check({ manual = true } = {}) {
    this.automatic = !manual;
    if (manual) this.backgroundError = null;
    if (!app.isPackaged) {
      this.automatic = false;
      this.error = { code: "not_packaged", message: this.reason, hint: "Run the installed build to check for updates." };
      this.onState?.(this.state());
      return this.state();
    }
    autoUpdater.logger?.info(`checking for updates via ${this.state().channel.label}`);
    try {
      await autoUpdater.checkForUpdates();
    } catch (error) {
      this.#fail(error, false);
    } finally {
      this.automatic = false;
    }
    return this.state();
  }

  async download() {
    if (!this.available) {
      this.error = { code: "nothing_to_download", message: "No update has been found yet.", hint: "" };
      this.onState?.(this.state());
      return this.state();
    }
    this.#set({ status: "downloading", progress: { percent: 0, transferred: 0, total: this.available.size ?? 0, bytesPerSecond: 0 }, error: null });
    try {
      await autoUpdater.downloadUpdate();
    } catch (error) {
      this.#fail(error, true);
    }
    return this.state();
  }

  /** Quit and let the installer take over. Refused where it cannot work. */
  install() {
    if (this.status !== "downloaded") {
      return { ok: false, error: { code: "not_downloaded", message: "No downloaded update to install.", hint: "" } };
    }
    if (!this.supported) {
      return {
        ok: false,
        error: {
          code: "not_installable",
          message: this.reason || "This build cannot install updates.",
          hint: this.downloadedFile ? `The downloaded file is at ${this.downloadedFile}.` : "",
        },
      };
    }
    // The engine keeps running: jobs are separate processes and the next launch
    // re-attaches to them.
    setImmediate(() => autoUpdater.quitAndInstall(false, true));
    return { ok: true, data: { installing: true, version: this.available?.version ?? null } };
  }
}
