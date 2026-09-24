/**
 * Job manager.
 *
 * Training, evaluation, conversion and installs all run as separate engine
 * processes. Their stdout is appended to a real log file inside the job
 * directory, which means:
 *   - the UI can stream events while the job runs,
 *   - the app can be closed and reopened and still find the job running,
 *   - nothing is ever buffered only in memory.
 *
 * The UI talks to this module over IPC; this module talks to the engine.
 */
import fs from "node:fs";
import path from "node:path";
import { engineSpawn } from "./python.js";
import { ensureWorkspace } from "./paths.js";

const EVENT_PREFIX = "@@event ";

export class JobManager {
  constructor({ onEvent, workspace, executable }) {
    this.workspace = workspace;
    this.executable = executable;
    this.onEvent = onEvent;
    this.running = new Map();
  }

  setWorkspace(workspace) {
    this.workspace = workspace;
  }

  setExecutable(executable) {
    this.executable = executable;
  }

  jobsRoot() {
    return path.join(ensureWorkspace(this.workspace), "jobs");
  }

  jobDir(jobId) {
    return path.join(this.jobsRoot(), jobId);
  }

  newJobId(kind) {
    const stamp = new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14);
    return `${kind}-${stamp}-${Math.random().toString(36).slice(2, 6)}`;
  }

  /** Create the job directory, write its spec, start the engine process. */
  start(spec) {
    if (!this.workspace) throw new Error("No workspace is configured.");
    const kind = spec.kind ?? "train";
    const jobId = spec.job_id ?? this.newJobId(kind);
    const dir = this.jobDir(jobId);
    fs.mkdirSync(path.join(dir, "output"), { recursive: true });

    const full = {
      ...spec,
      job_id: jobId,
      job_dir: dir,
      output_dir: spec.output_dir ?? path.join(dir, "output"),
      workspace: this.workspace,
      created_at: new Date().toISOString(),
    };
    fs.writeFileSync(path.join(dir, "spec.json"), JSON.stringify(full, null, 2), "utf8");
    fs.writeFileSync(
      path.join(dir, "status.json"),
      JSON.stringify({
        job_id: jobId,
        kind,
        state: "queued",
        message: "Queued",
        started_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        checkpoints: [],
        metrics: {},
        job_dir: dir,
        output_dir: full.output_dir,
      }, null, 2),
      "utf8",
    );
    fs.writeFileSync(
      path.join(dir, "control.json"),
      JSON.stringify({ pause: false, stop: false, save_now: false }, null, 2),
      "utf8",
    );

    const command = kind === "install"
      ? ["-m", "zxtrain.cli", "install"]
      : ["-m", "zxtrain.cli", "run", path.join(dir, "spec.json")];

    const logPath = path.join(dir, "console.log");
    const logStream = fs.createWriteStream(logPath, { flags: "a" });
    const child = engineSpawn(this.executable, command, {
      stdio: ["pipe", "pipe", "pipe"],
    });
    if (kind === "install") {
      child.stdin.write(JSON.stringify(full));
      child.stdin.end();
    } else {
      child.stdin.end();
    }
    child.stdout.pipe(logStream);
    child.stderr.pipe(logStream);

    const watcher = this.#watch(jobId, logPath, child);
    this.running.set(jobId, { child, watcher, logPath, startedAt: Date.now() });
    child.on("close", (code) => {
      logStream.end();
      const entry = this.running.get(jobId);
      if (entry) entry.exitCode = code;
      this.#emit(jobId, {
        type: "process",
        code,
        message: code === 0 ? "Engine process finished" : `Engine process exited with code ${code}`,
      });
    });
    this.#emit(jobId, { type: "log", level: "info", message: `Job ${jobId} started (${kind}).` });
    return { jobId, jobDir: dir, spec: full };
  }

  /** Tail a job log: every @@event line becomes a live UI event. */
  #watch(jobId, logPath, child) {
    let offset = 0;
    let buffer = "";
    let stopped = false;
    try {
      // Skip everything that was written before this session only when the file
      // is being re-attached to a live job (child may be undefined then).
      if (!child && fs.existsSync(logPath)) offset = fs.statSync(logPath).size;
    } catch {
      offset = 0;
    }
    const timer = setInterval(() => {
      if (stopped) return;
      let stats;
      try {
        stats = fs.statSync(logPath);
      } catch {
        return;
      }
      if (stats.size <= offset) return;
      let chunk = "";
      try {
        const handle = fs.openSync(logPath, "r");
        const length = stats.size - offset;
        const target = Buffer.alloc(length);
        fs.readSync(handle, target, 0, length, offset);
        fs.closeSync(handle);
        chunk = target.toString("utf8");
        offset = stats.size;
      } catch {
        return;
      }
      buffer += chunk;
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (line.startsWith(EVENT_PREFIX)) {
          try {
            this.#emit(jobId, JSON.parse(line.slice(EVENT_PREFIX.length)));
          } catch {
            this.#emit(jobId, { type: "raw", message: line });
          }
        } else if (line.trim()) {
          this.#emit(jobId, { type: "raw", message: line });
        }
      }
    }, 400);
    return {
      stop: () => {
        stopped = true;
        clearInterval(timer);
      },
    };
  }

  #emit(jobId, event) {
    try {
      this.onEvent?.({ jobId, event });
    } catch {
      /* the renderer may be gone; the log file is the source of truth */
    }
  }

  /** Re-attach to jobs that were running before this session started. */
  async reattach(states) {
    for (const state of states ?? []) {
      if (!state.job_id) continue;
      const dir = this.jobDir(state.job_id);
      const logPath = path.join(dir, "console.log");
      if (!fs.existsSync(logPath)) continue;
      if (state.action === "reconnected") {
        const watcher = this.#watch(state.job_id, logPath, null);
        this.running.set(state.job_id, { watcher, logPath, reattached: true });
        this.#emit(state.job_id, {
          type: "log",
          level: "info",
          message: "Reconnected to a training process that is still running.",
        });
      }
    }
  }

  /** Pause / stop / checkpoint-now. Written to the control file the engine polls. */
  control(jobId, patch) {
    const dir = this.jobDir(jobId);
    const file = path.join(dir, "control.json");
    let current = { pause: false, stop: false, save_now: false };
    try {
      current = { ...current, ...JSON.parse(fs.readFileSync(file, "utf8")) };
    } catch {
      /* first time */
    }
    const next = { ...current, ...patch };
    fs.writeFileSync(file, JSON.stringify(next, null, 2), "utf8");
    this.#emit(jobId, {
      type: "log",
      level: "info",
      message: `Requested: ${Object.entries(patch).map(([key, value]) => `${key}=${value}`).join(", ")}`,
    });
    return next;
  }

  /** Hard stop: mark it, then terminate the process group if it is ours. */
  terminate(jobId, force = false) {
    this.control(jobId, { stop: true });
    const entry = this.running.get(jobId);
    if (entry?.child && !entry.child.killed) {
      try {
        if (force) entry.child.kill("SIGKILL");
        else entry.child.kill("SIGTERM");
      } catch {
        /* already gone */
      }
    }
    return { jobId, forced: force };
  }

  status(jobId) {
    const file = path.join(this.jobDir(jobId), "status.json");
    try {
      return JSON.parse(fs.readFileSync(file, "utf8"));
    } catch {
      return { job_id: jobId, state: "unknown", message: "No status file yet." };
    }
  }

  list() {
    const root = this.jobsRoot();
    if (!fs.existsSync(root)) return [];
    const jobs = [];
    for (const name of fs.readdirSync(root)) {
      const dir = path.join(root, name);
      let stats;
      try {
        stats = fs.statSync(dir);
      } catch {
        continue;
      }
      if (!stats.isDirectory()) continue;
      const status = this.status(name);
      let spec = {};
      try {
        spec = JSON.parse(fs.readFileSync(path.join(dir, "spec.json"), "utf8"));
      } catch {
        /* spec may be missing for interrupted runs */
      }
      jobs.push({
        ...status,
        job_id: status.job_id ?? name,
        kind: status.kind ?? spec.kind,
        backend: status.backend ?? spec.backend,
        method: status.method ?? spec.method,
        model: spec.base_model,
        datasets: spec.dataset_paths ?? [],
        config: spec,
        created: status.started_at ?? stats.mtime.toISOString(),
        attached: this.running.has(name),
      });
    }
    jobs.sort((a, b) => String(b.created).localeCompare(String(a.created)));
    return jobs;
  }

  logs(jobId, lines = 400) {
    const file = path.join(this.jobDir(jobId), "console.log");
    try {
      const content = fs.readFileSync(file, "utf8").split("\n");
      return content.slice(-lines).join("\n");
    } catch {
      return "";
    }
  }

  events(jobId, limit = 300) {
    const file = path.join(this.jobDir(jobId), "events.jsonl");
    try {
      const content = fs.readFileSync(file, "utf8").trim().split("\n").slice(-limit);
      return content.map((line) => {
        try {
          return JSON.parse(line);
        } catch {
          return { type: "raw", message: line };
        }
      });
    } catch {
      return [];
    }
  }

  metrics(jobId, limit = 4000) {
    const file = path.join(this.jobDir(jobId), "metrics.jsonl");
    try {
      return fs
        .readFileSync(file, "utf8")
        .trim()
        .split("\n")
        .slice(-limit)
        .map((line) => {
          try {
            return JSON.parse(line);
          } catch {
            return null;
          }
        })
        .filter(Boolean);
    } catch {
      return [];
    }
  }

  activeCount() {
    return this.list().filter((job) => ["running", "queued", "paused"].includes(job.state)).length;
  }
}
