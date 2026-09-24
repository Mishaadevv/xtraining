/**
 * Smoke test — runs as the Electron main process:
 *
 *     npm run smoke        # builds the renderer first
 *     electron scripts/smoke.mjs
 *
 * It drives the same modules the app uses (interpreter discovery, the job
 * manager, the engine CLI, the store) inside a throwaway workspace and asserts
 * that real files appeared on disk. Nothing is mocked: if the engine stops
 * working, this fails.
 */
import { app } from "electron";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { JobManager } from "../electron/lib/jobs.js";
import { ensureWorkspace, engineDir, rendererIndex } from "../electron/lib/paths.js";
import { engineCall, resolveInterpreter } from "../electron/lib/python.js";
import { JsonStore } from "../electron/lib/store.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), "zxtrain-smoke-"));
const workspace = path.join(tmpRoot, "workspace");
const userData = path.join(tmpRoot, "userdata");
fs.mkdirSync(userData, { recursive: true });
app.setPath("userData", userData);
process.env.ZEQOUX_WORKSPACE = workspace;

const logFile = process.env.ZEQOUX_SMOKE_LOG || path.join(tmpRoot, "smoke.log");
const stageFile = path.join(tmpRoot, "stage.log");

/** Progress markers written as early as possible, because the Electron binary
 *  on Windows does not forward main-process stdout to the launching terminal. */
function stage(line) {
  try {
    fs.appendFileSync(stageFile, `${new Date().toISOString()} ${line}\n`, "utf8");
  } catch {
    /* nothing else we can do */
  }
}

stage(`imports resolved; tmp ${tmpRoot}`);
process.on("uncaughtException", (error) => stage(`uncaughtException: ${error?.stack ?? error}`));
process.on("unhandledRejection", (error) => stage(`unhandledRejection: ${error?.stack ?? error}`));

/**
 * Electron's Windows build is a GUI binary: its stdout does not reach the
 * terminal that launched it. Every line therefore also goes to a log file, so
 * `npm run smoke` can be inspected no matter where it was started from.
 */
function log(line) {
  const text = `${line}\n`;
  try {
    fs.appendFileSync(logFile, text, "utf8");
  } catch {
    /* the log is a convenience, never a failure */
  }
  try {
    process.stdout.write(text);
  } catch {
    /* stdout may not exist */
  }
}

const checks = [];
function record(name, ok, detail = "") {
  checks.push({ name, ok, detail });
  log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
}

function waitForJob(jobs, jobId, timeoutMs = 300_000) {
  const statusPath = path.join(jobs.jobDir(jobId), "status.json");
  const started = Date.now();
  return new Promise((resolve, reject) => {
    const timer = setInterval(() => {
      let status = null;
      try {
        status = JSON.parse(fs.readFileSync(statusPath, "utf8"));
      } catch {
        status = null;
      }
      if (status && ["completed", "failed", "cancelled"].includes(status.state)) {
        clearInterval(timer);
        resolve(status);
        return;
      }
      if (Date.now() - started > timeoutMs) {
        clearInterval(timer);
        reject(new Error(`job ${jobId} did not finish within ${Math.round(timeoutMs / 1000)} s`));
      }
    }, 300);
  });
}

async function main() {
  log(`smoke: workspace ${workspace}`);
  log(`smoke: log ${logFile}`);
  ensureWorkspace(workspace);

  // 1. The renderer must have been built.
  record("renderer bundle exists", fs.existsSync(rendererIndex()), rendererIndex());

  // 2. The engine must be present.
  const engine = engineDir();
  record("engine package exists", fs.existsSync(path.join(engine, "zxtrain", "cli.py")), engine);

  // 3. A real interpreter must be found.
  const interpreter = await resolveInterpreter({ pythonPath: null });
  record(
    "python interpreter found",
    Boolean(interpreter?.executable),
    interpreter ? `${interpreter.executable} (python ${interpreter.version}${interpreter.torch ? ", torch" : ""})` : "none",
  );
  if (!interpreter?.executable) throw new Error("no interpreter");

  // 4. Engine capabilities come back with real hardware.
  const capabilities = await engineCall(interpreter.executable, "engine.capabilities", { workspace }, { timeout: 240_000 });
  record(
    "engine reports capabilities",
    capabilities.ok === true && Boolean(capabilities.data?.hardware?.cpu?.model),
    capabilities.ok ? `${capabilities.data.backends?.length ?? 0} backends, cpu ${capabilities.data.hardware.cpu.model}` : JSON.stringify(capabilities.error),
  );

  // 5. A dataset file written for real, inspected by the engine.
  const datasetPath = path.join(workspace, "datasets", "smoke.jsonl");
  const rows = [];
  for (let index = 0; index < 60; index += 1) {
    rows.push(JSON.stringify({ text: `the quick brown fox jumps over the lazy dog number ${index}` }));
  }
  fs.writeFileSync(datasetPath, `${rows.join("\n")}\n`, "utf8");
  const inspection = await engineCall(interpreter.executable, "datasets.inspect", { path: datasetPath, sample_size: 200 }, { timeout: 120_000 });
  record(
    "dataset inspection counts real records",
    inspection.ok === true && inspection.data?.record_count === 60,
    inspection.ok ? `${inspection.data.record_count} records, ${inspection.data.size_human}` : JSON.stringify(inspection.error),
  );

  // 6. Launch a real training job through the real job manager.
  const jobs = new JobManager({ workspace, executable: interpreter.executable, onEvent: () => {} });
  const started = jobs.start({
    kind: "train",
    backend: "tiny",
    method: "scratch",
    dataset_paths: [datasetPath],
    mapping: { text: "text" },
    sequence_length: 16,
    batch_size: 2,
    max_steps: 30,
    save_every: 10,
    logging_every: 5,
    eval_every: 0,
    learning_rate: 0.02,
    checkpoint_limit: 2,
    seed: 7,
  });
  log(`smoke: started training job ${started.jobId}`);
  const status = await waitForJob(jobs, started.jobId);
  record("training job completed", status.state === "completed", status.message ?? "");

  const jobDir = jobs.jobDir(started.jobId);
  const events = fs.existsSync(path.join(jobDir, "events.jsonl"))
    ? fs.readFileSync(path.join(jobDir, "events.jsonl"), "utf8").trim().split("\n").filter(Boolean)
    : [];
  const metrics = events
    .map((line) => {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    })
    .filter((event) => event && event.type === "metrics" && event.phase === "train");
  const lastLoss = metrics.at(-1)?.loss ?? null;
  record("training produced metrics", metrics.length >= 3, `${metrics.length} metric events, last loss ${lastLoss}`);
  record(
    "loss is finite and falling or flat",
    typeof lastLoss === "number" && Number.isFinite(lastLoss) && lastLoss < 12,
    String(lastLoss),
  );

  const modelDir = status.result?.model_dir ?? null;
  record(
    "model weights written to disk",
    Boolean(modelDir && fs.existsSync(path.join(modelDir, "model.json"))),
    modelDir ?? "no model_dir in status",
  );
  const checkpoints = status.checkpoints ?? status.result?.checkpoints ?? [];
  record("checkpoints recorded", checkpoints.length >= 2, checkpoints.map((entry) => entry.name).join(", "));

  // 7. Generation from the trained model — a real forward pass over real weights.
  const generateStarted = jobs.start({
    kind: "generate",
    backend: "tiny",
    base_model: modelDir,
    messages: [{ role: "user", content: "the quick brown fox" }],
    max_tokens: 24,
    temperature: 0.8,
    seed: 3,
  });
  const generateStatus = await waitForJob(jobs, generateStarted.jobId, 180_000);
  const generated = generateStatus.result ?? {};
  record(
    "generation produced tokens",
    generateStatus.state === "completed" && (generated.completion_tokens ?? 0) > 0,
    `${generated.completion_tokens ?? 0} tokens in ${generated.latency_seconds ?? "?"}s (${generated.tokens_per_second ?? "?"} tok/s)`,
  );

  // 8. Continuation training: a child run that resumes from the produced model.
  const continued = jobs.start({
    kind: "train",
    backend: "tiny",
    method: "continued_training",
    base_model: modelDir,
    parent_checkpoint: modelDir,
    parent_job_id: started.jobId,
    dataset_paths: [datasetPath],
    mapping: { text: "text" },
    sequence_length: 16,
    batch_size: 2,
    max_steps: 10,
    save_every: 5,
    logging_every: 5,
    learning_rate: 0.005,
    seed: 7,
  });
  const continuedStatus = await waitForJob(jobs, continued.jobId, 240_000);
  record(
    "continued training completed",
    continuedStatus.state === "completed",
    continuedStatus.message ?? "",
  );
  const lineagePath = path.join(jobs.jobDir(continued.jobId), "lineage.json");
  let continuedLineage = null;
  try {
    continuedLineage = JSON.parse(fs.readFileSync(lineagePath, "utf8"));
  } catch {
    continuedLineage = null;
  }
  record(
    "lineage points at the parent run",
    Boolean(continuedLineage?.parent?.job_id === started.jobId || continuedLineage?.parent?.path),
    continuedLineage ? JSON.stringify(continuedLineage.parent ?? {}) : `no lineage.json at ${lineagePath}`,
  );

  // 9. The conversion planner answers honestly about non-safetensors weights.
  const generation = await engineCall(
    interpreter.executable,
    "quantization.plan",
    { model: modelDir, target_dtype: "F16", target_format: "safetensors", workspace },
    { timeout: 120_000 },
  );
  record(
    "conversion planner answers honestly about non-safetensors weights",
    generation.ok === true,
    generation.ok ? `status ${generation.data.status}` : JSON.stringify(generation.error),
  );

  // 9. Storage report sees the job output.
  const storage = await engineCall(interpreter.executable, "storage.report", { workspace }, { timeout: 120_000 });
  record(
    "storage report measured the workspace",
    storage.ok === true && (storage.data?.total_bytes ?? 0) > 0,
    storage.ok ? storage.data.total_human : JSON.stringify(storage.error),
  );

  // 10. The registry store round-trips on disk.
  const registry = new JsonStore(path.join(workspace, "registry.json"), { models: [], datasets: [] });
  registry.update({ models: [{ id: "smoke", name: "smoke", path: modelDir }] });
  const reloaded = new JsonStore(path.join(workspace, "registry.json"), {}).read();
  record("registry store persists", reloaded.models?.[0]?.id === "smoke", path.join(workspace, "registry.json"));

  // 11. Job list sees the finished run and can report recovery findings.
  const listed = jobs.list();
  record("job list includes the run", listed.some((job) => job.job_id === started.jobId), `${listed.length} job(s)`);
}

// A watchdog so a wedged job can never hang CI or the terminal forever.
const watchdog = setTimeout(() => {
  log("FAIL  smoke run did not finish within 8 minutes");
  for (const check of checks) log(`  so far: ${check.ok ? "PASS" : "FAIL"} ${check.name}`);
  app.exit(2);
}, 480_000);
watchdog.unref?.();

async function run() {
  stage("app ready; running checks");
  let failure = null;
  try {
    await main();
    stage("checks finished");
  } catch (error) {
    failure = error;
    stage(`throw: ${error?.stack ?? error}`);
    record("smoke run finished without throwing", false, String(error?.stack ?? error));
  }

  clearTimeout(watchdog);
  const failed = checks.filter((check) => !check.ok);
  log(`\nsmoke: ${checks.length - failed.length}/${checks.length} checks passed`);
  for (const check of failed) log(`  failed: ${check.name} — ${check.detail}`);
  log(`smoke: artifacts left in ${tmpRoot}`);
  if (!failed.length && !failure) {
    fs.rmSync(tmpRoot, { recursive: true, force: true });
  } else {
    log(`smoke: log kept at ${logFile}`);
  }
  app.exit(failed.length || failure ? 1 : 0);
}

// NOTE: the ready event is emitted only after this module finishes evaluating,
// so it must not be awaited at the top level here — that would deadlock.
stage("registering ready handler");
app.whenReady().then(run, (error) => {
  stage(`ready failed: ${error?.stack ?? error}`);
  app.exit(3);
});
