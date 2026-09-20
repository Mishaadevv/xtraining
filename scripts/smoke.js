/**
 * End-to-end smoke test for the desktop shell.
 *
 * Runs *inside* Electron's main process so the real code paths are exercised:
 * settings persistence, interpreter discovery, the Python process manager, the
 * dataset/model registry and the training job manager. Nothing is stubbed and
 * nothing is simulated — if the ML runtime is missing, the expected outcome is a
 * clean, readable failure, and that is what this test asserts.
 *
 *   npx electron scripts/smoke.js
 *
 * The real user data directory is never touched: the app is pointed at a
 * throwaway temp directory for the duration of the run.
 */
const { app } = require("electron");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const ROOT = path.join(__dirname, "..");
const FIXTURES = path.join(ROOT, "python", "tests", "fixtures");
const SANDBOX = fs.mkdtempSync(path.join(os.tmpdir(), "zeqoux-smoke-"));

// Redirect all app state before anything reads it.
app.setPath("userData", SANDBOX);
app.commandLine.appendSwitch("disable-gpu");

const results = [];
let failures = 0;

function record(name, ok, detail) {
  results.push({ name, ok, detail });
  if (!ok) failures += 1;
  const mark = ok ? "PASS" : "FAIL";
  console.log(`  ${mark}  ${name}${detail ? ` — ${detail}` : ""}`);
}

function section(title) {
  console.log(`\n${title}`);
}

/** Poll until `fn` returns truthy or the timeout expires. */
async function waitFor(fn, { timeoutMs = 30000, intervalMs = 250 } = {}) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const value = await fn();
    if (value) return value;
    if (Date.now() > deadline) return null;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

function writeMinimalModelDir() {
  const dir = path.join(SANDBOX, "tiny-model");
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(
    path.join(dir, "config.json"),
    JSON.stringify(
      {
        model_type: "llama",
        architectures: ["LlamaForCausalLM"],
        hidden_size: 64,
        num_hidden_layers: 2,
        max_position_embeddings: 512,
        torch_dtype: "float32",
        vocab_size: 128,
      },
      null,
      2,
    ),
    "utf8",
  );
  fs.writeFileSync(path.join(dir, "tokenizer_config.json"), "{}", "utf8");
  return dir;
}

async function main() {
  const { ensureDirs, dirs, files } = require("../electron/lib/paths");
  const store = require("../electron/lib/store");
  const python = require("../electron/lib/python");
  const hardware = require("../electron/lib/hardware");
  const registry = require("../electron/lib/registry");
  const exporter = require("../electron/lib/exporter");
  const jobs = require("../electron/lib/jobs");

  ensureDirs();

  /* ------------------------------------------------------------ settings */
  section("Settings and storage");
  const settings = store.settings();
  record("settings file is created with defaults", settings.get().version === 1);
  settings.merge({ simpleMode: false });
  const reloaded = new store.JsonStore(files.settings(), {});
  record("settings round-trip to disk", reloaded.get().simpleMode === false);
  record(
    "writes are atomic (no temp files left behind)",
    fs.readdirSync(dirs.root()).filter((name) => name.endsWith(".tmp")).length === 0,
  );
  record("secret storage reports its encryption state", typeof store.encryptionAvailable() === "boolean");

  /* ------------------------------------------------------------- runtime */
  section("Python runtime and hardware");
  const health = await python.health();
  record(
    "a Python interpreter was found",
    health.python.available === true,
    health.python.available ? `${health.python.version} (${health.python.executable})` : health.python.reason,
  );
  const discovered = await python.discover(true);
  record(
    "interpreter discovery returns a ranked list",
    Array.isArray(discovered) && discovered.length > 0,
    `${discovered.filter((entry) => entry.available).length} usable of ${discovered.length} candidates`,
  );

  const system = hardware.systemSnapshot();
  record(
    "cpu and memory are detected",
    Boolean(system.cpu.model) && system.memory.total_mb > 0,
    `${system.cpu.model} · ${Math.round(system.memory.total_mb / 1024)} GB RAM`,
  );

  const sample = await hardware.sample();
  record(
    sample.available ? "nvidia-smi reports a GPU" : "absence of an NVIDIA GPU is reported, not hidden",
    typeof sample.available === "boolean" && (sample.available || Boolean(sample.reason)),
    sample.available ? sample.gpu.name : sample.reason,
  );

  const envCheck = await python.run(["env-check"]);
  record(
    "env-check answers over the JSON protocol",
    envCheck.ok === true && envCheck.result && envCheck.result.dependencies,
    `torch installed: ${envCheck.result?.dependencies?.packages?.torch?.installed}`,
  );

  const backends = await python.run(["backends"]);
  const hfPeft = (backends.result?.backends || []).find((entry) => entry.name === "hf-peft");
  record(
    "the hf-peft training backend is registered",
    Boolean(hfPeft),
    hfPeft ? `available: ${hfPeft.available} (missing: ${(hfPeft.missing || []).join(", ") || "none"})` : "not found",
  );

  /* ------------------------------------------------------------ datasets */
  section("Dataset import and validation");
  const good = await registry.importDataset(path.join(FIXTURES, "good.jsonl"));
  record("a JSONL dataset imports", good.ok === true && Boolean(good.dataset));

  const goodReport = await registry.validateDataset(good.dataset.id);
  record(
    "the JSONL dataset validates as clean",
    goodReport.ok === true && goodReport.report.status === "ok",
    `${goodReport.report?.dataset?.records} records · ${goodReport.report?.stats?.usable} usable · avg ${goodReport.report?.stats?.avg_chars} chars`,
  );
  record(
    "the pair mapping is detected automatically",
    goodReport.report?.mapping?.kind === "pair" && goodReport.report.mapping.output_field === "output",
    `${goodReport.report?.mapping?.kind} · ${goodReport.report?.mapping?.instruction_field} → ${goodReport.report?.mapping?.output_field}`,
  );
  record(
    "the persisted dataset row is updated with the report",
    registry.listDatasets()[0].status === "ok" && registry.listDatasets()[0].usable > 0,
  );

  const preview = await registry.previewDataset(good.dataset.id, 3);
  record(
    "normalised samples can be previewed",
    preview.ok === true && Array.isArray(preview.samples) && preview.samples.length === 3,
    preview.samples?.[0] ? `${JSON.stringify(preview.samples[0]).slice(0, 70)}…` : "no samples",
  );

  const broken = await registry.importDataset(path.join(FIXTURES, "broken.json"));
  const brokenReport = await registry.validateDataset(broken.dataset.id);
  const errorIssues = (brokenReport.report?.issues || []).filter((issue) => issue.severity === "error");
  record(
    "a malformed JSON file is rejected with a precise error",
    brokenReport.ok === true && brokenReport.report.status === "errors" && errorIssues.length > 0,
    errorIssues[0] ? `${errorIssues[0].code}: ${errorIssues[0].message}` : "no error issue reported",
  );
  record(
    "the error is actionable, not a traceback",
    Boolean(errorIssues[0]?.hint) && !/Traceback/.test(JSON.stringify(errorIssues)),
    errorIssues[0]?.hint,
  );

  const folder = await registry.importDataset(path.join(FIXTURES, "folder_clean"));
  const folderReport = await registry.validateDataset(folder.dataset.id);
  record(
    "a folder of shards imports and validates",
    folderReport.ok === true && folderReport.report?.dataset?.files?.length === 2,
    `${folderReport.report?.dataset?.records} records from ${folderReport.report?.dataset?.files?.length} files`,
  );

  record("a duplicate import refreshes instead of duplicating", (await registry.importDataset(path.join(FIXTURES, "good.jsonl"))).refreshed === true);

  /* ------------------------------------------------------ built-in datasets */
  section("Built-in datasets");
  const bundled = registry.listBuiltinDatasets();
  record(
    "bundled datasets ship with the app",
    bundled.length >= 20,
    `${bundled.length} files in ${path.basename(dirs.datasets())}`,
  );
  record("every bundled dataset file exists on disk", bundled.every((item) => fs.existsSync(item.path)));
  const defaultBundled = bundled.find((item) => item.default);
  record("one bundled dataset is marked as the default", Boolean(defaultBundled), defaultBundled?.name);

  if (defaultBundled) {
    const bundledReport = await registry.validateDataset(defaultBundled.id);
    record(
      "the default bundled dataset validates through the backend",
      bundledReport.ok === true && Boolean(bundledReport.report) && bundledReport.report.status !== "errors",
      `${bundledReport.report?.dataset?.records} records · mapping ${bundledReport.report?.mapping?.kind}`,
    );
    const bundledRemove = registry.removeDataset(defaultBundled.id);
    record(
      "bundled datasets cannot be removed",
      bundledRemove.ok === false && bundledRemove.error?.code === "builtin",
      bundledRemove.error?.message,
    );
  }

  const missing = await registry.importDataset(path.join(SANDBOX, "does-not-exist.jsonl"));
  record("a missing path fails with a clear message", missing.ok === false && missing.error.code === "not_found", missing.error?.message);

  /* ------------------------------------------------- Hugging Face datasets */
  section("Hugging Face datasets");
  const badId = registry.addHfDataset("not a dataset id");
  record("a malformed Hub id is refused", badId.ok === false && badId.error.code === "bad_id", badId.error?.message);
  record("an empty Hub id is refused", registry.addHfDataset("  ").ok === false);

  const hub = registry.addHfDataset("tatsu-lab/alpaca", "train");
  record(
    "a Hub dataset is registered by id without downloading",
    hub.ok === true && hub.dataset.format === "hf" && hub.dataset.sizeBytes === null,
    hub.ok ? hub.dataset.name : hub.error?.message,
  );
  record(
    "re-adding the same Hub dataset refreshes instead of duplicating",
    registry.addHfDataset("tatsu-lab/alpaca").refreshed === true,
  );
  record(
    "a full Hub URL is accepted",
    registry.addHfDataset("https://huggingface.co/datasets/openai/gsm8k", "test").ok === true,
  );

  const hubReport = await registry.validateDataset(hub.dataset.id);
  const hubIssue = hubReport.report?.issues?.[0];
  record(
    "a Hub dataset is validated against the real backend, not assumed clean",
    hubReport.ok === true && hubReport.report.ok === false && hubReport.report.dataset.format === "hf",
    `${hubReport.report?.dataset?.path} · ${hubReport.report?.status}`,
  );
  record(
    "the missing Hub dependency is named with an install hint",
    Boolean(hubIssue && hubIssue.code === "missing_dependency" && /pip install/.test(hubIssue.hint || "")),
    hubIssue ? `${hubIssue.code}: ${hubIssue.message}` : "no issue reported",
  );
  record(
    "a Hub dataset never claims a local byte size",
    hubReport.report?.dataset?.bytes === null,
  );
  record(
    "Hub and local datasets share one list",
    registry.listDatasets().some((item) => item.format === "hf") &&
      registry.listDatasets().some((item) => item.format === "jsonl"),
  );

  /* -------------------------------------------------------------- models */
  section("Model inspection");
  const modelDir = writeMinimalModelDir();
  const model = await registry.addModel(modelDir);
  record(
    "a local model folder is inspected",
    model.ok === true && Boolean(model.model),
    model.ok ? `${model.model.architecture} · ${model.model.params} params (exact: ${model.model.paramsExact})` : model.error?.message,
  );
  // A Hub id that cannot be inspected yet (no huggingface_hub, no network) must
  // be recorded as unverified rather than presented as ready to train.
  const hfModel = await registry.addModel("not-a-real-org/definitely-not-a-real-model");
  const hubWarning = (hfModel.model?.issues || []).find((issue) => issue.code === "hub_missing");
  record(
    "an uninspectable Hub id is recorded as unverified, not as ready",
    hfModel.ok === true && hfModel.model.trainable === false && Boolean(hubWarning),
    hubWarning ? hubWarning.message : "no warning attached",
  );
  const noSource = await registry.addModel("");
  record("an empty model source is refused", noSource.ok === false, noSource.error?.message);

  /* -------------------------------------------------------------- export */
  section("Model export");
  const { runs: runsStore } = store;
  const runDir = path.join(SANDBOX, "runs", "export-demo");
  fs.mkdirSync(path.join(runDir, "checkpoint-20"), { recursive: true });
  fs.writeFileSync(
    path.join(runDir, "adapter_config.json"),
    JSON.stringify({ base_model_name_or_path: "meta-llama/Llama-3.2-1B", r: 16, lora_alpha: 32, lora_dropout: 0.05 }),
    "utf8",
  );
  fs.writeFileSync(path.join(runDir, "adapter_model.safetensors"), Buffer.alloc(4096));
  fs.writeFileSync(path.join(runDir, "tokenizer_config.json"), "{}", "utf8");

  const runRecord = {
    id: "export-demo",
    projectId: null,
    name: "export-demo",
    method: "lora",
    baseModel: "meta-llama/Llama-3.2-1B",
    datasetName: "good.jsonl",
    datasetPath: path.join(FIXTURES, "good.jsonl").replace(/\\/g, "/"),
    status: "completed",
    phase: "done",
    progress: 100,
    totalSteps: 240,
    finalLoss: 0.87,
    startedAt: Date.now() - 60000,
    finishedAt: Date.now(),
    runDir: runDir.replace(/\\/g, "/"),
    outputDir: runDir.replace(/\\/g, "/"),
    config: { method: "lora", base_model: "meta-llama/Llama-3.2-1B", epochs: 1 },
  };
  runsStore().replace([runRecord]);
  const trained = registry.registerTrainedModel(runRecord);
  record("a finished run registers as a trained model", Boolean(trained && trained.trained));

  const described = await exporter.describe(trained.id);
  record(
    "an artefact folder is described before exporting",
    described.ok === true && described.info.is_adapter === true,
    described.ok ? `${described.info.files.length} files · base ${described.info.base_model}` : described.error?.message,
  );
  record(
    "copy mode is always offered",
    described.info?.modes.copy.ready === true,
  );
  record(
    "merge is offered but honestly marked unavailable without the runtime",
    described.info?.modes.merge.applicable === true && described.info.modes.merge.ready === false,
    described.info ? `missing: ${described.info.modes.merge.missing.join(", ")}` : undefined,
  );

  const exportTarget = path.join(SANDBOX, "exported");
  const copied = await exporter.exportModel({ modelId: trained.id, outputDir: exportTarget });
  record("a copy export succeeds", copied.ok === true && copied.mode === "copy", copied.ok ? copied.output_dir : copied.error?.message);
  record("weights and config are written", fs.existsSync(path.join(exportTarget, "adapter_model.safetensors")) && fs.existsSync(path.join(exportTarget, "adapter_config.json")));
  record("a README is written for the user", fs.existsSync(path.join(exportTarget, "README.md")));

  const manifest = JSON.parse(fs.readFileSync(path.join(exportTarget, "export.json"), "utf8"));
  record(
    "the manifest carries real provenance from run history",
    manifest.run.runId === "export-demo" && manifest.run.finalLoss === 0.87 && manifest.run.datasetName === "good.jsonl",
    JSON.stringify(manifest.run).slice(0, 120),
  );
  const readme = fs.readFileSync(path.join(exportTarget, "README.md"), "utf8");
  record(
    "the README explains how to load the weights",
    readme.includes("PeftModel") && readme.includes("meta-llama/Llama-3.2-1B") && readme.includes("0.87"),
  );
  record("checkpoints are not duplicated into the export", !fs.existsSync(path.join(exportTarget, "checkpoint-20")));

  const refused = await exporter.exportModel({ modelId: trained.id, outputDir: exportTarget });
  record(
    "exporting over an existing folder is refused, not silently merged",
    refused.ok === false && refused.error.code === "not_empty",
    refused.error?.message,
  );

  const mergeBlocked = await exporter.exportModel({
    modelId: trained.id,
    outputDir: path.join(SANDBOX, "merged"),
    merge: true,
  });
  record(
    "merging without the ML runtime is refused with the missing packages named",
    mergeBlocked.ok === false && mergeBlocked.error.code === "missing_dependency",
    mergeBlocked.error?.message,
  );
  record("the refused merge left no folder behind", !fs.existsSync(path.join(SANDBOX, "merged")));

  const noFolder = await exporter.exportModel({ modelId: trained.id, outputDir: "" });
  record("exporting without a destination is refused", noFolder.ok === false && noFolder.error.code === "no_output");

  /* ------------------------------------------------------------ training */
  section("Training job lifecycle");
  const config = {
    method: "lora",
    base_model: modelDir.replace(/\\/g, "/"),
    output_name: "smoke-run",
    epochs: 1,
    batch_size: 1,
    gradient_accumulation: 1,
    learning_rate: 0.0002,
    context_length: 128,
    lora_r: 8,
    lora_alpha: 16,
    quantization: "none",
    precision: "auto",
    save_steps: 1,
    logging_steps: 1,
    dataset: { path: path.join(FIXTURES, "good.jsonl").replace(/\\/g, "/"), name: "good.jsonl", format: "jsonl" },
  };

  const started = await jobs.startRun({ name: "smoke-run", config, hardware: system });
  record("a run is accepted and a run directory is allocated", started.ok === true && Boolean(started.runId), started.ok ? started.runId : started.error?.message);

  if (started.ok) {
    const finished = await waitFor(
      () => {
        const run = jobs.getRun(started.runId);
        return run && !["starting", "running"].includes(run.status) ? run : null;
      },
      { timeoutMs: 120000 },
    );
    record("the run reaches a terminal state", Boolean(finished), finished ? `status: ${finished.status} · phase: ${finished.phase}` : "timed out");

    if (finished) {
      const runtimeMissing =
        finished.error && ["missing_dependency", "no_python", "backend_unavailable"].includes(finished.error.code);
      if (runtimeMissing) {
        record(
          "a missing ML runtime fails cleanly with a readable message",
          Boolean(finished.error.hint || finished.error.message) && !/Traceback \(most recent call last\)/.test(finished.error.message),
          `${finished.error.code}: ${finished.error.message}`.slice(0, 130),
        );
      } else if (finished.status === "failed") {
        record("the run failed with a translated error", Boolean(finished.error), `${finished.error?.code}: ${finished.error?.message}`.slice(0, 130));
      } else {
        record(
          "a real training run completed",
          finished.finalLoss !== null,
          `final loss ${finished.finalLoss} · ${finished.totalSteps} steps · ${finished.device}`,
        );
      }

      record("run metadata is written to the run directory", fs.existsSync(path.join(finished.runDir, "job.json")));
      record("the run is persisted to history", jobs.listRuns().some((run) => run.id === finished.id));

      const artifactError = await jobs.listCheckpoints(path.join(SANDBOX, "missing-run"));
      record(
        "checkpoint listing tolerates an unknown run directory",
        artifactError.ok === true && Array.isArray(artifactError.checkpoints) && artifactError.checkpoints.length === 0,
      );

      const deleted = await jobs.deleteRun(started.runId);
      record("a run can be removed from history", deleted.ok === true && !jobs.listRuns().some((run) => run.id === started.runId));
    }
  }

  const log = jobs.readLog("nonexistent-run");
  record(
    "reading the log of an unknown run is empty rather than a crash",
    Array.isArray(log.events) && log.events.length === 0,
  );

  /* ------------------------------------------------------------- summary */
  const passed = results.length - failures;
  console.log("\n" + "=".repeat(62));
  console.log(`passed: ${passed}   failed: ${failures}`);
  console.log(failures === 0 ? "smoke test passed" : "smoke test FAILED");
  return failures;
}

app.whenReady().then(async () => {
  let code = 1;
  try {
    code = await main();
  } catch (error) {
    console.error("\nsmoke test crashed:", error && error.stack ? error.stack : error);
    code = 1;
  } finally {
    try {
      fs.rmSync(SANDBOX, { recursive: true, force: true });
    } catch {
      /* temp dir, best effort */
    }
    app.exit(code);
  }
});
