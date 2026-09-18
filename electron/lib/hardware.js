/**
 * Hardware probing from the Node side.
 *
 * Live monitoring must not depend on the Python runtime: `nvidia-smi` is a
 * standalone binary, so sampling GPU utilisation and VRAM costs one short-lived
 * process per tick and works even when torch is missing and no training is
 * possible.
 *
 * The authoritative "can PyTorch actually use CUDA" answer still comes from the
 * Python backend, because only torch knows which CUDA version its wheel was
 * built against.
 */
const { execFile } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");

const SMI_FIELDS = [
  "index",
  "name",
  "memory.total",
  "memory.used",
  "memory.free",
  "utilization.gpu",
  "utilization.memory",
  "temperature.gpu",
  "power.draw",
  "power.limit",
  "driver_version",
  "compute_cap",
];

const SMI_TIMEOUT_MS = 8000;

function nvidiaSmiPath() {
  const candidates = [
    "nvidia-smi",
    process.env.ProgramFiles && `${process.env.ProgramFiles}\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe`,
    "C:\\Windows\\System32\\nvidia-smi.exe",
    "/usr/bin/nvidia-smi",
    "/usr/local/bin/nvidia-smi",
  ].filter(Boolean);

  for (const candidate of candidates) {
    if (!candidate.includes("\\") && !candidate.includes("/")) return candidate; // rely on PATH
    try {
      if (fs.existsSync(candidate)) return candidate;
    } catch {
      /* keep looking */
    }
  }
  return null;
}

function toNumber(value) {
  const cleaned = String(value ?? "").trim();
  if (!cleaned || /^(n\/a|\[n\/a\]|unknown)$/i.test(cleaned)) return null;
  const parsed = Number.parseFloat(cleaned);
  return Number.isFinite(parsed) ? parsed : null;
}

function runSmi(binary, args, timeout = SMI_TIMEOUT_MS) {
  return new Promise((resolve) => {
    execFile(binary, args, { timeout, windowsHide: true }, (error, stdout, stderr) => {
      if (error) {
        resolve({ ok: false, reason: (stderr || error.message || "").trim() || "nvidia-smi failed" });
        return;
      }
      resolve({ ok: true, stdout: stdout || "" });
    });
  });
}

/** One-shot query of every NVIDIA GPU. */
async function queryGpus() {
  const binary = nvidiaSmiPath();
  if (!binary) {
    return {
      available: false,
      reason:
        "nvidia-smi was not found. Either there is no NVIDIA GPU or the NVIDIA driver is not installed.",
      gpus: [],
    };
  }

  const result = await runSmi(binary, [
    `--query-gpu=${SMI_FIELDS.join(",")}`,
    "--format=csv,noheader,nounits",
  ]);
  if (!result.ok) {
    return { available: false, reason: result.reason, gpus: [] };
  }

  const gpus = [];
  for (const line of result.stdout.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const parts = line.split(",").map((part) => part.trim());
    if (parts.length < SMI_FIELDS.length) continue;
    gpus.push({
      index: toNumber(parts[0]) ?? 0,
      name: parts[1],
      memory_total_mb: toNumber(parts[2]),
      memory_used_mb: toNumber(parts[3]),
      memory_free_mb: toNumber(parts[4]),
      utilization_gpu: toNumber(parts[5]),
      utilization_memory: toNumber(parts[6]),
      temperature_c: toNumber(parts[7]),
      power_draw_w: toNumber(parts[8]),
      power_limit_w: toNumber(parts[9]),
      driver_version: parts[10] || null,
      compute_capability: parts[11] || null,
    });
  }

  if (!gpus.length) {
    return { available: false, reason: "nvidia-smi returned no devices.", gpus: [] };
  }
  return { available: true, binary, gpus, sampled_at: Date.now() };
}

/** Snapshots used by the live monitor; cheap enough to call once a second. */
async function sample() {
  const gpus = await queryGpus();
  const primary = gpus.gpus[0] || null;
  return {
    available: gpus.available,
    reason: gpus.reason || null,
    gpu: primary,
    gpus: gpus.gpus,
    sampled_at: Date.now(),
  };
}

function cpuSnapshot() {
  const cpus = os.cpus() || [];
  return {
    model: cpus[0] ? cpus[0].model.trim() : "Unknown CPU",
    logical_cores: cpus.length,
    load_average: os.loadavg ? os.loadavg().map((v) => Math.round(v * 100) / 100) : null,
    architecture: os.arch(),
  };
}

function memorySnapshot() {
  return {
    total_mb: Math.round(os.totalmem() / (1024 * 1024)),
    free_mb: Math.round(os.freemem() / (1024 * 1024)),
  };
}

function systemSnapshot() {
  return {
    platform: process.platform,
    release: os.release(),
    version: os.version ? os.version() : null,
    hostname: os.hostname(),
    cpu: cpuSnapshot(),
    memory: memorySnapshot(),
    uptime_seconds: Math.round(os.uptime()),
    electron: process.versions.electron,
    node: process.versions.node,
  };
}

module.exports = { nvidiaSmiPath, queryGpus, sample, systemSnapshot };
