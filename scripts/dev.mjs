/**
 * Development launcher: starts Vite, waits for it to answer, then starts
 * Electron pointed at that URL. Ctrl+C stops both.
 *
 * Nothing here is required by the app itself — `npm run build && electron .`
 * runs the packaged path. This script only exists so the renderer can use HMR
 * while the Python engine stays a real subprocess.
 */
import { spawn } from "node:child_process";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const port = Number(process.env.ZEQOUX_DEV_PORT ?? 5273);
const url = `http://127.0.0.1:${port}`;

const viteBin = path.join(root, "node_modules", "vite", "bin", "vite.js");
const electronBin = path.join(
  root,
  "node_modules",
  "electron",
  "cli.js",
);

let vite;
let electron;
let stopping = false;

function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  try {
    electron?.kill();
  } catch {
    /* already gone */
  }
  try {
    vite?.kill();
  } catch {
    /* already gone */
  }
  process.exit(code);
}

async function waitForServer(target, timeoutMs = 60_000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try {
      const response = await fetch(target, { method: "GET" });
      if (response.ok || response.status === 404) return true;
    } catch {
      /* not up yet */
    }
    await new Promise((resolve) => setTimeout(resolve, 400));
  }
  return false;
}

vite = spawn(process.execPath, [viteBin, "--port", String(port), "--strictPort"], {
  cwd: root,
  stdio: ["ignore", "pipe", "pipe"],
  env: process.env,
});

vite.stdout.setEncoding("utf8");
vite.stderr.setEncoding("utf8");
vite.stdout.on("data", (chunk) => process.stdout.write(`[vite] ${chunk}`));
vite.stderr.on("data", (chunk) => process.stderr.write(`[vite] ${chunk}`));
vite.on("close", (code) => {
  if (!stopping) {
    console.error(`[dev] Vite exited with code ${code}.`);
    stop(code ?? 1);
  }
});

const ready = await waitForServer(url);
if (!ready) {
  console.error(`[dev] The dev server did not answer at ${url} within 60 s.`);
  stop(1);
}

console.log(`[dev] Renderer ready at ${url}`);
console.log("[dev] Starting Electron. Renderer changes hot-reload; main-process changes need a restart.");

electron = spawn(process.execPath, [electronBin, root], {
  cwd: root,
  stdio: "inherit",
  env: { ...process.env, ZEQOU_DEV_SERVER_URL: url },
});

electron.on("close", (code) => {
  console.log(`[dev] Electron exited with code ${code}.`);
  stop(code ?? 0);
});

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    console.log(`[dev] ${signal} received — shutting down.`);
    stop(0);
  });
}
