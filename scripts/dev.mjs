/**
 * Dev launcher: starts the Vite dev server, then boots Electron pointed at it.
 * Keeps the two processes in sync and shuts both down together.
 * No extra dependencies — Vite's Node API and Electron's CLI are enough.
 */
import { createServer } from "vite";
import { spawn } from "node:child_process";
import process from "node:process";

const server = await createServer({ configFile: "vite.config.ts" });
await server.listen();
const info = server.resolvedUrls?.local?.[0] ?? `http://localhost:5273/`;
server.printUrls();

console.log(`\n[dev] ZeqouXTraining renderer at ${info}\n`);

const child = spawn(process.platform === "win32" ? "npx.cmd" : "npx", ["electron", "."], {
  stdio: "inherit",
  env: { ...process.env, ZEQOUX_DEV_SERVER_URL: info, ZEQOUX_DEV: "1" },
});

const shutdown = async (code = 0) => {
  await server.close().catch(() => {});
  process.exit(code);
};

child.on("exit", (code) => void shutdown(code ?? 0));
process.on("SIGINT", () => {
  child.kill();
  void shutdown(0);
});
