/**
 * Syntax check for the Electron main process.
 *
 * The renderer is type-checked by TypeScript, but everything under `electron/`
 * is plain JavaScript that no compiler looks at. A single unbalanced template
 * literal there reaches a build and only shows itself as "App threw an error
 * during load" in the packaged app — so the same `node --check` the runtime
 * would do is run here, over every file.
 *
 *   node scripts/check-js.mjs
 */
import { spawnSync } from "node:child_process";
import { readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const root = join(fileURLToPath(new URL(".", import.meta.url)), "..");
const roots = ["electron", "scripts"];
const skip = new Set(["node_modules", "dist", "release", ".git"]);

function walk(dir) {
  const found = [];
  let entries = [];
  try {
    entries = readdirSync(dir, { withFileTypes: true });
  } catch {
    return found;
  }
  for (const entry of entries) {
    if (skip.has(entry.name)) continue;
    const full = join(dir, entry.name);
    if (entry.isDirectory()) found.push(...walk(full));
    else if (/\.(m?js)$/.test(entry.name)) found.push(full);
  }
  return found;
}

const files = roots.flatMap((name) => {
  const dir = join(root, name);
  try {
    return statSync(dir).isDirectory() ? walk(dir) : [dir];
  } catch {
    return [];
  }
});

const failures = [];
for (const file of files) {
  const result = spawnSync(process.execPath, ["--check", file], { encoding: "utf8" });
  const label = relative(root, file).replace(/\\/g, "/");
  if (result.status === 0) {
    console.log(`  ok    ${label}`);
  } else {
    failures.push(label);
    console.log(`  FAIL  ${label}`);
    console.log(String(result.stderr || "").split("\n").slice(0, 6).join("\n"));
  }
}

console.log(`\n${files.length - failures.length}/${files.length} files parse`);
if (failures.length) {
  console.log(`\nnot valid JavaScript:\n  ${failures.join("\n  ")}`);
  process.exit(1);
}
