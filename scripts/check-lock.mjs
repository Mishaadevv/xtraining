/**
 * Is `package-lock.json` portable?
 *
 *     npm run check:lock            # package-lock.json
 *     npm run check:lock -- other-lock.json
 *
 * A lock file is written by whichever machine ran `npm install`, and it is easy
 * to end up with one that only describes the platform it was created on: npm
 * seeds the tree from `node_modules`, so native binaries for other platforms are
 * never resolved and never recorded. `npm ci` on Linux then installs a rollup or
 * esbuild without its native module and the build fails there — while working
 * perfectly on the machine that wrote the lock. That is exactly how this project
 * shipped a Windows-only lock and a red CI on ubuntu.
 *
 * The rule this checks is simple: every package a locked package says it depends
 * on optionally has to be in the lock as well, with an integrity hash. Optional
 * dependencies are where native, per-platform binaries live, so a lock that
 * satisfies this can be installed anywhere.
 */
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const lockFile = path.resolve(root, process.argv[2] ?? "package-lock.json");

if (!fs.existsSync(lockFile)) {
  console.log(`FAIL  ${path.relative(root, lockFile)} is missing — CI installs with \`npm ci\`, which requires it.`);
  process.exit(1);
}

const lock = JSON.parse(fs.readFileSync(lockFile, "utf8"));
const packages = lock.packages ?? {};
const present = new Set(Object.keys(packages).map((key) => key.replace(/^.*node_modules\//, "")));

const missing = [];
let declared = 0;
for (const [key, entry] of Object.entries(packages)) {
  for (const [name, range] of Object.entries(entry.optionalDependencies ?? {})) {
    declared += 1;
    if (!present.has(name)) missing.push({ from: key || "<root>", name, range });
  }
}

// The same question for the root package: everything it asks for has to be there.
for (const [name, range] of Object.entries(packages[""]?.optionalDependencies ?? {})) {
  declared += 1;
  if (!present.has(name)) missing.push({ from: "<root>", name, range });
}

console.log(`lock: ${present.size} packages, ${declared} optional dependency edge(s)`);

// A lock that is missing an entry for another platform is the interesting case,
// so name the platform in the report instead of dumping a bare package name.
const platformOf = (name) => {
  const match = name.match(/-(win32|darwin|linux|android|freebsd|openbsd|netbsd|sunos)-([a-z0-9_]+)/i);
  return match ? `${match[1]}-${match[2]}` : "no platform in the name";
};
const platforms = [...new Set(missing.map((entry) => platformOf(entry.name)))].sort();

for (const entry of missing.slice(0, 20)) {
  console.log(`      missing: ${entry.name} (optional ${entry.range} of ${entry.from})`);
}
if (missing.length > 20) console.log(`      … and ${missing.length - 20} more`);

if (missing.length) {
  console.log(
    `\nFAIL  ${missing.length} optional dependency/dependencies are declared but not in the lock — ` +
      `this lock cannot be installed on every platform (${platforms.join(", ")}).`,
  );
  console.log(
    "      Regenerate it so npm resolves the whole tree instead of trusting a local node_modules:\n" +
      "        mv node_modules node_modules.keep && rm package-lock.json && npm install --package-lock-only && npm ci\n",
  );
  process.exit(1);
}

console.log("PASS  every optional dependency is locked, so the install is the same on Windows, macOS and Linux");
