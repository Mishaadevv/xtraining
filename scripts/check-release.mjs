/**
 * Does every release manifest name a file the release actually carries?
 *
 *     npm run check:release            # checks release/
 *     npm run check:release -- artifacts
 *
 * `latest.yml` and its macOS and Linux counterparts are the whole update channel:
 * an installed copy reads the manifest, and downloads the file its `url:` names.
 * electron-builder writes those names itself, and it writes *safe* names — no
 * spaces — because GitHub rejects them: the installer on disk is
 * "ZeqouXTraining Setup 2.0.0.exe" while the manifest asks for
 * "ZeqouXTraining-Setup-2.0.0.exe". Uploaded by a script rather than by
 * electron-builder's own publisher, the asset keeps the spaced name (GitHub turns
 * the spaces into dots), so the manifest points at a file that is not there and
 * every update 404s — silently, because nothing else reads the manifest.
 *
 * This checks the pairing, which is the only thing that makes the channel work.
 */
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const directory = path.resolve(root, process.argv[2] ?? "release");

if (!fs.existsSync(directory)) {
  console.log(`FAIL  ${path.relative(root, directory)} does not exist — build first, or pass the directory to check.`);
  process.exit(2);
}

/** Every file below a directory, by base name. */
function collect(dir) {
  const found = new Map();
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      for (const [name, file] of collect(full)) found.set(name, file);
    } else {
      found.set(entry.name, full);
    }
  }
  return found;
}

const files = collect(directory);
const manifests = [...files.keys()].filter((name) => /^latest(-mac|-linux)?\.yml$/.test(name));

if (!manifests.length) {
  console.log(`FAIL  no latest.yml in ${path.relative(root, directory)} — there is no update channel to check.`);
  process.exit(1);
}

let failed = 0;
for (const manifest of manifests.sort()) {
  const text = fs.readFileSync(files.get(manifest), "utf8");
  // Only the file names matter here, and they always sit on a `url:` or `path:` line.
  const referenced = [...text.matchAll(/^\s*(?:url|path):\s*(\S+)\s*$/gm)].map((match) => match[1]);
  const version = /\bversion:\s*(\S+)/.exec(text)?.[1] ?? "?";
  const missing = [...new Set(referenced)].filter((name) => !files.has(name));

  console.log(`${missing.length ? "FAIL" : "PASS"}  ${manifest} — version ${version}, ${referenced.length} reference(s)`);
  for (const name of missing) {
    const closest = [...files.keys()].find((candidate) => candidate.replace(/[.\s]+/g, "-") === name.replace(/[.\s]+/g, "-"));
    console.log(`      names ${name}, which the release does not carry`);
    if (closest) console.log(`      the file here is called ${closest} — an installed app would download nothing`);
  }
  failed += missing.length;
}

if (failed) {
  console.log(
    `\nFAIL  ${failed} reference(s) point at files that do not exist. The update channel would 404.\n` +
      "      Give the target an `artifactName` without spaces so the manifest and the file agree:\n" +
      '        "nsis": { "artifactName": "${productName}-Setup-${version}.${ext}" }\n',
  );
  process.exit(1);
}

console.log(`\nPASS  every manifest names a file that is here (${manifests.length} manifest(s), ${files.size} file(s))`);
