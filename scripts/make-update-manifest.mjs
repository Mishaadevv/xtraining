/**
 * Build an update channel directory from a built installer.
 *
 *     node scripts/make-update-manifest.mjs \
 *       --installer "release/ZeqouXTraining Setup 2.0.0.exe" \
 *       --version 2.0.1 --out release/channel
 *
 * A generic channel is a `latest.yml` plus the file it names, and `electron-updater`
 * verifies the download against the `sha512` in that manifest. This writes both:
 * the installer is copied under the URL-safe name the manifest refers to (spaces
 * would otherwise have to survive URL encoding), hashed while it is copied, and
 * `releaseNotes` can be attached with `--notes`.
 *
 * electron-builder already writes a `latest.yml` for the builds it publishes; this
 * is for hosting a build somewhere else, and for testing the update path against a
 * channel whose version differs from the installed one.
 */
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

export const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function argOf(args, name, fallback = null) {
  const index = args.indexOf(`--${name}`);
  if (index === -1) return fallback;
  const value = args[index + 1];
  return value && !value.startsWith("--") ? value : true;
}

export function productName() {
  try {
    return JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8")).productName ?? "app";
  } catch {
    return "app";
  }
}

/** `ZeqouXTraining Setup 2.0.1.exe` — the name electron-builder puts in a manifest. */
export function safeArtifactName(version, extension = "exe") {
  return `${productName().replace(/\s+/g, "-")}-Setup-${version}.${extension}`;
}

/** Hash a file without holding all of it in memory. */
export async function sha512(file) {
  const hash = crypto.createHash("sha512");
  await new Promise((resolve, reject) => {
    const stream = fs.createReadStream(file);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("error", reject);
    stream.on("end", resolve);
  });
  return hash.digest("base64");
}

/**
 * Copy `installer` into `dir` as `latest.yml`'s payload and write the manifest.
 * Returns the manifest that was written.
 */
export async function writeChannel({ dir, installer, version, notes = null, extension = "exe" }) {
  if (!fs.existsSync(installer)) throw new Error(`no installer at ${installer}`);
  fs.mkdirSync(dir, { recursive: true });

  const name = safeArtifactName(version, extension);
  const target = path.join(dir, name);
  fs.copyFileSync(installer, target);

  const digest = await sha512(target);
  const size = fs.statSync(target).size;
  const manifest = {
    version,
    files: [{ url: name, sha512: digest, size }],
    path: name,
    sha512: digest,
    releaseDate: new Date().toISOString(),
  };

  const lines = [
    `version: ${manifest.version}`,
    "files:",
    `  - url: ${name}`,
    `    sha512: ${digest}`,
    `    size: ${size}`,
    `path: ${name}`,
    `sha512: ${digest}`,
    `releaseDate: '${manifest.releaseDate}'`,
  ];
  if (notes) {
    lines.push("releaseNotes: |-");
    for (const line of String(notes).split(/\r?\n/)) lines.push(`  ${line}`);
  }
  fs.writeFileSync(path.join(dir, "latest.yml"), `${lines.join("\n")}\n`, "utf8");
  return { ...manifest, dir, file: target, name };
}

const invokedDirectly = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (invokedDirectly) {
  const args = process.argv.slice(2);
  const installer = argOf(args, "installer");
  const version = argOf(args, "version");
  if (!installer || !version) {
    console.error("usage: node scripts/make-update-manifest.mjs --installer <file.exe> --version <x.y.z> [--out dir] [--notes text]");
    process.exit(2);
  }
  const out = path.resolve(String(argOf(args, "out", path.join(root, "release", "channel"))));
  const notes = argOf(args, "notes", null);

  try {
    const channel = await writeChannel({
      dir: out,
      installer: path.resolve(String(installer)),
      version: String(version),
      notes: notes ? String(notes) : null,
    });
    const appVersion = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8")).version;
    console.log(`channel written to ${channel.dir}`);
    console.log(`  latest.yml     version ${channel.version}`);
    console.log(`  ${channel.name}  ${(channel.files[0].size / 1024 / 1024).toFixed(1)} MiB`);
    console.log(`  sha512         ${channel.sha512}`);
    if (String(version) !== appVersion) {
      console.log("");
      console.log(`  note: the copied file is the ${appVersion} build wearing version ${version}. The updater will`);
      console.log("        detect, download and verify it, but installing it would put the older build back.");
      console.log("        Use this for testing the channel; publish real releases with `npm run dist`.");
    }
    console.log("");
    console.log(`  serve it:  node scripts/update-server.mjs "${channel.dir}"`);
  } catch (error) {
    console.error(`Could not build the channel: ${error.message}`);
    process.exit(1);
  }
}
