#!/usr/bin/env node
/**
 * Build the installers — the one entry point for packing.
 *
 *     node scripts/dist.mjs             # typecheck, build the renderer, installers
 *     node scripts/dist.mjs --dir       # unpacked build only (no installers)
 *     node scripts/dist.mjs --sign      # sign with certs/dev-codesign.pfx
 *     node scripts/dist.mjs --no-build  # skip the renderer step (already built)
 *
 * `electron-builder` signs when it is handed a certificate through `CSC_LINK` and
 * `CSC_KEY_PASSWORD`, and decides whether an update counts as trusted based on the
 * publisher name baked into `app-update.yml`. Leaving that name out of
 * `package.json` matters: an unsigned build must not claim a publisher it never
 * had. So both values are injected here, from the real certificate, at the moment
 * it is actually used. `node scripts/make-cert.mjs` creates the certificate.
 */
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const certDir = path.join(root, "certs");
const pfxFile = path.join(certDir, "dev-codesign.pfx");
const metaFile = path.join(certDir, "dev-codesign.json");
const passwordFile = path.join(certDir, "dev-codesign.password");

const args = process.argv.slice(2);
// Certificates come from the environment the way electron-builder expects them,
// which is also how CI provides them; a local certificate is the fallback.
const fromEnvironment = Boolean(process.env.CSC_LINK);
const sign = args.includes("--sign") || fromEnvironment;
let publisher = null;
const dirOnly = args.includes("--dir");
const skipBuild = args.includes("--no-build");

function run(command, commandArgs) {
  console.log(`\n$ ${command} ${commandArgs.join(" ")}`);
  execFileSync(command, commandArgs, { cwd: root, stdio: "inherit" });
}

if (!skipBuild) {
  // npm is a shell script on Windows, so this one step goes through the shell.
  // Its arguments carry no spaces, which is what makes that safe here.
  console.log("\n$ npm run build");
  execFileSync(process.platform === "win32" ? "npm.cmd" : "npm", ["run", "build"], {
    cwd: root,
    stdio: "inherit",
    shell: process.platform === "win32",
  });
}

const builderArgs = [dirOnly ? "--dir" : "--win", "--publish", "never"];

if (sign) {
  if (!fromEnvironment && !fs.existsSync(pfxFile)) {
    console.error(`\nNo certificate at ${path.relative(root, pfxFile)} — run \`node scripts/make-cert.mjs\` first.`);
    process.exit(2);
  }
  const password = process.env.CSC_KEY_PASSWORD || (fs.existsSync(passwordFile) ? fs.readFileSync(passwordFile, "utf8").trim() : "");
  if (!password) {
    console.error(
      fromEnvironment
        ? "\nCSC_LINK is set but CSC_KEY_PASSWORD is not — electron-builder needs both."
        : `\nNo certificate password: set CSC_KEY_PASSWORD or keep ${path.relative(root, passwordFile)} next to the certificate.`,
    );
    process.exit(2);
  }
  publisher =
    process.env.ZEQOUX_PUBLISHER ||
    (fs.existsSync(metaFile) ? JSON.parse(fs.readFileSync(metaFile, "utf8")).name : null);
  if (!publisher) {
    console.error(`\nNo publisher name: pass ZEQOUX_PUBLISHER or regenerate the certificate with scripts/make-cert.mjs.`);
    process.exit(2);
  }
  if (!fromEnvironment) process.env.CSC_LINK = pfxFile;
  process.env.CSC_KEY_PASSWORD = password;
  // The publisher name in app-update.yml has to match the certificate's subject,
  // otherwise electron-updater rejects its own updates as unsigned by us.
  builderArgs.push(`-c.win.signtoolOptions.publisherName=${publisher}`);
  console.log(
    `\nsigning with ${fromEnvironment ? "CSC_LINK" : path.relative(root, pfxFile)} (publisher “${publisher}”)`,
  );
} else if (fs.existsSync(pfxFile)) {
  console.log(`\nbuilding unsigned — pass --sign to use ${path.relative(root, pfxFile)}`);
}

// electron-builder is started through its own entry point rather than through
// npx: a shell on Windows splits arguments at the spaces inside the publisher
// name, and a quoted publisher name would then reach electron-builder mangled.
const builderCli = path.join(path.dirname(require.resolve("electron-builder/package.json")), "cli.js");
run(process.execPath, [builderCli, ...builderArgs]);

/**
 * electron-builder bakes `app-update.yml` into the installer targets, so an
 * installed app knows its release channel. An unpacked build (`--dir`) gets no
 * such file, and `electron-updater` refuses to work without it — it reads the
 * update cache directory from there even when the channel is overridden. Writing
 * the same file from the same build configuration keeps `release/win-unpacked`
 * behaving like an installation, which is what the packed-build tests drive.
 */
function writeUpdateConfig() {
  const pkg = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8"));
  const publish = Array.isArray(pkg.build?.publish) ? pkg.build.publish[0] : null;
  if (!publish) return null;

  const appOutDir = path.join(root, "release", process.platform === "win32" ? "win-unpacked" : process.platform === "darwin" ? "mac" : "linux-unpacked");
  const resources = process.platform === "darwin" ? path.join(appOutDir, "Contents", "Resources") : path.join(appOutDir, "resources");
  if (!fs.existsSync(resources)) return null;

  const lines = [];
  for (const key of ["owner", "repo", "provider", "url", "channel", "releaseType"]) {
    if (publish[key]) lines.push(`${key}: ${publish[key]}`);
  }
  lines.push(`updaterCacheDirName: ${pkg.name}-updater`);
  if (sign && publisher) lines.push("publisherName:", `  - ${publisher}`);

  const file = path.join(resources, "app-update.yml");
  // A full build already wrote its own copy from the same configuration.
  if (fs.existsSync(file)) return null;
  fs.writeFileSync(file, `${lines.join("\n")}\n`, "utf8");
  return file;
}

const updateConfig = writeUpdateConfig();
if (updateConfig) console.log(`\n  wrote ${path.relative(root, updateConfig)}`);

const release = path.join(root, "release");
const artifacts = fs.existsSync(release)
  ? fs.readdirSync(release).filter((entry) => !entry.startsWith(".") && fs.statSync(path.join(release, entry)).isFile())
  : [];
console.log("");
for (const entry of artifacts) {
  const size = fs.statSync(path.join(release, entry)).size;
  console.log(`  ${entry}  ${size > 1024 * 1024 ? `${(size / 1024 / 1024).toFixed(1)} MiB` : `${size} B`}`);
}
if (sign) {
  console.log("");
  console.log("  verify: powershell -NoProfile -Command \"Get-AuthenticodeSignature 'release\\…' | Format-List\"");
  console.log("  a self-signed certificate reports UntrustedRoot until it is trusted on the machine reading it");
}
