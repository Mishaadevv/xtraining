/**
 * Create a local code-signing certificate.
 *
 *     node scripts/make-cert.mjs                 # write certs/dev-codesign.*
 *     node scripts/make-cert.mjs --trust         # also trust it for this user
 *     node scripts/make-cert.mjs --password s3cret --name "Zeqou Development"
 *
 * Windows refuses to run an unsigned installer without a SmartScreen warning, and
 * `electron-builder` refuses to sign without a certificate. This script produces a
 * self-signed code-signing certificate so a local or internal build can be signed
 * and so the update path can be exercised the way it runs in production.
 *
 * A self-signed certificate is not a substitute for a real one: only machines that
 * have been told to trust it will treat the signature as valid, and SmartScreen
 * will still warn until enough machines have seen the binary. Everything it writes
 * goes to `certs/`, which is git-ignored.
 */
import { execFileSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const certDir = path.join(root, "certs");

function argOf(name, fallback = null) {
  const index = process.argv.indexOf(`--${name}`);
  if (index === -1) return fallback;
  const value = process.argv[index + 1];
  return value && !value.startsWith("--") ? value : true;
}

const name = String(argOf("name", "Zeqou Development"));
const days = Number(argOf("days", 1095));
const force = Boolean(argOf("force", false));
const trust = Boolean(argOf("trust", false));
const password = String(argOf("password") || crypto.randomBytes(18).toString("base64url"));

const keyFile = path.join(certDir, "dev-codesign.key");
const pemFile = path.join(certDir, "dev-codesign.crt");
const pfxFile = path.join(certDir, "dev-codesign.pfx");
const passwordFile = path.join(certDir, "dev-codesign.password");

function openssl(args, options = {}) {
  try {
    return execFileSync("openssl", args, {
      cwd: certDir,
      encoding: "utf8",
      stdio: options.quiet ? ["ignore", "pipe", "pipe"] : ["ignore", "pipe", "pipe"],
      // MSYS would otherwise turn `/CN=…` style arguments into Windows paths.
      env: { ...process.env, MSYS_NO_PATHCONV: "1", MSYS2_ARG_CONV_EXCL: "*" },
    });
  } catch (error) {
    const detail = `${error?.stderr ?? ""}${error?.stdout ?? ""}`.trim();
    throw new Error(`openssl ${args[0]} failed: ${detail || error.message}`);
  }
}

fs.mkdirSync(certDir, { recursive: true });

if (fs.existsSync(pfxFile) && !force) {
  console.log(`${path.relative(root, pfxFile)} already exists — pass --force to replace it.`);
  process.exit(0);
}

try {
  openssl(["-version"], { quiet: true });
} catch {
  console.error("openssl is not on PATH. Install OpenSSL (Git for Windows ships one) and try again.");
  process.exit(2);
}

// A configuration file rather than `-subj`: the subject then means the same thing
// on every shell, and Windows path mangling cannot reach it.
const configFile = path.join(certDir, "dev-codesign.cnf");
fs.writeFileSync(
  configFile,
  [
    "[req]",
    "distinguished_name = dn",
    "x509_extensions = ext",
    "prompt = no",
    "[dn]",
    `CN = ${name}`,
    "O = Zeqou",
    "[ext]",
    "basicConstraints = critical,CA:FALSE",
    "keyUsage = critical,digitalSignature,keyEncipherment",
    "extendedKeyUsage = critical,codeSigning",
    "subjectKeyIdentifier = hash",
    "",
  ].join("\n"),
  "utf8",
);

console.log(`generating a ${days}-day code-signing certificate for “${name}”…`);
openssl([
  "req", "-x509", "-newkey", "rsa:3072", "-sha256", "-days", String(days), "-nodes",
  "-keyout", path.basename(keyFile),
  "-out", path.basename(pemFile),
  "-config", path.basename(configFile),
]);

// signtool reads PKCS#12. The explicitly chosen ciphers keep it readable by the
// older CryptoAPI inside signtool, which still rejects OpenSSL 3's AES defaults in
// some Windows builds.
try {
  openssl([
    "pkcs12", "-export", "-out", path.basename(pfxFile),
    "-inkey", path.basename(keyFile), "-in", path.basename(pemFile),
    "-name", name, "-passout", `pass:${password}`,
    "-keypbe", "PBE-SHA1-3DES", "-certpbe", "PBE-SHA1-3DES", "-macalg", "sha1",
  ]);
} catch (error) {
  console.log(`  (modern PKCS#12 ciphers refused, falling back: ${error.message.split("\n")[0]})`);
  openssl([
    "pkcs12", "-export", "-out", path.basename(pfxFile),
    "-inkey", path.basename(keyFile), "-in", path.basename(pemFile),
    "-name", name, "-passout", `pass:${password}`,
  ]);
}

fs.writeFileSync(passwordFile, `${password}\n`, "utf8");

// The build script needs the publisher name without parsing openssl output, and
// electron-updater compares it against the certificate's subject.
const certMeta = { name, days, createdAt: new Date().toISOString(), passwordFile: path.basename(passwordFile) };
const metaFile = path.join(certDir, "dev-codesign.json");
fs.writeFileSync(metaFile, `${JSON.stringify(certMeta, null, 2)}\n`, "utf8");

const subject = openssl(["x509", "-in", path.basename(pemFile), "-noout", "-subject"]).trim();
const fingerprint = openssl(["x509", "-in", path.basename(pemFile), "-noout", "-fingerprint", "-sha256"]).trim();

console.log("");
console.log(`  certificate  ${path.relative(root, pemFile)}`);
console.log(`  private key  ${path.relative(root, keyFile)}`);
console.log(`  pkcs#12      ${path.relative(root, pfxFile)}`);
console.log(`  password     ${path.relative(root, passwordFile)}  (generated)`);
console.log(`  ${subject}`);
console.log(`  ${fingerprint}`);
console.log("");
console.log("  node scripts/dist.mjs --sign     # sign installers with this certificate");

if (trust) {
  if (process.platform !== "win32") {
    console.log("\n--trust only applies to Windows; on other platforms add the certificate to your trust store manually.");
  } else {
    // CurrentUser stores need no administrator rights; the OS shows its own
    // confirmation dialog for the root store.
    const script = [
      "$ErrorActionPreference = 'Stop'",
      `Import-Certificate -FilePath '${pemFile}' -CertStoreLocation 'Cert:\\CurrentUser\\TrustedPublisher' | Out-Null`,
      `Import-Certificate -FilePath '${pemFile}' -CertStoreLocation 'Cert:\\CurrentUser\\Root' | Out-Null`,
      "Write-Output 'certificate imported into CurrentUser\\TrustedPublisher and CurrentUser\\Root'",
    ].join("; ");
    try {
      const output = execFileSync("powershell", ["-NoProfile", "-NonInteractive", "-Command", script], { encoding: "utf8" });
      console.log(`\n${output.trim()}`);
      console.log("Locally signed builds now verify on this machine. Other machines still need the certificate.");
    } catch (error) {
      console.error(`\nCould not trust the certificate: ${String(error?.stderr ?? error.message).trim()}`);
      console.error("Import it by hand from the certificate file if you need this machine to trust the signature.");
    }
  }
} else {
  console.log("  (add --trust to make this machine accept the signature)");
}
