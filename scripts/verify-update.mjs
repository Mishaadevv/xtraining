/**
 * Update test — does the installed app really follow a release channel?
 *
 *     npm run pack            # the packed build must exist
 *     npm run verify:update
 *
 * It stands up a local update channel (a `latest.yml` and its installer, served
 * over loopback), starts the packed application pointed at that channel through
 * `ZEQOUX_UPDATE_URL`, and drives the update from the renderer as a user would:
 * check, refuse to install something that is not downloaded, download, verify.
 *
 * What this proves: the channel is read, the version is compared, the payload is
 * downloaded and its hash matches the manifest, and the progress the UI shows is
 * real. What it cannot prove: that the installer replaces the running build —
 * that needs a real release, because the payload here is the current build wearing
 * the next version number (see `make-update-manifest.mjs`).
 */
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { PackedApp, Reporter, packedDir, packedExecutable, root, withPackedApp } from "./lib/packed.mjs";
import { sha512, writeChannel } from "./make-update-manifest.mjs";
import { createUpdateServer } from "./update-server.mjs";

const report = new Reporter("update");

const appVersion = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8")).version;
const [major, minor, patch] = appVersion.split(".").map(Number);
const nextVersion = `${major}.${minor}.${patch + 1}`;

function findInstaller() {
  const release = path.join(root, "release");
  if (!fs.existsSync(release)) return null;
  const candidates = fs
    .readdirSync(release)
    .filter((entry) => /\.exe$/i.test(entry) && /setup/i.test(entry) && fs.statSync(path.join(release, entry)).isFile())
    .map((entry) => path.join(release, entry));
  return candidates.sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs)[0] ?? null;
}

const installer = findInstaller();
if (!fs.existsSync(packedExecutable())) {
  console.log(`FAIL  ${path.relative(root, packedExecutable())} does not exist — run \`npm run pack\` first.`);
  process.exit(2);
}
if (!installer) {
  console.log("FAIL  no installer in release/ to publish into the test channel — run `npm run dist` first.");
  process.exit(2);
}

let server = null;
const failure = await withPackedApp({ env: { ZEQOUX_UPDATE_SKIP_SIGNATURE: "1" } }, async (app) => {
  // The channel: the current installer, announced as the next version.
  const channel = await writeChannel({
    dir: app.channel,
    installer,
    version: nextVersion,
    notes: "Local channel created by scripts/verify-update.mjs.",
  });
  let requests = 0;
  server = await createUpdateServer({
    dir: app.channel,
    log: (line) => {
      requests += 1;
      if (requests <= 25) report.note(`channel: ${line}`);
    },
  });
  report.note(`channel ${server.url} announces ${channel.version} (${(channel.files[0].size / 1024 / 1024).toFixed(1)} MiB)`);

  // Only the explicit checks this test performs should run.
  app.seedSettings({ checkForUpdatesOnStart: false });
  app.env.ZEQOUX_UPDATE_URL = server.url;

  await app.start();
  const booted = await app.waitForInterface();
  report.record("the packed app started with an update channel configured", booted);
  if (!booted) return;

  // The bridge returns `{ ok, data }`; only the payload is interesting here.
  const before = (await app.js("window.zx.updates.state()"))?.data ?? null;
  report.record(
    "the channel override is what the updater talks to",
    before?.channel?.source === "override" && before.channel.provider === "generic" && before.channel.url === server.url,
    `${before?.channel?.source} · ${before?.channel?.label}`,
  );
  report.record(
    "the install reports itself and its version honestly",
    before?.currentVersion === appVersion && before.mode === "installed" && before.supported === true,
    `version ${before?.currentVersion}, mode ${before?.mode}`,
  );
  report.record(
    "signature verification is off because the test asked for it",
    before?.signatureVerification === false,
    "ZEQOUX_UPDATE_SKIP_SIGNATURE=1",
  );

  // Record every state the main process pushes, so progress can be checked later.
  await app.js("(() => { window.__updates = []; window.zx.updates.onState((state) => window.__updates.push(state)); return true; })()");

  // 1. Check.
  const checked = (await app.js("window.zx.updates.check()"))?.data ?? null;
  const sawAvailable = await app.waitFor(
    `window.__updates.some((state) => state.status === "available")`,
    60_000,
    "the update to be announced",
  );
  report.record(
    "the channel is read and the newer version is found",
    sawAvailable && checked?.available?.version === nextVersion,
    `${appVersion} → ${checked?.available?.version ?? "nothing"}${checked?.error ? ` (${checked.error.message})` : ""}`,
  );

  // 2. Installing something that was never downloaded has to be refused.
  const premature = await app.js("window.zx.updates.install()");
  report.record(
    "installing before the download is refused, not attempted",
    premature?.ok === false && premature.error?.code === "not_downloaded",
    premature?.ok === false ? premature.error.code : "it tried to install",
  );

  // 3. Download. The promise is not awaited here: progress is what matters, and
  //    the state stream reports the end of it.
  await app.js(`(() => { window.zx.updates.download().then((state) => { window.__downloadResult = state; }); return true; })()`);
  const downloaded = await app.waitFor(
    `window.__updates.some((state) => state.status === "downloaded")`,
    120_000,
    "the update to finish downloading",
    { quiet: false },
  );
  if (!downloaded) {
    // The updater writes why it stopped to its own log; that is the honest answer
    // to “the download did not finish”.
    const logFile = path.join(app.workspace, "logs", "updater.log");
    if (fs.existsSync(logFile)) {
      const tail = fs.readFileSync(logFile, "utf8").trim().split(/\r?\n/).slice(-12);
      for (const line of tail) report.note(`updater: ${line}`);
    } else {
      report.note("updater: no log file was written");
    }
  }
  const progressStates = await app.js(`window.__updates.filter((state) => state.progress && state.progress.total > 0).length`);
  const lastProgress = await app.js(`(() => {
    const withProgress = window.__updates.filter((state) => state.progress && state.progress.total > 0);
    const last = withProgress[withProgress.length - 1];
    return last ? { percent: last.progress.percent, total: last.progress.total, speed: last.progress.bytesPerSecond } : null;
  })()`);
  report.record("the download completed", downloaded === true, `${progressStates} progress report(s), last ${lastProgress?.percent?.toFixed?.(1) ?? "?"}%`);

  const final = (await app.js("window.zx.updates.state()"))?.data ?? null;
  report.record(
    "the app reports a downloaded update ready to install",
    final?.status === "downloaded" && final.available?.version === nextVersion && Boolean(final.downloadedFile),
    final?.downloadedFile ??
      `status ${final?.status}${final?.progress ? `, ${final.progress.percent.toFixed(1)}%, ${final.progress.transferred} B of ${final.progress.total} B` : ""}${
        final?.error ? `, error ${final.error.message}` : ""
      }`,
  );

  // 4. The downloaded payload must be the file the manifest described.
  const downloadedFile = final?.downloadedFile ?? null;
  if (downloadedFile && fs.existsSync(downloadedFile)) {
    const digest = await sha512(downloadedFile);
    report.record(
      "the downloaded installer matches the hash in the manifest",
      digest === channel.sha512,
      digest === channel.sha512 ? "sha512 verified" : `expected ${channel.sha512.slice(0, 16)}…, got ${digest.slice(0, 16)}…`,
    );
    report.record(
      "nothing was written outside the throwaway profile",
      path.resolve(downloadedFile).toLowerCase().startsWith(app.profile.toLowerCase()),
      path.dirname(downloadedFile),
    );
  } else {
    report.record("the downloaded installer is on disk", false, downloadedFile ?? "the app reported no file");
  }

  report.record("the renderer logged no errors", app.pageErrors.length === 0, app.pageErrors.slice(0, 4).join(" | "));

  /* ------------------- phase two: an untrusted publisher ---------------- */

  // Signature verification is only performed when the build names a publisher,
  // so this half is skipped on an unsigned build rather than pretending.
  const updateConfig = path.join(packedDir(), "resources", "app-update.yml");
  const namesPublisher = fs.existsSync(updateConfig) && /publisherName:/.test(fs.readFileSync(updateConfig, "utf8"));
  if (!namesPublisher) {
    report.note("the packed build names no publisher — signature verification cannot be exercised here");
    return;
  }

  const guarded = new PackedApp({});
  try {
    guarded.seedSettings({ checkForUpdatesOnStart: false });
    guarded.env.ZEQOUX_UPDATE_URL = server.url;
    await guarded.start();
    if (!(await guarded.waitForInterface())) {
      report.record("the app restarted with signature verification on", false, "it never booted");
      return;
    }
    const strict = (await guarded.js("window.zx.updates.state()"))?.data ?? null;
    report.record(
      "the same update is attempted with signature verification on",
      strict?.signatureVerification === true && strict?.channel?.source === "override",
      `signature verification ${strict?.signatureVerification ? "on" : "off"}`,
    );

    // The check has to succeed first, or a refused download proves nothing.
    await guarded.js("window.zx.updates.check()");
    const announced = await guarded.waitFor(
      `window.zx.updates.state().then((result) => result?.data?.available?.version === ${JSON.stringify(nextVersion)})`,
      60_000,
      "the update to be announced with verification on",
    );
    report.record("the channel announces the same update here", announced === true, nextVersion);

    await guarded.js(
      `(() => { window.zx.updates.download().then((state) => { window.__strictResult = state; }); return true; })()`,
    );
    const refused = await guarded.waitFor(
      `(() => { const result = window.__strictResult; return Boolean(result) && result.status !== "downloading"; })()`,
      180_000,
      "the unverifiable update to be refused",
      { quiet: false },
    );
    const after = (await guarded.js("window.zx.updates.state()"))?.data ?? null;
    const message = String(after?.error?.message ?? "");
    report.record(
      "an update whose publisher cannot be verified is refused",
      refused === true && after?.status !== "downloaded" && /signature|publisher|not signed|cannot verify/i.test(message),
      message ? message.replace(/\s+/g, " ").slice(0, 240) : `status ${after?.status}, no error reported`,
    );
    report.record("the refused download is not kept as an installed update", !after?.downloadedFile);
  } finally {
    await guarded.cleanup();
  }
});

await server?.close();
if (failure) report.record("the update test finished without throwing", false, String(failure?.message ?? failure));
const failed = report.finish();
console.log(`  (the ${nextVersion} payload is the ${appVersion} build; installing it needs a real release)`);
console.log(`  (packed build under test: ${path.relative(root, packedDir())})`);
process.exit(failed ? 1 : 0);
