/**
 * Packed-build test — the last mile that `npm run dist` cannot check itself:
 *
 *     npm run pack          # produces release/win-unpacked
 *     npm run verify:packed
 *
 * electron-builder exiting with code 0 only proves the archive was written. This
 * script starts the *packed executable* with a throwaway profile and workspace,
 * attaches to it over the Chrome DevTools protocol, and asserts what a user would
 * notice: the window opens, the bridge answers, the bundled engine under
 * `resources/python` really runs and, if the renderer misbehaves, that shows up
 * here as a failure instead of on someone's desktop.
 */
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { PackedApp, Reporter, packedExecutable, root, sleep, withPackedApp } from "./lib/packed.mjs";

const report = new Reporter("packed");

if (!fs.existsSync(packedExecutable())) {
  console.log(`FAIL  ${path.relative(root, packedExecutable())} does not exist — run \`npm run pack\` first.`);
  process.exit(2);
}

const failure = await withPackedApp({}, async (app) => {
  const target = await app.start();
  report.record("the packed app opened a debuggable window", Boolean(target.url), target.url);

  const booted = await app.waitForInterface();
  report.record("the bundled interface loads from the packed archive", booted);
  if (!booted) return;

  const onScreen = String(await app.js("document.body.innerText"));
  report.record(
    "the app did not come up on an error screen",
    !onScreen.includes("could not reach its Python engine") && !onScreen.includes("The desktop bridge is not available"),
    onScreen.slice(0, 120).replace(/\s+/g, " "),
  );

  const bridge = await app.js(`(() => ({
    ok: typeof window.zx === "object" && typeof window.zx.engine?.call === "function",
    nodeExposed: typeof window.require === "function" || typeof window.process?.versions?.node === "string",
  }))()`);
  report.record("the preload bridge is present and Node is not exposed", bridge.ok === true && bridge.nodeExposed !== true);

  const info = await app.js("window.zx.app.info()");
  const engineDir = String(info?.data?.engineDir ?? "");
  report.record(
    "the app runs as a real install",
    info?.ok === true && info.data.packaged === true && Boolean(info.data.version),
    `version ${info?.data?.version}, electron ${info?.data?.electron}`,
  );
  report.record(
    "the engine ships inside the install and not next to the sources",
    engineDir.toLowerCase().includes(`${path.sep}resources${path.sep}python`.toLowerCase()),
    engineDir,
  );
  report.record(
    "the run is confined to a throwaway profile",
    String(info?.data?.workspace ?? "").toLowerCase() === app.workspace.toLowerCase() &&
      String(info?.data?.userData ?? "").toLowerCase() === app.profile.toLowerCase(),
    `workspace ${info?.data?.workspace}`,
  );

  // Renderer -> IPC -> bundled Python engine -> back.
  const capabilities = await app.js(`window.zx.engine.call("engine.capabilities", {}, { timeout: 240000 })`);
  report.record(
    "the bundled engine answers for real",
    capabilities?.ok === true && Boolean(capabilities.data?.hardware?.cpu?.model),
    capabilities?.ok
      ? `cpu ${capabilities.data.hardware.cpu.model}, ${capabilities.data.backends?.length ?? 0} backends, python ${capabilities.data.hardware.os?.python ?? "?"}`
      : JSON.stringify(capabilities?.error ?? capabilities).slice(0, 300),
  );

  // Interpreter discovery lives in the main process, so it is not an engine
  // command: it goes through its own channel.
  const interpreters = await app.js("window.zx.engine.interpreters(true)");
  const found = Array.isArray(interpreters?.data) ? interpreters.data : [];
  report.record(
    "the packed app still finds a Python interpreter on this machine",
    interpreters?.ok === true && found.length > 0,
    found.length ? found.map((item) => `${item.version}${item.torch ? " +torch" : ""}`).join(", ") : "none found",
  );

  /* ------------------------------------------------------------ pages --- */

  const navCount = await app.js(`document.querySelectorAll("aside nav button").length`);
  report.record("the sidebar lists every page", navCount >= 19, `${navCount} nav items`);

  let previousHeading = null;
  let renderedPages = 0;
  for (let index = 0; index < navCount; index += 1) {
    const label = String(await app.js(`(document.querySelectorAll("aside nav button")[${index}].innerText || "").trim()`));
    await app.js(`document.querySelectorAll("aside nav button")[${index}].click() || true`);
    const rendered = await app.waitFor(
      `(() => {
         const main = document.querySelector("main");
         const heading = main?.querySelector("h1");
         return Boolean(heading) && heading.textContent.trim().length > 0 && (main.innerText || "").trim().length > 120;
       })()`,
      30_000,
      `page “${label}”`,
    );
    const heading = String(await app.js(`(document.querySelector("main h1")?.textContent || "").trim()`));
    if (rendered && heading !== previousHeading) renderedPages += 1;
    previousHeading = heading;
    await sleep(120);
  }
  report.record(
    "every page renders distinct content from the packed build",
    renderedPages === navCount,
    `${renderedPages}/${navCount} pages`,
  );

  /* ------------------------------------------------- updates panel --- */

  // The update panel has to show what the main process really knows — version,
  // channel and log path all come from there — rather than a placeholder.
  await app.js(`(() => {
    const settings = [...document.querySelectorAll("aside nav button")].find((button) => button.innerText.trim() === "Settings");
    settings.click();
    return true;
  })()`);
  await app.waitFor(`(document.querySelector("main h1")?.textContent || "").includes("Settings")`, 20_000, "Settings");
  const openedUpdates = await app.js(`(() => {
    const button = [...document.querySelectorAll("main button")].find((entry) => entry.innerText.trim() === "Updates");
    if (!button) return false;
    button.click();
    return true;
  })()`);
  await sleep(400);
  // The interface uppercases its statistic labels, so compare without case.
  const updatesPanel = String(await app.js(`document.querySelector("main").innerText`)).toLowerCase();
  const version = String(info?.data?.version ?? "");
  const shown = {
    installed: updatesPanel.includes("installed version"),
    version: updatesPanel.includes(version.toLowerCase()),
    signature: updatesPanel.includes("signature verification"),
    log: updatesPanel.includes("updater.log"),
  };
  report.record(
    "the Updates panel reports the real version, channel and log",
    openedUpdates === true && Object.values(shown).every(Boolean),
    openedUpdates
      ? `${version} · ${Object.entries(shown).map(([key, ok]) => `${key}=${ok}`).join(" ")}`
      : "the section never opened",
  );

  /* -------------------------------------------------------- palette --- */

  await app.js(`window.dispatchEvent(new KeyboardEvent("keydown", { key: "k", ctrlKey: true, bubbles: true })) || true`);
  const opened = await app.waitFor(`Boolean(document.querySelector('input[placeholder^="Search pages"]'))`, 10_000, "the palette");
  const filtered = await app.js(`(() => {
    const input = document.querySelector('input[placeholder^="Search pages"]');
    if (!input) return false;
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
    setter.call(input, "hardware");
    input.dispatchEvent(new Event("input", { bubbles: true }));
    return true;
  })()`);
  await sleep(300);
  const matched = String(await app.js("document.body.innerText")).includes("Hardware");
  await app.js(`window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true })) || true`);
  await sleep(300);
  const closed = await app.js(`!document.querySelector('input[placeholder^="Search pages"]')`);
  report.record("the command palette opens, filters and closes", opened && filtered && matched && closed === true);

  report.record("the renderer logged no errors", app.pageErrors.length === 0, app.pageErrors.slice(0, 4).join(" | "));
});

if (failure) report.record("the packed-build test finished without throwing", false, String(failure?.message ?? failure));
const failed = report.finish();
process.exit(failed ? 1 : 0);
