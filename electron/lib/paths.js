/**
 * Path resolution for the app: where the engine lives, where the workspace
 * lives, and which directories exist inside it.
 */
import { app } from "electron";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
export const projectRoot = path.resolve(here, "..", "..");

export const WORKSPACE_DIRS = ["models", "datasets", "jobs", "runtime", "exports", "logs", "projects", "servers"];

/** The Python engine ships next to the app, in `python/zxtrain`. */
export function engineDir() {
  const packaged = path.join(process.resourcesPath ?? "", "python");
  if (app.isPackaged && fs.existsSync(path.join(packaged, "zxtrain"))) return packaged;
  return path.join(projectRoot, "python");
}

/** Where user data lives: settings, workspace, logs. */
export function userDataDir() {
  return app.getPath("userData");
}

export function settingsPath() {
  return path.join(userDataDir(), "settings.json");
}

export function defaultWorkspace() {
  const preferred = path.join(app.getPath("documents"), "ZeqouXTraining");
  return process.env.ZEQOUX_WORKSPACE || preferred;
}

export function ensureWorkspace(workspace) {
  const root = workspace || defaultWorkspace();
  fs.mkdirSync(root, { recursive: true });
  for (const name of WORKSPACE_DIRS) {
    fs.mkdirSync(path.join(root, name), { recursive: true });
  }
  return root;
}

export function workspaceFile(workspace, name) {
  return path.join(workspace, name);
}

/**
 * The version of *this* application.
 *
 * `app.getVersion()` reads it from the package.json of the app being run, but
 * when Electron is handed a file instead of a directory — which is how the
 * development harness and the tests start it — there is no such package.json in
 * scope and Electron answers with its own version. Falling back to the project's
 * package.json keeps the number honest in development instead of claiming to be
 * the browser runtime.
 */
export function appVersion() {
  if (app.isPackaged) return app.getVersion();
  try {
    const pkg = JSON.parse(fs.readFileSync(path.join(projectRoot, "package.json"), "utf8"));
    if (typeof pkg.version === "string" && pkg.version) return pkg.version;
  } catch {
    /* an unreadable package.json is not worth failing over */
  }
  return app.getVersion();
}

export function tmpDir() {
  const dir = path.join(os.tmpdir(), "zxtrain-electron");
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

export function rendererIndex() {
  return path.join(projectRoot, "dist", "index.html");
}
