/**
 * A tiny static server for an update channel.
 *
 *     node scripts/update-server.mjs [dir] [--port 8123] [--host 127.0.0.1]
 *
 * `electron-builder` publishes releases to GitHub, and the app talks to that by
 * default. Any other host — a file share, an internal mirror, the machine you are
 * testing on — needs the same two things: `latest.yml` and the installer it
 * names, at the same base URL. This serves a directory that way (including byte
 * ranges, which `electron-updater` uses for differential downloads) so a channel
 * can be hosted or tested without a release pipeline.
 *
 * `createUpdateServer` is exported so tests can start one on a random port.
 */
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const TYPES = {
  ".yml": "text/yaml; charset=utf-8",
  ".yaml": "text/yaml; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".exe": "application/octet-stream",
  ".dmg": "application/octet-stream",
  ".zip": "application/octet-stream",
  ".blockmap": "application/octet-stream",
  ".txt": "text/plain; charset=utf-8",
};

function resolveInside(rootDir, requestPath) {
  const decoded = decodeURIComponent(requestPath.split("?")[0]);
  const target = path.resolve(rootDir, `.${decoded}`);
  // Never serve anything outside the channel directory, however the URL is built.
  if (!target.startsWith(path.resolve(rootDir))) return null;
  return target;
}

export function createUpdateServer({ dir, host = "127.0.0.1", port = 0, log = null } = {}) {
  if (!fs.existsSync(dir)) throw new Error(`no channel directory at ${dir}`);
  const server = http.createServer((request, response) => {
    const target = resolveInside(dir, request.url ?? "/");
    const file = target && fs.existsSync(target) && fs.statSync(target).isFile() ? target : null;

    if (!file) {
      if (log) log(`404 ${request.method} ${request.url}`);
      response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      response.end("not found\n");
      return;
    }

    const size = fs.statSync(file).size;
    const type = TYPES[path.extname(file).toLowerCase()] ?? "application/octet-stream";
    const range = /^bytes=(\d*)-(\d*)$/.exec(request.headers.range ?? "");

    if (range && (range[1] || range[2])) {
      const start = range[1] ? Number(range[1]) : Math.max(0, size - Number(range[2]));
      const end = range[1] && range[2] ? Math.min(Number(range[2]), size - 1) : size - 1;
      if (log) log(`206 ${request.method} ${request.url} (${start}-${end}/${size})`);
      response.writeHead(206, {
        "Content-Type": type,
        "Content-Length": end - start + 1,
        "Content-Range": `bytes ${start}-${end}/${size}`,
        "Accept-Ranges": "bytes",
      });
      if (request.method === "HEAD") response.end();
      else fs.createReadStream(file, { start, end }).pipe(response);
      return;
    }

    if (log) log(`${request.method} ${request.url} (${size} bytes)`);
    response.writeHead(200, { "Content-Type": type, "Content-Length": size, "Accept-Ranges": "bytes" });
    if (request.method === "HEAD") response.end();
    else fs.createReadStream(file).pipe(response);
  });

  return new Promise((resolve, reject) => {
    server.on("error", reject);
    server.listen(port, host, () => {
      const address = server.address();
      resolve({
        port: address.port,
        url: `http://${host}:${address.port}`,
        close: () => new Promise((done) => server.close(() => done())),
      });
    });
  });
}

const invokedDirectly = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (invokedDirectly) {
  const args = process.argv.slice(2);
  const flagOf = (name, fallback) => {
    const index = args.indexOf(`--${name}`);
    return index === -1 ? fallback : args[index + 1];
  };
  const dir = path.resolve(args.find((arg) => !arg.startsWith("--") && arg !== flagOf("port") && arg !== flagOf("host")) ?? "release/channel");
  const port = Number(flagOf("port", 8123));
  const host = String(flagOf("host", "127.0.0.1"));

  try {
    const files = fs.readdirSync(dir);
    const server = await createUpdateServer({ dir, host, port, log: (line) => console.log(line) });
    console.log(`serving ${dir}`);
    console.log(`  ${files.length} file(s): ${files.join(", ")}`);
    console.log(`  channel URL: ${server.url}`);
    console.log("  set it in Settings → Updates, or start the app with ZEQOUX_UPDATE_URL=" + server.url);
    console.log("  press Ctrl+C to stop");
  } catch (error) {
    console.error(`Could not serve ${dir}: ${error.message}`);
    process.exit(2);
  }
}
