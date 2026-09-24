import { useEffect, useMemo, useState } from "react";
import { ExternalLink, Play, RefreshCw, Server, Square, Terminal, Zap } from "lucide-react";
import { api } from "../lib/api";
import { basename, clock, duration } from "../lib/format";
import { useApp } from "../state/app";
import {
  Badge,
  Button,
  Callout,
  CodeBlock,
  CopyButton,
  Field,
  KeyValue,
  NumberInput,
  Panel,
  SectionHeader,
  Select,
  Stat,
  Table,
  Td,
  TextInput,
  Th,
} from "../components/ui";
import { ErrorPanel, ModelPicker, useEngine } from "./common";

interface RequestRecord {
  at: string;
  path: string;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  latency_seconds: number | null;
  tokens_per_second: number | null;
  preview?: string;
}

export function DeployPage() {
  const { jobs, refreshJobs, toast, reportError, registry, refreshRegistry } = useApp();
  const [modelPath, setModelPath] = useState<string | null>(null);
  const [host, setHost] = useState("127.0.0.1");
  const [port, setPort] = useState(8080);
  const [concurrency, setConcurrency] = useState(2);
  const [starting, setStarting] = useState(false);
  const [testPrompt, setTestPrompt] = useState("Say hello in one short sentence.");
  const [testResult, setTestResult] = useState<any>(null);
  const [requestLog, setRequestLog] = useState<RequestRecord[]>([]);
  const [ending, setEnding] = useState(false);
  const tools = useEngine<any>("tools.report", { model: modelPath }, { deps: [modelPath], timeout: 120_000 });

  const servers = useMemo(() => jobs.filter((job) => job.kind === "server"), [jobs]);
  const active = useMemo(
    () => servers.find((job) => ["running", "queued", "paused"].includes(job.state)) ?? servers[0] ?? null,
    [servers],
  );
  const running = Boolean(active && ["running", "queued", "paused"].includes(active.state));
  const serverInfo = (active?.result as any)?.server ?? (active as any)?.plan?.server ?? null;
  const requestUrl = active?.job_dir ? `${active.job_dir}\\requests.jsonl`.replace(/\//g, "\\") : null;
  const endpointHost = serverInfo?.host ?? host;
  const endpointPort = serverInfo?.port ?? port;

  // Poll the job's request log while a server is up.
  useEffect(() => {
    if (!running || !active?.job_dir) {
      setRequestLog([]);
      return;
    }
    let cancelled = false;
    const load = async () => {
      try {
        const text = await api.readText(`${active.job_dir}\\requests.jsonl`, 512 * 1024);
        if (cancelled) return;
        const records = text
          .split("\n")
          .filter(Boolean)
          .map((line) => {
            try {
              return JSON.parse(line);
            } catch {
              return null;
            }
          })
          .filter((entry) => entry && entry.type === "request")
          .reverse()
          .slice(0, 40) as RequestRecord[];
        setRequestLog(records);
      } catch {
        if (!cancelled) setRequestLog([]);
      }
    };
    void load();
    const timer = window.setInterval(load, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [running, active?.job_dir]);

  const startServer = async () => {
    if (!modelPath) {
      toast({ title: "Pick a model", body: "The server needs a model to load.", tone: "warn" });
      return;
    }
    setStarting(true);
    try {
      const job = await api.jobs.start({
        kind: "server",
        base_model: modelPath,
        host,
        port,
        concurrency,
      });
      await api.registry.add("servers", {
        id: job.jobId,
        model: modelPath,
        host,
        port,
        createdAt: new Date().toISOString(),
        state: "starting",
      });
      await refreshRegistry();
      await refreshJobs();
      toast({
        title: "Local server starting",
        body: `Job ${job.jobId} is loading ${basename(modelPath)} and will bind ${host}:${port}.`,
        tone: "info",
      });
    } catch (error) {
      reportError(error, "The server could not be started");
    } finally {
      setStarting(false);
    }
  };

  const stopServer = async () => {
    if (!active) return;
    setEnding(true);
    try {
      await api.jobs.control(active.job_id, { stop: true });
      await refreshJobs();
      toast({ title: "Shutdown requested", body: "The engine stops accepting requests and releases the model.", tone: "info" });
    } catch (error) {
      reportError(error, "The server did not stop");
    } finally {
      setEnding(false);
    }
  };

  const sendTest = async () => {
    const url = `http://${endpointHost}:${endpointPort}/v1/chat/completions`;
    try {
      const started = performance.now();
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [{ role: "user", content: testPrompt }],
          max_tokens: 64,
          temperature: 0.7,
        }),
      });
      const payload = await response.json();
      setTestResult({ status: response.status, ms: Math.round(performance.now() - started), payload });
      if (!response.ok) reportError({ structured: payload.error ?? { message: `HTTP ${response.status}` } }, "The API answered with an error");
    } catch (error: any) {
      setTestResult({
        status: 0,
        ms: null,
        payload: { error: { message: String(error?.message ?? error) } },
      });
      reportError(error, "The request did not reach the server");
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Deploy</h1>
          <p className="mt-0.5 max-w-3xl text-xs text-ink-2">
            Serves the loaded model over HTTP with an OpenAI-compatible surface. The server is a real engine process:
            it binds the port, logs every request to its job folder, and releases the model when stopped.
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="subtle" icon={<RefreshCw size={12} />} onClick={() => void refreshJobs()}>
            Refresh
          </Button>
          {running ? (
            <Button size="sm" variant="danger" icon={<Square size={12} />} loading={ending} onClick={() => void stopServer()}>
              Stop server
            </Button>
          ) : (
            <Button size="sm" variant="primary" icon={<Play size={12} />} loading={starting} onClick={() => void startServer()}>
              Start server
            </Button>
          )}
        </div>
      </div>

      <div className="grid gap-3 lg:grid-cols-[380px_1fr]">
        <div className="space-y-3">
          <Panel>
            <SectionHeader title="Server" subtitle="Loopback only unless you change the host." />
            <div className="space-y-3">
              <ModelPicker value={modelPath} onChange={setModelPath} label="Model to serve" />
              <div className="grid grid-cols-2 gap-3">
                <Field label="Host">
                  <Select value={host} onChange={(event) => setHost(event.target.value)}>
                    <option value="127.0.0.1">127.0.0.1 (loopback)</option>
                    <option value="0.0.0.0">0.0.0.0 (all interfaces)</option>
                  </Select>
                </Field>
                <Field label="Port">
                  <NumberInput value={port} min={1} max={65535} onChange={(value) => setPort(Number(value) || 8080)} />
                </Field>
              </div>
              <Field label="Concurrency" hint="How many requests the threaded server will handle at once; the backend still runs generations one at a time.">
                <NumberInput value={concurrency} min={1} max={16} onChange={(value) => setConcurrency(Number(value) || 1)} />
              </Field>
              {host === "0.0.0.0" ? (
                <Callout tone="warn" title="This binds to every interface">
                  Anyone on the same network could reach the API. Prefer loopback unless you deliberately want that.
                </Callout>
              ) : null}
            </div>
          </Panel>

          <Panel>
            <SectionHeader title="Status" subtitle="Read from the engine process, not cached." />
            {active ? (
              <>
                <KeyValue
                  items={[
                    ["Job", active.job_id],
                    ["State", active.state],
                    ["Model", active.model ? basename(active.model) : serverInfo?.model ? basename(serverInfo.model) : "—"],
                    ["Backend", (active as any).backend ?? serverInfo?.backend ?? "auto"],
                    ["Endpoint", `http://${endpointHost}:${endpointPort}`],
                    ["Requests", requestLog.length ? requestLog.length : String((active as any).metrics_state?.requests ?? 0)],
                    ["Uptime", duration(active.elapsed_seconds)],
                    ["Started", active.started_at ? clock(active.started_at) : "—"],
                  ]}
                  columns={1}
                />
                {active.error ? (
                  <div className="mt-3">
                    <Callout tone="danger" title={active.error.message} hint={active.error.hint}>
                      {active.error.code}
                    </Callout>
                  </div>
                ) : null}
                <div className="mt-3 flex flex-wrap gap-2">
                  <Button size="sm" variant="subtle" onClick={() => void api.shell.reveal(active.job_dir!)}>
                    Open job folder
                  </Button>
                  <Button
                    size="sm"
                    variant="subtle"
                    icon={<ExternalLink size={11} />}
                    onClick={() => void api.shell.openExternal(`http://${endpointHost}:${endpointPort}/docs`)}
                  >
                    Open API docs
                  </Button>
                </div>
              </>
            ) : (
              <Callout tone="info" title="No server has been started yet">
                Starting one runs <span className="font-mono text-2xs">python -m zxtrain.cli run &lt;spec.json&gt;</span>{" "}
                with kind <span className="font-mono text-2xs">server</span>. It appears in Jobs like any other run.
              </Callout>
            )}
          </Panel>

          <Panel>
            <SectionHeader title="Capabilities" subtitle="What this model declares, checked on the engine side." />
            {tools.error ? <ErrorPanel error={tools.error} onRetry={() => void tools.reload()} /> : null}
            {tools.data ? (
              <KeyValue
                items={[
                  ["Tool calling", tools.data.tool_calling?.available ? "declared by the model" : "not declared"],
                  ["Reasoning field", tools.data.tool_calling?.available ? "see template" : "not declared"],
                  ["JSON mode", tools.data.json_mode?.available ? "configuration hints found" : "not declared"],
                  ["Structured output", tools.data.json_mode?.reason ?? "—"],
                ]}
                columns={1}
              />
            ) : null}
            {tools.data?.tool_calling?.evidence?.length ? (
              <div className="mt-2 space-y-1">
                {tools.data.tool_calling.evidence.map((line: string) => (
                  <div key={line} className="font-mono text-2xs text-ink-3">
                    {line}
                  </div>
                ))}
              </div>
            ) : null}
          </Panel>
        </div>

        <div className="space-y-3">
          <Panel>
            <SectionHeader
              title="Request tester"
              subtitle="Sends a real HTTP request to the running server and shows the raw response."
              actions={
                <Button size="sm" variant="primary" icon={<Zap size={11} />} disabled={!running} onClick={() => void sendTest()}>
                  Send request
                </Button>
              }
            />
            <div className="space-y-3">
              <Field label="Prompt">
                <TextInput value={testPrompt} onChange={(event) => setTestPrompt(event.target.value)} />
              </Field>
              <CodeBlock max="max-h-40">{`POST http://${endpointHost}:${endpointPort}/v1/chat/completions
Content-Type: application/json

{"messages":[{"role":"user","content":${JSON.stringify(testPrompt)}}],"max_tokens":64}`}</CodeBlock>
              {testResult ? (
                <>
                  <div className="grid gap-2 sm:grid-cols-2">
                    <Stat label="HTTP status" value={testResult.status} tone={testResult.status === 200 ? "ok" : "danger"} />
                    <Stat label="Round trip" value={testResult.ms !== null ? `${testResult.ms} ms` : "—"} />
                  </div>
                  <CodeBlock max="max-h-64">{JSON.stringify(testResult.payload, null, 2)}</CodeBlock>
                </>
              ) : (
                <div className="text-xs text-ink-2">
                  Start the server and send a request — the response below is exactly what the engine returned.
                </div>
              )}
            </div>
          </Panel>

          <Panel>
            <SectionHeader
              title="Requests"
              subtitle="Tail of the server's own request log."
              actions={<Badge tone="muted">{requestLog.length} shown</Badge>}
            />
            {requestLog.length ? (
              <Table>
                <thead>
                  <tr>
                    <Th>Time</Th>
                    <Th>Path</Th>
                    <Th align="right">Tokens</Th>
                    <Th align="right">Latency</Th>
                    <Th align="right">Tok/s</Th>
                  </tr>
                </thead>
                <tbody>
                  {requestLog.map((record, index) => (
                    <tr key={`${record.at}-${index}`}>
                      <Td>{clock(record.at)}</Td>
                      <Td>
                        <span className="font-mono text-2xs">{record.path}</span>
                      </Td>
                      <Td align="right">
                        {record.prompt_tokens ?? "—"} / {record.completion_tokens ?? "—"}
                      </Td>
                      <Td align="right">{record.latency_seconds !== null ? `${record.latency_seconds}s` : "—"}</Td>
                      <Td align="right">{record.tokens_per_second ?? "—"}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            ) : (
              <div className="text-xs text-ink-2">
                {running ? "No requests yet." : requestUrl ? `The log lives at ${requestUrl}.` : "Start the server to collect requests."}
              </div>
            )}
          </Panel>

          <Panel>
            <SectionHeader title="Client examples" subtitle="Copy-paste for other tools." actions={<CopyButton value={`curl http://${endpointHost}:${endpointPort}/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":64}'`} />} />
            <CodeBlock>{`# health
curl http://${endpointHost}:${endpointPort}/health

# model list (OpenAI compatible)
curl http://${endpointHost}:${endpointPort}/v1/models

# chat
curl http://${endpointHost}:${endpointPort}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":64,"stream":true}'

# raw completion
curl http://${endpointHost}:${endpointPort}/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{"prompt":"Once upon a time","max_tokens":32}'`}</CodeBlock>
          </Panel>

          <Panel>
            <SectionHeader title="Serving history" subtitle="Servers started from this workspace." />
            {registry?.servers?.length ? (
              <Table>
                <thead>
                  <tr>
                    <Th>Model</Th>
                    <Th>Endpoint</Th>
                    <Th>Started</Th>
                  </tr>
                </thead>
                <tbody>
                  {registry.servers.slice(0, 10).map((entry: any) => (
                    <tr key={entry.id}>
                      <Td>{entry.model ? basename(entry.model) : "—"}</Td>
                      <Td>
                        <span className="font-mono text-2xs">
                          {entry.host}:{entry.port}
                        </span>
                      </Td>
                      <Td>{clock(entry.createdAt)}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            ) : (
              <div className="text-xs text-ink-2">Nothing served yet.</div>
            )}
          </Panel>

          <Callout tone="info" title="Honesty about capabilities">
            <span className="flex items-center gap-2">
              <Server size={12} /> Tool calling and structured output are reported only when the model's own template or
              configuration declares them. The engine never fabricates tool calls, hidden reasoning or JSON mode.
            </span>
          </Callout>

          <div className="flex items-center gap-2 text-2xs text-ink-3">
            <Terminal size={11} /> Every request above was produced by the engine you can inspect in the Jobs page, not by
            a mock server.
          </div>
        </div>
      </div>
    </div>
  );
}
