import { useState } from "react";
import { Cpu, Package, ShieldCheck, Zap } from "lucide-react";
import { api } from "../lib/api";
import { bytes } from "../lib/format";
import { useApp } from "../state/app";
import { useRouter } from "../state/router";
import { Badge, Button, Callout, Field, Panel, SectionHeader, Select, Stat, Table, Td, Th, Toggle } from "../components/ui";
import { ErrorPanel, Loading, RefreshButton, useEngine } from "./common";

export function EnvironmentPage() {
  const { environment, appInfo, refreshJobs, toast, saveSettings, settings } = useApp();
  const report = useEngine<any>("env.report", {}, { timeout: 180_000 });
  const interpreters = useEngine<any[]>("interpreters:list", {}, { timeout: 120_000 });
  const [cuda, setCuda] = useState(false);
  const [basePython, setBasePython] = useState("");
  const [starting, setStarting] = useState(false);
  const [probe, setProbe] = useState<any>(null);
  const router = useRouter();

  const data = report.data ?? environment;
  const packages = Object.entries(data?.packages ?? {}) as Array<[string, { installed: boolean; version: string | null }]>;

  const startInstall = async () => {
    setStarting(true);
    try {
      const job = await api.jobs.start({
        kind: "install",
        workspace: settings?.workspace,
        cuda,
        base_python: basePython || null,
      });
      toast({
        title: "Runtime installation started",
        body: `Job ${job.jobId} is downloading into ${data?.install_plan?.venv ?? "the workspace environment"}.`,
        tone: "info",
      });
      await refreshJobs();
      router.navigate("/jobs");
    } catch (error: any) {
      toast({
        title: "The installation could not be started",
        body: error?.structured?.message ?? String(error?.message ?? error),
        hint: error?.structured?.hint,
        detail: error?.structured?.detail,
        tone: "danger",
      });
    } finally {
      setStarting(false);
    }
  };

  const probeInterpreter = async (executable: string) => {
    const result = await api.probeInterpreter(executable).catch((error) => ({ error: error?.structured }));
    setProbe(result);
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Environment</h1>
          <p className="mt-0.5 max-w-3xl text-xs text-ink-2">
            The engine runs inside a Python interpreter. This page shows which interpreters exist on this
            machine, what is installed in the active one, and performs a real installation into the app's own
            environment when you ask for it — never into the system Python.
          </p>
        </div>
        <RefreshButton onClick={() => void report.reload()} loading={report.loading} />
      </div>

      {report.error ? <ErrorPanel error={report.error} onRetry={report.reload} /> : null}
      {!data ? <Loading label="Reading interpreters and packages…" lines={5} /> : null}

      {data ? (
        <>
          <div className="grid gap-3 lg:grid-cols-4">
            <Stat label="Engine interpreter" value={`python ${data.engine_python.version}`} hint={data.engine_python.in_venv ? "inside a virtual environment" : "system interpreter"} />
            <Stat label="Transformers training" value={data.ready.transformers_training ? "ready" : "not ready"} tone={data.ready.transformers_training ? "ok" : "warn"} />
            <Stat label="Adapter training" value={data.ready.peft ? "ready" : "needs peft"} tone={data.ready.peft ? "ok" : "warn"} />
            <Stat label="App environment" value={data.workspace_venv.exists ? "created" : "not created"} hint={bytes(null)} />
          </div>

          {data.recommendations?.length ? (
            <Callout tone="warn" title="What this runtime still needs">
              <ul className="space-y-1">
                {data.recommendations.map((note: string) => (
                  <li key={note}>· {note}</li>
                ))}
              </ul>
            </Callout>
          ) : (
            <Callout tone="ok" title="Runtime is complete">
              Every capability the app exposes is backed by an installed package in the active interpreter.
            </Callout>
          )}

          <Panel>
            <SectionHeader
              title="Python interpreters found"
              subtitle="Detected on PATH, through the Windows launcher and in the usual install locations"
              icon={<Cpu size={13} className="text-accent" />}
              actions={<RefreshButton onClick={() => void interpreters.reload()} loading={interpreters.loading} />}
            />
            <Table>
              <thead>
                <tr>
                  <Th>Interpreter</Th>
                  <Th>Version</Th>
                  <Th>PyTorch</Th>
                  <Th>Transformers</Th>
                  <Th>Engine importable</Th>
                  <Th>Virtual env</Th>
                  <Th align="right">Use</Th>
                </tr>
              </thead>
              <tbody>
                {(interpreters.data ?? []).map((item: any) => (
                  <tr key={item.executable}>
                    <Td>
                      <div className="max-w-[320px] truncate font-mono text-2xs" title={item.executable}>
                        {item.executable}
                      </div>
                      {item.torch_note ? <div className="text-2xs text-warn">{item.torch_note}</div> : null}
                    </Td>
                    <Td>{item.version}</Td>
                    <Td>
                      <Badge tone={item.torch ? "ok" : "muted"}>{item.torch ? "installed" : "missing"}</Badge>
                    </Td>
                    <Td>
                      <Badge tone={item.transformers ? "ok" : "muted"}>{item.transformers ? "installed" : "missing"}</Badge>
                    </Td>
                    <Td>
                      <Badge tone={item.zxtrain ? "ok" : "muted"}>{item.zxtrain ? "yes" : "no"}</Badge>
                    </Td>
                    <Td>{item.inVenv ? "yes" : "no"}</Td>
                    <Td align="right">
                      <div className="flex justify-end gap-1">
                        <Button size="sm" variant="ghost" onClick={() => void probeInterpreter(item.executable)}>
                          Probe
                        </Button>
                        <Button
                          size="sm"
                          variant="subtle"
                          onClick={async () => {
                            await saveSettings({ pythonPath: item.executable });
                            toast({ title: "Interpreter selected", body: item.executable, tone: "info" });
                            await report.reload();
                          }}
                        >
                          Use
                        </Button>
                      </div>
                    </Td>
                  </tr>
                ))}
              </tbody>
            </Table>
            {probe ? (
              <div className="mt-3 rounded-md border border-line-soft bg-surface-2 p-2.5 text-2xs">
                <div className="mb-1 uppercase tracking-wide text-ink-3">Probe result</div>
                <pre className="overflow-auto font-mono text-2xs text-ink-1">{JSON.stringify(probe, null, 2)}</pre>
              </div>
            ) : null}
            <div className="mt-2 text-2xs text-ink-3">
              Active interpreter: <span className="font-mono">{data.engine_python.executable}</span>
            </div>
          </Panel>

          <div className="grid gap-3 lg:grid-cols-2">
            <Panel>
              <SectionHeader title="Packages" subtitle="Versions read from the interpreter itself" icon={<Package size={13} className="text-accent" />} />
              <Table>
                <thead>
                  <tr>
                    <Th>Package</Th>
                    <Th>State</Th>
                    <Th>Version</Th>
                  </tr>
                </thead>
                <tbody>
                  {packages.map(([name, info]) => (
                    <tr key={name}>
                      <Td>{name}</Td>
                      <Td>
                        <Badge tone={info.installed ? "ok" : "muted"}>{info.installed ? "installed" : "missing"}</Badge>
                      </Td>
                      <Td>{info.version ?? "—"}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            </Panel>

            <Panel>
              <SectionHeader
                title="Install the ML runtime"
                subtitle="Downloads PyTorch and the training stack into the app's own environment"
                icon={<Zap size={13} className="text-accent" />}
              />
              <div className="space-y-3">
                <Field label="Base interpreter" hint="PyTorch supports Python 3.9 – 3.13. Pick an interpreter with matching wheels.">
                  <Select value={basePython} onChange={(event) => setBasePython(event.target.value)}>
                    <option value="">Engine interpreter ({data.engine_python.version})</option>
                    {(interpreters.data ?? []).map((item: any) => (
                      <option key={item.executable} value={item.executable}>
                        python {item.version} — {item.executable}
                      </option>
                    ))}
                  </Select>
                </Field>
                <Toggle
                  checked={cuda}
                  onChange={setCuda}
                  label="Install the CUDA build"
                  hint={
                    data.engine_python
                      ? "Only useful on a machine with an NVIDIA GPU. The CUDA wheels are several gigabytes."
                      : undefined
                  }
                />
                <div className="rounded-md border border-line-soft bg-surface-2 p-2.5">
                  <div className="mb-1 text-2xs uppercase tracking-wide text-ink-3">Commands that will run</div>
                  <pre className="max-h-40 overflow-auto whitespace-pre-wrap font-mono text-2xs text-ink-1">
                    {(data.install_plan?.steps ?? []).join("\n")}
                  </pre>
                </div>
                <div className="space-y-1">
                  {(data.install_plan?.notes ?? []).map((note: string) => (
                    <div key={note} className="text-2xs text-ink-2">
                      · {note}
                    </div>
                  ))}
                </div>
                <Button variant="primary" loading={starting} onClick={() => void startInstall()} icon={<Zap size={13} />}>
                  Start installation as a job
                </Button>
                <div className="text-2xs text-ink-3">
                  The install runs as a normal job, so it streams into the Jobs page and survives a UI reload.
                </div>
              </div>
            </Panel>
          </div>

          <Panel>
            <SectionHeader title="Capability readiness" subtitle="Read from real imports in the active interpreter" icon={<ShieldCheck size={13} className="text-accent" />} />
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {Object.entries(data.ready ?? {}).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between rounded-md border border-line-soft bg-surface-2 px-2.5 py-2">
                  <span className="text-xs capitalize">{key.replace(/_/g, " ")}</span>
                  <Badge tone={value ? "ok" : "muted"}>{value ? "ready" : "unavailable"}</Badge>
                </div>
              ))}
            </div>
            <div className="mt-3 text-2xs text-ink-3">
              App version {appInfo?.version} · engine directory {appInfo?.engineDir}
            </div>
          </Panel>
        </>
      ) : null}
    </div>
  );
}
