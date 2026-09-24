import { useEffect, useMemo, useRef, useState } from "react";
import { Activity, FileText, FolderOpen, Pause, Play, Save, Shield, Square } from "lucide-react";
import { LineChart, UsageBar } from "../components/charts";
import { api } from "../lib/api";
import { basename, bytes, duration, number } from "../lib/format";
import { useApp } from "../state/app";
import { useRouter } from "../state/router";
import {
  Badge,
  Button,
  Callout,
  CodeBlock,
  KeyValue,
  Panel,
  ProgressBar,
  SectionHeader,
  Stat,
  Tab,
  Table,
  Tabs,
  Td,
  Th,
  cx,
} from "../components/ui";
import { ErrorPanel, JobStateBadge, Loading, NotFound, PathText, useInterval } from "./common";

export function TrainingRunPage({ jobId }: { jobId: string }) {
  const { jobs, live, refreshJobs, toast, reportError } = useApp();
  const router = useRouter();
  const [tab, setTab] = useState("monitor");
  const [events, setEvents] = useState<any[]>([]);
  const [metrics, setMetrics] = useState<any[]>([]);
  const [logs, setLogs] = useState("");
  const [busy] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);
  const job = jobs.find((item) => item.job_id === jobId) ?? null;

  // Load the full event/metric history once, then keep appending live events.
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const [loadedEvents, loadedMetrics, loadedLogs] = await Promise.all([
          api.jobs.events(jobId, 600),
          api.jobs.metrics(jobId, 5000),
          api.jobs.logs(jobId, 400),
        ]);
        if (!active) return;
        setEvents(loadedEvents ?? []);
        setMetrics(loadedMetrics ?? []);
        setLogs(loadedLogs ?? "");
      } catch (error) {
        reportError(error, "Could not read the job history");
      }
    })();
    return () => {
      active = false;
    };
  }, [jobId, reportError]);

  useEffect(() => {
    return api.jobs.onEvent(({ jobId: id, event }) => {
      if (id !== jobId) return;
      setEvents((current) => [...current.slice(-1200), event]);
      if (event.type === "metrics") setMetrics((current) => [...current.slice(-5000), event]);
      if (["log", "raw", "error", "process", "done"].includes(event.type)) {
        setLogs((current) => {
          const line = event.message ?? JSON.stringify(event);
          return `${current}\n${line}`.split("\n").slice(-500).join("\n");
        });
      }
    });
  }, [jobId]);

  useInterval(() => {
    void (async () => {
      try {
        setLogs(await api.jobs.logs(jobId, 400));
      } catch {
        /* the job directory may still be being created */
      }
    })();
  }, 4000);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs, tab]);

  const series = useMemo(() => {
    const loss = metrics
      .filter((entry) => entry.loss !== undefined && entry.loss !== null)
      .map((entry) => ({ x: entry.step ?? 0, y: entry.loss }));
    return {
      loss,
      eval: metrics
        .filter((entry) => entry.eval_loss !== undefined && entry.eval_loss !== null)
        .map((entry) => ({ x: entry.step ?? 0, y: entry.eval_loss })),
      lr: metrics
        .filter((entry) => entry.learning_rate !== undefined && entry.learning_rate !== null)
        .map((entry) => ({ x: entry.step ?? 0, y: entry.learning_rate })),
      throughput: metrics
        .filter((entry) => entry.tokens_per_second !== undefined && entry.tokens_per_second !== null)
        .map((entry) => ({ x: entry.step ?? 0, y: entry.tokens_per_second })),
      gradNorm: metrics
        .filter((entry) => entry.grad_norm !== undefined && entry.grad_norm !== null)
        .map((entry) => ({ x: entry.step ?? 0, y: entry.grad_norm })),
    };
  }, [metrics]);

  if (!job) {
    return <NotFound what="This run" onBack={() => router.navigate("/training")} />;
  }

  const control = async (patch: Record<string, unknown>) => {
    try {
      await api.jobs.control(jobId, patch);
      await refreshJobs();
    } catch (error) {
      reportError(error, "Could not send the control signal");
    }
  };

  const gpu = live?.gpus?.[0] ?? null;
  const finished = ["completed", "failed", "cancelled", "interrupted"].includes(job.state);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-2xs uppercase tracking-wider text-ink-3">
            {job.kind ?? "train"} · {job.backend ?? "auto"} · {job.method ?? "method"}
          </div>
          <h1 className="text-lg font-semibold tracking-tight">{job.job_id}</h1>
          <div className="mt-0.5 flex flex-wrap items-center gap-2">
            <JobStateBadge state={job.state} />
            {job.attached ? <Badge tone="info">attached to live process</Badge> : null}
            {job.model ? <Badge tone="muted">{basename(job.model)}</Badge> : null}
            {(job.datasets ?? []).map((dataset) => (
              <Badge key={dataset} tone="muted">
                {basename(dataset)}
              </Badge>
            ))}
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {["running", "paused"].includes(job.state) ? (
            <>
              <Button icon={job.state === "paused" ? <Play size={13} /> : <Pause size={13} />} onClick={() => void control({ pause: job.state !== "paused" })}>
                {job.state === "paused" ? "Resume" : "Pause"}
              </Button>
              <Button icon={<Save size={13} />} onClick={() => void control({ save_now: true })}>
                Checkpoint now
              </Button>
              <Button
                variant="danger"
                icon={<Square size={13} />}
                onClick={async () => {
                  if (!window.confirm("Stop this run? A checkpoint is written if the backend supports it.")) return;
                  await control({ stop: true });
                }}
              >
                Stop
              </Button>
            </>
          ) : null}
          {finished && job.state !== "completed" ? (
            <Button
              variant="primary"
              onClick={() =>
                router.navigate(
                  `/training/new?resume=${encodeURIComponent((job.checkpoints ?? []).slice(-1)[0]?.path ?? "")}&parent=${encodeURIComponent(job.job_id)}`,
                )
              }
            >
              Resume from last checkpoint
            </Button>
          ) : null}
          {job.state === "completed" ? (
            <>
              <Button
                variant="primary"
                onClick={() => router.navigate(`/evaluation?model=${encodeURIComponent(job.result?.model_dir ?? "")}`)}
              >
                Evaluate the result
              </Button>
              <Button onClick={() => router.navigate(`/playground?model=${encodeURIComponent(job.result?.model_dir ?? "")}`)}>
                Open in playground
              </Button>
              <Button onClick={() => router.navigate(`/compare?left=${encodeURIComponent(job.model ?? "")}&right=${encodeURIComponent(job.result?.model_dir ?? "")}`)}>
                Compare with base
              </Button>
            </>
          ) : null}
          <Button
            icon={<FolderOpen size={13} />}
            onClick={async () => {
              try {
                const output = job.result?.model_dir ?? job.result?.output_dir;
                if (output) await api.shell.reveal(output);
                else if (job.job_dir) await api.shell.reveal(job.job_dir);
              } catch (error) {
                reportError(error, "Could not open the folder");
              }
            }}
          >
            Open output folder
          </Button>
        </div>
      </div>

      {job.error ? <ErrorPanel error={job.error} /> : null}

      <div className="grid gap-3 lg:grid-cols-5">
        <Stat label="Step" value={job.total_steps ? `${job.step ?? 0} / ${job.total_steps}` : "—"} hint={job.epoch ? `epoch ${job.epoch}` : undefined} />
        <Stat label="Loss" value={job.loss !== null && job.loss !== undefined ? job.loss.toFixed(4) : "—"} />
        <Stat label="Eval loss" value={job.eval_loss !== null && job.eval_loss !== undefined ? job.eval_loss.toFixed(4) : "—"} />
        <Stat label="Throughput" value={job.tokens_per_second ? `${number(job.tokens_per_second, 1)} tok/s` : "—"} />
        <Stat label="Elapsed / ETA" value={duration(job.elapsed_seconds ?? null)} hint={job.eta_seconds ? `ETA ${duration(job.eta_seconds)}` : undefined} />
      </div>

      {job.total_steps ? (
        <Panel>
          <ProgressBar
            value={job.step ?? 0}
            max={job.total_steps}
            tone={job.state === "failed" ? "danger" : job.state === "paused" ? "warn" : "accent"}
            label={`${number(job.step ?? 0)} of ${number(job.total_steps)} optimizer steps`}
          />
        </Panel>
      ) : null}

      <Tabs value={tab} onChange={setTab}>
        <Tab value="monitor">Monitor</Tab>
        <Tab value="checkpoints" count={(job.checkpoints ?? []).length}>
          Checkpoints
        </Tab>
        <Tab value="logs">Logs</Tab>
        <Tab value="config">Configuration</Tab>
        <Tab value="result">Result</Tab>
      </Tabs>

      {tab === "monitor" ? (
        <div className="grid gap-3 lg:grid-cols-2">
          <Panel className="lg:col-span-2">
            <SectionHeader
              title="Loss"
              subtitle={series.loss.length ? `${series.loss.length} logged points from the training process` : "Waiting for the first metrics"}
              icon={<Activity size={13} className="text-accent" />}
            />
            <LineChart
              height={220}
              emptyLabel="No metrics yet — the chart fills as soon as the engine reports a step"
              series={[
                { name: "loss", color: "rgb(var(--accent))", points: series.loss },
                { name: "eval loss", color: "rgb(var(--warn))", points: series.eval },
              ]}
              formatY={(value) => value.toFixed(3)}
            />
          </Panel>
          <Panel>
            <SectionHeader title="Learning rate" subtitle="Exactly the value the optimizer used" />
            <LineChart
              height={150}
              series={[{ name: "learning rate", color: "rgb(var(--ok))", points: series.lr }]}
              formatY={(value) => value.toExponential(2)}
            />
          </Panel>
          <Panel>
            <SectionHeader title="Throughput" subtitle="Tokens per second reported per logging step" />
            <LineChart
              height={150}
              series={[{ name: "tokens/s", color: "rgb(var(--info))", points: series.throughput }]}
              formatY={(value) => value.toFixed(0)}
            />
          </Panel>
          <Panel>
            <SectionHeader title="Gradient norm" subtitle="Before clipping; spikes usually precede loss spikes" />
            <LineChart
              height={150}
              series={[{ name: "grad norm", color: "rgb(var(--accent))", points: series.gradNorm }]}
              formatY={(value) => value.toFixed(2)}
            />
          </Panel>
          <Panel>
            <SectionHeader title="Machine while training" subtitle="Sampled live, not from the training process" />
            <div className="space-y-3">
              <UsageBar label="System memory" value={live?.ram?.used ?? null} max={live?.ram?.total ?? null} format={(value) => bytes(value, 1)} />
              {gpu ? (
                <UsageBar
                  label={`${gpu.name} VRAM`}
                  value={gpu.memory_used_mb ? gpu.memory_used_mb * 1024 * 1024 : null}
                  max={gpu.memory_total}
                  format={(value) => bytes(value, 1)}
                />
              ) : (
                <div className="text-2xs text-ink-2">
                  No CUDA device: this run is using the CPU, so system RAM is the constraint that matters.
                </div>
              )}
              <UsageBar label="CPU load" value={live?.cpu?.percent ?? null} max={100} format={(value) => `${value.toFixed(0)}%`} />
              <div className="text-2xs text-ink-3">
                Engine process memory: {bytes(live?.process?.rss ?? null)} · {live?.gpus?.length ? `${live.gpus.length} GPU(s) visible` : "CPU only"}
              </div>
            </div>
          </Panel>
        </div>
      ) : null}

      {tab === "checkpoints" ? (
        <Panel padded={false}>
          <div className="p-3">
            <SectionHeader
              title="Checkpoints"
              subtitle="Every row is a real folder the engine wrote, including its state files"
              actions={
                <Button size="sm" variant="subtle" onClick={() => void refreshJobs()}>
                  Refresh
                </Button>
              }
            />
          </div>
          {!(job.checkpoints ?? []).length ? (
            <div className="p-4 text-2xs text-ink-3">
              No checkpoints yet. They appear as soon as the run reaches its save interval.
            </div>
          ) : (
            <Table>
              <thead>
                <tr>
                  <Th>Checkpoint</Th>
                  <Th align="right">Step</Th>
                  <Th>Kind</Th>
                  <Th align="right">Actions</Th>
                </tr>
              </thead>
              <tbody>
                {(job.checkpoints ?? []).map((checkpoint) => (
                  <tr key={checkpoint.path}>
                    <Td>
                      <div className="text-xs">{checkpoint.name}</div>
                      <PathText value={checkpoint.path} />
                    </Td>
                    <Td align="right">{checkpoint.step ?? "—"}</Td>
                    <Td>
                      <Badge tone={checkpoint.kind === "final" ? "ok" : "muted"}>{checkpoint.kind}</Badge>
                    </Td>
                    <Td align="right">
                      <div className="flex justify-end gap-1">
                        <Button size="sm" variant="ghost" onClick={() => api.shell.reveal(checkpoint.path)}>
                          Reveal
                        </Button>
                        <Button
                          size="sm"
                          variant="subtle"
                          onClick={() => router.navigate(`/training/new?resume=${encodeURIComponent(checkpoint.path)}&parent=${encodeURIComponent(jobId)}`)}
                        >
                          Continue from here
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={async () => {
                            try {
                              const result = await api.call<any>("models.card", { path: checkpoint.path, write: true });
                              toast({ title: "Checkpoint card written", body: result.written ? "README.md created next to the weights." : "", tone: "ok" });
                            } catch (error) {
                              reportError(error, "Could not write the checkpoint card");
                            }
                          }}
                        >
                          Card
                        </Button>
                      </div>
                    </Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          )}
        </Panel>
      ) : null}

      {tab === "logs" ? (
        <Panel padded={false}>
          <div className="flex items-center justify-between border-b border-line-soft p-3">
            <SectionHeader title="Event stream" subtitle={`${events.length} events · console.log is the source of truth`} icon={<FileText size={13} className="text-accent" />} />
            <div className="flex gap-2">
              <Button size="sm" variant="subtle" onClick={() => void navigator.clipboard.writeText(logs)}>
                Copy console log
              </Button>
              <Button size="sm" variant="subtle" onClick={() => api.shell.reveal(`${job.job_dir}\\console.log`)}>
                Reveal log file
              </Button>
            </div>
          </div>
          <div className="grid gap-0 lg:grid-cols-2">
            <div ref={logRef} className="max-h-[420px] overflow-auto border-r border-line-soft p-3 font-mono text-2xs leading-relaxed text-ink-1">
              {logs || "No console output yet."}
            </div>
            <div className="max-h-[420px] overflow-auto p-3">
              {events.slice(-200).map((event, index) => (
                <div key={index} className="mb-1 flex items-start gap-2 border-b border-line-soft/40 pb-1">
                  <Badge tone={event.type === "error" ? "danger" : event.type === "done" ? "ok" : "muted"}>{event.type}</Badge>
                  <span className={cx("min-w-0 flex-1 truncate text-2xs", event.type === "error" && "text-danger")} title={JSON.stringify(event)}>
                    {event.message ?? JSON.stringify(event).slice(0, 180)}
                  </span>
                </div>
              ))}
              {!events.length ? <div className="text-2xs text-ink-3">No events recorded yet.</div> : null}
            </div>
          </div>
        </Panel>
      ) : null}

      {tab === "config" ? (
        <div className="grid gap-3 lg:grid-cols-2">
          <Panel>
            <SectionHeader title="Requested configuration" subtitle="Exactly what was written into spec.json" />
            <CodeBlock max="max-h-[520px]">{JSON.stringify(job.config ?? {}, null, 2)}</CodeBlock>
          </Panel>
          <Panel>
            <SectionHeader title="Effective settings" subtitle="Values the engine actually used, from its own metrics" />
            <KeyValue
              items={[
                ["Backend", job.backend ?? "auto"],
                ["Method", job.method ?? "—"],
                ["Total steps", job.total_steps ? number(job.total_steps) : "unknown"],
                ["Effective batch", job.metrics_state?.effective_batch_size ? number(job.metrics_state.effective_batch_size as number) : "see plan"],
                ["Precision", String(job.config?.precision ?? "—")],
                ["Quantisation", String(job.config?.quantization ?? "none")],
                ["Sequence length", String(job.config?.sequence_length ?? "—")],
                ["Learning rate", String(job.config?.learning_rate ?? "—")],
                ["Seed", String(job.config?.seed ?? "—")],
              ]}
            />
            <Callout tone="info" title="Metric history is stored on disk">
              metrics.jsonl in the job folder holds every logged point, so charts survive restarts. Export a
              reproducibility bundle to also capture package versions and hardware.
            </Callout>
            <Button
              size="sm"
              variant="subtle"
              onClick={async () => {
                try {
                  const destination = await api.dialog.saveFile({
                    title: "Save reproducibility bundle",
                    defaultPath: `${job.job_dir}\\reproducibility.json`,
                  });
                  if (!destination) return;
                  const bundle = await api.call<any>("reproducibility.bundle", { job_dir: job.job_dir, write: true, output: destination });
                  toast({ title: "Reproducibility bundle written", body: bundle.written_to, tone: "ok" });
                } catch (error) {
                  reportError(error, "Could not write the bundle");
                }
              }}
            >
              Export reproducibility bundle
            </Button>
          </Panel>
        </div>
      ) : null}

      {tab === "result" ? (
        <div className="space-y-3">
          {job.result ? (
            <>
              <Panel>
                <SectionHeader title="Run result" subtitle={job.result.lineage_note ?? ""} />
                <KeyValue
                  items={[
                    ["Status", job.result.status],
                    ["Backend", job.result.backend],
                    ["Steps", number(job.result.steps ?? 0)],
                    ["Epochs", String(job.result.epochs ?? "—")],
                    ["Final loss", job.result.final_loss !== undefined && job.result.final_loss !== null ? Number(job.result.final_loss).toFixed(5) : "—"],
                    ["Eval loss", job.result.eval_loss !== undefined && job.result.eval_loss !== null ? Number(job.result.eval_loss).toFixed(5) : job.result.final_eval_loss !== undefined && job.result.final_eval_loss !== null ? Number(job.result.final_eval_loss).toFixed(5) : "—"],
                    ["Perplexity", job.result.perplexity ? String(job.result.perplexity) : "—"],
                    ["Parameters", job.result.parameters ? number(job.result.parameters) : "—"],
                    ["Tokens seen", job.result.tokens_seen ? number(job.result.tokens_seen) : "—"],
                    ["Duration", duration(job.result.duration_seconds ?? null)],
                    ["Output", job.result.model_dir ?? job.result.output_dir],
                  ]}
                />
                <div className="mt-3 flex flex-wrap gap-2">
                  <Button onClick={() => api.shell.reveal(job.result.model_dir ?? job.result.output_dir)}>Reveal output</Button>
                  <Button
                    onClick={async () => {
                      try {
                        await api.registry.add("models", {
                          id: `${Date.now().toString(36)}`,
                          name: basename(job.result.model_dir ?? job.result.output_dir ?? jobId),
                          path: job.result.model_dir ?? job.result.output_dir,
                          addedAt: new Date().toISOString(),
                          kind: job.result.backend === "tiny" ? "zx-tiny" : "transformers-safetensors",
                          summary: { parameters: job.result.parameters ?? null, architecture: job.result.backend },
                        });
                        toast({ title: "Result added to the model library", tone: "ok" });
                      } catch (error) {
                        reportError(error, "Could not add the model");
                      }
                    }}
                  >
                    Add to model library
                  </Button>
                  <Button
                    onClick={async () => {
                      try {
                        await api.call("models.card", { path: job.result.model_dir ?? job.result.output_dir, write: true });
                        toast({ title: "Model card written", tone: "ok" });
                      } catch (error) {
                        reportError(error, "Could not write the model card");
                      }
                    }}
                  >
                    Write model card
                  </Button>
                </div>
              </Panel>
              {job.result.history?.length ? (
                <Panel>
                  <SectionHeader title="Full metric history" subtitle="As recorded in metrics.jsonl" />
                  <CodeBlock max="max-h-72">{JSON.stringify(job.result.history.slice(-40), null, 2)}</CodeBlock>
                </Panel>
              ) : null}
            </>
          ) : (
            <Panel>
              <Callout tone="info" title="No result yet">
                When the run finishes, this tab shows its measured outcome and the follow-up actions: validate
                the checkpoint, evaluate it, compare it with the base model or write its model card.
              </Callout>
            </Panel>
          )}
          <Panel>
            <SectionHeader title="Next steps" subtitle="Each of these runs against the real artefact" icon={<Shield size={13} className="text-accent" />} />
            <div className="grid gap-2 md:grid-cols-3">
              <Button disabled={busy} onClick={() => router.navigate(`/evaluation?model=${encodeURIComponent(job.result?.model_dir ?? "")}`)}>
                Evaluate checkpoint
              </Button>
              <Button disabled={busy} onClick={() => router.navigate(`/playground?model=${encodeURIComponent(job.result?.model_dir ?? "")}`)}>
                Sample chat
              </Button>
              <Button disabled={busy} onClick={() => router.navigate(`/compare?left=${encodeURIComponent(job.model ?? "")}&right=${encodeURIComponent(job.result?.model_dir ?? "")}`)}>
                Compare with parent
              </Button>
            </div>
          </Panel>
        </div>
      ) : null}

      {!events.length && job.state === "running" ? <Loading label="Waiting for the engine to report progress…" /> : null}
    </div>
  );
}
