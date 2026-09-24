import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, FolderOpen, Pause, Play, RefreshCw, RotateCcw, Square, Terminal, X } from "lucide-react";
import { api } from "../lib/api";
import { clock, duration } from "../lib/format";
import { useApp } from "../state/app";
import { useRouter } from "../state/router";
import {
  Badge,
  Button,
  Callout,
  CodeBlock,
  CopyButton,
  EmptyState,
  Panel,
  ProgressBar,
  SectionHeader,
  Select,
  Stat,
  Table,
  Td,
  Th,
  cx,
} from "../components/ui";
import { ErrorPanel, JobRow, JobStateBadge, useEngine, useInterval } from "./common";

const STATE_FILTERS = ["all", "running", "queued", "paused", "completed", "failed", "cancelled"];

export function JobsPage() {
  const { jobs, refreshJobs, jobsLoading, toast, reportError } = useApp();
  const router = useRouter();
  const [filter, setFilter] = useState("all");
  const [selected, setSelected] = useState<string | null>(null);
  const [logs, setLogs] = useState<string>("");
  const engineIndex = useEngine<any>("jobs.list", {}, { timeout: 120_000 });
  const reconcile = useEngine<any>("jobs.reconcile", {}, { timeout: 120_000 });

  useInterval(() => {
    void refreshJobs();
  }, 8000);

  const list = useMemo(() => {
    const shown = filter === "all" ? jobs : jobs.filter((job) => job.state === filter);
    return [...shown].sort((a, b) => String(b.updated_at ?? b.started_at ?? "").localeCompare(String(a.updated_at ?? a.started_at ?? "")));
  }, [jobs, filter]);

  const current = selected ? jobs.find((job) => job.job_id === selected) ?? null : null;

  useEffect(() => {
    if (!selected) {
      setLogs("");
      return;
    }
    let cancelled = false;
    const load = async () => {
      try {
        const text = await api.jobs.logs(selected, 500);
        if (!cancelled) setLogs(text ?? "");
      } catch (error) {
        if (!cancelled) setLogs(`(the log could not be read)\n${String((error as Error).message)}`);
      }
    };
    void load();
    const timer = window.setInterval(load, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [selected]);

  const control = async (jobId: string, patch: Record<string, unknown>) => {
    try {
      await api.jobs.control(jobId, patch);
      await refreshJobs();
      toast({
        title: patch.pause ? "Pause requested" : patch.stop ? "Stop requested" : "Request sent",
        body: "The engine reads the request at the next step boundary, so state changes are recorded on disk.",
        tone: "info",
      });
    } catch (error) {
      reportError(error, "The job did not accept the request");
    }
  };

  const terminate = async (jobId: string) => {
    if (!window.confirm(`Terminate ${jobId}?\n\nThe process is killed. Any checkpoint already written stays usable, and the engine will report the run as interrupted.`)) return;
    try {
      await api.jobs.terminate(jobId, true);
      toast({ title: "Process terminated", body: "Checkpoints written before the kill remain on disk.", tone: "warn" });
      await refreshJobs();
    } catch (error) {
      reportError(error, "Could not terminate the process");
    }
  };

  const counts = jobs.reduce<Record<string, number>>((accumulator, job) => {
    accumulator[job.state] = (accumulator[job.state] ?? 0) + 1;
    return accumulator;
  }, {});
  const engineJobs: any[] = engineIndex.data?.jobs ?? [];
  const findings: any[] = (reconcile.data?.findings ?? []).filter((entry: any) => entry.action !== "healthy");

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Jobs</h1>
          <p className="mt-0.5 max-w-3xl text-xs text-ink-2">
            Every long operation runs in its own engine process. State lives on disk, so a job survives an app
            restart — when the app comes back it re-attaches to the process, or marks it interrupted and offers a
            resume from the last real checkpoint.
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" icon={<RefreshCw size={12} />} loading={jobsLoading} onClick={() => void refreshJobs()}>
            Refresh
          </Button>
          <Button size="sm" variant="subtle" loading={engineIndex.loading} onClick={() => void engineIndex.reload()}>
            Rescan job folders
          </Button>
        </div>
      </div>

      {engineIndex.error ? <ErrorPanel error={engineIndex.error} onRetry={() => void engineIndex.reload()} /> : null}

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Running" value={counts.running ?? jobs.filter((job) => job.state === "running").length} tone="info" />
        <Stat label="Queued" value={counts.queued ?? jobs.filter((job) => job.state === "queued").length} />
        <Stat label="Completed" value={counts.completed ?? jobs.filter((job) => job.state === "completed").length} tone="ok" />
        <Stat
          label="Failed / interrupted"
          value={(counts.failed ?? 0) + (counts.interrupted ?? 0)}
          tone={(counts.failed ?? 0) + (counts.interrupted ?? 0) ? "danger" : "muted"}
        />
      </div>

      {engineJobs.length > jobs.length ? (
        <Callout
          tone="info"
          title="Job folders on disk"
          hint={`The engine found ${engineJobs.length} job director${engineJobs.length === 1 ? "y" : "ies"} in the workspace; the list above shows the ${jobs.length} the desktop process knows about.`}
        >
          Use “Rescan job folders” after copying a run in from elsewhere.
        </Callout>
      ) : null}

      <div className="grid gap-3 lg:grid-cols-[420px_1fr]">
        <Panel padded={false} className="flex max-h-[640px] flex-col">
          <div className="flex items-center justify-between gap-2 border-b border-line-soft p-2.5">
            <Select value={filter} onChange={(event) => setFilter(event.target.value)}>
              {STATE_FILTERS.map((state) => (
                <option key={state} value={state}>
                  {state === "all" ? "All states" : state}
                </option>
              ))}
            </Select>
            <Badge tone="muted">{list.length}</Badge>
          </div>
          <div className="flex-1 space-y-1.5 overflow-y-auto p-2">
            {!list.length ? (
              <EmptyState
                title="No jobs yet"
                hint="Start a training run, an evaluation, a conversion or a local server from their pages."
              />
            ) : null}
            {list.map((job) => (
              <div
                key={job.job_id}
                className={cx(
                  "rounded-md border p-1",
                  selected === job.job_id ? "border-accent/40 bg-surface-3" : "border-line-soft bg-surface-2",
                )}
              >
                <JobRow job={job} onOpen={() => setSelected(job.job_id)} />
                {job.total_steps ? (
                  <div className="px-1 pb-1 pt-1.5">
                    <ProgressBar value={job.step ?? 0} max={job.total_steps} />
                  </div>
                ) : null}
              </div>
            ))}
          </div>
        </Panel>

        {!current ? (
          <Panel>
            <EmptyState title="Select a job" hint="Its configuration, live metrics, checkpoints and raw log appear here." />
          </Panel>
        ) : (
          <div className="space-y-3">
            <Panel>
              <SectionHeader
                title={current.job_id}
                subtitle={`${current.kind ?? "job"} · ${current.method ?? "n/a"} · ${current.backend ?? "auto"}`}
                actions={
                  <div className="flex flex-wrap gap-2">
                    <JobStateBadge state={current.state} />
                    <Button
                      size="sm"
                      variant="subtle"
                      icon={<Pause size={11} />}
                      disabled={current.state !== "running"}
                      onClick={() => void control(current.job_id, { pause: true })}
                    >
                      Pause
                    </Button>
                    <Button
                      size="sm"
                      variant="subtle"
                      icon={<Play size={11} />}
                      disabled={current.state !== "paused"}
                      onClick={() => void control(current.job_id, { pause: false })}
                    >
                      Resume
                    </Button>
                    <Button
                      size="sm"
                      variant="subtle"
                      icon={<RotateCcw size={11} />}
                      disabled={current.state !== "running"}
                      onClick={() => void control(current.job_id, { save_now: true })}
                    >
                      Checkpoint now
                    </Button>
                    <Button
                      size="sm"
                      variant="subtle"
                      icon={<Square size={11} />}
                      disabled={["completed", "failed", "cancelled"].includes(current.state)}
                      onClick={() => void control(current.job_id, { stop: true })}
                    >
                      Stop safely
                    </Button>
                    <Button size="sm" variant="danger" icon={<X size={11} />} onClick={() => void terminate(current.job_id)}>
                      Kill
                    </Button>
                  </div>
                }
              />
              <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <Stat label="Step" value={current.step ?? "—"} hint={current.total_steps ? `of ${current.total_steps}` : undefined} />
                <Stat
                  label="Loss"
                  value={typeof current.loss === "number" ? current.loss.toFixed(4) : "—"}
                  hint={typeof current.eval_loss === "number" ? `eval ${current.eval_loss.toFixed(4)}` : undefined}
                />
                <Stat label="Tokens / s" value={typeof current.tokens_per_second === "number" ? current.tokens_per_second.toFixed(1) : "—"} />
                <Stat label="Elapsed" value={duration(current.elapsed_seconds)} hint={current.eta_seconds ? `ETA ${duration(current.eta_seconds)}` : undefined} />
              </div>
              <div className="mt-3 text-xs text-ink-2">{current.message ?? "No message recorded yet."}</div>
              <div className="mt-3 flex flex-wrap gap-2">
                {current.job_dir ? (
                  <>
                    <Button size="sm" variant="subtle" icon={<FolderOpen size={11} />} onClick={() => void api.shell.reveal(current.job_dir!)}>
                      Open job folder
                    </Button>
                    <CopyButton value={current.job_dir} label="Copy job path" />
                  </>
                ) : null}
                {current.model ? (
                  <Button size="sm" variant="subtle" onClick={() => router.navigate(`/playground?model=${encodeURIComponent(current.model!)}`)}>
                    Open in playground
                  </Button>
                ) : null}
                {current.state === "failed" && current.error ? (
                  <Button size="sm" variant="primary" onClick={() => router.navigate(`/training?resume=${current.job_id}`)}>
                    Resume from checkpoint
                  </Button>
                ) : null}
              </div>
              {current.error ? (
                <div className="mt-3">
                  <Callout tone="danger" title={current.error.message} hint={current.error.hint} detail={(current.error as any).detail}>
                    <div className="mt-1 font-mono text-2xs">{current.error.code}</div>
                  </Callout>
                </div>
              ) : null}
            </Panel>

            {current.checkpoints?.length ? (
              <Panel>
                <SectionHeader title="Checkpoints" subtitle="Written by the engine at real save points." />
                <Table>
                  <thead>
                    <tr>
                      <Th>Name</Th>
                      <Th>Step</Th>
                      <Th>Kind</Th>
                      <Th>Path</Th>
                    </tr>
                  </thead>
                  <tbody>
                    {current.checkpoints.map((checkpoint) => (
                      <tr key={checkpoint.path ?? checkpoint.name}>
                        <Td>{checkpoint.name}</Td>
                        <Td align="right">{checkpoint.step ?? "—"}</Td>
                        <Td>{checkpoint.kind ?? "checkpoint"}</Td>
                        <Td>
                          <span className="font-mono text-2xs text-ink-2">{checkpoint.path}</span>
                        </Td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              </Panel>
            ) : null}

            <Panel>
              <SectionHeader
                title="Log"
                subtitle="Raw engine output, including the @@event stream."
                actions={
                  <div className="flex gap-2">
                    <Badge tone="muted">
                      <Terminal size={10} /> tail 500 lines
                    </Badge>
                    <CopyButton value={logs} label="Copy log" />
                  </div>
                }
              />
              <CodeBlock max="max-h-[320px]">{logs || "(no output yet)"}</CodeBlock>
            </Panel>
          </div>
        )}
      </div>

      {findings.length ? (
        <Panel>
          <SectionHeader
            title="Recovery findings"
            subtitle="Runs whose engine process is gone, and what state they were left in."
          />
          <div className="space-y-2">
            {findings.map((finding) => (
              <div key={finding.job_id} className="rounded-md border border-line-soft bg-surface-2 p-2.5">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-mono text-2xs">{finding.job_id}</span>
                  <Badge tone={finding.state === "running" ? "warn" : "muted"}>
                    {finding.action ?? finding.state}
                  </Badge>
                </div>
                <div className="mt-1 text-xs text-ink-2">{finding.message}</div>
                {finding.resumable ? (
                  <div className="mt-2">
                    <Button size="sm" variant="primary" onClick={() => router.navigate(`/training?resume=${finding.job_id}`)}>
                      Resume from last checkpoint
                    </Button>
                  </div>
                ) : null}
              </div>
            ))}
          </div>
        </Panel>
      ) : null}

      <div className="flex items-center gap-2 text-2xs text-ink-3">
        <AlertTriangle size={11} /> Heavy jobs share one device. The engine warns before starting a second training
        process on the same GPU, and queued jobs wait rather than competing for memory.
      </div>
      <div className="text-2xs text-ink-3">
        {engineJobs.length} job folder{engineJobs.length === 1 ? "" : "s"} on disk · refreshed {clock(new Date().toISOString())}
      </div>
    </div>
  );
}
