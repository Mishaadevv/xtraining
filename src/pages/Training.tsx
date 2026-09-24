import { useMemo, useState } from "react";
import { Activity, Play, RotateCcw, Square, Trash2 } from "lucide-react";
import { api } from "../lib/api";
import { basename, clock, duration } from "../lib/format";
import { useApp } from "../state/app";
import { useRouter } from "../state/router";
import { Badge, Button, EmptyState, Panel, SectionHeader, Select, Table, Td, TextInput, Th } from "../components/ui";
import { JobStateBadge, useEngine } from "./common";

export function TrainingPage() {
  const { jobs, refreshJobs, toast, reportError } = useApp();
  const router = useRouter();
  const [filter, setFilter] = useState("all");
  const [query, setQuery] = useState("");
  const backends = useEngine<any>("backends.list", {}, { timeout: 120_000 });

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return jobs.filter((job) => {
      if (filter === "running" && !["running", "queued", "paused"].includes(job.state)) return false;
      if (filter === "resumable" && !(job.state === "interrupted" || job.state === "failed" || job.state === "cancelled")) return false;
      if (filter === "completed" && job.state !== "completed") return false;
      if (filter === "failed" && !["failed", "interrupted"].includes(job.state)) return false;
      if (needle && !`${job.job_id} ${job.method ?? ""} ${job.model ?? ""}`.toLowerCase().includes(needle)) return false;
      return true;
    });
  }, [jobs, filter, query]);

  const running = jobs.filter((job) => ["running", "queued", "paused"].includes(job.state));

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Training</h1>
          <p className="mt-0.5 max-w-3xl text-xs text-ink-2">
            Every run below is a real engine process writing to a real log file. Closing the app does not stop
            it, and reopening the app re-attaches to it.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="primary" icon={<Play size={13} />} onClick={() => router.navigate("/training/new")}>
            New training run
          </Button>
          <Button icon={<RotateCcw size={13} />} onClick={() => void refreshJobs()}>
            Re-scan jobs
          </Button>
        </div>
      </div>

      <div className="grid gap-3 lg:grid-cols-4">
        <Panel quiet className="flex items-center justify-between">
          <div>
            <div className="text-2xs uppercase tracking-wide text-ink-3">Active</div>
            <div className="text-sm">{running.length}</div>
          </div>
          <Activity size={16} className="text-info" />
        </Panel>
        <Panel quiet>
          <div className="text-2xs uppercase tracking-wide text-ink-3">Completed</div>
          <div className="text-sm">{jobs.filter((job) => job.state === "completed").length}</div>
        </Panel>
        <Panel quiet>
          <div className="text-2xs uppercase tracking-wide text-ink-3">Interrupted / failed</div>
          <div className="text-sm">{jobs.filter((job) => ["interrupted", "failed"].includes(job.state)).length}</div>
        </Panel>
        <Panel quiet>
          <div className="text-2xs uppercase tracking-wide text-ink-3">Backends available</div>
          <div className="text-sm">
            {(backends.data?.backends ?? []).filter((backend: any) => backend.available).length} /{" "}
            {(backends.data?.backends ?? []).length}
          </div>
        </Panel>
      </div>

      <Panel padded={false}>
        <div className="flex flex-wrap items-center gap-2 border-b border-line-soft p-3">
          <TextInput className="max-w-xs" placeholder="Filter by id, method or model…" value={query} onChange={(event) => setQuery(event.target.value)} />
          <Select className="w-44" value={filter} onChange={(event) => setFilter(event.target.value)}>
            <option value="all">All runs</option>
            <option value="running">Active</option>
            <option value="resumable">Interrupted / resumable</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </Select>
          <Badge tone="muted">{filtered.length} runs</Badge>
          <span className="ml-auto text-2xs text-ink-3">
            {(backends.data?.methods ?? [])
              .filter((method: any) => method.available)
              .map((method: any) => method.method)
              .slice(0, 6)
              .join(" · ")}
          </span>
        </div>

        {!filtered.length ? (
          <div className="p-4">
            <EmptyState title="No runs match this filter">
              Create a run, or clear the filter. Runs appear here the moment the engine writes their spec.
            </EmptyState>
          </div>
        ) : (
          <Table>
            <thead>
              <tr>
                <Th>Run</Th>
                <Th>State</Th>
                <Th>Progress</Th>
                <Th align="right">Loss</Th>
                <Th>Started</Th>
                <Th align="right">Duration</Th>
                <Th align="right">Actions</Th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((job) => {
                const progress = job.total_steps ? `${job.step ?? 0} / ${job.total_steps}` : "—";
                return (
                  <tr key={job.job_id} className="hover:bg-surface-2">
                    <Td>
                      <button className="text-left" onClick={() => router.navigate(`/training/${encodeURIComponent(job.job_id)}`)}>
                        <div className="text-xs text-ink-0">{job.job_id}</div>
                        <div className="text-2xs text-ink-3">
                          {job.method ?? job.kind} · {job.backend ?? "auto"} · {job.model ? basename(job.model) : "no model"}
                        </div>
                      </button>
                    </Td>
                    <Td>
                      <JobStateBadge state={job.state} />
                    </Td>
                    <Td>
                      <span className="tabular-nums text-2xs">{progress}</span>
                      {job.epoch ? <span className="ml-2 text-2xs text-ink-3">epoch {job.epoch}</span> : null}
                    </Td>
                    <Td align="right">{job.loss !== null && job.loss !== undefined ? job.loss.toFixed(4) : "—"}</Td>
                    <Td>{clock(job.started_at ?? job.created)}</Td>
                    <Td align="right">{duration(job.elapsed_seconds ?? null)}</Td>
                    <Td align="right">
                      <div className="flex justify-end gap-1">
                        {["running", "paused", "queued"].includes(job.state) ? (
                          <>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={async () => {
                                await api.jobs.control(job.job_id, { pause: job.state !== "paused", stop: false });
                                await refreshJobs();
                              }}
                            >
                              {job.state === "paused" ? "Resume" : "Pause"}
                            </Button>
                            <Button
                              size="sm"
                              variant="danger"
                              icon={<Square size={11} />}
                              onClick={async () => {
                                if (!window.confirm(`Stop ${job.job_id}? A checkpoint is saved if the backend supports it.`)) return;
                                await api.jobs.control(job.job_id, { stop: true });
                                await refreshJobs();
                              }}
                            >
                              Stop
                            </Button>
                          </>
                        ) : null}
                        <Button size="sm" variant="subtle" onClick={() => router.navigate(`/training/${encodeURIComponent(job.job_id)}`)}>
                          Open
                        </Button>
                        {["interrupted", "failed", "cancelled"].includes(job.state) ? (
                          <Button
                            size="sm"
                            variant="primary"
                            onClick={() =>
                              router.navigate(
                                `/training/new?resume=${encodeURIComponent((job.checkpoints ?? []).slice(-1)[0]?.path ?? "")}&parent=${encodeURIComponent(job.job_id)}`,
                              )
                            }
                          >
                            Resume
                          </Button>
                        ) : null}
                        <Button
                          size="sm"
                          variant="ghost"
                          icon={<Trash2 size={11} />}
                          onClick={async () => {
                            if (!window.confirm(`Delete the job directory of ${job.job_id}?\n\nCheckpoints inside it are removed. This cannot be undone.`)) return;
                            try {
                              const result = await api.call<any>("storage.clean", { paths: [job.job_dir], confirm: true });
                              toast({
                                title: result.removed?.length ? "Job directory deleted" : "Nothing deleted",
                                body: result.removed?.length ? `Freed ${result.freed_human}.` : (result.skipped ?? []).join(", "),
                                tone: result.removed?.length ? "ok" : "warn",
                              });
                              await refreshJobs();
                            } catch (error) {
                              reportError(error, "Deletion refused");
                            }
                          }}
                        >
                          Delete
                        </Button>
                      </div>
                    </Td>
                  </tr>
                );
              })}
            </tbody>
          </Table>
        )}
      </Panel>

      <Panel>
        <SectionHeader title="How runs behave" />
        <div className="grid gap-3 md:grid-cols-3 text-2xs leading-relaxed text-ink-2">
          <div>
            <div className="mb-1 font-medium text-ink-1">Pausing</div>
            Only backends that support it can pause mid-step: the tiny backend does, the Transformers backend
            stops between evaluation points instead of pretending to pause instantly.
          </div>
          <div>
            <div className="mb-1 font-medium text-ink-1">Recovery</div>
            On startup the app checks every job's process. A live process is re-attached; a dead one is marked
            interrupted with the last resumable checkpoint offered as the resume point.
          </div>
          <div>
            <div className="mb-1 font-medium text-ink-1">Resources</div>
            The pre-flight check estimates VRAM, RAM and disk from the real model, dataset and configuration, and
            refuses a run that cannot fit instead of letting it fail halfway. Every estimate is labelled; everything
            reported after a run started is measured from the process.
          </div>
        </div>
      </Panel>
    </div>
  );
}
