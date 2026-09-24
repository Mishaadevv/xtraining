import { useMemo } from "react";
import {
  Activity,
  Brain,
  Cpu,
  Database,
  FolderOpen,
  Gauge,
  HardDrive,
  ListChecks,
  MessageSquare,
  Rocket,
  Thermometer,
} from "lucide-react";
import { BarChart, LineChart, UsageBar } from "../components/charts";
import { Badge, Button, EmptyState, KeyValue, Panel, SectionHeader, Stat, cx } from "../components/ui";
import { bytes, compact, duration, percent, relative } from "../lib/format";
import { useApp } from "../state/app";
import { useRouter } from "../state/router";
import { ErrorPanel, JobRow, Loading, PathText, useEngine } from "./common";

export function DashboardPage() {
  const { hardware, live, jobs, registry, settings, appInfo, environment, refreshHardware } = useApp();
  const router = useRouter();
  const storage = useEngine<any>("storage.report", {}, { timeout: 120_000 });

  const running = jobs.filter((job) => ["running", "queued", "paused"].includes(job.state));
  const recent = jobs.slice(0, 6);
  const checkpoints = useMemo(
    () =>
      jobs
        .flatMap((job) => (job.checkpoints ?? []).map((checkpoint) => ({ ...checkpoint, jobId: job.job_id })))
        .slice(0, 8),
    [jobs],
  );
  const lossHistory = useMemo(() => {
    const job = jobs.find((item) => item.state === "running" && (item.metrics ?? []).length) ?? jobs[0];
    return (job?.metrics ?? [])
      .filter((entry: any) => entry.loss !== undefined && entry.loss !== null)
      .map((entry: any) => ({ x: entry.step ?? 0, y: entry.loss }));
  }, [jobs]);

  const gpu = live?.gpus?.[0] ?? hardware?.gpus?.[0] ?? null;
  const disk = hardware?.disks?.[0] ?? null;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-2xs uppercase tracking-wider text-ink-3">
            {appInfo?.packaged ? "release build" : "development build"} · engine {appInfo?.version ?? "…"}
          </div>
          <h1 className="text-lg font-semibold tracking-tight">Local training environment</h1>
          <p className="mt-0.5 max-w-3xl text-xs text-ink-2">
            Everything on this page is read from this machine and this workspace. Nothing is cached from a
            previous version, and unsupported operations are reported instead of shown as available.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="primary" icon={<Rocket size={13} />} onClick={() => router.navigate("/training/new")}>
            New training run
          </Button>
          <Button icon={<Brain size={13} />} onClick={() => router.navigate("/models?action=import")}>
            Import model
          </Button>
          <Button icon={<Database size={13} />} onClick={() => router.navigate("/datasets?action=import")}>
            Import dataset
          </Button>
          <Button icon={<MessageSquare size={13} />} onClick={() => router.navigate("/playground")}>
            Playground
          </Button>
        </div>
      </div>

      {!hardware ? <Loading label="Reading hardware…" /> : null}

      <div className="grid gap-3 lg:grid-cols-4">
        <Stat
          label="CPU"
          value={percent(live?.cpu?.percent ?? null)}
          hint={`${hardware?.cpu?.logical_cores ?? "?"} threads`}
        />
        <Stat
          label="RAM used"
          value={bytes(live?.ram?.used ?? null, 1)}
          hint={`of ${bytes(live?.ram?.total ?? hardware?.memory?.total ?? null, 0)}`}
          tone={(live?.ram?.percent ?? 0) > 85 ? "warn" : "muted"}
        />
        <Stat
          label={gpu ? "GPU memory" : "GPU"}
          value={gpu ? bytes((gpu.memory_used_mb ?? 0) * 1024 * 1024, 1) : "none detected"}
          hint={gpu ? `of ${bytes(gpu.memory_total ?? null, 0)}` : "CPU only"}
          tone={gpu ? "muted" : "warn"}
        />
        <Stat
          label="Free disk"
          value={bytes(disk?.free ?? null, 0)}
          hint={disk ? `of ${bytes(disk.total, 0)} · ${percent(disk.percent)} used` : undefined}
          tone={disk && disk.percent && disk.percent > 90 ? "danger" : "muted"}
        />
      </div>

      <div className="grid gap-3 lg:grid-cols-3">
        <Panel className="lg:col-span-2">
          <SectionHeader
            title="Live resources"
            subtitle="Sampled from the operating system and nvidia-smi every few seconds"
            icon={<Gauge size={13} className="text-accent" />}
            actions={
              <Button size="sm" variant="subtle" onClick={() => void refreshHardware(false)}>
                Re-detect
              </Button>
            }
          />
          <div className="space-y-3">
            <UsageBar label="System memory" value={live?.ram?.used ?? null} max={live?.ram?.total ?? null} format={(value) => bytes(value, 1)} />
            <UsageBar
              label={gpu ? `${gpu.name} VRAM` : "GPU memory (none)"}
              value={gpu?.memory_used_mb ? gpu.memory_used_mb * 1024 * 1024 : null}
              max={gpu?.memory_total ?? null}
              format={(value) => bytes(value, 1)}
            />
            <UsageBar label="Workspace disk" value={disk?.used ?? null} max={disk?.total ?? null} format={(value) => bytes(value, 0)} />
            <div className="grid gap-2 sm:grid-cols-3">
              <div className="rounded-md border border-line-soft bg-surface-2 px-2.5 py-2">
                <div className="flex items-center gap-1.5 text-2xs uppercase tracking-wide text-ink-3">
                  <Cpu size={11} /> CPU load
                </div>
                <div className="mt-0.5 text-sm tabular-nums">{percent(live?.cpu?.percent ?? null, 1)}</div>
              </div>
              <div className="rounded-md border border-line-soft bg-surface-2 px-2.5 py-2">
                <div className="flex items-center gap-1.5 text-2xs uppercase tracking-wide text-ink-3">
                  <Thermometer size={11} /> GPU temperature
                </div>
                <div className="mt-0.5 text-sm tabular-nums">
                  {gpu?.temperature_c !== null && gpu?.temperature_c !== undefined ? `${gpu.temperature_c} °C` : "not reported"}
                </div>
              </div>
              <div className="rounded-md border border-line-soft bg-surface-2 px-2.5 py-2">
                <div className="flex items-center gap-1.5 text-2xs uppercase tracking-wide text-ink-3">
                  <Activity size={11} /> GPU utilisation
                </div>
                <div className="mt-0.5 text-sm tabular-nums">
                  {gpu?.utilization_gpu !== null && gpu?.utilization_gpu !== undefined ? `${gpu.utilization_gpu}%` : "not reported"}
                </div>
              </div>
            </div>
            {(hardware?.notes ?? []).length ? (
              <div className="space-y-1">
                {hardware?.notes.map((note) => (
                  <div key={note} className="text-2xs leading-relaxed text-ink-2">
                    · {note}
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        </Panel>

        <Panel>
          <SectionHeader title="This machine" subtitle="Detected once per session and on demand" icon={<Cpu size={13} className="text-accent" />} />
          <KeyValue
            columns={1}
            items={[
              ["CPU", <span className="truncate">{hardware?.cpu?.model ?? "unknown"}</span>],
              ["Cores", `${hardware?.cpu?.physical_cores ?? "?"} physical / ${hardware?.cpu?.logical_cores ?? "?"} logical`],
              ["System RAM", bytes(hardware?.memory?.total ?? null, 0)],
              ["CUDA", hardware?.capabilities?.cuda ? `yes (CC ${hardware.capabilities.cuda_compute_capability ?? "?"})` : "not available"],
              ["PyTorch", hardware?.torch?.installed ? `${hardware.torch.version} · cuda build ${hardware.torch.cuda_build ?? "none"}` : "not installed"],
              ["Hardware backend", environment?.ready?.transformers_training ? "Transformers ready" : "tiny backend only"],
            ]}
          />
          <div className="mt-3 flex flex-wrap gap-1.5">
            {hardware?.capabilities?.precisions?.map((precision) => (
              <Badge key={precision.name} tone={precision.available ? "ok" : "muted"} title={precision.reason}>
                {precision.name}
              </Badge>
            ))}
          </div>
        </Panel>
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <Panel>
          <SectionHeader
            title="Training runs"
            subtitle={running.length ? `${running.length} active` : "Nothing running right now"}
            icon={<Activity size={13} className="text-accent" />}
            actions={
              <Button size="sm" variant="subtle" onClick={() => router.navigate("/training")}>
                All runs
              </Button>
            }
          />
          {!recent.length ? (
            <EmptyState title="No runs yet" action={<Button size="sm" variant="primary" onClick={() => router.navigate("/training/new")}>Create the first run</Button>}>
              Configure a run and this panel starts streaming real metrics from the training process.
            </EmptyState>
          ) : (
            <div className="space-y-2">
              {recent.map((job) => (
                <JobRow key={job.job_id} job={job} onOpen={() => router.navigate(`/training/${encodeURIComponent(job.job_id)}`)} />
              ))}
            </div>
          )}
          {lossHistory.length > 1 ? (
            <div className="mt-3">
              <div className="mb-1 text-2xs uppercase tracking-wide text-ink-3">Loss so far</div>
              <LineChart
                height={120}
                series={[{ name: "loss", color: "rgb(var(--accent))", points: lossHistory }]}
                formatY={(value) => value.toFixed(3)}
              />
            </div>
          ) : null}
        </Panel>

        <Panel>
          <SectionHeader
            title="Library"
            subtitle={`${registry?.models?.length ?? 0} models · ${registry?.datasets?.length ?? 0} datasets`}
            icon={<HardDrive size={13} className="text-accent" />}
            actions={
              <Button size="sm" variant="subtle" onClick={() => router.navigate("/files")}>
                Storage
              </Button>
            }
          />
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <div className="mb-1.5 text-2xs uppercase tracking-wide text-ink-3">Recent models</div>
              <div className="space-y-1.5">
                {(registry?.models ?? []).slice(0, 4).map((model) => (
                  <button
                    key={model.id}
                    onClick={() => router.navigate(`/models/${encodeURIComponent(model.id)}`)}
                    className="w-full rounded-md border border-line-soft bg-surface-2 px-2 py-1.5 text-left hover:bg-surface-3"
                  >
                    <div className="truncate text-xs">{model.name}</div>
                    <PathText value={model.path} />
                  </button>
                ))}
                {!(registry?.models ?? []).length ? <div className="text-2xs text-ink-3">Nothing imported yet.</div> : null}
              </div>
            </div>
            <div>
              <div className="mb-1.5 text-2xs uppercase tracking-wide text-ink-3">Recent datasets</div>
              <div className="space-y-1.5">
                {(registry?.datasets ?? []).slice(0, 4).map((dataset) => (
                  <button
                    key={dataset.id}
                    onClick={() => router.navigate(`/datasets/${encodeURIComponent(dataset.id)}`)}
                    className="w-full rounded-md border border-line-soft bg-surface-2 px-2 py-1.5 text-left hover:bg-surface-3"
                  >
                    <div className="truncate text-xs">{dataset.name}</div>
                    <PathText value={dataset.path} />
                  </button>
                ))}
                {!(registry?.datasets ?? []).length ? <div className="text-2xs text-ink-3">Nothing imported yet.</div> : null}
              </div>
            </div>
          </div>
          {storage.error ? (
            <div className="mt-3">
              <ErrorPanel error={storage.error} onRetry={storage.reload} />
            </div>
          ) : storage.data ? (
            <div className="mt-3 space-y-1.5">
              {(storage.data.entries ?? [])
                .filter((entry: any) => entry.exists)
                .slice(0, 6)
                .map((entry: any) => (
                  <div key={entry.name} className="flex items-center justify-between text-2xs">
                    <span className="text-ink-2">{entry.name}</span>
                    <span className="tabular-nums text-ink-1">{entry.human}</span>
                  </div>
                ))}
            </div>
          ) : null}
        </Panel>
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <Panel>
          <SectionHeader title="Recent checkpoints" subtitle="Written by real training processes" icon={<FolderOpen size={13} className="text-accent" />} />
          {!checkpoints.length ? (
            <div className="text-2xs text-ink-3">No checkpoints yet. They appear here as soon as a run saves one.</div>
          ) : (
            <div className="space-y-1.5">
              {checkpoints.map((checkpoint) => (
                <div key={`${checkpoint.jobId}-${checkpoint.name}`} className="flex items-center justify-between gap-2 rounded-md border border-line-soft bg-surface-2 px-2 py-1.5">
                  <div className="min-w-0">
                    <div className="truncate text-xs">{checkpoint.name}</div>
                    <PathText value={checkpoint.path} />
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge tone="muted">step {checkpoint.step ?? "?"}</Badge>
                    <Button size="sm" variant="ghost" onClick={() => router.navigate(`/training/${encodeURIComponent(checkpoint.jobId)}`)}>
                      Open
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Panel>

        <Panel>
          <SectionHeader title="Readiness" subtitle="What the engine can do without further installs" icon={<ListChecks size={13} className="text-accent" />} />
          <div className="grid gap-2 sm:grid-cols-2">
            {Object.entries(environment?.ready ?? {}).map(([key, value]) => (
              <div key={key} className={cx("flex items-center justify-between rounded-md border px-2 py-1.5 text-2xs", value ? "border-ok/25 bg-ok/8" : "border-line-soft bg-surface-2")}>
                <span className="truncate text-ink-1">{key.replace(/_/g, " ")}</span>
                <Badge tone={value ? "ok" : "muted"}>{value ? "ready" : "unavailable"}</Badge>
              </div>
            ))}
          </div>
          {(environment?.recommendations ?? []).length ? (
            <div className="mt-3 space-y-1.5">
              {environment?.recommendations.map((note) => (
                <div key={note} className="rounded-md border border-warn/25 bg-warn/8 px-2 py-1.5 text-2xs leading-relaxed text-ink-1">
                  {note}
                </div>
              ))}
            </div>
          ) : (
            <div className="mt-3 text-2xs text-ink-3">No recommendations — the detected runtime matches what the app needs.</div>
          )}
        </Panel>
      </div>

      {hardware ? (
        <Panel>
          <SectionHeader title="Capability matrix" subtitle="Reported exactly as detected; nothing is assumed" icon={<Gauge size={13} className="text-accent" />} />
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            <CapabilityGroup
              title="Memory optimisation"
              items={Object.entries(hardware.capabilities.memory_optimizations).map(([key, value]) => [key.replace(/_/g, " "), value])}
            />
            <CapabilityGroup
              title="Attention kernels"
              items={Object.entries(hardware.capabilities.attention).map(([key, value]) => [key.replace(/_/g, " "), value])}
            />
            <CapabilityGroup
              title="Quantisation"
              items={Object.entries(hardware.capabilities.quantization_backends).map(([key, value]) => [key, value])}
            />
            <CapabilityGroup
              title="Distributed"
              items={Object.entries(hardware.capabilities.distribution).map(([key, value]) => [key, typeof value === "number" ? compact(value) : value])}
            />
          </div>
        </Panel>
      ) : null}

      <Panel>
        <SectionHeader title="Workspace" subtitle={settings?.workspace ?? "not configured"} icon={<FolderOpen size={13} className="text-accent" />} />
        <div className="grid gap-3 md:grid-cols-3">
          <div className="rounded-md border border-line-soft bg-surface-2 p-2.5">
            <div className="text-2xs uppercase tracking-wide text-ink-3">Total artefact size</div>
            <div className="mt-1 text-sm tabular-nums">{storage.data ? bytes(storage.data.total_bytes, 1) : "…"}</div>
          </div>
          <div className="rounded-md border border-line-soft bg-surface-2 p-2.5">
            <div className="text-2xs uppercase tracking-wide text-ink-3">Interpreter</div>
            <div className="mt-1 truncate font-mono text-2xs" title={environment?.engine_python?.executable}>
              {environment?.engine_python?.version ? `python ${environment.engine_python.version}` : "…"}
            </div>
          </div>
          <div className="rounded-md border border-line-soft bg-surface-2 p-2.5">
            <div className="text-2xs uppercase tracking-wide text-ink-3">Longest running job</div>
            <div className="mt-1 text-2xs">
              {running.length ? `${running[0].job_id} · ${duration(running[0].elapsed_seconds ?? null)}` : "idle"}
            </div>
          </div>
        </div>
        {jobs.length ? (
          <div className="mt-3">
            <div className="mb-1 text-2xs uppercase tracking-wide text-ink-3">Jobs per hour (last 12h, real start times)</div>
            <BarChart
              formatLabel={(value) => String(value).slice(-5)}
              bars={bucketJobs(jobs).map((bucket) => ({ label: bucket.label, value: bucket.count }))}
            />
          </div>
        ) : null}
        <div className="mt-3 text-2xs text-ink-3">
          Last hardware detection: {relative(hardware ? new Date().toISOString() : null)}
        </div>
      </Panel>
    </div>
  );
}

function CapabilityGroup({ title, items }: { title: string; items: Array<[string, unknown]> }) {
  return (
    <div className="rounded-md border border-line-soft bg-surface-2 p-2.5">
      <div className="mb-1.5 text-2xs uppercase tracking-wide text-ink-3">{title}</div>
      <div className="space-y-1">
        {items.map(([label, value]) => (
          <div key={label} className="flex items-center justify-between gap-2 text-2xs">
            <span className="truncate text-ink-2">{label}</span>
            {typeof value === "boolean" ? (
              <Badge tone={value ? "ok" : "muted"}>{value ? "yes" : "no"}</Badge>
            ) : (
              <span className="text-ink-1">{String(value)}</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function bucketJobs(jobs: Array<{ created?: string; started_at?: string }>) {
  const buckets: Array<{ label: string; count: number }> = [];
  const now = Date.now();
  for (let index = 5; index >= 0; index -= 1) {
    const start = now - (index + 1) * 2 * 3600 * 1000;
    const end = now - index * 2 * 3600 * 1000;
    const count = jobs.filter((job) => {
      const stamp = new Date(job.started_at ?? job.created ?? 0).getTime();
      return stamp >= start && stamp < end;
    }).length;
    buckets.push({ label: `-${index * 2}h`, count });
  }
  return buckets;
}
