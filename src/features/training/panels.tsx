import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  Copy,
  FolderOpen,
  HardDrive,
  Pause,
  Play,
  Square,
  Terminal,
  Thermometer,
} from "lucide-react";

import { LineChart, MeterBar, Sparkline } from "@/components/charts/Charts";
import {
  Badge,
  Button,
  Dot,
  IconButton,
  KeyValue,
  Note,
  Panel,
  PanelHeader,
  ProgressBar,
  Segmented,
  Stat,
} from "@/components/ui/primitives";
import type { GpuPoint, SeriesPoint } from "@/state/appStore";
import type { HardwareSnapshot, LogEntry, RunRecord, VramEstimate } from "@/lib/types";
import {
  cn,
  formatCount,
  formatDuration,
  formatLearningRate,
  formatLoss,
  formatRate,
} from "@/lib/utils";
import { useSelect } from "@/state/store";
import { appStore, openPath, pauseRun, resumeRun, revealPath, stopRun } from "@/state/appStore";

export function MetricsGrid({ run }: { run: RunRecord }) {
  const running = run.status === "running" || run.status === "starting";
  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
      <Stat
        label="Loss"
        value={formatLoss(run.loss)}
        tone={run.loss != null && run.loss < 1 ? "good" : undefined}
        sub={run.evalLoss != null ? `eval ${formatLoss(run.evalLoss)}` : undefined}
      />
      <Stat label="Learning rate" value={formatLearningRate(run.learningRate)} sub={run.optimizer ?? undefined} />
      <Stat
        label="Step"
        value={`${run.step}${run.totalSteps ? ` / ${run.totalSteps}` : ""}`}
        sub={`epoch ${run.epoch ? run.epoch.toFixed(2) : "0.00"}`}
      />
      <Stat
        label="Speed"
        value={run.samplesPerSecond != null ? `${formatRate(run.samplesPerSecond, 1)}/s` : "—"}
        sub={run.secondsPerStep != null ? `${formatRate(run.secondsPerStep, 2)} s/step` : undefined}
      />
      <Stat label="Elapsed" value={formatDuration(run.elapsedSeconds)} sub={run.device ?? undefined} />
      <Stat
        label="ETA"
        value={running && run.etaSeconds != null ? formatDuration(run.etaSeconds) : run.status === "completed" ? "done" : "—"}
        sub={run.precision ?? undefined}
      />
    </div>
  );
}

export function ChartsPanel({ series }: { series: SeriesPoint[] }) {
  const lossPoints = useMemo(
    () => series.filter((point) => point.loss != null).map((point) => ({ x: point.step, y: point.loss as number })),
    [series],
  );
  const lrPoints = useMemo(
    () => series.filter((point) => point.lr != null).map((point) => ({ x: point.step, y: point.lr as number })),
    [series],
  );

  const lastLoss = lossPoints.length ? lossPoints[lossPoints.length - 1].y : null;
  const firstLoss = lossPoints.length ? lossPoints[0].y : null;
  const delta = lastLoss != null && firstLoss != null ? lastLoss - firstLoss : null;

  return (
    <Panel>
      <PanelHeader
        icon={<Activity className="h-4 w-4" />}
        title="Training curves"
        description="Recorded every logging step. Hover for the exact value."
        actions={
          delta != null ? (
            <Badge tone={delta < 0 ? "good" : "warn"}>
              {delta < 0 ? "▼" : "▲"} {Math.abs(delta).toFixed(4)}
            </Badge>
          ) : null
        }
      />
      <LineChart
        points={lossPoints}
        height={216}
        color="var(--acc)"
        labelY="loss"
        labelX="step"
        formatY={(value) => value.toFixed(3)}
        emptyLabel="Loss will appear once the first logging step completes"
        tooltip={(index: number) => {
          const point = lossPoints[index];
          if (!point) return [];
          const lr = lrPoints[index];
          return [
            { label: "step", value: String(point.x) },
            { label: "loss", value: point.y.toFixed(5) },
            ...(lr ? [{ label: "lr", value: formatLearningRate(lr.y) }] : []),
          ];
        }}
      />
      {lrPoints.length > 1 ? (
        <div className="mt-3 border-t border-[var(--border-soft)] pt-3">
          <LineChart
            points={lrPoints}
            height={110}
            color="var(--blue)"
            labelY="learning rate"
            labelX="step"
            formatY={(value) => value.toExponential(1)}
            emptyLabel="Learning rate schedule"
          />
        </div>
      ) : null}
    </Panel>
  );
}

export function GpuPanel({
  liveGpu,
  gpuSeries,
  hardware,
  run,
}: {
  liveGpu: GpuPoint | null;
  gpuSeries: GpuPoint[];
  hardware: HardwareSnapshot | null;
  run: RunRecord | null;
}) {
  const gpu = hardware?.gpu;
  const device = gpu?.gpus?.[0];
  const utilSeries = gpuSeries.map((point) => point.utilization ?? 0);
  const vramSeries = gpuSeries.map((point) => point.vramUsedMb ?? 0);

  if (!gpu || !gpu.available || !device) {
    return (
      <Panel>
        <PanelHeader icon={<HardDrive className="h-4 w-4" />} title="GPU" description="Live NVIDIA telemetry" />
        <Note tone="warn" title="No NVIDIA GPU detected">
          {gpu?.reason ?? "nvidia-smi was not found on this machine."}{" "}
          Training will use the CPU, and GPU metrics are unavailable.
        </Note>
        {run?.gpuSummary ? (
          <div className="mt-3">
            <KeyValue label="Recorded peak VRAM" value={`${(run.gpuSummary.peakVramMb / 1024).toFixed(1)} GB`} />
          </div>
        ) : null}
      </Panel>
    );
  }

  const totalVram = device.memory_total_mb ?? 0;
  const usedVram = liveGpu?.vramUsedMb ?? device.memory_used_mb ?? 0;

  return (
    <Panel>
      <PanelHeader
        icon={<HardDrive className="h-4 w-4" />}
        title={device.name}
        description={`${(totalVram / 1024).toFixed(1)} GB · driver ${device.driver_version ?? "—"} · compute ${device.compute_capability ?? "—"}`}
        actions={
          liveGpu ? (
            <Badge tone={liveGpu.utilization && liveGpu.utilization > 5 ? "good" : "neutral"}>
              <Dot tone={liveGpu.utilization && liveGpu.utilization > 5 ? "good" : "neutral"} />
              sampling
            </Badge>
          ) : (
            <Badge>idle</Badge>
          )
        }
      />

      <div className="space-y-3">
        <MeterBar
          value={liveGpu?.utilization ?? device.utilization_gpu ?? 0}
          max={100}
          label="GPU utilisation"
          display={`${Math.round(liveGpu?.utilization ?? device.utilization_gpu ?? 0)}%`}
        />
        <MeterBar
          value={usedVram}
          max={totalVram || 1}
          label="VRAM"
          display={`${(usedVram / 1024).toFixed(2)} / ${(totalVram / 1024).toFixed(1)} GB`}
        />
        <div className="grid grid-cols-3 gap-3 border-t border-[var(--border-soft)] pt-3">
          <div>
            <div className="mb-0.5 flex items-center gap-1 text-[11px] text-[var(--text-3)]">
              <Thermometer className="h-3 w-3" />
              Temperature
            </div>
            <div className="zq-mono text-[13px]">
              {liveGpu?.temperature != null ? `${Math.round(liveGpu.temperature)} °C` : "—"}
            </div>
          </div>
          <div>
            <div className="mb-0.5 text-[11px] text-[var(--text-3)]">Torch allocated</div>
            <div className="zq-mono text-[13px]">
              {run?.gpuMemoryAllocatedMb != null ? `${(run.gpuMemoryAllocatedMb / 1024).toFixed(2)} GB` : "—"}
            </div>
          </div>
          <div>
            <div className="mb-0.5 text-[11px] text-[var(--text-3)]">Reserved</div>
            <div className="zq-mono text-[13px]">
              {run?.gpuMemoryReservedMb != null ? `${(run.gpuMemoryReservedMb / 1024).toFixed(2)} GB` : "—"}
            </div>
          </div>
        </div>

        {gpuSeries.length > 2 ? (
          <div className="border-t border-[var(--border-soft)] pt-3">
            <div className="mb-1 text-[11px] uppercase tracking-[0.06em] text-[var(--text-3)]">
              Utilisation history
            </div>
            <Sparkline values={utilSeries} color="var(--acc)" height={36} />
            <div className="mb-1 mt-2 text-[11px] uppercase tracking-[0.06em] text-[var(--text-3)]">
              VRAM history
            </div>
            <Sparkline values={vramSeries} color="var(--blue)" height={36} />
          </div>
        ) : null}

        {run?.gpuSummary ? (
          <div className="border-t border-[var(--border-soft)] pt-2">
            <KeyValue label="Peak utilisation" value={`${run.gpuSummary.peakUtilization}%`} />
            <KeyValue label="Average utilisation" value={`${run.gpuSummary.averageUtilization}%`} />
            <KeyValue label="Peak VRAM" value={`${(run.gpuSummary.peakVramMb / 1024).toFixed(2)} GB`} />
            <KeyValue label="Peak temperature" value={`${run.gpuSummary.peakTemperatureC} °C`} />
          </div>
        ) : null}
      </div>
    </Panel>
  );
}

export function CheckpointsPanel({ run }: { run: RunRecord }) {
  const checkpoints = run.checkpoints || [];
  return (
    <Panel>
      <PanelHeader
        icon={<HardDrive className="h-4 w-4" />}
        title="Checkpoints"
        description={
          checkpoints.length
            ? `${checkpoints.length} saved · newest first · older ones are pruned by the retention limit`
            : "Written automatically while training runs"
        }
        actions={
          run.runDir ? (
            <Button size="sm" variant="ghost" icon={<FolderOpen className="h-3.5 w-3.5" />} onClick={() => void openPath(run.runDir)}>
              Open run folder
            </Button>
          ) : null
        }
      />
      {checkpoints.length === 0 ? (
        <p className="text-[12.5px] text-[var(--text-3)]">
          No checkpoints yet. The first one is written after the configured interval, and one is always
          written when you pause or stop.
        </p>
      ) : (
        <div className="space-y-1.5">
          {checkpoints.slice(0, 10).map((checkpoint) => (
            <div
              key={checkpoint.path}
              className="flex items-center gap-3 rounded-[10px] border border-[var(--border-soft)] bg-[var(--panel-2)] px-2.5 py-2"
            >
              <span className="zq-mono shrink-0 text-[11.5px] font-medium">step {checkpoint.step}</span>
              <span className="min-w-0 flex-1 truncate text-[11px] text-[var(--text-3)]">
                loss {formatLoss(checkpoint.loss ?? null)} · epoch {checkpoint.epoch ?? "—"} ·{" "}
                {checkpoint.sizeBytes ? `${(checkpoint.sizeBytes / (1024 * 1024)).toFixed(0)} MB` : "size unknown"}
                {checkpoint.hasOptimizer === false ? " · optimizer state missing" : ""}
              </span>
              <IconButton title="Reveal in file manager" onClick={() => void revealPath(checkpoint.path)}>
                <FolderOpen className="h-3.5 w-3.5" />
              </IconButton>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}

export function LogPanel({ logs }: { logs: LogEntry[] }) {
  const [tab, setTab] = useState<"events" | "stderr">("events");
  const [autoscroll, setAutoscroll] = useState(true);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const filtered = useMemo(
    () => logs.filter((entry) => (tab === "events" ? entry.stream !== "stderr" : entry.stream === "stderr")),
    [logs, tab],
  );

  useEffect(() => {
    if (!autoscroll || !containerRef.current) return;
    containerRef.current.scrollTop = containerRef.current.scrollHeight;
  }, [filtered, autoscroll]);

  const copyAll = () => {
    const text = filtered.map((entry) => entry.message).join("\n");
    void navigator.clipboard?.writeText(text);
  };

  return (
    <Panel>
      <PanelHeader
        icon={<Terminal className="h-4 w-4" />}
        title="Logs"
        description="Events come from the trainer's protocol stream; standard output and errors are kept separately."
        actions={
          <>
            <Segmented
              size="sm"
              value={tab}
              onChange={setTab}
              options={[
                { value: "events", label: "Events" },
                { value: "stderr", label: "stdout / stderr" },
              ]}
            />
            <Button size="sm" variant="ghost" icon={<Copy className="h-3.5 w-3.5" />} onClick={copyAll}>
              Copy
            </Button>
          </>
        }
      />
      <div
        ref={containerRef}
        onScroll={(event) => {
          const element = event.currentTarget;
          const atBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 32;
          setAutoscroll(atBottom);
        }}
        className="zq-mono max-h-[320px] min-h-[160px] overflow-auto rounded-[10px] border border-[var(--border-soft)] bg-[var(--code-bg)] p-3 text-[11.5px] leading-[18px]"
      >
        {filtered.length === 0 ? (
          <p className="text-[var(--text-3)]">Nothing logged yet.</p>
        ) : (
          filtered.map((entry) => (
            <div
              key={entry.key}
              className={cn(
                "whitespace-pre-wrap break-words",
                entry.level === "error"
                  ? "text-[var(--red)]"
                  : entry.level === "warn"
                    ? "text-[var(--amber)]"
                    : "text-[var(--text-2)]",
              )}
            >
              {entry.message}
            </div>
          ))
        )}
      </div>
      {!autoscroll ? (
        <button
          type="button"
          onClick={() => setAutoscroll(true)}
          className="mt-2 text-[11.5px] text-[var(--acc)] hover:underline"
        >
          Jump to latest
        </button>
      ) : null}
    </Panel>
  );
}

export function RunControls({ run }: { run: RunRecord }) {
  const running = run.status === "running" || run.status === "starting";
  const pending = run.pendingAction;
  const resumable =
    (run.status === "paused" || run.status === "stopped" || run.status === "failed") &&
    Boolean(run.lastCheckpoint);
  const resuming = useBusyFlag("resume");

  return (
    <div className="flex items-center gap-2">
      {running ? (
        <>
          <Button
            size="sm"
            variant="quiet"
            icon={<Pause className="h-3.5 w-3.5" />}
            disabled={pending === "pausing"}
            onClick={() => void pauseRun(run.id)}
          >
            {pending === "pausing" ? "Pausing…" : "Pause"}
          </Button>
          <Button
            size="sm"
            variant="danger"
            icon={<Square className="h-3.5 w-3.5" />}
            disabled={pending === "stopping"}
            onClick={() => void stopRun(run.id)}
          >
            Stop
          </Button>
        </>
      ) : null}
      {resumable ? (
        <Button
          size="sm"
          variant="secondary"
          icon={<Play className="h-3.5 w-3.5" />}
          loading={resuming}
          onClick={() => void resumeRun(run.id)}
        >
          Resume from checkpoint
          </Button>
      ) : null}
    </div>
  );
}

/** Read a busy flag from the store without re-rendering on every tick. */
function useBusyFlag(key: string): boolean {
  return useSelect(appStore, (state) => Boolean(state.busy[key]));
}

export function RunErrorNote({ run }: { run: RunRecord }) {
  if (!run.error) return null;
  return (
    <Note tone="bad" title="Training failed" className="mb-4">
      <p>{run.error.message}</p>
      {run.error.hint ? <p className="mt-1">{run.error.hint}</p> : null}
      <p className="mt-1 text-[11px] text-[var(--text-3)]">
        The full traceback is in the “stdout / stderr” tab below.
      </p>
    </Note>
  );
}

export function EstimatePanel({ estimate }: { estimate: VramEstimate | null }) {
  if (!estimate) return null;
  return (
    <Panel>
      <PanelHeader icon={<Activity className="h-4 w-4" />} title="Estimated memory" />
      {estimate.available ? (
        <>
          <Stat
            label="Total"
            value={`${((estimate.estimated_total_mb ?? 0) / 1024).toFixed(2)} GB`}
            sub={estimate.available_vram_mb ? `of ${(estimate.available_vram_mb / 1024).toFixed(1)} GB` : undefined}
            tone={estimate.verdict === "exceeds" ? "bad" : estimate.verdict === "tight" ? "warn" : "good"}
          />
          <div className="mt-2">
            <ProgressBar
              value={estimate.available_vram_mb ? ((estimate.estimated_total_mb ?? 0) / estimate.available_vram_mb) * 100 : 0}
              tone={estimate.verdict === "exceeds" ? "bad" : estimate.verdict === "tight" ? "warn" : "good"}
            />
          </div>
          <div className="mt-3">
            <KeyValue label="Trainable parameters" value={formatCount(estimate.trainable_params ?? null)} />
            <KeyValue label="Weights" value={`${((estimate.weights_mb ?? 0) / 1024).toFixed(2)} GB`} />
            <KeyValue label="Optimizer" value={`${((estimate.optimizer_mb ?? 0) / 1024).toFixed(2)} GB`} />
            <KeyValue label="Activations" value={`${((estimate.activations_mb ?? 0) / 1024).toFixed(2)} GB`} />
          </div>
        </>
      ) : (
        <p className="text-[12.5px] text-[var(--text-3)]">{estimate.reason}</p>
      )}
    </Panel>
  );
}
