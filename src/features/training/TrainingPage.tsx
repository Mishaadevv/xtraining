import { useEffect } from "react";
import {
  Activity,
  FolderOpen,
  Play,
  RefreshCw,
  Trash2,
} from "lucide-react";

import { PageBody, PageHeader } from "@/components/layout/Shell";
import {
  Badge,
  Button,
  Dot,
  EmptyState,
  IconButton,
  Panel,
  PanelHeader,
  Stat,
} from "@/components/ui/primitives";
import type { RunRecord, RunStatus } from "@/lib/types";
import { cn, formatDuration, formatLoss, formatRelative } from "@/lib/utils";
import { useStore } from "@/state/store";
import {
  appStore,
  clearLogs,
  deleteRun,
  loadRunDetails,
  navigate,
  openPath,
  refreshRuns,
  resetWizard,
  selectRun,
  setWizard,
} from "@/state/appStore";
import { bridge, isDesktop } from "@/lib/bridge";

import {
  ChartsPanel,
  CheckpointsPanel,
  GpuPanel,
  LogPanel,
  MetricsGrid,
  RunControls,
  RunErrorNote,
} from "./panels";

const STATUS_TONE: Record<RunStatus, "good" | "warn" | "bad" | "neutral" | "accent"> = {
  completed: "good",
  running: "accent",
  starting: "accent",
  failed: "bad",
  paused: "warn",
  stopped: "neutral",
};

function RunListItem({
  run,
  active,
  onSelect,
  onDelete,
}: {
  run: RunRecord;
  active: boolean;
  onSelect: () => void;
  onDelete: () => void;
}) {
  const tone = STATUS_TONE[run.status] ?? "neutral";
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        "group flex w-full items-start gap-2.5 rounded-[11px] border p-2.5 text-left transition-colors",
        active
          ? "border-[var(--border)] bg-[var(--panel-2)]"
          : "border-transparent hover:bg-[var(--hover)]",
      )}
    >
      <span className="mt-[6px] shrink-0">
        {run.status === "running" || run.status === "starting" ? (
          <span className="relative flex h-1.5 w-1.5">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[var(--acc)] opacity-70" />
            <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-[var(--acc)]" />
          </span>
        ) : (
          <Dot tone={tone === "accent" ? "accent" : tone} />
        )}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[12px] font-medium">{run.name}</span>
        <span className="zq-mono block truncate text-[10.5px] text-[var(--text-3)]">
          {run.method} · loss {formatLoss(run.finalLoss ?? run.loss)} · {formatRelative(run.startedAt)}
        </span>
        {run.status === "running" ? (
          <span className="mt-1 block h-[3px] w-full overflow-hidden rounded-full bg-[var(--panel)]">
            <span
              className="block h-full rounded-full bg-[var(--acc)] transition-[width] duration-500"
              style={{ width: `${run.progress}%` }}
            />
          </span>
        ) : null}
      </span>
      <span
        role="button"
        tabIndex={0}
        title="Delete this run"
        onClick={(event) => {
          event.stopPropagation();
          onDelete();
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.stopPropagation();
            onDelete();
          }
        }}
        className="mt-[2px] hidden h-6 w-6 shrink-0 items-center justify-center rounded-[7px] text-[var(--text-3)] transition-colors hover:text-[var(--red)] group-hover:flex"
      >
        <Trash2 className="h-3 w-3" />
      </span>
    </button>
  );
}

export function TrainingPage() {
  const { runs, selectedRun, selectedRunId, series, gpuSeries, liveGpu, logs, env, busy } = useStore(appStore);

  const activeRun = runs.find((run) => run.status === "running" || run.status === "starting") ?? null;
  const isLive = selectedRun?.status === "running" || selectedRun?.status === "starting";

  // A safety net for the live view: if an event is ever missed (window reload,
  // renderer restart), polling brings the displayed state back in sync.
  useEffect(() => {
    if (!isDesktop || !selectedRunId || !isLive) return undefined;
    const timer = setInterval(() => {
      void bridge.training.progress(selectedRunId).then((result) => {
        const record = result.record as RunRecord | undefined;
        if (record) appStore.set({ selectedRun: record });
      });
    }, 2500);
    return () => clearInterval(timer);
  }, [selectedRunId, isLive]);

  useEffect(() => {
    if (isDesktop && selectedRunId && logs.length === 0 && !isLive) {
      void loadRunDetails(selectedRunId);
    }
  }, [selectedRunId, logs.length, isLive]);

  return (
    <>
      <PageHeader
        icon={<Activity className="h-4 w-4" />}
        title={selectedRun ? selectedRun.name : "Training"}
        subtitle={
          selectedRun
            ? `${selectedRun.method} · ${selectedRun.baseModel}`
            : activeRun
              ? `${activeRun.name} is running`
              : "Live metrics, GPU telemetry, logs and checkpoints"
        }
        actions={
          <>
            <Button
              size="sm"
              variant="quiet"
              icon={<RefreshCw className="h-3.5 w-3.5" />}
              onClick={() => void refreshRuns()}
            >
              Refresh
            </Button>
            {selectedRun ? <RunControls run={selectedRun} /> : null}
            <Button
              size="sm"
              variant="primary"
              icon={<Play className="h-3.5 w-3.5" />}
              onClick={() => {
                resetWizard();
                navigate("new");
              }}
            >
              New training
            </Button>
          </>
        }
      />

      <PageBody wide>
        {runs.length === 0 ? (
          <EmptyState
            icon={<Activity className="h-7 w-7" />}
            title="No training runs yet"
            description="Start a run and this screen becomes a live monitor: loss curve, learning rate, steps, throughput, GPU and VRAM usage, ETA, logs and checkpoints."
            action={
              <Button
                variant="primary"
                icon={<Play className="h-3.5 w-3.5" />}
                onClick={() => {
                  resetWizard();
                  navigate("new");
                }}
              >
                Start a training run
              </Button>
            }
          />
        ) : (
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-[268px_minmax(0,1fr)]">
            <div className="space-y-1">
              <div className="flex items-center justify-between px-2 pb-1">
                <span className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[var(--text-3)]">
                  Runs
                </span>
                <span className="zq-mono text-[10.5px] text-[var(--text-3)]">{runs.length}</span>
              </div>
              <div className="max-h-[calc(100vh-220px)] space-y-0.5 overflow-y-auto pr-1">
                {runs.map((run) => (
                  <RunListItem
                    key={run.id}
                    run={run}
                    active={run.id === selectedRunId}
                    onSelect={() => {
                      void selectRun(run.id);
                      clearLogs();
                    }}
                    onDelete={() => void deleteRun(run.id)}
                  />
                ))}
              </div>
            </div>

            <div className="min-w-0 space-y-4">
              {!selectedRun ? (
                <Panel>
                  <p className="text-[12.5px] text-[var(--text-3)]">
                    Select a run on the left to inspect its metrics, logs and checkpoints.
                  </p>
                </Panel>
              ) : (
                <div className="zq-rise space-y-4">
                  <Panel>
                    <PanelHeader
                      title={selectedRun.name}
                      description={
                        <span className="zq-mono">
                          {selectedRun.method} · {selectedRun.baseModel} · started{" "}
                          {formatRelative(selectedRun.startedAt)}
                          {selectedRun.resumedFrom ? " · resumed from a checkpoint" : ""}
                        </span>
                      }
                      icon={<Activity className="h-4 w-4" />}
                      actions={
                        <>
                          <Badge tone={STATUS_TONE[selectedRun.status] ?? "neutral"}>
                            <Dot
                              tone={
                                selectedRun.status === "running"
                                  ? "accent"
                                  : STATUS_TONE[selectedRun.status] === "neutral"
                                    ? "neutral"
                                    : STATUS_TONE[selectedRun.status]
                              }
                            />
                            {selectedRun.status}
                          </Badge>
                          {selectedRun.runDir ? (
                            <IconButton title="Open the run folder" onClick={() => void openPath(selectedRun.runDir)}>
                              <FolderOpen className="h-3.5 w-3.5" />
                            </IconButton>
                          ) : null}
                        </>
                      }
                    />

                    {isLive || selectedRun.status === "starting" ? (
                      <div className="mb-4">
                        <div className="mb-1.5 flex items-baseline justify-between gap-3">
                          <span className="truncate text-[12px] text-[var(--text-2)]">
                            {selectedRun.message ?? selectedRun.phase}
                          </span>
                          <span className="zq-mono shrink-0 text-[12px] font-medium">
                            {selectedRun.progress}%
                          </span>
                        </div>
                        <span className="block h-[6px] w-full overflow-hidden rounded-full bg-[var(--panel-2)]">
                          <span
                            className="block h-full rounded-full bg-[var(--acc)] transition-[width] duration-500 ease-out"
                            style={{ width: `${selectedRun.progress}%` }}
                          />
                        </span>
                      </div>
                    ) : null}

                    <MetricsGrid run={selectedRun} />

                    {!isLive && selectedRun.finalLoss != null ? (
                      <div className="mt-4 grid grid-cols-2 gap-4 border-t border-[var(--border-soft)] pt-4 sm:grid-cols-4">
                        <Stat label="Final loss" value={formatLoss(selectedRun.finalLoss)} tone="good" />
                        <Stat label="Total steps" value={selectedRun.totalSteps} />
                        <Stat label="Duration" value={formatDuration(selectedRun.elapsedSeconds)} />
                        <Stat
                          label="Trainable params"
                          value={selectedRun.trainableParams ? `${(selectedRun.trainableParams / 1_000_000).toFixed(2)}M` : "—"}
                          sub={selectedRun.totalParams ? `of ${(selectedRun.totalParams / 1_000_000).toFixed(0)}M` : undefined}
                        />
                      </div>
                    ) : null}
                  </Panel>

                  <RunErrorNote run={selectedRun} />

                  <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
                    <ChartsPanel series={series} />
                    <GpuPanel
                      liveGpu={liveGpu}
                      gpuSeries={gpuSeries}
                      hardware={env.hardware}
                      run={selectedRun}
                    />
                  </div>

                  <CheckpointsPanel run={selectedRun} />
                  <LogPanel logs={logs} />

                  {selectedRun.datasetReport ? (
                    <Panel>
                      <PanelHeader title="Dataset used" description="Recorded at the moment the run started." />
                      <div className="grid grid-cols-2 gap-x-6 sm:grid-cols-4">
                        <Stat label="Records" value={selectedRun.datasetReport.records ?? "—"} />
                        <Stat label="Usable" value={selectedRun.datasetReport.usable ?? "—"} />
                        <Stat label="Mapping" value={selectedRun.datasetReport.mapping?.kind ?? "—"} mono={false} />
                        <Stat
                          label="Target modules"
                          value={selectedRun.targetModules?.length ? String(selectedRun.targetModules.length) : "—"}
                          sub={selectedRun.targetModules?.join(", ")}
                          mono={false}
                        />
                      </div>
                    </Panel>
                  ) : null}

                  <div className="flex flex-wrap items-center gap-2 pb-2">
                    <Button
                      size="sm"
                      variant="quiet"
                      icon={<Play className="h-3.5 w-3.5" />}
                      onClick={() => {
                        setWizard({
                          baseModel: selectedRun.baseModel,
                          method: selectedRun.method,
                          projectId: selectedRun.projectId,
                          runName: `${selectedRun.name}-again`,
                        });
                        navigate("new");
                      }}
                    >
                      Reuse this configuration
                    </Button>
                    {selectedRun.outputDir ? (
                      <Button
                        size="sm"
                        variant="ghost"
                        icon={<FolderOpen className="h-3.5 w-3.5" />}
                        onClick={() => void openPath(selectedRun.outputDir)}
                      >
                        Open the trained model folder
                      </Button>
                    ) : null}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {busy.resume ? <p className="mt-2 text-[12px] text-[var(--text-3)]">Preparing the resumed run…</p> : null}
      </PageBody>
    </>
  );
}
