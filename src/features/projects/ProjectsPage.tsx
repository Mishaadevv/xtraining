import { useState } from "react";
import {
  Activity,
  Clock,
  FolderKanban,
  Play,
  RefreshCw,
  Trash2,
  TrendingDown,
} from "lucide-react";

import { PageBody, PageHeader } from "@/components/layout/Shell";
import { ConfirmDialog } from "@/components/ui/Overlay";
import {
  Badge,
  Button,
  Dot,
  EmptyState,
  IconButton,
  Note,
  Panel,
  PanelHeader,
} from "@/components/ui/primitives";
import type { Project, RunRecord } from "@/lib/types";
import { basename, formatDuration, formatLoss, formatRelative } from "@/lib/utils";
import { useStore } from "@/state/store";
import {
  appStore,
  navigate,
  refreshProjects,
  refreshRuns,
  removeProject,
  resetWizard,
  selectRun,
  validateDataset,
  wizardSelectDataset,
  wizardSelectModel,
  wizardSelectMethod,
} from "@/state/appStore";

const STATUS_TONE: Record<string, "good" | "warn" | "bad" | "neutral" | "accent"> = {
  completed: "good",
  running: "accent",
  starting: "accent",
  failed: "bad",
  paused: "warn",
  stopped: "neutral",
};

function StatusBadge({ status }: { status: string | null | undefined }) {
  if (!status) return <Badge>never run</Badge>;
  const tone = STATUS_TONE[status] ?? "neutral";
  return (
    <Badge tone={tone}>
      <Dot tone={tone === "accent" || tone === "neutral" ? tone : tone} />
      {status}
    </Badge>
  );
}

export function ProjectsPage() {
  const { projects, runs, env, busy } = useStore(appStore);
  const [pendingDelete, setPendingDelete] = useState<Project | null>(null);

  const trainAgain = async (project: Project) => {
    resetWizard({
      projectId: project.id,
      runName: `${project.name}-again`,
      method: project.method ?? "lora",
      baseModel: project.baseModel ?? "",
      datasetId: project.datasetPath
        ? appStore.get().datasets.find((dataset) => dataset.path === project.datasetPath)?.id ?? null
        : null,
    });
    navigate("new");
    if (project.baseModel) await wizardSelectModel(project.baseModel, null);
    if (project.method) await wizardSelectMethod(project.method);
    const datasetId = appStore.get().wizard.datasetId;
    if (datasetId) await wizardSelectDataset(datasetId);
  };

  return (
    <>
      <PageHeader
        icon={<FolderKanban className="h-4 w-4" />}
        title="Projects"
        subtitle={
          projects.length
            ? `${projects.length} project${projects.length === 1 ? "" : "s"} · ${runs.length} run${runs.length === 1 ? "" : "s"} recorded`
            : "Every training configuration becomes a project with its own run history"
        }
        actions={
          <>
            <Button
              size="sm"
              variant="quiet"
              icon={<RefreshCw className="h-3.5 w-3.5" />}
              busy={Boolean(busy.refresh)}
              onClick={() => {
                void refreshProjects();
                void refreshRuns();
              }}
            >
              Refresh
            </Button>
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
        {env.hardware && !env.hardware.cuda_ready ? (
          <Note
            tone="warn"
            title="No usable NVIDIA GPU detected"
            className="mb-4"
            actions={
              <Button size="sm" variant="quiet" onClick={() => navigate("hardware")}>
                Diagnose
              </Button>
            }
          >
            Training will fall back to the CPU, which is very slow. {env.hardware.cuda_blockers[0] ?? ""}
          </Note>
        ) : null}

        {projects.length === 0 ? (
          <EmptyState
            icon={<FolderKanban className="h-7 w-7" />}
            title="No projects yet"
            description="A project groups a model, a dataset and a training configuration, together with every run made from it."
            action={
              <Button
                variant="primary"
                icon={<Play className="h-3.5 w-3.5" />}
                onClick={() => {
                  resetWizard();
                  navigate("new");
                }}
              >
                Create the first project
              </Button>
            }
          />
        ) : (
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
            {projects.map((project) => {
              const projectRuns = runs.filter((run) => run.projectId === project.id);
              const lastRun = projectRuns[0] ?? null;
              return (
                <Panel key={project.id} className="zq-rise flex flex-col">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <h3 className="truncate text-[13.5px] font-semibold">{project.name}</h3>
                      <p className="zq-mono mt-0.5 truncate text-[11px] text-[var(--text-3)]">
                        {project.baseModel || "no base model"} · {basename(project.datasetPath) || "no dataset"}
                      </p>
                    </div>
                    <StatusBadge status={project.lastStatus} />
                  </div>

                  <div className="mt-3 grid grid-cols-3 gap-3">
                    <div>
                      <div className="text-[10.5px] uppercase tracking-[0.06em] text-[var(--text-3)]">Runs</div>
                      <div className="zq-mono text-[13px]">{project.runCount || projectRuns.length}</div>
                    </div>
                    <div>
                      <div className="text-[10.5px] uppercase tracking-[0.06em] text-[var(--text-3)]">Best loss</div>
                      <div className="zq-mono text-[13px]">
                        {project.bestLoss == null ? "—" : formatLoss(project.bestLoss)}
                      </div>
                    </div>
                    <div>
                      <div className="text-[10.5px] uppercase tracking-[0.06em] text-[var(--text-3)]">Last run</div>
                      <div className="text-[12px] text-[var(--text-2)]">{formatRelative(project.lastRunAt)}</div>
                    </div>
                  </div>

                  {lastRun ? (
                    <button
                      type="button"
                      onClick={() => {
                        void selectRun(lastRun.id);
                        navigate("training");
                      }}
                      className="mt-3 flex items-center gap-2.5 rounded-[10px] border border-[var(--border-soft)] bg-[var(--panel-2)] px-2.5 py-2 text-left transition-colors hover:bg-[var(--hover)]"
                    >
                      <Activity className="h-3.5 w-3.5 shrink-0 text-[var(--text-3)]" />
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-[12px]">{lastRun.name}</span>
                        <span className="zq-mono block truncate text-[10.5px] text-[var(--text-3)]">
                          {lastRun.method} · loss {formatLoss(lastRun.finalLoss ?? lastRun.loss)} ·{" "}
                          {formatDuration(lastRun.elapsedSeconds)}
                        </span>
                      </span>
                      <StatusBadge status={lastRun.status} />
                    </button>
                  ) : null}

                  <div className="mt-3 flex items-center gap-2 border-t border-[var(--border-soft)] pt-3">
                    <Button size="sm" icon={<Play className="h-3.5 w-3.5" />} onClick={() => void trainAgain(project)}>
                      Train again
                    </Button>
                    {project.datasetPath ? (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => {
                          const dataset = appStore.get().datasets.find((item) => item.path === project.datasetPath);
                          if (dataset) void validateDataset(dataset.id);
                          navigate("datasets");
                        }}
                      >
                        Dataset
                      </Button>
                    ) : null}
                    <span className="flex-1" />
                    <IconButton title="Delete project" onClick={() => setPendingDelete(project)}>
                      <Trash2 className="h-3.5 w-3.5" />
                    </IconButton>
                  </div>
                </Panel>
              );
            })}
          </div>
        )}

        <div className="mt-6">
          <Panel>
            <PanelHeader
              icon={<Clock className="h-4 w-4" />}
              title="Run history"
              description="Every training run this machine has produced, newest first."
            />
            {runs.length === 0 ? (
              <p className="py-4 text-center text-[12.5px] text-[var(--text-3)]">No runs recorded yet.</p>
            ) : (
              <div className="-mx-1 overflow-x-auto">
                <table className="w-full min-w-[720px] border-collapse">
                  <thead>
                    <tr className="text-left text-[10.5px] uppercase tracking-[0.06em] text-[var(--text-3)]">
                      <th className="px-1 pb-2 font-medium">Run</th>
                      <th className="px-1 pb-2 font-medium">Method</th>
                      <th className="px-1 pb-2 font-medium">Status</th>
                      <th className="px-1 pb-2 font-medium">Steps</th>
                      <th className="px-1 pb-2 font-medium">Loss</th>
                      <th className="px-1 pb-2 font-medium">Duration</th>
                      <th className="px-1 pb-2 font-medium">When</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.slice(0, 40).map((run: RunRecord) => (
                      <tr
                        key={run.id}
                        onClick={() => {
                          void selectRun(run.id);
                          navigate("training");
                        }}
                        className="cursor-pointer border-t border-[var(--border-soft)] transition-colors hover:bg-[var(--hover)]"
                      >
                        <td className="max-w-[220px] truncate px-1 py-2 text-[12.5px]">{run.name}</td>
                        <td className="px-1 py-2">
                          <Badge tone="accent">{run.method}</Badge>
                        </td>
                        <td className="px-1 py-2">
                          <StatusBadge status={run.status} />
                        </td>
                        <td className="zq-mono px-1 py-2 text-[11.5px] text-[var(--text-2)]">
                          {run.step}/{run.totalSteps || "?"}
                        </td>
                        <td className="zq-mono px-1 py-2 text-[11.5px]">
                          {run.finalLoss != null || run.loss != null ? (
                            <span className="inline-flex items-center gap-1">
                              <TrendingDown className="h-3 w-3 text-[var(--green)]" />
                              {formatLoss(run.finalLoss ?? run.loss)}
                            </span>
                          ) : (
                            "—"
                          )}
                        </td>
                        <td className="zq-mono px-1 py-2 text-[11.5px] text-[var(--text-2)]">
                          {formatDuration(run.elapsedSeconds)}
                        </td>
                        <td className="px-1 py-2 text-[11.5px] text-[var(--text-3)]">
                          {formatRelative(run.finishedAt ?? run.startedAt)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Panel>
        </div>
      </PageBody>

      <ConfirmDialog
        open={Boolean(pendingDelete)}
        title="Delete this project?"
        description="The project grouping is removed. Its runs stay in history and no model files are touched."
        confirmLabel="Delete project"
        onConfirm={() => {
          if (pendingDelete) void removeProject(pendingDelete.id);
          setPendingDelete(null);
        }}
        onCancel={() => setPendingDelete(null)}
      />
    </>
  );
}
