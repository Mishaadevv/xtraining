import type { ReactNode } from "react";
import {
  Activity,
  Brain,
  Cpu,
  Database,
  FlaskConical,
  FolderKanban,
  Play,
  Plug,
  Settings as SettingsIcon,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Badge, Dot } from "@/components/ui/primitives";
import { useStore } from "@/state/store";
import { navigate, type Page } from "@/state/appStore";
import { appStore } from "@/state/appStore";

const NAV: { page: Page; label: string; icon: ReactNode; group: string }[] = [
  { page: "projects", label: "Projects", icon: <FolderKanban className="h-3.5 w-3.5" />, group: "Work" },
  { page: "new", label: "New training", icon: <Play className="h-3.5 w-3.5" />, group: "Work" },
  { page: "training", label: "Training", icon: <Activity className="h-3.5 w-3.5" />, group: "Work" },
  { page: "models", label: "Models", icon: <Brain className="h-3.5 w-3.5" />, group: "Library" },
  { page: "datasets", label: "Datasets", icon: <Database className="h-3.5 w-3.5" />, group: "Library" },
  { page: "playground", label: "Playground", icon: <FlaskConical className="h-3.5 w-3.5" />, group: "Library" },
  { page: "hardware", label: "Hardware", icon: <Cpu className="h-3.5 w-3.5" />, group: "System" },
  { page: "settings", label: "Settings", icon: <SettingsIcon className="h-3.5 w-3.5" />, group: "System" },
];

const GROUPS = ["Work", "Library", "System"];

export function Sidebar() {
  const { page, env, runs, wizard, busy, datasets, models, projects } = useStore(appStore);
  const activeRun = runs.find((run) => run.status === "running" || run.status === "starting") ?? null;

  const cudaReady = Boolean(env.hardware?.cuda_ready);
  const pythonReady = Boolean(env.python?.available);

  return (
    <aside className="flex w-[236px] shrink-0 flex-col border-r border-[var(--border-soft)] bg-[var(--bg)] px-3 py-4">
      <div className="flex items-center gap-2.5 px-1.5 pb-4">
        <img src="./ico.png" alt="Zeqou" className="h-6 w-6 rounded-[7px] object-cover" />
        <div className="flex min-w-0 flex-col leading-[1.15]">
          <span className="text-[11px] font-extrabold tracking-[0.14em]">ZEQOU</span>
          <span className="text-[11px] text-[var(--text-3)]">XTraining</span>
        </div>
      </div>

      {activeRun ? (
        <button
          type="button"
          onClick={() => navigate("training")}
          className="mb-3 flex w-full items-center gap-2.5 rounded-[10px] border border-[var(--border)] bg-[var(--panel-2)] px-2.5 py-2 text-left transition-colors hover:bg-[var(--hover)]"
        >
          <span className="relative flex h-1.5 w-1.5 shrink-0">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[var(--acc)] opacity-70" />
            <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-[var(--acc)]" />
          </span>
          <span className="min-w-0 flex-1">
            <span className="block truncate text-[12px] font-medium">{activeRun.name}</span>
            <span className="zq-mono block text-[10.5px] text-[var(--text-3)]">
              {Math.round(activeRun.progress)}% · step {activeRun.step}/{activeRun.totalSteps || "?"}
            </span>
          </span>
        </button>
      ) : (
        <button
          type="button"
          onClick={() => navigate("new")}
          className="mb-3 flex w-full items-center gap-2 rounded-[10px] border border-transparent px-2.5 py-2 text-[12.5px] font-medium text-[var(--text-2)] transition-colors hover:bg-[var(--hover)] hover:text-[var(--text)]"
        >
          <Play className="h-3.5 w-3.5" />
          Start a training run
        </button>
      )}

      <nav className="zq-scroll flex min-h-0 flex-1 flex-col gap-0.5">
        {GROUPS.map((group) => (
          <div key={group} className="mt-3 first:mt-0">
            <div className="px-2 pb-1 text-[10px] font-semibold uppercase tracking-[0.1em] text-[var(--text-3)]">
              {group}
            </div>
            {NAV.filter((item) => item.group === group).map((item) => {
              const active = page === item.page;
              const count =
                item.page === "datasets" ? datasets.length
                  : item.page === "models" ? models.length
                    : item.page === "projects" ? projects.length
                      : 0;
              return (
                <button
                  key={item.page}
                  type="button"
                  onClick={() => navigate(item.page)}
                  className={cn(
                    "group flex w-full items-center gap-2.5 rounded-[9px] px-2.5 py-[7px] text-left",
                    "transition-colors duration-[130ms]",
                    active
                      ? "bg-[var(--panel-2)] text-[var(--text)]"
                      : "text-[var(--text-2)] hover:bg-[var(--hover)] hover:text-[var(--text)]",
                  )}
                >
                  <span
                    className={cn(
                      "shrink-0 transition-colors",
                      active ? "text-[var(--acc)]" : "text-[var(--text-3)] group-hover:text-[var(--text-2)]",
                    )}
                  >
                    {item.icon}
                  </span>
                  <span className="flex-1 truncate text-[12.5px] font-medium">{item.label}</span>
                  {item.page === "new" && busy.autoConfig ? (
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[var(--acc)]" />
                  ) : null}
                  {count > 0 ? (
                    <span className="zq-mono text-[10.5px] text-[var(--text-3)]">{count}</span>
                  ) : null}
                  {item.page === "new" && wizard.config && !active ? (
                    <span className="h-1.5 w-1.5 rounded-full bg-[var(--acc)]" />
                  ) : null}
                </button>
              );
            })}
          </div>
        ))}
      </nav>

      <div className="mt-3 space-y-2 border-t border-[var(--border-soft)] pt-3">
        <div className="flex items-center justify-between px-1.5">
          <span className="flex items-center gap-1.5 text-[11px] text-[var(--text-3)]">
            <Plug className="h-3 w-3" />
            CUDA
          </span>
          <Badge tone={cudaReady ? "good" : "warn"}>
            <Dot tone={cudaReady ? "good" : "warn"} />
            {cudaReady ? "Ready" : "CPU only"}
          </Badge>
        </div>
        <div className="flex items-center justify-between px-1.5">
          <span className="text-[11px] text-[var(--text-3)]">Python</span>
          <Badge tone={pythonReady ? "good" : "bad"}>
            <Dot tone={pythonReady ? "good" : "bad"} />
            {env.python?.version ?? "not found"}
          </Badge>
        </div>
      </div>
    </aside>
  );
}

export function PageHeader({
  title,
  subtitle,
  actions,
  icon,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <header className="flex h-[62px] shrink-0 items-center justify-between gap-4 border-b border-[var(--border-soft)] px-6">
      <div className="flex min-w-0 items-center gap-3">
        {icon ? (
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-[10px] bg-[var(--panel-2)] text-[var(--text-2)]">
            {icon}
          </span>
        ) : null}
        <div className="min-w-0">
          <h1 className="truncate text-[14px] font-semibold leading-5">{title}</h1>
          {subtitle ? (
            <p className="truncate text-[12px] leading-[18px] text-[var(--text-3)]">{subtitle}</p>
          ) : null}
        </div>
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
    </header>
  );
}

export function PageBody({
  children,
  wide = false,
  className,
}: {
  children: ReactNode;
  wide?: boolean;
  className?: string;
}) {
  return (
    <div className="zq-scroll min-h-0 flex-1">
      <div className={cn("mx-auto px-6 py-5", wide ? "max-w-[1400px]" : "max-w-[1080px]", className)}>
        {children}
      </div>
    </div>
  );
}
