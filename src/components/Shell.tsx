import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  Activity,
  Bell,
  ChevronLeft,
  ChevronRight,
  Command,
  Cpu,
  FolderOpen,
  Gauge,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  Search,
  Sun,
  Terminal,
} from "lucide-react";
import { useApp } from "../state/app";
import { useRouter } from "../state/router";
import { NAV_GROUPS, NAV_ITEMS, navItemFor } from "./nav";
import { Badge, Button, Dot, IconButton, Mono, ProgressBar, cx } from "./ui";
import { bytes, duration, percent } from "../lib/format";

export function Shell({ children, onOpenPalette }: { children: ReactNode; onOpenPalette: () => void }) {
  const { hardware, live, jobs, settings, saveSettings, appInfo, registry, ready } = useApp();
  const router = useRouter();
  const [collapsed, setCollapsed] = useState(false);
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [search, setSearch] = useState("");

  const activeJob = useMemo(
    () => jobs.find((job) => ["running", "paused", "queued"].includes(job.state)) ?? null,
    [jobs],
  );
  const current = navItemFor(router.path);
  const project = registry?.projects?.find((item) => item.id === settings?.defaultBackend) ?? null;

  const theme = settings?.theme ?? "system";
  const cycleTheme = useCallback(() => {
    const order: Array<"system" | "light" | "dark"> = ["system", "dark", "light"];
    const next = order[(order.indexOf(theme as any) + 1) % order.length];
    void saveSettings({ theme: next });
  }, [saveSettings, theme]);

  useEffect(() => {
    const stored = localStorage.getItem("zx.sidebar");
    if (stored === "collapsed") setCollapsed(true);
  }, []);
  useEffect(() => {
    localStorage.setItem("zx.sidebar", collapsed ? "collapsed" : "expanded");
  }, [collapsed]);

  const gpu = live?.gpus?.[0] ?? hardware?.gpus?.[0] ?? null;
  const gpuTotal = gpu?.memory_total ?? null;
  const gpuUsed = gpu?.memory_used_mb ? gpu.memory_used_mb * 1024 * 1024 : null;

  return (
    <div className="flex h-full w-full overflow-hidden bg-surface-0">
      {/* Sidebar */}
      <aside
        className={cx(
          "flex shrink-0 flex-col border-r border-line-soft bg-surface-1 transition-[width] duration-200",
          collapsed ? "w-[52px]" : "w-56",
        )}
      >
        <div className="flex h-11 items-center gap-2 border-b border-line-soft px-2.5">
          <div className="flex h-6 w-6 items-center justify-center rounded-md border border-line-soft bg-surface-3 text-accent">
            <Activity size={13} />
          </div>
          {!collapsed ? (
            <div className="min-w-0 flex-1">
              <div className="truncate text-xs font-semibold tracking-tight">ZeqouXTraining</div>
              <div className="truncate text-[9.5px] uppercase tracking-wider text-ink-3">Train. Tune. Test. Build.</div>
            </div>
          ) : null}
          <IconButton
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            icon={collapsed ? <PanelLeftOpen size={13} /> : <PanelLeftClose size={13} />}
            onClick={() => setCollapsed((value) => !value)}
          />
        </div>

        <nav className="flex-1 overflow-y-auto px-1.5 py-2">
          {NAV_GROUPS.map((group) => {
            const items = NAV_ITEMS.filter((item) => item.group === group);
            return (
              <div key={group} className="mb-2">
                {!collapsed ? (
                  <div className="px-2 py-1 text-[9.5px] font-medium uppercase tracking-wider text-ink-3">{group}</div>
                ) : null}
                {items.map((item) => {
                  const active = current.id === item.id;
                  const running = item.id === "training" && activeJob;
                  const badge =
                    item.id === "jobs" ? jobs.filter((job) => ["running", "queued"].includes(job.state)).length : 0;
                  return (
                    <button
                      key={item.id}
                      title={collapsed ? `${item.label} — ${item.description}` : item.description}
                      onClick={() => router.navigate(item.path)}
                      className={cx(
                        "group relative flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs transition-colors",
                        active ? "bg-accent/12 text-ink-0" : "text-ink-1 hover:bg-surface-2 hover:text-ink-0",
                      )}
                    >
                      <span className={cx("shrink-0", active ? "text-accent" : "text-ink-2 group-hover:text-ink-1")}>
                        {item.icon}
                      </span>
                      {!collapsed ? <span className="flex-1 truncate">{item.label}</span> : null}
                      {running ? <Dot tone="info" pulse /> : null}
                      {badge ? <Badge tone="muted">{badge}</Badge> : null}
                    </button>
                  );
                })}
              </div>
            );
          })}
        </nav>

        {!collapsed ? (
          <div className="border-t border-line-soft p-2">
            <div className="rounded-md border border-line-soft bg-surface-2 p-2">
              <div className="flex items-center justify-between text-2xs text-ink-3">
                <span className="uppercase tracking-wide">Workspace</span>
                <FolderOpen size={11} />
              </div>
              <div className="mt-0.5 truncate font-mono text-[10px] text-ink-1" title={settings?.workspace ?? ""}>
                {settings?.workspace ?? (ready ? "not configured" : "loading…")}
              </div>
              <div className="mt-1.5 truncate text-2xs text-ink-3">
                {registry?.models?.length ?? 0} models · {registry?.datasets?.length ?? 0} datasets ·{" "}
                {jobs.length} jobs
              </div>
            </div>
          </div>
        ) : null}
      </aside>

      {/* Main column */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-11 shrink-0 items-center gap-2 border-b border-line-soft bg-surface-1 px-3">
          <div className="flex items-center gap-1">
            <IconButton title="Back" icon={<ChevronLeft size={14} />} disabled={!router.canGoBack} onClick={router.back} />
            <IconButton
              title="Forward"
              icon={<ChevronRight size={14} />}
              disabled={!router.canGoForward}
              onClick={router.forward}
            />
          </div>
          <div className="hidden items-center gap-2 lg:flex">
            <span className="text-xs font-medium">{current.label}</span>
            <span className="text-2xs text-ink-3">{current.description}</span>
          </div>
          <form
            className="relative ml-2 max-w-md flex-1"
            onSubmit={(event) => {
              event.preventDefault();
              if (search.trim()) {
                onOpenPalette();
              }
            }}
          >
            <Search size={13} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-3" />
            <input
              className="field cursor-pointer pl-8 pr-16"
              placeholder={project ? `Search ${project.name}…` : "Search models, datasets, jobs, commands…"}
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              onClick={onOpenPalette}
              onFocus={onOpenPalette}
              readOnly
            />
            <span className="pointer-events-none absolute right-2 top-1/2 flex -translate-y-1/2 items-center gap-1">
              <span className="kbd flex items-center gap-0.5">
                <Command size={9} />K
              </span>
            </span>
          </form>

          <div className="ml-auto flex items-center gap-2">
            <ResourcePill
              icon={<Cpu size={12} />}
              label={gpu ? gpu.name : "CPU"}
              value={
                gpu
                  ? gpuUsed !== null
                    ? `${percent(((gpuUsed / (gpuTotal || 1)) * 100) as number)} VRAM`
                    : "GPU"
                  : `RAM ${percent(live?.ram?.percent ?? null)}`
              }
              title={
                gpu
                  ? `GPU memory used${gpu.utilization_gpu !== null && gpu.utilization_gpu !== undefined ? `, utilisation ${gpu.utilization_gpu}%` : ""}`
                  : `RAM total ${bytes(hardware?.memory?.total ?? null)}`
              }
              onClick={() => router.navigate("/hardware")}
            />
            <ResourcePill
              icon={<Activity size={12} />}
              label="CPU"
              value={percent(live?.cpu?.percent ?? null)}
              title={`${live?.cpu?.logical_cores ?? hardware?.cpu?.logical_cores ?? "?"} logical cores`}
              onClick={() => router.navigate("/hardware")}
            />
            {activeJob ? (
              <button
                className="flex items-center gap-2 rounded-md border border-info/30 bg-info/10 px-2 py-1 text-2xs text-info transition-colors hover:bg-info/15"
                onClick={() => router.navigate(`/training/${encodeURIComponent(activeJob.job_id)}`)}
                title="Open the live training monitor"
              >
                <Dot tone="info" pulse />
                <span className="max-w-[220px] truncate">
                  {activeJob.method ?? activeJob.kind} · {activeJob.job_id}
                </span>
                {activeJob.total_steps ? (
                  <span className="tabular-nums">
                    {activeJob.step ?? 0}/{activeJob.total_steps}
                  </span>
                ) : null}
              </button>
            ) : null}
            <div className="relative">
              <IconButton
                title="Recent activity"
                icon={<Bell size={14} />}
                onClick={() => setNotificationsOpen((value) => !value)}
              />
              {notificationsOpen ? (
                <div className="absolute right-0 top-7 z-40 w-80 animate-scale-in rounded-lg border border-line-soft bg-surface-1 p-2 shadow-2xl">
                  <div className="px-1 pb-1 text-2xs uppercase tracking-wide text-ink-3">Recent jobs</div>
                  {jobs.slice(0, 6).map((job) => (
                    <button
                      key={job.job_id}
                      className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-2xs hover:bg-surface-2"
                      onClick={() => {
                        setNotificationsOpen(false);
                        router.navigate(`/training/${encodeURIComponent(job.job_id)}`);
                      }}
                    >
                      <Dot tone={job.state === "completed" ? "ok" : job.state === "failed" ? "danger" : "info"} />
                      <span className="flex-1 truncate">{job.job_id}</span>
                      <span className="text-ink-3">{job.state}</span>
                    </button>
                  ))}
                  {!jobs.length ? <div className="px-2 py-3 text-2xs text-ink-3">No jobs yet.</div> : null}
                </div>
              ) : null}
            </div>
            <IconButton
              title={`Theme: ${theme} (click to change)`}
              icon={theme === "dark" ? <Moon size={14} /> : theme === "light" ? <Sun size={14} /> : <Gauge size={14} />}
              onClick={cycleTheme}
            />
          </div>
        </header>

        <main className="min-h-0 flex-1 overflow-y-auto p-4">{children}</main>

        <footer className="flex h-7 shrink-0 items-center gap-3 border-t border-line-soft bg-surface-1 px-3 text-2xs text-ink-3">
          <span className="flex items-center gap-1.5">
            <Dot tone={appInfo ? "ok" : "warn"} />
            engine zxtrain {appInfo?.version ? `v${appInfo.version}` : ""}
          </span>
          <Mono className="text-ink-3">
            {hardware?.os?.python ? `python ${hardware.os.python}` : "python unknown"}
          </Mono>
          <span className="truncate" title={hardware?.torch?.version ? `torch ${hardware.torch.version}` : "PyTorch not installed"}>
            {hardware?.torch?.installed ? `torch ${hardware.torch.version}` : "torch not installed"}
          </span>
          <span className="truncate">
            {gpu ? `${gpu.name}` : hardware?.cpu?.model ?? "CPU unknown"}
          </span>
          <span className="ml-auto flex items-center gap-3">
            <span>RAM {bytes(live?.ram?.used ?? null)} / {bytes(live?.ram?.total ?? hardware?.memory?.total ?? null)}</span>
            {activeJob?.eta_seconds ? <span>ETA {duration(activeJob.eta_seconds)}</span> : null}
            <button className="flex items-center gap-1 hover:text-ink-1" onClick={() => router.navigate("/jobs")}>
              <Terminal size={11} /> {jobs.filter((job) => ["running", "queued"].includes(job.state)).length} active
            </button>
          </span>
        </footer>
      </div>
    </div>
  );
}

function ResourcePill({
  icon,
  label,
  value,
  title,
  onClick,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  title?: string;
  onClick?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className="hidden items-center gap-1.5 rounded-md border border-line-soft bg-surface-2 px-2 py-1 text-2xs text-ink-1 transition-colors hover:bg-surface-3 md:flex"
    >
      <span className="text-ink-3">{icon}</span>
      <span className="max-w-[120px] truncate">{label}</span>
      <span className="tabular-nums text-ink-0">{value}</span>
    </button>
  );
}

export function JobProgressInline({ job }: { job: { step?: number | null; total_steps?: number | null; loss?: number | null } }) {
  if (!job.total_steps) return null;
  return (
    <div className="w-40">
      <ProgressBar
        value={job.step ?? 0}
        max={job.total_steps}
        label={
          <span className="tabular-nums">
            {job.step ?? 0}/{job.total_steps} steps{job.loss !== null && job.loss !== undefined ? ` · loss ${job.loss.toFixed(4)}` : ""}
          </span>
        }
      />
    </div>
  );
}

export function Banner({ children, tone = "info" }: { children: ReactNode; tone?: "info" | "warn" | "danger" }) {
  const tones = {
    info: "border-info/30 bg-info/8 text-info",
    warn: "border-warn/30 bg-warn/8 text-warn",
    danger: "border-danger/30 bg-danger/8 text-danger",
  } as const;
  return <div className={cx("rounded-lg border px-3 py-2 text-xs", tones[tone])}>{children}</div>;
}

export function HeaderButton({ children, ...rest }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <Button variant="default" size="sm" {...rest}>
      {children}
    </Button>
  );
}
