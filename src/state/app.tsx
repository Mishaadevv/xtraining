/**
 * Application state.
 *
 * One provider holds everything the shell needs: the detected environment, the
 * live resource sample, the job list with live events, and the workspace
 * registry. Long operations always go through the engine; this store only keeps
 * the latest real answer.
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { api, bridgeAvailable } from "../lib/api";
import type {
  Backend,
  EnvironmentReport,
  HardwareReport,
  JobSummary,
  LiveSample,
  MethodAvailability,
  Registry,
  Settings,
  UpdateState,
} from "../lib/types";

export interface Toast {
  id: string;
  title: string;
  body?: string;
  tone: "info" | "ok" | "warn" | "danger";
  hint?: string;
  detail?: string;
}

interface AppState {
  ready: boolean;
  bootError: { code: string; message: string; hint?: string; detail?: string } | null;
  appInfo: any;
  settings: Settings | null;
  registry: Registry | null;
  hardware: HardwareReport | null;
  environment: EnvironmentReport | null;
  live: LiveSample | null;
  backends: Backend[];
  methods: MethodAvailability[];
  jobs: JobSummary[];
  jobsLoading: boolean;
  updates: UpdateState | null;
  toasts: Toast[];
  activeJobId: string | null;
  refreshHardware: (live?: boolean) => Promise<void>;
  refreshJobs: () => Promise<void>;
  refreshRegistry: () => Promise<void>;
  refreshUpdates: () => Promise<void>;
  saveSettings: (patch: Partial<Settings>) => Promise<void>;
  updateJob: (jobId: string, event: any) => void;
  setActiveJob: (jobId: string | null) => void;
  toast: (toast: Omit<Toast, "id"> & { id?: string }) => void;
  dismiss: (id: string) => void;
  reportError: (error: any, title?: string) => void;
}

const AppContext = createContext<AppState | null>(null);

const emptyRegistry: Registry = {
  projects: [],
  models: [],
  datasets: [],
  presets: [],
  evaluations: [],
  servers: [],
  conversations: [],
  workflows: [],
  notes: [],
};

export function AppProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [bootError, setBootError] = useState<AppState["bootError"]>(null);
  const [appInfo, setAppInfo] = useState<any>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [registry, setRegistry] = useState<Registry | null>(null);
  const [hardware, setHardware] = useState<HardwareReport | null>(null);
  const [environment, setEnvironment] = useState<EnvironmentReport | null>(null);
  const [live, setLive] = useState<LiveSample | null>(null);
  const [backends, setBackends] = useState<Backend[]>([]);
  const [methods, setMethods] = useState<MethodAvailability[]>([]);
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [jobsLoading, setJobsLoading] = useState(false);
  const [updates, setUpdates] = useState<UpdateState | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const toastsRef = useRef<Toast[]>([]);
  toastsRef.current = toasts;

  const toast = useCallback((entry: Omit<Toast, "id"> & { id?: string }) => {
    const id = entry.id ?? `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    setToasts((current) => {
      const next = [{ ...entry, id } as Toast, ...current.filter((item) => item.id !== id)];
      return next.slice(0, 4);
    });
    window.setTimeout(() => setToasts((current) => current.filter((item) => item.id !== id)), 9000);
  }, []);

  const dismiss = useCallback((id: string) => {
    setToasts((current) => current.filter((item) => item.id !== id));
  }, []);

  const reportError = useCallback(
    (error: any, title = "Operation failed") => {
      const structured = error?.structured ?? error ?? {};
      toast({
        title,
        body: structured.message ?? String(error?.message ?? error),
        hint: structured.hint,
        detail: structured.detail,
        tone: "danger",
      });
    },
    [toast],
  );

  const refreshHardware = useCallback(async (liveOnly = false) => {
    if (!bridgeAvailable) return;
    if (!liveOnly) {
      const { data } = await api.callSafe<HardwareReport>("hardware.detect", {}, { timeout: 120_000 });
      if (data) setHardware(data);
    }
    const { data: sample } = await api.callSafe<LiveSample>("hardware.live", {}, { timeout: 60_000 });
    if (sample) setLive(sample);
  }, []);

  const refreshJobs = useCallback(async () => {
    if (!bridgeAvailable) return;
    setJobsLoading(true);
    try {
      const list = await api.jobs.list();
      setJobs(list ?? []);
    } catch (error) {
      reportError(error, "Job list unavailable");
    } finally {
      setJobsLoading(false);
    }
  }, [reportError]);

  const refreshRegistry = useCallback(async () => {
    if (!bridgeAvailable) return;
    try {
      const data = await api.registry.get();
      setRegistry({ ...emptyRegistry, ...data });
    } catch (error) {
      reportError(error, "Workspace registry unavailable");
    }
  }, [reportError]);

  const refreshUpdates = useCallback(async () => {
    if (!bridgeAvailable) return;
    try {
      setUpdates(await api.updates.state());
    } catch {
      // The update state is informational; a missing one must not look like a crash.
    }
  }, []);

  const saveSettings = useCallback(
    async (patch: Partial<Settings>) => {
      try {
        const updated = await api.settings.set(patch as Record<string, unknown>);
        setSettings(updated);
        if (patch.workspace) {
          await refreshRegistry();
          await refreshJobs();
        }
        return undefined;
      } catch (error) {
        reportError(error, "Settings were not saved");
      }
    },
    [refreshJobs, refreshRegistry, reportError],
  );

  /** Fold one live job event into the job list. */
  const updateJob = useCallback((jobId: string, event: any) => {
    setJobs((current) =>
      current.map((job) => {
        if (job.job_id !== jobId) return job;
        const next: JobSummary = { ...job, metrics_state: { ...(job.metrics_state ?? {}) } };
        next.message = event.message ?? next.message;
        if (event.type === "metrics") {
          for (const key of [
            "step",
            "total_steps",
            "loss",
            "eval_loss",
            "learning_rate",
            "tokens_per_second",
            "eta_seconds",
            "elapsed_seconds",
            "epoch",
          ]) {
            if (event[key] !== undefined && event[key] !== null) {
              (next as any)[key] = event[key];
              next.metrics_state![key] = event[key];
            }
          }
          next.state = "running";
        }
        if (event.type === "checkpoint") {
          next.checkpoints = [
            ...(next.checkpoints ?? []).filter((item) => item.name !== event.name),
            { name: event.name, path: event.path, step: event.step, kind: event.kind },
          ];
        }
        if (event.type === "done") {
          next.state = event.result?.status ?? "completed";
          next.result = event.result;
        }
        if (event.type === "error") {
          next.state = "failed";
          next.error = { code: "job_failed", message: event.message ?? "Job failed" };
        }
        return next;
      }),
    );
  }, []);

  // Bootstrap: settings, registry, environment, capabilities, jobs.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!bridgeAvailable) {
        setBootError({
          code: "bridge_unavailable",
          message: "This window is not running inside the desktop app.",
          hint: "Start it with `npm run dev`, or build it with `npm run build && electron .`.",
        });
        setReady(true);
        return;
      }
      try {
        const [info, loadedSettings, loadedRegistry] = await Promise.all([
          api.appInfo(),
          api.settings.get(),
          api.registry.get(),
        ]);
        if (cancelled) return;
        setAppInfo(info);
        setSettings(loadedSettings);
        setRegistry({ ...emptyRegistry, ...loadedRegistry });

        const capabilities = await api.callSafe<any>("engine.capabilities", {}, { timeout: 180_000 });
        if (cancelled) return;
        if (capabilities.error) {
          setBootError(capabilities.error);
        } else {
          const payload = capabilities.data;
          setHardware(payload?.hardware ?? null);
          setLive(payload?.hardware_live ?? null);
          setBackends(payload?.backends ?? []);
          setMethods(payload?.methods ?? []);
          setEnvironment(payload?.environment ?? null);
        }
        const list = await api.jobs.list();
        if (!cancelled) setJobs(list ?? []);
        await refreshUpdates();
      } catch (error: any) {
        if (!cancelled) {
          setBootError(error?.structured ?? { code: "boot_failed", message: String(error?.message ?? error) });
        }
      } finally {
        if (!cancelled) setReady(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [refreshUpdates]);

  // Update state pushed by the main process (checking, downloading, ready).
  useEffect(() => {
    if (!bridgeAvailable) return;
    const off = api.updates.onState((state) => setUpdates(state));
    return off;
  }, []);

  // Live job events from the main process.
  useEffect(() => {
    if (!bridgeAvailable) return;
    const offEvent = api.jobs.onEvent(({ jobId, event }) => {
      updateJob(jobId, event);
      if (event?.type === "done" || event?.type === "error") void refreshJobs();
    });
    const offReconciled = api.jobs.onReconciled((findings) => {
      const interesting = (findings ?? []).filter((item) => item.action !== "reconnected");
      for (const finding of interesting) {
        toast({
          title: finding.state === "running" ? "Job reconnected" : "Job interrupted",
          body: `${finding.job_id}: ${finding.message}`,
          tone: finding.state === "running" ? "info" : "warn",
        });
      }
      void refreshJobs();
    });
    return () => {
      offEvent();
      offReconciled();
    };
  }, [refreshJobs, toast, updateJob]);

  // Live resource sampling while the app is visible.
  useEffect(() => {
    if (!bridgeAvailable) return;
    let cancelled = false;
    const tick = async () => {
      if (cancelled || document.hidden) return;
      const { data } = await api.callSafe<LiveSample>("hardware.live", {}, { timeout: 60_000 });
      if (!cancelled && data) setLive(data);
    };
    const timer = window.setInterval(tick, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  // Refresh the job list periodically as a safety net behind the event stream.
  useEffect(() => {
    if (!bridgeAvailable) return;
    const timer = window.setInterval(() => void refreshJobs(), 15000);
    return () => window.clearInterval(timer);
  }, [refreshJobs]);

  const value = useMemo<AppState>(
    () => ({
      ready,
      bootError,
      appInfo,
      settings,
      registry,
      hardware,
      environment,
      live,
      backends,
      methods,
      jobs,
      jobsLoading,
      updates,
      toasts,
      activeJobId,
      refreshHardware,
      refreshJobs,
      refreshRegistry,
      refreshUpdates,
      saveSettings,
      updateJob,
      setActiveJob: setActiveJobId,
      toast,
      dismiss,
      reportError,
    }),
    [
      ready,
      bootError,
      appInfo,
      settings,
      registry,
      hardware,
      environment,
      live,
      backends,
      methods,
      jobs,
      jobsLoading,
      updates,
      toasts,
      activeJobId,
      refreshHardware,
      refreshJobs,
      refreshRegistry,
      refreshUpdates,
      saveSettings,
      updateJob,
      toast,
      dismiss,
      reportError,
    ],
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp(): AppState {
  const context = useContext(AppContext);
  if (!context) throw new Error("useApp must be used inside AppProvider");
  return context;
}

/** Subscribe to raw sidecar stream events for the playground. */
export function useSidecarStream(handler: (event: any) => void) {
  const ref = useRef(handler);
  ref.current = handler;
  useEffect(() => {
    if (!bridgeAvailable) return;
    return api.sidecar.onStream((event) => ref.current(event));
  }, []);
}
