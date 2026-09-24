/** Helpers shared by the pages: async engine calls, entity pickers, error surfaces. */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { RefreshCw } from "lucide-react";
import { api, bridgeAvailable } from "../lib/api";
import { basename } from "../lib/format";
import type { JobSummary, Registry } from "../lib/types";
import { useApp } from "../state/app";
import { useRouter } from "../state/router";
import { Badge, Button, Callout, Dot, Field, Select, SkeletonBlock, cx } from "../components/ui";

/** Run an engine command and keep its result, error and loading state. */
export function useEngine<T = any>(
  method: string,
  payload: Record<string, unknown> = {},
  options: { auto?: boolean; timeout?: number; deps?: unknown[] } = {},
) {
  const { auto = true } = options;
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<any | null>(null);
  const [loading, setLoading] = useState(auto);
  const payloadRef = useRef(payload);
  payloadRef.current = payload;
  const deps = options.deps ?? [method, JSON.stringify(payload)];

  const run = useCallback(async () => {
    if (!bridgeAvailable) {
      setError({
        code: "bridge_unavailable",
        message: "The desktop bridge is not available in this window.",
        hint: "Run the app with `npm run dev`.",
      });
      setLoading(false);
      return null;
    }
    setLoading(true);
    const result = await api.callSafe<T>(method, payloadRef.current, { timeout: options.timeout ?? 180_000 });
    if (result.error) {
      setError(result.error);
      setData(null);
    } else {
      setError(null);
      setData(result.data);
    }
    setLoading(false);
    return result.data;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    if (auto) void run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run, auto]);

  return { data, error, loading, reload: run, setData };
}

export function ErrorPanel({ error, onRetry, title = "The engine reported a problem" }: { error: any; onRetry?: () => void; title?: string }) {
  if (!error) return null;
  return (
    <Callout
      tone="danger"
      title={error.message ?? title}
      hint={error.hint}
      detail={error.detail}
      onRetry={onRetry}
    >
      {error.context && Object.keys(error.context).length ? (
        <span className="font-mono text-2xs">{JSON.stringify(error.context)}</span>
      ) : null}
    </Callout>
  );
}

export function Loading({ lines = 3, label }: { lines?: number; label?: string }) {
  return (
    <div className="space-y-2">
      {label ? <div className="text-2xs text-ink-3">{label}</div> : null}
      <SkeletonBlock lines={lines} />
    </div>
  );
}

export function JobStateBadge({ state }: { state: string }) {
  const tone =
    state === "completed"
      ? "ok"
      : state === "failed"
        ? "danger"
        : state === "running"
          ? "info"
          : state === "paused" || state === "interrupted"
            ? "warn"
            : "muted";
  return (
    <Badge tone={tone as any}>
      <Dot tone={tone as any} pulse={state === "running"} />
      {state}
    </Badge>
  );
}

export function PathText({ value, className }: { value: string | null | undefined; className?: string }) {
  if (!value) return <span className="text-ink-3">—</span>;
  return (
    <span className={cx("font-mono text-2xs text-ink-2", className)} title={value}>
      {value}
    </span>
  );
}

export function ModelPicker({
  value,
  onChange,
  label = "Base model",
  filter,
  allowScratch,
}: {
  value: string | null;
  onChange: (path: string | null) => void;
  label?: string;
  filter?: (entry: Registry["models"][number]) => boolean;
  allowScratch?: boolean;
}) {
  const { registry } = useApp();
  const models = useMemo(() => (registry?.models ?? []).filter((entry) => (filter ? filter(entry) : true)), [registry, filter]);
  return (
    <Field label={label} hint={models.length ? `${models.length} models in the library` : "No models imported yet"}>
      <Select value={value ?? ""} onChange={(event) => onChange(event.target.value || null)}>
        <option value="">{allowScratch ? "Train from scratch (no base model)" : "Select a model…"}</option>
        {models.map((entry) => (
          <option key={entry.id} value={entry.path}>
            {entry.name} — {basename(entry.path)}
          </option>
        ))}
      </Select>
    </Field>
  );
}

export function DatasetPicker({
  value,
  onChange,
  label = "Dataset",
  multiple,
  multipleValues,
  onMultipleChange,
}: {
  value?: string | null;
  onChange?: (path: string | null) => void;
  label?: string;
  multiple?: boolean;
  multipleValues?: string[];
  onMultipleChange?: (paths: string[]) => void;
}) {
  const { registry } = useApp();
  const datasets = registry?.datasets ?? [];
  if (multiple) {
    const selected = multipleValues ?? [];
    return (
      <Field
        label={label}
        hint={selected.length > 1 ? `${selected.length} datasets will be mixed` : "Select one or more datasets"}
      >
        <div className="max-h-40 overflow-y-auto rounded-md border border-line-soft bg-surface-2 p-1.5">
          {!datasets.length ? <div className="px-1 py-2 text-2xs text-ink-3">No datasets imported yet.</div> : null}
          {datasets.map((entry) => {
            const checked = selected.includes(entry.path);
            return (
              <label key={entry.id} className="flex cursor-pointer items-center gap-2 rounded px-1.5 py-1 hover:bg-surface-3">
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => {
                    const next = checked
                      ? selected.filter((item) => item !== entry.path)
                      : [...selected, entry.path];
                    onMultipleChange?.(next);
                  }}
                />
                <span className="min-w-0 flex-1 truncate text-xs">{entry.name}</span>
                <PathText value={entry.path} />
              </label>
            );
          })}
        </div>
      </Field>
    );
  }
  return (
    <Field label={label} hint={datasets.length ? `${datasets.length} datasets in the library` : "No datasets imported yet"}>
      <Select value={value ?? ""} onChange={(event) => onChange?.(event.target.value || null)}>
        <option value="">Select a dataset…</option>
        {datasets.map((entry) => (
          <option key={entry.id} value={entry.path}>
            {entry.name}
          </option>
        ))}
      </Select>
    </Field>
  );
}

export function RefreshButton({ onClick, loading }: { onClick: () => void; loading?: boolean }) {
  return (
    <Button size="sm" variant="subtle" icon={<RefreshCw size={12} />} loading={loading} onClick={onClick}>
      Refresh
    </Button>
  );
}

/** A compact row of job facts used on several pages. */
export function JobRow({ job, onOpen }: { job: JobSummary; onOpen: () => void }) {
  return (
    <button
      onClick={onOpen}
      className="flex w-full items-center gap-3 rounded-md border border-line-soft bg-surface-2 px-3 py-2 text-left transition-colors hover:bg-surface-3"
    >
      <JobStateBadge state={job.state} />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-xs text-ink-0">{job.job_id}</span>
        <span className="block truncate text-2xs text-ink-3">
          {job.method ?? job.kind} · {job.backend ?? "auto"} · {job.model ? basename(job.model) : "no model"}
        </span>
      </span>
      <span className="text-2xs tabular-nums text-ink-2">
        {job.step !== null && job.step !== undefined && job.total_steps ? `${job.step}/${job.total_steps} steps` : ""}
      </span>
    </button>
  );
}

export function useInterval(callback: () => void, delay: number | null) {
  const ref = useRef(callback);
  ref.current = callback;
  useEffect(() => {
    if (delay === null) return;
    const timer = window.setInterval(() => ref.current(), delay);
    return () => window.clearInterval(timer);
  }, [delay]);
}

export function NotFound({ what, onBack }: { what: string; onBack: () => void }) {
  return (
    <Callout tone="warn" title={`${what} is no longer in the workspace`} hint="It may have been moved or deleted outside the app.">
      <Button size="sm" onClick={onBack}>
        Go back
      </Button>
    </Callout>
  );
}

export function useNavigatePath() {
  const router = useRouter();
  return (path: string) => router.navigate(path);
}
