import { useEffect, useMemo, useRef, useState } from "react";
import { Command, CornerDownLeft, Database, Activity, Brain, Search } from "lucide-react";
import { useApp } from "../state/app";
import { useRouter } from "../state/router";
import { NAV_ITEMS, QUICK_ACTIONS } from "./nav";
import { Badge, Mono, cx } from "./ui";

interface Entry {
  id: string;
  label: string;
  hint?: string;
  group: string;
  icon: React.ReactNode;
  run: () => void;
  keywords: string[];
}

export function CommandPalette({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { registry, jobs } = useApp();
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [index, setIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const entries = useMemo<Entry[]>(() => {
    const items: Entry[] = [];
    for (const item of NAV_ITEMS) {
      items.push({
        id: `nav:${item.id}`,
        label: item.label,
        hint: item.description,
        group: "Pages",
        icon: item.icon,
        keywords: [item.label, ...(item.keywords ?? []), item.group],
        run: () => router.navigate(item.path),
      });
    }
    for (const action of QUICK_ACTIONS) {
      items.push({
        id: `action:${action.label}`,
        label: action.label,
        group: "Actions",
        icon: action.icon,
        keywords: [action.label, ...action.keywords],
        run: () => router.navigate(action.path),
      });
    }
    for (const model of registry?.models ?? []) {
      items.push({
        id: `model:${model.id}`,
        label: model.name,
        hint: model.path,
        group: "Models",
        icon: <Brain size={13} />,
        keywords: [model.name, model.path, ...(model.tags ?? [])],
        run: () => router.navigate(`/models/${encodeURIComponent(model.id)}`),
      });
    }
    for (const dataset of registry?.datasets ?? []) {
      items.push({
        id: `dataset:${dataset.id}`,
        label: dataset.name,
        hint: dataset.path,
        group: "Datasets",
        icon: <Database size={13} />,
        keywords: [dataset.name, dataset.path],
        run: () => router.navigate(`/datasets/${encodeURIComponent(dataset.id)}`),
      });
    }
    for (const job of jobs.slice(0, 40)) {
      items.push({
        id: `job:${job.job_id}`,
        label: job.job_id,
        hint: `${job.method ?? job.kind ?? "run"} · ${job.state}`,
        group: "Runs",
        icon: <Activity size={13} />,
        keywords: [job.job_id, job.method ?? "", job.state, job.backend ?? ""],
        run: () => router.navigate(`/training/${encodeURIComponent(job.job_id)}`),
      });
    }
    return items;
  }, [jobs, registry, router]);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) {
      return entries.filter((entry) => entry.group === "Actions" || entry.group === "Pages").slice(0, 24);
    }
    const scored = entries
      .map((entry) => {
        const haystack = [entry.label, entry.hint ?? "", ...entry.keywords].join(" ").toLowerCase();
        const position = haystack.indexOf(needle);
        if (position === -1) return null;
        const labelMatch = entry.label.toLowerCase().startsWith(needle) ? 0 : 1;
        return { entry, score: labelMatch * 100 + position };
      })
      .filter(Boolean) as Array<{ entry: Entry; score: number }>;
    return scored.sort((a, b) => a.score - b.score).slice(0, 40).map((item) => item.entry);
  }, [entries, query]);

  useEffect(() => {
    if (open) {
      setQuery("");
      setIndex(0);
      window.setTimeout(() => inputRef.current?.focus(), 20);
    }
  }, [open]);

  useEffect(() => {
    setIndex(0);
  }, [query]);

  const grouped = useMemo(() => {
    const map = new Map<string, Entry[]>();
    for (const entry of filtered) {
      const list = map.get(entry.group) ?? [];
      list.push(entry);
      map.set(entry.group, list);
    }
    return Array.from(map.entries());
  }, [filtered]);

  if (!open) return null;

  const flat = grouped.flatMap(([, list]) => list);

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/50 p-6 backdrop-blur-[2px] animate-fade-in"
      onMouseDown={onClose}
    >
      <div
        className="mt-12 w-full max-w-2xl animate-slide-up overflow-hidden rounded-xl border border-line-soft bg-surface-1 shadow-2xl"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="flex items-center gap-2 border-b border-line-soft px-3 py-2.5">
          <Search size={14} className="text-ink-3" />
          <input
            ref={inputRef}
            className="flex-1 bg-transparent text-sm outline-none placeholder:text-ink-3"
            placeholder="Search pages, models, datasets, runs and actions…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "ArrowDown") {
                event.preventDefault();
                setIndex((value) => Math.min(flat.length - 1, value + 1));
              } else if (event.key === "ArrowUp") {
                event.preventDefault();
                setIndex((value) => Math.max(0, value - 1));
              } else if (event.key === "Enter") {
                event.preventDefault();
                flat[index]?.run();
                onClose();
              } else if (event.key === "Escape") {
                onClose();
              }
            }}
          />
          <span className="kbd">esc</span>
        </div>

        <div ref={listRef} className="max-h-[52vh] overflow-y-auto p-1.5">
          {!flat.length ? (
            <div className="px-3 py-6 text-center text-xs text-ink-3">
              Nothing matches “{query}”. Only real workspace items are searchable.
            </div>
          ) : null}
          {grouped.map(([group, list]) => (
            <div key={group} className="mb-1.5">
              <div className="px-2 py-1 text-[9.5px] font-medium uppercase tracking-wider text-ink-3">{group}</div>
              {list.map((entry) => {
                const flatIndex = flat.indexOf(entry);
                const active = flatIndex === index;
                return (
                  <button
                    key={entry.id}
                    onMouseEnter={() => setIndex(flatIndex)}
                    onClick={() => {
                      entry.run();
                      onClose();
                    }}
                    className={cx(
                      "flex w-full items-center gap-2.5 rounded-md px-2 py-1.5 text-left transition-colors",
                      active ? "bg-accent/12" : "hover:bg-surface-2",
                    )}
                  >
                    <span className={cx("shrink-0", active ? "text-accent" : "text-ink-3")}>{entry.icon}</span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-xs text-ink-0">{entry.label}</span>
                      {entry.hint ? (
                        <span className="block truncate text-2xs text-ink-3" title={entry.hint}>
                          {entry.hint}
                        </span>
                      ) : null}
                    </span>
                    {active ? <CornerDownLeft size={12} className="text-ink-3" /> : null}
                  </button>
                );
              })}
            </div>
          ))}
        </div>

        <div className="flex items-center justify-between border-t border-line-soft px-3 py-1.5 text-2xs text-ink-3">
          <span className="flex items-center gap-2">
            <Mono>
              <Command size={10} className="inline" />K
            </Mono>
            to open
            <span className="text-ink-3">·</span>
            <span>↑↓ navigate</span>
            <span className="text-ink-3">·</span>
            <span>↵ open</span>
          </span>
          <Badge tone="muted">{filtered.length} results</Badge>
        </div>
      </div>
    </div>
  );
}
