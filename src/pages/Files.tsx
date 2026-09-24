import { useMemo, useState } from "react";
import {
  AlertTriangle,
  File as FileIcon,
  FolderOpen,
  HardDrive,
  Lock,
  Search,
  ShieldCheck,
  Trash2,
  Unlock,
} from "lucide-react";
import { api } from "../lib/api";
import { bytes, clock } from "../lib/format";
import { useApp } from "../state/app";
import { useRouter } from "../state/router";
import {
  Badge,
  Button,
  Callout,
  CodeBlock,
  EmptyState,
  Input,
  Panel,
  ProgressBar,
  SectionHeader,
  Select,
  Stat,
  Table,
  Td,
  TextInput,
  Th,
  Toggle,
  cx,
} from "../components/ui";
import { BarChart } from "../components/charts";
import { ErrorPanel, Loading, useEngine } from "./common";

interface TreeEntry {
  name: string;
  path: string;
  relative: string;
  is_dir: boolean;
  size: number;
  size_human: string;
  modified: string;
  depth: number;
}

export function FilesPage() {
  const { settings, toast, reportError } = useApp();
  const router = useRouter();
  const [filter, setFilter] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [depth, setDepth] = useState(2);
  const [confirmText, setConfirmText] = useState("");
  const [includeJobs, setIncludeJobs] = useState(false);

  const report = useEngine<any>("storage.report", {}, { timeout: 180_000 });
  const tree = useEngine<any>("storage.tree", { path: settings?.workspace, depth, max_entries: 600 }, {
    deps: [settings?.workspace, depth],
    timeout: 180_000,
  });
  const orphans = useEngine<any>("storage.orphans", {}, { timeout: 180_000 });

  const entries: TreeEntry[] = tree.data?.entries ?? [];
  const shown = useMemo(
    () =>
      filter
        ? entries.filter((entry) => entry.name.toLowerCase().includes(filter.toLowerCase()))
        : entries,
    [entries, filter],
  );
  const findings: any[] = (orphans.data?.findings ?? []).filter(
    (entry: any) => includeJobs || entry.kind !== "job",
  );

  const run = async (paths: string[], label: string) => {
    if (!paths.length) return;
    if (confirmText.trim().toUpperCase() !== "DELETE") {
      toast({
        title: "Type DELETE to confirm",
        body: "Deletion is never silent: the confirmation box gates the engine call.",
        tone: "warn",
      });
      return;
    }
    try {
      const result = await api.call<any>("storage.clean", { paths, confirm: true });
      toast({
        title: result.removed?.length ? `Freed ${result.freed_human}` : "Nothing was removed",
        body: result.removed?.length
          ? `${result.removed.length} ${label} deleted.`
          : (result.skipped ?? []).slice(0, 4).join("; ") || "The engine refused every path.",
        tone: result.removed?.length ? "ok" : "warn",
      });
      setConfirmText("");
      await report.reload();
      await orphans.reload();
      await tree.reload();
    } catch (error) {
      reportError(error, "Cleanup refused");
    }
  };

  const protect = async (path: string, value: boolean) => {
    try {
      await api.call("storage.protect", { path, protected: value });
      await report.reload();
      toast({
        title: value ? "Path protected" : "Protection removed",
        body: value ? "Cleanup tools will refuse it from now on." : "The path can be cleaned again.",
        tone: value ? "ok" : "info",
      });
    } catch (error) {
      reportError(error, "Could not change protection");
    }
  };

  if (!settings?.workspace) {
    return (
      <Callout tone="warn" title="No workspace configured" hint="The workspace folder holds models, datasets, jobs and checkpoints.">
        <Button size="sm" variant="primary" onClick={() => router.navigate("/settings")}>
          Open settings
        </Button>
      </Callout>
    );
  }

  const disk = report.data?.disk;
  const lowDisk = typeof disk?.free === "number" && disk.free < 5 * 1024 ** 3;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Files</h1>
          <p className="mt-0.5 max-w-3xl text-xs text-ink-2">
            The workspace keeps everything as ordinary files — no proprietary container. Every size on this page was
            measured by walking the real directories.
          </p>
        </div>
        <Button size="sm" variant="subtle" icon={<FolderOpen size={12} />} onClick={() => void api.shell.reveal(settings.workspace!)}>
          Open in file manager
        </Button>
      </div>

      {report.error ? <ErrorPanel error={report.error} onRetry={() => void report.reload()} /> : null}

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat
          label="Workspace size"
          value={report.data ? bytes(report.data.total_bytes) : "—"}
          hint={report.data?.workspace}
        />
        <Stat label="Models" value={bytes(report.data?.categories?.models ?? 0)} />
        <Stat label="Jobs & checkpoints" value={bytes(report.data?.categories?.jobs ?? 0)} />
        <Stat
          label="Free on volume"
          value={disk ? bytes(disk.free) : "—"}
          tone={lowDisk ? "danger" : undefined}
          hint={report.data?.volume ?? undefined}
        />
      </div>

      {disk ? (
        <Panel>
          <SectionHeader title="Disk" subtitle={`Volume ${report.data.volume}`} />
          <ProgressBar
            value={disk.total - disk.free}
            max={disk.total}
            tone={lowDisk ? "danger" : "accent"}
            label={`${bytes(disk.total - disk.free)} used of ${bytes(disk.total)} (${disk.used_percent}%)`}
          />
          {lowDisk ? (
            <div className="mt-2">
              <Callout tone="warn" title="Low disk space">
                Training writes checkpoints continuously; below 5 GiB free a long run can fail mid-save. Prune old
                checkpoints from the Jobs page or remove orphaned folders below.
              </Callout>
            </div>
          ) : null}
        </Panel>
      ) : null}

      <div className="grid gap-3 lg:grid-cols-[1fr_420px]">
        <Panel padded={false}>
          <div className="flex flex-wrap items-center gap-2 border-b border-line-soft p-2.5">
            <div className="relative min-w-[180px] flex-1">
              <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-3" />
              <Input className="pl-7" placeholder="Filter entries by name…" value={filter} onChange={(event) => setFilter(event.target.value)} />
            </div>
            <Select value={String(depth)} onChange={(event) => setDepth(Number(event.target.value))}>
              <option value="1">1 level</option>
              <option value="2">2 levels</option>
              <option value="3">3 levels</option>
              <option value="4">4 levels</option>
            </Select>
          </div>

          <div className="max-h-[520px] overflow-y-auto">
            {tree.loading && !entries.length ? <div className="p-3"><Loading label="Walking the workspace…" lines={4} /></div> : null}
            {tree.error ? (
              <div className="p-3">
                <ErrorPanel error={tree.error} onRetry={() => void tree.reload()} />
              </div>
            ) : null}
            {!tree.loading && !entries.length && !tree.error ? (
              <div className="p-3">
                <EmptyState title="Nothing in the workspace yet" hint="Import a model or a dataset, or start a training run." />
              </div>
            ) : null}
            {shown.length ? (
              <div className="divide-y divide-line-soft">
                {shown.map((entry) => (
                  <div
                    key={entry.path}
                    className={cx(
                      "flex items-center gap-2 px-2.5 py-1.5 text-xs transition-colors hover:bg-surface-3",
                      selected === entry.path && "bg-accent/10",
                    )}
                  >
                    <button
                      className="flex min-w-0 flex-1 items-center gap-2 text-left"
                      onClick={() => setSelected(entry.path)}
                      style={{ paddingLeft: `${entry.depth * 12}px` }}
                    >
                      {entry.is_dir ? <FolderOpen size={11} className="text-ink-3" /> : <FileIcon size={11} className="text-ink-3" />}
                      <span className="min-w-0 flex-1 truncate">{entry.relative || entry.name}</span>
                      <span className="hidden text-2xs text-ink-3 sm:inline">{entry.modified}</span>
                      <span className="w-16 text-right tabular-nums text-2xs text-ink-2">{entry.size_human}</span>
                    </button>
                  </div>
                ))}
              </div>
            ) : null}
            {tree.data?.truncated ? (
              <div className="p-2 text-2xs text-ink-3">Listing truncated at 600 entries — reduce the depth to see the rest.</div>
            ) : null}
          </div>

          {selected ? (
            <div className="flex flex-wrap items-center gap-2 border-t border-line-soft p-2.5">
              <span className="min-w-0 flex-1 truncate font-mono text-2xs text-ink-2">{selected}</span>
              <Button size="sm" variant="subtle" onClick={() => void api.shell.reveal(selected)}>
                Reveal
              </Button>
              <Button size="sm" variant="subtle" onClick={() => router.navigate(`/models?path=${encodeURIComponent(selected)}`)}>
                Inspect as model
              </Button>
              <Button size="sm" variant="subtle" icon={<Lock size={11} />} onClick={() => void protect(selected, true)}>
                Protect
              </Button>
            </div>
          ) : null}
        </Panel>

        <div className="space-y-3">
          <Panel>
            <SectionHeader title="Largest folders" subtitle="Measured sizes of the workspace folders." />
            {!report.data?.entries?.length ? <EmptyState title="Nothing measured yet" /> : null}
            {report.data?.entries?.length ? (
              <BarChart
                bars={report.data.entries
                  .filter((entry: any) => entry.bytes > 0)
                  .slice(0, 8)
                  .map((entry: any) => ({ label: entry.name, value: entry.bytes }))}
                formatLabel={(value) => `${value}: ${bytes(Number(value))}`}
                height={160}
              />
            ) : null}
          </Panel>

          <Panel>
            <SectionHeader
              title="Orphans"
              subtitle="Folders nothing references any more, plus temporary files from interrupted operations."
              actions={<Badge tone="muted">{bytes(orphans.data?.total_bytes ?? 0)}</Badge>}
            />
            {orphans.error ? <ErrorPanel error={orphans.error} onRetry={() => void orphans.reload()} /> : null}
            {orphans.loading && !orphans.data ? <Loading lines={2} /> : null}
            {orphans.data ? (
              <>
                <div className="mb-2">
                  <Toggle checked={includeJobs} onChange={setIncludeJobs} label="Include finished job folders" />
                </div>
                {findings.length ? (
                  <Table>
                    <thead>
                      <tr>
                        <Th>Entry</Th>
                        <Th>Why the engine flagged it</Th>
                        <Th align="right">Size</Th>
                      </tr>
                    </thead>
                    <tbody>
                      {findings.slice(0, 30).map((entry: any) => (
                        <tr key={entry.path}>
                          <Td>
                            <span className="block truncate text-xs">{entry.name}</span>
                            <span className="block truncate font-mono text-2xs text-ink-3">{entry.kind}</span>
                          </Td>
                          <Td>
                            <span className="text-2xs text-ink-2">{entry.reason}</span>
                          </Td>
                          <Td align="right">
                            <div className="flex items-center justify-end gap-2">
                              <span className="tabular-nums text-2xs">{entry.human}</span>
                              <Button size="sm" variant="subtle" icon={<Lock size={10} />} onClick={() => void protect(entry.path, true)}>
                                Keep
                              </Button>
                            </div>
                          </Td>
                        </tr>
                      ))}
                    </tbody>
                  </Table>
                ) : (
                  <div className="text-xs text-ink-2">Nothing orphaned — the workspace is fully accounted for.</div>
                )}
                {findings.length ? (
                  <div className="mt-3 space-y-2">
                    <div className="text-2xs text-ink-3">
                      Type DELETE and press the button to remove the {findings.length} entries listed above. Protected
                      paths, registered models and running jobs are refused by the engine even if they slip through.
                    </div>
                    <div className="flex gap-2">
                      <TextInput
                        placeholder="Type DELETE"
                        value={confirmText}
                        onChange={(event) => setConfirmText(event.target.value)}
                        className="max-w-[160px]"
                      />
                      <Button
                        size="sm"
                        variant="danger"
                        icon={<Trash2 size={11} />}
                        onClick={() => void run(findings.map((entry: any) => entry.path), "orphaned entries")}
                      >
                        Delete orphans
                      </Button>
                    </div>
                  </div>
                ) : null}
                <div className="mt-3 text-2xs text-ink-3">Checked {clock(orphans.data.checked_at)}</div>
              </>
            ) : null}
          </Panel>

          <Panel>
            <SectionHeader
              title="Protection"
              subtitle="Protected artifacts are skipped by every cleanup path."
              actions={<ShieldCheck size={14} className="text-ok" />}
            />
            {report.data?.protected?.length ? (
              <div className="space-y-1.5">
                {report.data.protected.map((entry: any) => (
                  <div key={entry.path} className="flex items-center gap-2 rounded border border-line-soft bg-surface-2 px-2 py-1.5">
                    <Lock size={11} className="text-warn" />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-xs">{entry.name}</span>
                      <span className="block truncate font-mono text-2xs text-ink-3">{entry.path}</span>
                    </span>
                    <Badge tone="muted">{entry.kind}</Badge>
                    <Button size="sm" variant="subtle" icon={<Unlock size={10} />} onClick={() => void protect(entry.path, false)}>
                      Unprotect
                    </Button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-xs text-ink-2">
                Nothing is protected yet. Select any entry in the browser and press Protect to lock it.
              </div>
            )}
          </Panel>
        </div>
      </div>

      <Panel>
        <SectionHeader title="Workspace layout" subtitle="Where each kind of artifact lives." />
        <CodeBlock>{`${settings.workspace}\\
  models\\       imported and trained models
  datasets\\     imported and built datasets
  jobs\\         one folder per run: spec, status, events, metrics, checkpoints
  exports\\      exported models, cards and reproducibility bundles
  plugins\\      optional engine plugins (plugin.json per plugin)
  workspace.json  library registry and settings`}</CodeBlock>
        <div className="mt-2 flex items-center gap-2 text-2xs text-ink-3">
          <HardDrive size={11} /> Sizes are read from the filesystem each time this page loads.
          {report.data?.reported_at ? <span>· measured {clock(report.data.reported_at)}</span> : null}
        </div>
      </Panel>

      <Callout tone="info" title="Deletion rules">
        <ul className="ml-4 list-disc space-y-1">
          <li>Deletion needs the typed confirmation, runs one path at a time, and reports what was skipped and why.</li>
          <li>Paths outside the artifact folders, protected entries and (when locked) job folders are refused by the engine.</li>
          <li>
            <AlertTriangle size={11} className="mr-1 inline" />
            No artifact is ever uploaded, and nothing is removed silently.
          </li>
        </ul>
      </Callout>
    </div>
  );
}
