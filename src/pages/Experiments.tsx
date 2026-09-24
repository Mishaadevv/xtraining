import { useMemo, useState } from "react";
import { FlaskConical, GitBranch, Star } from "lucide-react";
import { Sparkline } from "../components/charts";
import { api } from "../lib/api";
import { basename, clock, duration, number } from "../lib/format";
import { useApp } from "../state/app";
import { useRouter } from "../state/router";
import { Badge, Button, EmptyState, Panel, SectionHeader, Select, Table, Td, TextInput, Th, cx } from "../components/ui";
import { JobStateBadge, useEngine } from "./common";

export function ExperimentsPage() {
  const { jobs, toast, reportError } = useApp();
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [methodFilter, setMethodFilter] = useState("all");
  const lineage = useEngine<any>("lineage.list", {}, { timeout: 120_000 });

  const experiments = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return jobs.filter((job) => {
      if (methodFilter !== "all" && (job.method ?? "") !== methodFilter) return false;
      return needle ? `${job.job_id} ${job.method} ${job.model}`.toLowerCase().includes(needle) : true;
    });
  }, [jobs, query, methodFilter]);

  const methods = useMemo(() => Array.from(new Set(jobs.map((job) => job.method).filter(Boolean))) as string[], [jobs]);

  const archive = async (jobId: string) => {
    try {
      await api.registry.add("notes", { id: `archive-${jobId}`, job: jobId, kind: "archived", at: new Date().toISOString() });
      toast({ title: `${jobId} archived`, body: "Archived runs are noted in the workspace registry.", tone: "info" });
    } catch (error) {
      reportError(error, "Could not archive the run");
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Experiments</h1>
          <p className="mt-0.5 max-w-3xl text-xs text-ink-2">
            Every run is an experiment with its configuration, datasets, metrics and lineage recorded on disk.
            The tree below is built from the real lineage records the engine wrote at the end of each run.
          </p>
        </div>
        <Button onClick={() => void lineage.reload()}>Rebuild lineage</Button>
      </div>

      <Panel padded={false}>
        <div className="flex flex-wrap items-center gap-2 border-b border-line-soft p-3">
          <TextInput className="max-w-xs" placeholder="Filter experiments…" value={query} onChange={(event) => setQuery(event.target.value)} />
          <Select className="w-48" value={methodFilter} onChange={(event) => setMethodFilter(event.target.value)}>
            <option value="all">All methods</option>
            {methods.map((method) => (
              <option key={method} value={method}>
                {method}
              </option>
            ))}
          </Select>
          <Badge tone="muted">{experiments.length} experiments</Badge>
        </div>

        {!experiments.length ? (
          <div className="p-4">
            <EmptyState title="No experiments yet" icon={<FlaskConical size={18} />}>
              Start a run and it will appear here with its configuration, metrics history and checkpoint list.
            </EmptyState>
          </div>
        ) : (
          <Table>
            <thead>
              <tr>
                <Th>Experiment</Th>
                <Th>State</Th>
                <Th>Config</Th>
                <Th align="right">Loss curve</Th>
                <Th align="right">Final loss</Th>
                <Th align="right">Duration</Th>
                <Th align="right">Actions</Th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((job) => {
                const lossValues = (job.metrics ?? [])
                  .filter((entry: any) => entry.loss !== undefined && entry.loss !== null)
                  .map((entry: any) => entry.loss);
                return (
                  <tr key={job.job_id} className="hover:bg-surface-2">
                    <Td>
                      <button className="text-left" onClick={() => router.navigate(`/training/${encodeURIComponent(job.job_id)}`)}>
                        <div className="text-xs text-ink-0">{job.job_id}</div>
                        <div className="text-2xs text-ink-3">{job.model ? basename(job.model) : "from scratch"}</div>
                      </button>
                    </Td>
                    <Td>
                      <JobStateBadge state={job.state} />
                    </Td>
                    <Td>
                      <div className="text-2xs text-ink-2">
                        {job.method} · {String(job.config?.precision ?? "—")} · lr {String(job.config?.learning_rate ?? "—")}
                      </div>
                      <div className="text-2xs text-ink-3">
                        batch {String(job.config?.batch_size ?? "?")} × {String(job.config?.gradient_accumulation ?? "?")} ·
                        seq {String(job.config?.sequence_length ?? "?")} · seed {String(job.config?.seed ?? "?")}
                      </div>
                    </Td>
                    <Td align="right">
                      <div className="ml-auto w-28">
                        <Sparkline values={lossValues} tone={job.state === "failed" ? "danger" : "accent"} />
                      </div>
                    </Td>
                    <Td align="right">
                      {job.loss !== null && job.loss !== undefined
                        ? job.loss.toFixed(4)
                        : job.result?.final_loss !== undefined
                          ? Number(job.result.final_loss).toFixed(4)
                          : "—"}
                    </Td>
                    <Td align="right">{duration(job.elapsed_seconds ?? job.result?.duration_seconds ?? null)}</Td>
                    <Td align="right">
                      <div className="flex justify-end gap-1">
                        <Button
                          size="sm"
                          variant="subtle"
                          onClick={() => router.navigate(`/training/${encodeURIComponent(job.job_id)}`)}
                        >
                          Open
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          title="Create a new run with the same configuration"
                          onClick={() => router.navigate(`/training/new?model=${encodeURIComponent(job.model ?? "")}&method=${encodeURIComponent(job.method ?? "lora")}`)}
                        >
                          Duplicate
                        </Button>
                        <Button size="sm" variant="ghost" icon={<Star size={11} />} onClick={() => void archive(job.job_id)}>
                          Archive
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={async () => {
                            try {
                              const destination = await api.dialog.saveFile({ title: "Export experiment metadata", defaultPath: `${job.job_id}-metadata.json` });
                              if (!destination) return;
                              await api.call("reproducibility.bundle", { job_dir: job.job_dir, write: true, output: destination });
                              toast({ title: "Metadata exported", body: destination, tone: "ok" });
                            } catch (error) {
                              reportError(error, "Export failed");
                            }
                          }}
                        >
                          Export
                        </Button>
                      </div>
                    </Td>
                  </tr>
                );
              })}
            </tbody>
          </Table>
        )}
      </Panel>

      <Panel>
        <SectionHeader
          title="Lineage"
          subtitle={lineage.data?.entries?.length ? `${lineage.data.entries.length} lineage records` : "No lineage records yet"}
          icon={<GitBranch size={13} className="text-accent" />}
          actions={
            <Button size="sm" variant="subtle" onClick={() => void lineage.reload()}>
              Refresh
            </Button>
          }
        />
        {lineage.data?.roots?.length ? (
          <div className="space-y-2">
            {lineage.data.roots.map((root: any) => (
              <LineageNode key={root.id} node={root} depth={0} onOpen={(id) => router.navigate(`/training/${encodeURIComponent(id)}`)} />
            ))}
          </div>
        ) : (
          <div className="text-2xs text-ink-3">
            Lineage appears after a run that continues from a model or checkpoint. Each record states what was
            restored: weights, optimizer state, RNG state and the parent artefact.
          </div>
        )}
        {lineage.data?.entries?.length ? (
          <div className="mt-3">
            <Table>
              <thead>
                <tr>
                  <Th>Run</Th>
                  <Th>Parent</Th>
                  <Th>Resumed</Th>
                  <Th>Method</Th>
                  <Th>Created</Th>
                </tr>
              </thead>
              <tbody>
                {lineage.data.entries.slice(-12).reverse().map((entry: any) => (
                  <tr key={entry.job_id}>
                    <Td>
                      <button className="text-xs hover:text-accent" onClick={() => router.navigate(`/training/${encodeURIComponent(entry.job_id)}`)}>
                        {entry.job_id}
                      </button>
                    </Td>
                    <Td>
                      <span className="max-w-[280px] truncate font-mono text-2xs" title={entry.parent?.path}>
                        {entry.parent?.path ? basename(entry.parent.path) : "—"}
                      </span>
                    </Td>
                    <Td>
                      <Badge tone={entry.resumed ? "ok" : "muted"}>{entry.resumed ? "yes" : "no"}</Badge>
                    </Td>
                    <Td>{entry.child?.method}</Td>
                    <Td>{clock(entry.created_at)}</Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </div>
        ) : null}
      </Panel>
    </div>
  );
}

function LineageNode({ node, depth, onOpen }: { node: any; depth: number; onOpen: (id: string) => void }) {
  return (
    <div style={{ marginLeft: depth * 18 }} className={cx("rounded-md border border-line-soft bg-surface-2 px-2.5 py-1.5")}>
      <div className="flex items-center justify-between gap-2">
        <button className="text-left" onClick={() => onOpen(node.id)}>
          <span className="text-xs text-ink-0">{node.id}</span>
          <span className="ml-2 text-2xs text-ink-3">
            {node.method} · {node.steps ? `${number(node.steps)} steps` : "steps unknown"}
            {node.resumed ? " · resumed" : ""}
          </span>
        </button>
        <span className="text-2xs text-ink-3">{clock(node.created_at)}</span>
      </div>
      {node.model_dir ? (
        <div className="mt-0.5 truncate font-mono text-2xs text-ink-3" title={node.model_dir}>
          {node.model_dir}
        </div>
      ) : null}
      {node.children?.length ? (
        <div className="mt-1.5 space-y-1.5">
          {node.children.map((child: any) => (
            <LineageNode key={child.id} node={child} depth={depth + 1} onOpen={onOpen} />
          ))}
        </div>
      ) : null}
    </div>
  );
}
