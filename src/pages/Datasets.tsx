import { useEffect, useMemo, useState } from "react";
import { Database, Import, Search, Trash2 } from "lucide-react";
import { api } from "../lib/api";
import { basename, bytes, clock, number } from "../lib/format";
import { useApp } from "../state/app";
import { useRouter } from "../state/router";
import {
  Badge,
  Button,
  Callout,
  EmptyState,
  Modal,
  Panel,
  SectionHeader,
  Select,
  Table,
  Td,
  TextInput,
  Th,
} from "../components/ui";
import { useEngine } from "./common";

const FORMAT_FILTERS = ["Any JSONL, JSON, CSV, TSV, TXT, Markdown, Parquet, Arrow, SQLite, Excel, YAML, or a folder of shards"];

export function DatasetsPage() {
  const { registry, refreshRegistry, toast, settings, reportError } = useApp();
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState("recent");
  const [importOpen, setImportOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [lastImport, setLastImport] = useState<any>(null);

  useEffect(() => {
    const action = new URLSearchParams(window.location.hash.split("?")[1] ?? "").get("action");
    if (action === "import") setImportOpen(true);
  }, []);

  const datasets = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const list = (registry?.datasets ?? []).filter((dataset) =>
      needle ? `${dataset.name} ${dataset.path}`.toLowerCase().includes(needle) : true,
    );
    return list.sort((a, b) =>
      sort === "name" ? a.name.localeCompare(b.name) : String(b.addedAt).localeCompare(String(a.addedAt)),
    );
  }, [registry, query, sort]);

  const register = async (path: string, report: any) => {
    await api.registry.add("datasets", {
      id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`,
      name: report?.name ?? basename(path),
      path,
      addedAt: new Date().toISOString(),
      kind: report?.kind,
      tags: [],
      favorite: false,
      summary: {
        records: report?.record_count ?? null,
        size_bytes: report?.size_bytes ?? null,
        fields: report?.field_names ?? [],
        tokens: report?.token_estimate?.total ?? null,
      },
    });
    await refreshRegistry();
  };

  const importFiles = async () => {
    const files = await api.dialog.pickFiles({
      title: "Select dataset files",
      filters: [{ name: "Datasets", extensions: ["jsonl", "ndjson", "json", "csv", "tsv", "txt", "md", "parquet", "arrow", "gz", "sqlite", "db", "xlsx", "yaml"] }],
    });
    if (!files?.length) return;
    setBusy(true);
    try {
      for (const file of files) {
        const report = await api.call<any>("datasets.inspect", { path: file, sample_size: 800 });
        await register(file, report);
        setLastImport(report);
      }
      toast({ title: `Imported ${files.length} dataset file(s)`, body: "Referenced in place — nothing was copied.", tone: "ok" });
    } catch (error) {
      reportError(error, "Dataset import failed");
    } finally {
      setBusy(false);
    }
  };

  const importFolder = async () => {
    const folder = await api.dialog.pickFolder({ title: "Select a dataset folder", defaultPath: settings?.workspace ?? undefined });
    if (!folder) return;
    setBusy(true);
    try {
      const report = await api.call<any>("datasets.inspect", { path: folder, sample_size: 800 });
      await register(folder, report);
      setLastImport(report);
      toast({
        title: `Imported ${report.name}`,
        body: `${number(report.record_count)} records${report.media_kind ? ` · ${report.media_kind} dataset` : ""}.`,
        tone: "ok",
      });
    } catch (error) {
      reportError(error, "Dataset import failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Datasets</h1>
          <p className="mt-0.5 max-w-3xl text-xs text-ink-2">{FORMAT_FILTERS[0]}</p>
        </div>
        <div className="flex gap-2">
          <Button variant="primary" icon={<Import size={13} />} loading={busy} onClick={() => void importFolder()}>
            Import folder
          </Button>
          <Button icon={<Import size={13} />} loading={busy} onClick={() => void importFiles()}>
            Import files
          </Button>
        </div>
      </div>

      <Panel padded={false}>
        <div className="flex flex-wrap items-center gap-2 border-b border-line-soft p-3">
          <div className="relative min-w-[220px] flex-1">
            <Search size={13} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-3" />
            <TextInput className="pl-8" placeholder="Filter datasets…" value={query} onChange={(event) => setQuery(event.target.value)} />
          </div>
          <Select className="w-40" value={sort} onChange={(event) => setSort(event.target.value)}>
            <option value="recent">Newest first</option>
            <option value="name">Name</option>
          </Select>
        </div>

        {!datasets.length ? (
          <div className="p-4">
            <EmptyState
              title="No datasets yet"
              icon={<Database size={18} />}
              action={
                <Button variant="primary" onClick={() => void importFolder()}>
                  Import a folder
                </Button>
              }
            >
              Datasets are referenced in place: the app reads them where they are and never copies or uploads
              your data. Folders containing shards (JSONL, CSV, Parquet) or images/audio are supported.
            </EmptyState>
          </div>
        ) : (
          <Table>
            <thead>
              <tr>
                <Th>Name</Th>
                <Th>Type</Th>
                <Th align="right">Records</Th>
                <Th align="right">Fields</Th>
                <Th align="right">Estimated tokens</Th>
                <Th align="right">Size</Th>
                <Th>Added</Th>
                <Th align="right">Actions</Th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((dataset) => (
                <tr key={dataset.id} className="hover:bg-surface-2">
                  <Td>
                    <button className="text-left" onClick={() => router.navigate(`/datasets/${encodeURIComponent(dataset.id)}`)}>
                      <div className="text-xs text-ink-0">{dataset.name}</div>
                      <div className="max-w-[380px] truncate font-mono text-2xs text-ink-3" title={dataset.path}>
                        {dataset.path}
                      </div>
                    </button>
                  </Td>
                  <Td>
                    <Badge tone="muted">{String(dataset.summary?.media_kind ?? dataset.kind ?? "file")}</Badge>
                  </Td>
                  <Td align="right">{dataset.summary?.records ? number(Number(dataset.summary.records)) : "—"}</Td>
                  <Td align="right">{Array.isArray(dataset.summary?.fields) ? (dataset.summary!.fields as string[]).length : "—"}</Td>
                  <Td align="right">{dataset.summary?.tokens ? number(Number(dataset.summary.tokens)) : "—"}</Td>
                  <Td align="right">{dataset.summary?.size_bytes ? bytes(Number(dataset.summary.size_bytes)) : "—"}</Td>
                  <Td>{clock(dataset.addedAt)}</Td>
                  <Td align="right">
                    <div className="flex justify-end gap-1">
                      <Button size="sm" variant="ghost" onClick={() => api.shell.reveal(dataset.path)}>
                        Reveal
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        icon={<Trash2 size={11} />}
                        onClick={async () => {
                          if (!window.confirm(`Remove “${dataset.name}” from the library?\n\nThe file stays on disk.`)) return;
                          await api.registry.remove("datasets", dataset.id);
                          await refreshRegistry();
                        }}
                      >
                        Unlist
                      </Button>
                    </div>
                  </Td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </Panel>

      <Modal
        open={importOpen}
        onClose={() => setImportOpen(false)}
        title="Import a dataset"
        footer={
          <>
            <Button onClick={() => setImportOpen(false)}>Close</Button>
            <Button variant="primary" loading={busy} onClick={() => void importFolder()}>
              Choose folder…
            </Button>
            <Button loading={busy} onClick={() => void importFiles()}>
              Choose files…
            </Button>
          </>
        }
      >
        <div className="space-y-3">
          <p className="text-xs text-ink-2">
            Supported: JSONL/NDJSON, JSON, CSV, TSV, PSV, TXT, Markdown, Parquet, Arrow/Feather, SQLite,
            Excel and YAML, plus gzip/bzip2/xz variants, plus folders of shards or of images/audio. Anything
            that needs an optional reader (Parquet, Excel, YAML) says so if the package is missing.
          </p>
          {lastImport ? (
            <Panel quiet>
              <SectionHeader title={lastImport.name} subtitle={lastImport.path} />
              <div className="grid gap-2 text-2xs sm:grid-cols-2">
                <div>Records: {number(lastImport.record_count)}</div>
                <div>Fields: {(lastImport.field_names ?? []).join(", ") || "none"}</div>
                <div>Duplicates: {lastImport.duplicates}</div>
                <div>Empty records: {lastImport.empty_records}</div>
                <div>Average length: {lastImport.length?.average ?? "—"} characters</div>
                <div>Mapping: {Object.entries(lastImport.detected_mapping ?? {}).map(([key, value]) => `${key}→${value}`).join(", ") || "not detected"}</div>
              </div>
            </Panel>
          ) : (
            <Callout tone="info" title="Nothing imported in this session yet">
              Every import is inspected immediately so the record count and field mapping you see are real.
            </Callout>
          )}
        </div>
      </Modal>
    </div>
  );
}

export function useDatasetReport(path: string | null) {
  return useEngine<any>("datasets.inspect", { path: path ?? "" }, { auto: Boolean(path), timeout: 600_000, deps: [path] });
}
