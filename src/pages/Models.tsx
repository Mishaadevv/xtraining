import { useEffect, useMemo, useState } from "react";
import { Brain, FolderSearch, Import, Search, Star, Trash2 } from "lucide-react";
import { api } from "../lib/api";
import { basename, bytes, clock, number } from "../lib/format";
import { useApp } from "../state/app";
import { useRouter } from "../state/router";
import {
  Badge,
  Button,
  Callout,
  EmptyState,
  Field,
  Modal,
  Panel,
  SectionHeader,
  Select,
  Table,
  Td,
  TextInput,
  Th,
} from "../components/ui";
import { ErrorPanel, Loading, useEngine } from "./common";

export function ModelsPage() {
  const { registry, refreshRegistry, toast, settings, reportError } = useApp();
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState("recent");
  const [importOpen, setImportOpen] = useState(false);
  const [scanOpen, setScanOpen] = useState(false);
  const [scanRoot, setScanRoot] = useState("");
  const [busy, setBusy] = useState(false);
  const [importSummary, setImportSummary] = useState<any>(null);
  const scan = useEngine<any>("models.scan", { roots: scanRoot ? [scanRoot] : [], max_depth: 3 }, { auto: false, timeout: 300_000 });

  useEffect(() => {
    const action = new URLSearchParams(window.location.hash.split("?")[1] ?? "").get("action");
    if (action === "import") setImportOpen(true);
  }, []);

  const models = useMemo(() => {
    const list = [...(registry?.models ?? [])];
    const needle = query.trim().toLowerCase();
    const filtered = needle
      ? list.filter((model) => `${model.name} ${model.path} ${(model.tags ?? []).join(" ")}`.toLowerCase().includes(needle))
      : list;
    const sorted = filtered.sort((a, b) => {
      if (sort === "name") return a.name.localeCompare(b.name);
      if (sort === "favorite") return Number(b.favorite ?? false) - Number(a.favorite ?? false);
      return String(b.addedAt).localeCompare(String(a.addedAt));
    });
    return sorted;
  }, [registry, query, sort]);

  const importModel = async (source: string) => {
    if (!settings?.workspace) return;
    setBusy(true);
    try {
      const result = await api.call<any>("models.import", {
        source,
        destination: `${settings.workspace}\\models`.replace(/\\+$/, ""),
        copy: true,
      });
      await api.registry.add("models", {
        id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`,
        name: result.inspection?.name ?? basename(source),
        path: result.imported,
        addedAt: new Date().toISOString(),
        kind: result.inspection?.format?.kind,
        tags: [],
        favorite: false,
        summary: {
          parameters: result.inspection?.weights?.parameter_count ?? null,
          architecture: result.inspection?.architecture?.model_type ?? null,
          size_bytes: result.inspection?.size_bytes ?? null,
        },
      });
      await refreshRegistry();
      setImportSummary(result.inspection);
      toast({
        title: `Imported ${basename(result.imported)}`,
        body: result.inspection?.weights?.parameter_count
          ? `${number(result.inspection.weights.parameter_count)} parameters · ${result.inspection.weights.source}`
          : "Model copied into the workspace library.",
        tone: "ok",
      });
    } catch (error) {
      reportError(error, "Model import failed");
    } finally {
      setBusy(false);
    }
  };

  const pickAndImport = async () => {
    const selected = await api.dialog.pickFolder({ title: "Select a model folder", defaultPath: settings?.workspace ?? undefined });
    if (selected) await importModel(selected);
  };

  const removeModel = async (id: string, path: string, name: string) => {
    const confirmed = window.confirm(
      `Remove “${name}” from the library?\n\nThe folder remains on disk at:\n${path}\n\nUse the storage tools if you also want the files deleted.`,
    );
    if (!confirmed) return;
    try {
      await api.registry.remove("models", id);
      await refreshRegistry();
      toast({ title: `Removed ${name} from the library`, body: "The files on disk were left untouched.", tone: "info" });
    } catch (error) {
      reportError(error, "Could not update the library");
    }
  };

  const deleteFiles = async (path: string, name: string) => {
    const confirmed = window.confirm(`Permanently delete the model folder “${name}”?\n\n${path}\n\nThis cannot be undone.`);
    if (!confirmed) return;
    try {
      const result = await api.call<any>("storage.clean", { paths: [path], confirm: true });
      toast({
        title: result.removed?.length ? `Deleted ${name}` : `Not deleted: ${name}`,
        body: result.removed?.length ? `Freed ${result.freed_human}.` : (result.skipped ?? []).join(", "),
        tone: result.removed?.length ? "ok" : "warn",
      });
      await refreshRegistry();
    } catch (error) {
      reportError(error, "Deletion refused");
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Models</h1>
          <p className="mt-0.5 text-xs text-ink-2">
            The library references models in place or copies them into the workspace. Importing never uploads
            anything and never modifies the source.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="primary" icon={<Import size={13} />} onClick={() => void pickAndImport()} loading={busy}>
            Import folder
          </Button>
          <Button icon={<FolderSearch size={13} />} onClick={() => setScanOpen(true)}>
            Scan for models
          </Button>
        </div>
      </div>

      <Panel padded={false}>
        <div className="flex flex-wrap items-center gap-2 border-b border-line-soft p-3">
          <div className="relative flex-1 min-w-[220px]">
            <Search size={13} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-3" />
            <TextInput className="pl-8" placeholder="Filter by name, path or tag…" value={query} onChange={(event) => setQuery(event.target.value)} />
          </div>
          <Select className="w-40" value={sort} onChange={(event) => setSort(event.target.value)}>
            <option value="recent">Newest first</option>
            <option value="name">Name</option>
            <option value="favorite">Favourites first</option>
          </Select>
          <Badge tone="muted">{models.length} shown</Badge>
        </div>

        {!models.length ? (
          <div className="p-4">
            <EmptyState
              title="No models in the library"
              icon={<Brain size={18} />}
              action={
                <div className="flex gap-2">
                  <Button variant="primary" onClick={() => void pickAndImport()}>
                    Import a folder
                  </Button>
                  <Button onClick={() => setScanOpen(true)}>Scan a directory</Button>
                </div>
              }
            >
              A model folder must contain config.json, or a GGUF file, to be inspectable. The engine reads
              configs and weight headers directly — nothing is loaded into memory to inspect it.
            </EmptyState>
          </div>
        ) : (
          <Table>
            <thead>
              <tr>
                <Th>Name</Th>
                <Th>Format</Th>
                <Th align="right">Parameters</Th>
                <Th align="right">Size</Th>
                <Th>Added</Th>
                <Th align="right">Actions</Th>
              </tr>
            </thead>
            <tbody>
              {models.map((model) => (
                <tr key={model.id} className="hover:bg-surface-2">
                  <Td>
                    <button className="text-left" onClick={() => router.navigate(`/models/${encodeURIComponent(model.id)}`)}>
                      <div className="flex items-center gap-1.5">
                        <span className="text-xs text-ink-0">{model.name}</span>
                        {model.favorite ? <Star size={11} className="text-warn" /> : null}
                      </div>
                      <div className="max-w-[380px] truncate font-mono text-2xs text-ink-3" title={model.path}>
                        {model.path}
                      </div>
                    </button>
                  </Td>
                  <Td>
                    <Badge tone="muted">{String(model.kind ?? model.summary?.architecture ?? "unknown")}</Badge>
                  </Td>
                  <Td align="right">
                    {model.summary?.parameters ? number(Number(model.summary.parameters)) : "—"}
                  </Td>
                  <Td align="right">{model.summary?.size_bytes ? bytes(Number(model.summary.size_bytes)) : "—"}</Td>
                  <Td>{clock(model.addedAt)}</Td>
                  <Td align="right">
                    <div className="flex justify-end gap-1">
                      <Button size="sm" variant="ghost" onClick={() => api.shell.reveal(model.path)}>
                        Reveal
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        icon={<Trash2 size={11} />}
                        onClick={() => void deleteFiles(model.path, model.name)}
                      >
                        Delete files
                      </Button>
                      <Button size="sm" variant="ghost" onClick={() => void removeModel(model.id, model.path, model.name)}>
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
        open={scanOpen}
        onClose={() => setScanOpen(false)}
        title="Scan for models"
        footer={
          <>
            <Button onClick={() => setScanOpen(false)}>Close</Button>
            <Button variant="primary" loading={scan.loading} onClick={() => void scan.reload()}>
              Scan
            </Button>
          </>
        }
      >
        <div className="space-y-3">
          <Field label="Root folder" hint="Every subfolder containing config.json or a GGUF file is listed.">
            <div className="flex gap-2">
              <TextInput value={scanRoot} onChange={(event) => setScanRoot(event.target.value)} placeholder="C:\\models" />
              <Button
                onClick={async () => {
                  const picked = await api.dialog.pickFolder({ title: "Choose a folder to scan" });
                  if (picked) setScanRoot(picked);
                }}
              >
                Browse
              </Button>
            </div>
          </Field>
          {scan.error ? <ErrorPanel error={scan.error} onRetry={scan.reload} /> : null}
          {scan.data?.models?.length ? (
            <div className="max-h-72 overflow-auto rounded-md border border-line-soft">
              {scan.data.models.map((found: any) => (
                <div key={found.path} className="flex items-center justify-between gap-2 border-b border-line-soft/60 px-2 py-1.5 last:border-0">
                  <div className="min-w-0">
                    <div className="truncate text-xs">{found.name}</div>
                    <div className="truncate font-mono text-2xs text-ink-3">{found.path}</div>
                  </div>
                  <div className="flex items-center gap-2">
                    {found.has_weights ? <Badge tone="ok">weights</Badge> : <Badge tone="warn">config only</Badge>}
                    <Button
                      size="sm"
                      variant="subtle"
                      loading={busy}
                      onClick={async () => {
                        await importModel(found.path);
                      }}
                    >
                      Import
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          ) : null}
          {scan.data && !scan.data.models?.length ? (
            <Callout tone="info" title="Nothing found yet">
              Point the scan at a folder that contains model directories (each with config.json) or GGUF files.
            </Callout>
          ) : null}
        </div>
      </Modal>

      <Modal
        open={importOpen}
        onClose={() => {
          setImportOpen(false);
          setImportSummary(null);
        }}
        title="Import a model"
        footer={
          <>
            <Button onClick={() => setImportOpen(false)}>Close</Button>
            <Button variant="primary" loading={busy} onClick={() => void pickAndImport()}>
              Choose folder…
            </Button>
          </>
        }
      >
        <div className="space-y-3">
          <p className="text-xs text-ink-2">
            Point the file dialog at a Hugging Face style folder (config.json + weights/tokenizer), a LoRA
            adapter folder, or a GGUF file. The engine inspects it immediately and reports exactly what it
            found, including any warning about custom code.
          </p>
          {importSummary ? (
            <Panel quiet>
              <SectionHeader title={importSummary.name} subtitle={importSummary.path} />
              <div className="grid gap-2 text-2xs sm:grid-cols-2">
                <div>Format: {importSummary.format?.kind}</div>
                <div>Size: {bytes(importSummary.size_bytes)}</div>
                <div>Parameters: {importSummary.weights?.parameter_count ? number(importSummary.weights.parameter_count) : "unknown"}</div>
                <div>Architecture: {importSummary.architecture?.model_type ?? "unknown"}</div>
              </div>
              {importSummary.weights?.errors?.length ? (
                <div className="mt-2 text-2xs text-warn">{importSummary.weights.errors.join("; ")}</div>
              ) : null}
            </Panel>
          ) : null}
        </div>
      </Modal>

      {scan.loading ? <Loading label="Scanning folders…" lines={2} /> : null}
    </div>
  );
}
