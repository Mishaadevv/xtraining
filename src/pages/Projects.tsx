import { useMemo, useState } from "react";
import { Boxes, Copy, FolderInput, Pencil, Plus, Trash2 } from "lucide-react";
import { api } from "../lib/api";
import { bytes, clock, relative } from "../lib/format";
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
  Stat,
  Table,
  Td,
  TextInput,
  Th,
  cx,
} from "../components/ui";
import { ErrorPanel, useEngine } from "./common";

interface Draft {
  id?: string;
  name: string;
  description: string;
  tags: string;
}

const emptyDraft: Draft = { name: "", description: "", tags: "" };

export function ProjectsPage() {
  const { registry, refreshRegistry, settings, toast, reportError } = useApp();
  const router = useRouter();
  const storage = useEngine<any>("storage.report", {}, { timeout: 180_000 });
  const [active, setActive] = useState<string | null>(null);
  const [dialog, setDialog] = useState<Draft | null>(null);

  const projects = registry?.projects ?? [];
  const selected = useMemo(
    () => projects.find((project) => project.id === (active ?? projects[0]?.id)) ?? null,
    [projects, active],
  );

  const members = useMemo(() => {
    if (!selected) return { models: [], datasets: [] };
    return {
      models: (registry?.models ?? []).filter((entry) => entry.project === selected.id),
      datasets: (registry?.datasets ?? []).filter((entry) => entry.project === selected.id),
    };
  }, [registry, selected]);

  const folderSize = useMemo(() => {
    if (!selected) return null;
    const entry = (storage.data?.entries ?? []).find((item: any) => item.name === selected.name);
    return entry ?? null;
  }, [storage.data, selected]);

  const save = async () => {
    if (!dialog) return;
    if (!dialog.name.trim()) {
      toast({ title: "A project needs a name", tone: "warn" });
      return;
    }
    const tags = dialog.tags
      .split(",")
      .map((tag) => tag.trim())
      .filter(Boolean);
    try {
      if (dialog.id) {
        const existing = projects.find((project) => project.id === dialog.id);
        await api.registry.update("projects", {
          ...existing,
          id: dialog.id,
          name: dialog.name.trim(),
          description: dialog.description.trim(),
          tags,
          updatedAt: new Date().toISOString(),
        });
      } else {
        await api.registry.add("projects", {
          id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
          name: dialog.name.trim(),
          description: dialog.description.trim(),
          tags,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        });
      }
      await refreshRegistry();
      toast({ title: dialog.id ? "Project updated" : "Project created", body: "Projects are metadata only — your files stay where they are.", tone: "ok" });
      setDialog(null);
    } catch (error) {
      reportError(error, "The project could not be saved");
    }
  };

  const remove = async (id: string, name: string) => {
    if (!window.confirm(`Delete the project “${name}”?\n\nModels and datasets keep their files; only the grouping is removed.`)) return;
    try {
      await api.registry.remove("projects", id);
      await refreshRegistry();
      toast({ title: `Project ${name} deleted`, body: "Artifacts were left on disk and in the library.", tone: "info" });
      setActive(null);
    } catch (error) {
      reportError(error, "The project could not be deleted");
    }
  };

  const duplicate = async (project: any) => {
    try {
      await api.registry.add("projects", {
        ...project,
        id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
        name: `${project.name} copy`,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        parent: project.id,
      });
      await refreshRegistry();
      toast({ title: "Project duplicated", body: "The copy keeps the description, tags and parent link.", tone: "ok" });
    } catch (error) {
      reportError(error, "Duplication failed");
    }
  };

  const assign = async (collection: "models" | "datasets", id: string, projectId: string | null) => {
    const entry = (registry?.[collection] ?? []).find((item) => item.id === id);
    if (!entry) return;
    try {
      await api.registry.update(collection, { ...entry, project: projectId });
      await refreshRegistry();
    } catch (error) {
      reportError(error, "Could not move the artifact");
    }
  };

  const importFolder = async () => {
    if (!selected) return;
    const folder = await api.dialog.pickFolder({ title: `Attach a folder to ${selected.name}`, defaultPath: settings?.workspace ?? undefined });
    if (!folder) return;
    try {
      await api.registry.update("projects", {
        ...selected,
        folders: [...new Set([...(selected as any).folders ?? [], folder])],
        updatedAt: new Date().toISOString(),
      });
      await refreshRegistry();
      toast({ title: "Folder attached", body: folder, tone: "ok" });
    } catch (error) {
      reportError(error, "Could not attach the folder");
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Projects</h1>
          <p className="mt-0.5 max-w-3xl text-xs text-ink-2">
            A project groups models, datasets and runs that belong together. Nothing is moved or rewritten on disk —
            existing folders can be attached exactly where they are.
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="subtle" icon={<FolderInput size={12} />} disabled={!selected} onClick={() => void importFolder()}>
            Attach existing folder
          </Button>
          <Button size="sm" variant="primary" icon={<Plus size={12} />} onClick={() => setDialog({ ...emptyDraft })}>
            New project
          </Button>
        </div>
      </div>

      {storage.error ? <ErrorPanel error={storage.error} onRetry={() => void storage.reload()} /> : null}

      <div className="grid gap-3 lg:grid-cols-[320px_1fr]">
        <Panel padded={false}>
          <div className="border-b border-line-soft p-2.5 text-2xs uppercase tracking-wide text-ink-3">
            {projects.length} project{projects.length === 1 ? "" : "s"}
          </div>
          <div className="max-h-[520px] space-y-1 overflow-y-auto p-2">
            {!projects.length ? (
              <EmptyState title="No projects yet" hint="Create one to group a base model with its fine-tunes and datasets." />
            ) : null}
            {projects.map((project) => (
              <button
                key={project.id}
                onClick={() => setActive(project.id)}
                className={cx(
                  "w-full rounded-md border px-2.5 py-2 text-left transition-colors",
                  selected?.id === project.id ? "border-accent/40 bg-surface-3" : "border-line-soft bg-surface-2 hover:bg-surface-3",
                )}
              >
                <div className="flex items-center gap-2">
                  <Boxes size={12} className="text-ink-3" />
                  <span className="min-w-0 flex-1 truncate text-xs">{project.name}</span>
                  {(registry?.models ?? []).filter((entry) => entry.project === project.id).length ? (
                    <Badge tone="muted">
                      {(registry?.models ?? []).filter((entry) => entry.project === project.id).length} models
                    </Badge>
                  ) : null}
                </div>
                {project.description ? (
                  <div className="mt-1 line-clamp-2 text-2xs text-ink-3">{project.description}</div>
                ) : null}
                <div className="mt-1 text-2xs text-ink-3">updated {relative(project.updatedAt ?? project.createdAt)}</div>
              </button>
            ))}
          </div>
        </Panel>

        {!selected ? (
          <Panel>
            <EmptyState title="Select a project" hint="Its models, datasets and notes appear here." />
          </Panel>
        ) : (
          <div className="space-y-3">
            <Panel>
              <SectionHeader
                title={selected.name}
                subtitle={selected.description || "No description yet."}
                actions={
                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      variant="subtle"
                      icon={<Pencil size={11} />}
                      onClick={() =>
                        setDialog({
                          id: selected.id,
                          name: selected.name,
                          description: selected.description ?? "",
                          tags: (selected.tags ?? []).join(", "),
                        })
                      }
                    >
                      Edit
                    </Button>
                    <Button size="sm" variant="subtle" icon={<Copy size={11} />} onClick={() => void duplicate(selected)}>
                      Duplicate
                    </Button>
                    <Button size="sm" variant="danger" icon={<Trash2 size={11} />} onClick={() => void remove(selected.id, selected.name)}>
                      Delete
                    </Button>
                  </div>
                }
              />
              <div className="mt-3 grid gap-3 sm:grid-cols-3">
                <Stat label="Models" value={members.models.length} />
                <Stat label="Datasets" value={members.datasets.length} />
                <Stat
                  label="Workspace folder size"
                  value={folderSize ? bytes(folderSize.bytes) : "—"}
                  hint={folderSize ? undefined : "No workspace folder with this project name"}
                />
              </div>
              <div className="mt-3 flex flex-wrap items-center gap-2">
                {(selected.tags ?? []).map((tag) => (
                  <Badge key={tag} tone="accent">
                    {tag}
                  </Badge>
                ))}
                <span className="text-2xs text-ink-3">created {clock(selected.createdAt)}</span>
              </div>
              {(selected as any).folders?.length ? (
                <div className="mt-3 space-y-1">
                  <div className="text-2xs uppercase tracking-wide text-ink-3">Attached folders</div>
                  {(selected as any).folders.map((folder: string) => (
                    <div key={folder} className="flex items-center gap-2 rounded border border-line-soft bg-surface-2 px-2 py-1.5">
                      <span className="min-w-0 flex-1 truncate font-mono text-2xs">{folder}</span>
                      <Button size="sm" variant="subtle" onClick={() => void api.shell.reveal(folder)}>
                        Reveal
                      </Button>
                      <Button
                        size="sm"
                        variant="subtle"
                        icon={<Trash2 size={10} />}
                        onClick={async () => {
                          await api.registry.update("projects", {
                            ...selected,
                            folders: (selected as any).folders.filter((item: string) => item !== folder),
                          });
                          await refreshRegistry();
                        }}
                      >
                        Detach
                      </Button>
                    </div>
                  ))}
                </div>
              ) : null}
            </Panel>

            <Panel>
              <SectionHeader
                title="Models in this project"
                subtitle="Assign a model to a project from here or from its detail page."
              />
              {members.models.length ? (
                <Table>
                  <thead>
                    <tr>
                      <Th>Model</Th>
                      <Th>Architecture</Th>
                      <Th align="right">Parameters</Th>
                      <Th align="right" />
                    </tr>
                  </thead>
                  <tbody>
                    {members.models.map((entry) => (
                      <tr key={entry.id}>
                        <Td>
                          <button className="text-left hover:text-accent" onClick={() => router.navigate(`/models/${encodeURIComponent(entry.id)}`)}>
                            {entry.name}
                          </button>
                          <span className="block truncate font-mono text-2xs text-ink-3">{entry.path}</span>
                        </Td>
                        <Td>{String((entry.summary as any)?.architecture ?? "—")}</Td>
                        <Td align="right">{entry.summary?.parameters ? Number(entry.summary.parameters).toLocaleString() : "—"}</Td>
                        <Td align="right">
                          <Button size="sm" variant="subtle" onClick={() => void assign("models", entry.id, null)}>
                            Remove from project
                          </Button>
                        </Td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              ) : (
                <EmptyState title="No models assigned" hint="Use “Assign” below to pull models from the library into this project." />
              )}
              <div className="mt-3 flex flex-wrap gap-2">
                {(registry?.models ?? [])
                  .filter((entry) => entry.project !== selected.id)
                  .slice(0, 12)
                  .map((entry) => (
                    <Button key={entry.id} size="sm" variant="subtle" onClick={() => void assign("models", entry.id, selected.id)}>
                      + {entry.name}
                    </Button>
                  ))}
              </div>
            </Panel>

            <Panel>
              <SectionHeader title="Datasets in this project" subtitle="Grouping does not copy any file." />
              {members.datasets.length ? (
                <Table>
                  <thead>
                    <tr>
                      <Th>Dataset</Th>
                      <Th align="right">Records</Th>
                      <Th align="right" />
                    </tr>
                  </thead>
                  <tbody>
                    {members.datasets.map((entry) => (
                      <tr key={entry.id}>
                        <Td>
                          <button className="text-left hover:text-accent" onClick={() => router.navigate(`/datasets/${encodeURIComponent(entry.id)}`)}>
                            {entry.name}
                          </button>
                          <span className="block truncate font-mono text-2xs text-ink-3">{entry.path}</span>
                        </Td>
                        <Td align="right">{entry.summary?.records ? Number(entry.summary.records).toLocaleString() : "—"}</Td>
                        <Td align="right">
                          <Button size="sm" variant="subtle" onClick={() => void assign("datasets", entry.id, null)}>
                            Remove from project
                          </Button>
                        </Td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              ) : (
                <EmptyState title="No datasets assigned" />
              )}
              <div className="mt-3 flex flex-wrap gap-2">
                {(registry?.datasets ?? [])
                  .filter((entry) => entry.project !== selected.id)
                  .slice(0, 12)
                  .map((entry) => (
                    <Button key={entry.id} size="sm" variant="subtle" onClick={() => void assign("datasets", entry.id, selected.id)}>
                      + {entry.name}
                    </Button>
                  ))}
              </div>
            </Panel>

            <Callout tone="info" title="Backup">
              Copying this project to another machine means copying the folders listed above plus{" "}
              <span className="font-mono text-2xs">{settings?.workspace}\\workspace.json</span>. Model weights are never
              duplicated by project operations.
            </Callout>
          </div>
        )}
      </div>

      <Modal open={Boolean(dialog)} onClose={() => setDialog(null)} title={dialog?.id ? "Edit project" : "New project"}>
        {dialog ? (
          <div className="space-y-3">
            <Field label="Name">
              <TextInput value={dialog.name} onChange={(event) => setDialog({ ...dialog, name: event.target.value })} placeholder="e.g. Support assistant" />
            </Field>
            <Field label="Description" hint="What this project is for.">
              <TextInput value={dialog.description} onChange={(event) => setDialog({ ...dialog, description: event.target.value })} />
            </Field>
            <Field label="Tags" hint="Comma separated.">
              <TextInput value={dialog.tags} onChange={(event) => setDialog({ ...dialog, tags: event.target.value })} placeholder="chat, russian, support" />
            </Field>
            <div className="flex justify-end gap-2">
              <Button variant="subtle" onClick={() => setDialog(null)}>
                Cancel
              </Button>
              <Button variant="primary" onClick={() => void save()}>
                {dialog.id ? "Save changes" : "Create project"}
              </Button>
            </div>
          </div>
        ) : null}
      </Modal>
    </div>
  );
}
