import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Archive,
  Brain,
  Cpu,
  Download,
  FlaskConical,
  FolderOpen,
  HardDrive,
  Play,
  Plus,
  RefreshCw,
  Sparkles,
  Trash2,
} from "lucide-react";

import { PageBody, PageHeader } from "@/components/layout/Shell";
import { ConfirmDialog, Modal } from "@/components/ui/Overlay";
import {
  Badge,
  Button,
  EmptyState,
  Field,
  IconButton,
  Input,
  KeyValue,
  Note,
  Panel,
  PanelHeader,
  Segmented,
  Switch,
} from "@/components/ui/primitives";
import type { ModelEntry, ModelExportInfo } from "@/lib/types";
import { formatBytes, formatCount, formatLoss, formatRelative } from "@/lib/utils";
import { useStore } from "@/state/store";
import {
  addModel,
  appStore,
  exportModel,
  inspectModelExport,
  loadPlaygroundModel,
  navigate,
  openPath,
  pickAndAddLocalModel,
  pickDirectory,
  pickModelPackPath,
  removeModel,
  resetWizard,
  revealPath,
  wizardSelectModel,
} from "@/state/appStore";

type Tab = "all" | "base" | "trained";

function ModelRow({
  model,
  onDelete,
  onExport,
}: {
  model: ModelEntry;
  onDelete: (model: ModelEntry) => void;
  onExport: (model: ModelEntry) => void;
}) {
  const errors = model.issues.filter((issue) => issue.severity === "error");
  const warnings = model.issues.filter((issue) => issue.severity === "warning");

  return (
    <Panel className="zq-rise">
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-start gap-2.5">
          <span className="mt-[2px] shrink-0 text-[var(--text-3)]">
            {model.trained ? <Sparkles className="h-4 w-4" style={{ color: "var(--acc)" }} /> : model.kind === "local" ? <HardDrive className="h-4 w-4" /> : <Download className="h-4 w-4" />}
          </span>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="truncate text-[13.5px] font-semibold">{model.name}</h3>
              {model.trained ? <Badge tone="accent">trained</Badge> : null}
              {model.kind === "huggingface" && !model.cached ? <Badge tone="info">downloads on use</Badge> : null}
              {model.adapter && model.trained ? <Badge>LoRA adapter</Badge> : null}
              {model.trained && model.isAdapterFolder ? <Badge tone="info">can be fine-tuned further</Badge> : null}
              {model.trained && !model.adapter ? <Badge>full model</Badge> : null}
            </div>
            <p className="zq-mono mt-0.5 truncate text-[11px] text-[var(--text-3)]" title={model.path ?? model.source}>
              {model.source}
            </p>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          {model.path ? (
            <IconButton title="Reveal in file manager" onClick={() => void revealPath(model.path)}>
              <FolderOpen className="h-3.5 w-3.5" />
            </IconButton>
          ) : null}
          <IconButton title="Remove from the library" onClick={() => onDelete(model)}>
            <Trash2 className="h-3.5 w-3.5" />
          </IconButton>
        </div>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-x-6 sm:grid-cols-4">
        <KeyValue label="Parameters" value={model.params ? formatCount(model.params) : "unknown"} />
        <KeyValue label="Architecture" value={model.architecture ?? "—"} />
        <KeyValue label="Context" value={model.maxPositionEmbeddings ? formatCount(model.maxPositionEmbeddings) : "—"} />
        <KeyValue
          label={model.trained ? "Final loss" : "Size on disk"}
          value={model.trained ? formatLoss(model.finalLoss ?? null) : formatBytes(model.sizeBytes)}
        />
        {model.trained ? (
          <>
            <KeyValue label="Method" value={model.method ?? "—"} />
            <KeyValue label="Base model" value={model.baseModel ? model.baseModel.split("/").pop() ?? "—" : "—"} />
            <KeyValue label="Steps" value={model.steps ?? "—"} />
            <KeyValue label="Trained" value={formatRelative(model.createdAt ?? model.addedAt)} />
          </>
        ) : null}
      </div>

      {errors.length || warnings.length ? (
        <div className="mt-3 space-y-1.5">
          {[...errors, ...warnings].slice(0, 3).map((issue, index) => (
            <div key={index} className="flex items-start gap-2 text-[11.5px] leading-[17px]">
              <AlertTriangle
                className="mt-[3px] h-3 w-3 shrink-0"
                style={{ color: issue.severity === "error" ? "var(--red)" : "var(--amber)" }}
              />
              <span className="text-[var(--text-2)]">{issue.message}</span>
            </div>
          ))}
        </div>
      ) : null}

      <div className="mt-3 flex items-center gap-2 border-t border-[var(--border-soft)] pt-3">
        {!model.trained ? (
          <Button
            size="sm"
            variant="primary"
            icon={<Play className="h-3.5 w-3.5" />}
            disabled={!model.trainable}
            onClick={() => {
              resetWizard({ baseModel: model.source, baseModelEntryId: model.id });
              void wizardSelectModel(model.source, model.id);
              navigate("new");
            }}
          >
            Use as base model
          </Button>
        ) : null}
        {model.trained && model.path ? (
          <Button
            size="sm"
            variant="primary"
            icon={<RefreshCw className="h-3.5 w-3.5" />}
            onClick={() => {
              resetWizard({ baseModel: model.path as string });
              void wizardSelectModel(model.path as string, null);
              navigate("new");
            }}
          >
            Continue training
          </Button>
        ) : null}
        {model.trained && model.path ? (
          <Button
            size="sm"
            variant="primary"
            icon={<FlaskConical className="h-3.5 w-3.5" />}
            onClick={() => {
              void loadPlaygroundModel(model.path as string, model.name);
              navigate("playground");
            }}
          >
            Test in playground
          </Button>
        ) : null}
        {model.path ? (
          <Button
            size="sm"
            variant="quiet"
            icon={<Archive className="h-3.5 w-3.5" />}
            onClick={() => onExport(model)}
          >
            Export
          </Button>
        ) : null}
        {model.trained && !model.adapter && model.path ? (
          <span className="text-[11px] text-[var(--text-3)]">
            Full fine-tune — every weight was updated.
          </span>
        ) : null}
      </div>
    </Panel>
  );
}

/**
 * Export dialog.
 *
 * Copy mode is always offered, either as a folder of files or packed into one
 * single .zip. Merging is offered only when the artefact is an adapter *and* the
 * ML runtime is installed — the reason is shown instead of a disabled control
 * with no explanation.
 */
function ExportModelDialog({ model, onClose }: { model: ModelEntry; onClose: () => void }) {
  const busy = useStore(appStore).busy;
  const [info, setInfo] = useState<ModelExportInfo | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [outputDir, setOutputDir] = useState("");
  const [merge, setMerge] = useState(false);
  const [pack, setPack] = useState(false);
  const [done, setDone] = useState<{ outputDir: string; files: string[]; archive: boolean } | null>(null);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const result = await inspectModelExport(model.id);
      if (cancelled) return;
      if (!result) setInfoError("The export folder could not be inspected.");
      else setInfo(result);
    })();
    return () => {
      cancelled = true;
    };
  }, [model.id]);

  const chooseFolder = async () => {
    const picked = await pickDirectory("Choose an empty folder for the export");
    if (picked) setOutputDir(picked);
  };

  const chooseFile = async () => {
    const picked = await pickModelPackPath(model.name);
    if (picked) setOutputDir(picked);
  };

  const run = async () => {
    const result = await exportModel(model.id, outputDir, { merge, pack });
    if (result) setDone({ outputDir: result.outputDir, files: result.files, archive: result.archive });
  };

  const mergeInfo = info?.modes.merge;
  const mergeBlocked = Boolean(mergeInfo && (!mergeInfo.applicable || !mergeInfo.ready));

  if (done) {
    return (
      <Modal
        open
        onClose={onClose}
        title="Export finished"
        description={
          merge
            ? "The adapter was merged into a standalone model."
            : done.archive
              ? "Everything was written into one .zip file."
              : "The adapter files and a description were written."
        }
        width={560}
        footer={
          <div className="flex items-center justify-end gap-2">
            <Button
              size="sm"
              variant="quiet"
              icon={<FolderOpen className="h-3.5 w-3.5" />}
              onClick={() => void (done.archive ? revealPath(done.outputDir) : openPath(done.outputDir))}
            >
              {done.archive ? "Show in folder" : "Open folder"}
            </Button>
            <Button size="sm" variant="primary" onClick={onClose}>
              Done
            </Button>
          </div>
        }
      >
        <Note tone="good" title="Written to">
          <span className="zq-mono break-all">{done.outputDir}</span>
        </Note>
        <div className="mt-3 space-y-1">
          {done.files.map((name) => (
            <p key={name} className="zq-mono text-[11.5px] text-[var(--text-2)]">
              {name}
            </p>
          ))}
        </div>
        <p className="mt-3 text-[11.5px] leading-[17px] text-[var(--text-3)]">
          {done.archive
            ? "Unpack the archive to get exactly the same export: README.md explains how to load the weights and export.json holds the full record of the run that produced them."
            : "README.md explains how to load these weights, and export.json holds the full record of the run that produced them."}
        </p>
      </Modal>
    );
  }

  return (
    <Modal
      open
      onClose={onClose}
      title={`Export ${model.name}`}
      description={
        pack
          ? "The whole export is written into one .zip file you choose. Nothing is overwritten."
          : "The export is written into a folder you choose. Nothing is overwritten."
      }
      width={600}
      footer={
        <div className="flex items-center justify-end gap-2">
          <Button size="sm" variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button
            size="sm"
            variant="primary"
            icon={<Archive className="h-3.5 w-3.5" />}
            loading={busy.exportModel}
            disabled={!info || !outputDir || info.blockers.length > 0}
            onClick={() => void run()}
          >
            {merge ? "Merge and export" : pack ? "Export as one file" : "Export"}
          </Button>
        </div>
      }
    >
      {infoError ? (
        <Note tone="bad" title="Cannot export this model">
          {infoError}
        </Note>
      ) : !info ? (
        <div className="space-y-2">
          <div className="zq-skeleton h-16" />
          <div className="zq-skeleton h-10" />
        </div>
      ) : (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-x-6 gap-y-1">
            <KeyValue label="Contents" value={info.is_adapter ? "LoRA adapter" : "Full model"} />
            <KeyValue label="Size" value={formatBytes(info.size_bytes)} />
            <KeyValue label="Base model" value={info.base_model ?? "unknown"} />
            <KeyValue label="Files" value={formatCount(info.files.length)} />
            {info.checkpoints.length ? (
              <KeyValue label="Checkpoints kept in place" value={formatCount(info.checkpoints.length)} />
            ) : null}
          </div>

          <Switch
            checked={pack}
            onChange={(value) => {
              setPack(value);
              setOutputDir("");
            }}
            label="Write one single file (.zip)"
            description="A transformers model is several files at runtime, so the single-file form is an archive. Unpacking it gives the same export, file for file."
          />

          <Field
            label={pack ? "Destination file" : "Destination folder"}
            hint={pack ? "The archive is created at this path." : "Pick an empty folder, or create a new one."}
          >
            <div className="flex items-center gap-2">
              <Input
                readOnly
                value={outputDir}
                placeholder={pack ? "No file chosen yet" : "No folder chosen yet"}
                spellCheck={false}
              />
              <Button
                size="sm"
                variant="quiet"
                icon={<FolderOpen className="h-3.5 w-3.5" />}
                onClick={() => void (pack ? chooseFile() : chooseFolder())}
              >
                Choose
              </Button>
            </div>
          </Field>

          <Switch
            checked={merge && !mergeBlocked}
            disabled={mergeBlocked}
            onChange={setMerge}
            label="Merge the adapter into the base model"
            description={
              mergeBlocked
                ? !mergeInfo?.applicable
                  ? "This artefact has no adapter to merge — it is already a full model."
                  : `Needs ${mergeInfo?.missing.join(", ")}. Install the ML runtime in Settings → Environment first.`
                : "Writes a standalone model that needs no PEFT. The base model must be reachable, and the weights are genuinely combined."
            }
          />

          {info.blockers.length ? (
            <Note tone="warn" title="This folder cannot be exported yet">
              {info.blockers[0].message} {info.blockers[0].hint}
            </Note>
          ) : null}
        </div>
      )}
    </Modal>
  );
}

export function ModelsPage() {
  const { models, busy, env } = useStore(appStore);
  const [tab, setTab] = useState<Tab>("all");
  const [source, setSource] = useState("");
  const [pendingDelete, setPendingDelete] = useState<ModelEntry | null>(null);
  const [pendingExport, setPendingExport] = useState<ModelEntry | null>(null);

  const filtered = useMemo(() => {
    if (tab === "base") return models.filter((model) => !model.trained);
    if (tab === "trained") return models.filter((model) => model.trained);
    return models;
  }, [models, tab]);

  const submit = async () => {
    const value = source.trim();
    if (!value) return;
    const added = await addModel(value);
    if (added) setSource("");
  };

  return (
    <>
      <PageHeader
        icon={<Brain className="h-4 w-4" />}
        title="Models"
        subtitle={`${models.length} in the library · ${models.filter((model) => model.trained).length} trained here`}
        actions={
          <>
            <Segmented
              size="sm"
              value={tab}
              onChange={setTab}
              options={[
                { value: "all", label: "All" },
                { value: "base", label: "Base" },
                { value: "trained", label: "Trained" },
              ]}
            />
            <Button
              size="sm"
              variant="quiet"
              icon={<HardDrive className="h-3.5 w-3.5" />}
              onClick={() => void pickAndAddLocalModel()}
            >
              Local folder
            </Button>
          </>
        }
      />

      <PageBody wide>
        <Panel className="mb-4">
          <PanelHeader
            icon={<Download className="h-4 w-4" />}
            title="Add a model from Hugging Face"
            description="Paste a repository id such as meta-llama/Llama-3.2-1B or Qwen/Qwen2.5-0.5B-Instruct. Nothing is downloaded until you start training."
          />
          <div className="flex flex-wrap items-end gap-2">
            <Field label="Repository id" className="min-w-[260px] flex-1">
              <Input
                value={source}
                placeholder="Qwen/Qwen2.5-0.5B-Instruct"
                onChange={(event) => setSource(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") void submit();
                }}
              />
            </Field>
            <Button
              variant="primary"
              icon={<Plus className="h-3.5 w-3.5" />}
              loading={busy.addModel}
              disabled={!source.trim()}
              onClick={() => void submit()}
            >
              Add model
            </Button>
          </div>
          {env.hardware && !env.hardware.cuda_ready ? (
            <div className="mt-3">
              <Note tone="warn" title="CPU-only machine">
                Large models will be extremely slow to train here. Prefer parameter counts under
                about 1B on CPU, or move training to a CUDA machine.
              </Note>
            </div>
          ) : null}
        </Panel>

        {filtered.length === 0 ? (
          <EmptyState
            icon={<Brain className="h-7 w-7" />}
            title={tab === "trained" ? "No trained models yet" : "No models in the library"}
            description={
              tab === "trained"
                ? "Once a training run finishes, the resulting adapter or model appears here and can be tested in the playground."
                : "Add a Hugging Face repository or point at a local folder that contains config.json and the weight files."
            }
          />
        ) : (
          <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
            {filtered.map((model) => (
              <ModelRow
                key={model.id}
                model={model}
                onDelete={setPendingDelete}
                onExport={setPendingExport}
              />
            ))}
          </div>
        )}

        {env.hardware ? (
          <p className="mt-4 flex items-center gap-1.5 text-[11.5px] text-[var(--text-3)]">
            <Cpu className="h-3 w-3" />
            Training device: {env.hardware.training_device === "cuda" ? "CUDA" : "CPU"} ·{" "}
            {env.hardware.cuda_ready
              ? env.hardware.cuda.devices.map((device) => device.name).filter(Boolean).join(", ")
              : "no CUDA device"}
          </p>
        ) : null}
      </PageBody>

      {pendingExport ? (
        <ExportModelDialog model={pendingExport} onClose={() => setPendingExport(null)} />
      ) : null}

      <ConfirmDialog
        open={Boolean(pendingDelete)}
        title="Remove this model from the library?"
        description={
          pendingDelete?.trained
            ? "Models trained by this app live in the app data folder; removing them deletes those files. Base models and their downloads are never touched."
            : "Only the library entry is removed. Downloaded or local files are never touched."
        }
        confirmLabel="Remove"
        onConfirm={() => {
          if (pendingDelete) void removeModel(pendingDelete.id);
          setPendingDelete(null);
        }}
        onCancel={() => setPendingDelete(null)}
      />
    </>
  );
}
