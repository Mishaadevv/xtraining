import { useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Cloud,
  Database,
  ExternalLink,
  Eye,
  FolderOpen,
  Play,
  RefreshCw,
  Trash2,
  Upload,
} from "lucide-react";

import { PageBody, PageHeader } from "@/components/layout/Shell";
import { ConfirmDialog, Modal } from "@/components/ui/Overlay";
import {
  Badge,
  Button,
  Dot,
  EmptyState,
  Field,
  IconButton,
  Input,
  KeyValue,
  Note,
  Panel,
  PanelHeader,
} from "@/components/ui/primitives";
import type { DatasetEntry, ValidationIssue } from "@/lib/types";
import { cn, formatBytes, formatCount } from "@/lib/utils";
import { useStore } from "@/state/store";
import {
  addHfDataset,
  appStore,
  navigate,
  openExternal,
  pickAndImportDatasets,
  previewDataset,
  removeDataset,
  resetWizard,
  revealPath,
  validateAllDatasets,
  validateDataset,
  wizardSelectDataset,
} from "@/state/appStore";

const STATUS_TONE = { ok: "good", warnings: "warn", errors: "bad", unvalidated: "neutral" } as const;

function IssueRow({ issue }: { issue: ValidationIssue }) {
  const tone =
    issue.severity === "error" ? "var(--red)" : issue.severity === "warning" ? "var(--amber)" : "var(--blue)";
  return (
    <div className="flex items-start gap-2.5 border-t border-[var(--border-soft)] py-2.5 first:border-t-0">
      <span className="mt-[3px] shrink-0" style={{ color: tone }}>
        <AlertTriangle className="h-3.5 w-3.5" />
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-[12.5px] leading-[18px]">{issue.message}</p>
        {issue.hint ? (
          <p className="mt-0.5 text-[11.5px] leading-[17px] text-[var(--text-2)]">{issue.hint}</p>
        ) : null}
        {issue.breakdown ? (
          <div className="mt-1 flex flex-wrap gap-1.5">
            {Object.entries(issue.breakdown).map(([reason, count]) => (
              <Badge key={reason}>
                {reason.replace(/_/g, " ")}: {count}
              </Badge>
            ))}
          </div>
        ) : null}
        {issue.samples && issue.samples.length ? (
          <p className="zq-mono mt-1 text-[10.5px] text-[var(--text-3)]">
            records {issue.samples.slice(0, 10).join(", ")}
            {issue.samples.length > 10 ? "…" : ""}
          </p>
        ) : null}
      </div>
      <Badge tone={issue.severity === "error" ? "bad" : issue.severity === "warning" ? "warn" : "info"}>
        {issue.severity}
      </Badge>
    </div>
  );
}

/** Hub datasets are referenced by id; nothing is copied into the app. */
function isHub(dataset: DatasetEntry): boolean {
  return dataset.format === "hf";
}

function DatasetDetail({ dataset }: { dataset: DatasetEntry }) {
  const report = dataset.report;
  const stats = report?.stats;
  const busy = useStore(appStore).busy;

  return (
    <Panel className="zq-rise">
      <PanelHeader
        title={dataset.name}
        description={
          <span className="zq-mono truncate text-[11px]">{dataset.path}</span>
        }
        icon={<Database className="h-4 w-4" />}
        actions={
          <>
            <Button
              size="sm"
              variant="quiet"
              loading={busy[`dataset:${dataset.id}`]}
              onClick={() => void validateDataset(dataset.id)}
            >
              Re-validate
            </Button>
            {isHub(dataset) ? (
              <IconButton
                title="Open this dataset on the Hugging Face Hub"
                onClick={() => void openExternal(`https://huggingface.co/datasets/${dataset.hfId ?? dataset.path}`)}
              >
                <ExternalLink className="h-3.5 w-3.5" />
              </IconButton>
            ) : (
              <IconButton title="Reveal in file manager" onClick={() => void revealPath(dataset.path)}>
                <FolderOpen className="h-3.5 w-3.5" />
              </IconButton>
            )}
          </>
        }
      />

      <div className="grid grid-cols-2 gap-x-6 gap-y-1 sm:grid-cols-3">
        <KeyValue label="Format" value={dataset.format} />
        <KeyValue label="Records" value={formatCount(stats?.records ?? dataset.records)} />
        <KeyValue
          label="Usable"
          value={formatCount(stats?.usable ?? dataset.usable)}
          tone={(stats?.usable ?? dataset.usable) === 0 ? "bad" : undefined}
        />
        <KeyValue
          label="Size"
          value={isHub(dataset) ? "Downloaded on demand" : formatBytes(dataset.sizeBytes)}
        />
        {isHub(dataset) ? <KeyValue label="Split" value={dataset.split ?? "train"} /> : null}
        <KeyValue label="Duplicates" value={formatCount(stats?.duplicates ?? 0)} tone={stats?.duplicates ? "warn" : undefined} />
        <KeyValue
          label="Over context"
          value={formatCount(stats?.over_length ?? 0)}
          tone={stats?.over_length ? "warn" : undefined}
        />
        <KeyValue label="Avg length" value={stats?.avg_chars ? `${stats.avg_chars} chars` : "—"} />
        <KeyValue label="Est. tokens / sample" value={formatCount(stats?.est_tokens_avg ?? null)} />
        <KeyValue
          label="Unusable"
          value={formatCount(stats?.unusable ?? 0)}
          tone={stats?.unusable ? "warn" : undefined}
        />
      </div>

      {report?.mapping ? (
        <div className="mt-3 rounded-[10px] border border-[var(--border-soft)] bg-[var(--panel-2)] p-3">
          <div className="mb-1.5 flex items-center gap-2">
            <span className="text-[11px] font-medium uppercase tracking-[0.06em] text-[var(--text-3)]">
              Field mapping
            </span>
            <Badge tone={report.mapping.auto ? "info" : "accent"}>
              {report.mapping.auto ? "detected automatically" : "manual"}
            </Badge>
          </div>
          <div className="zq-mono text-[11.5px] text-[var(--text-2)]">
            {report.mapping.kind === "pair" ? (
              <>
                prompt = <span className="text-[var(--text)]">{report.mapping.instruction_field}</span>
                {report.mapping.input_field ? (
                  <>
                    {" · input = "}
                    <span className="text-[var(--text)]">{report.mapping.input_field}</span>
                  </>
                ) : null}
                {" · target = "}
                <span className="text-[var(--text)]">{report.mapping.output_field}</span>
                {" · template = "}
                <span className="text-[var(--text)]">{report.mapping.template}</span>
              </>
            ) : report.mapping.kind === "chat" ? (
              <>
                messages = <span className="text-[var(--text)]">{report.mapping.messages_field}</span>
                {report.mapping.messages_field_alternatives?.length ? (
                  <span className="text-[var(--text-3)]">
                    {" "}
                    (also: {report.mapping.messages_field_alternatives.join(", ")})
                  </span>
                ) : null}
              </>
            ) : (
              <>
                text = <span className="text-[var(--text)]">{report.mapping.text_field ?? "joined fields"}</span>
              </>
            )}
          </div>
          {stats?.roles && Object.keys(stats.roles).length ? (
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {Object.entries(stats.roles).map(([role, count]) => (
                <Badge key={role}>
                  {role}: {count}
                </Badge>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}

      {report && report.issues.length ? (
        <div className="mt-3">
          <div className="mb-1 text-[11px] font-medium uppercase tracking-[0.06em] text-[var(--text-3)]">
            Findings
          </div>
          {report.issues.map((issue, index) => (
            <IssueRow key={`${issue.code}_${index}`} issue={issue} />
          ))}
        </div>
      ) : dataset.status === "ok" ? (
        <div className="mt-3">
          <Note tone="good" title="No problems found">
            Every record produced usable training text.
          </Note>
        </div>
      ) : null}

      {report?.preview?.length ? (
        <div className="mt-3">
          <div className="mb-1.5 text-[11px] font-medium uppercase tracking-[0.06em] text-[var(--text-3)]">
            Preview
          </div>
          <div className="space-y-2">
            {report.preview.slice(0, 3).map((sample, index) => (
              <pre
                key={index}
                className="zq-mono max-h-[150px] overflow-auto whitespace-pre-wrap rounded-[10px] border border-[var(--border-soft)] bg-[var(--code-bg)] p-2.5 text-[11.5px] leading-[18px] text-[var(--text-2)]"
              >
                {sample}
              </pre>
            ))}
          </div>
        </div>
      ) : null}

      <div className="mt-4 flex items-center gap-2 border-t border-[var(--border-soft)] pt-3">
        <Button
          size="sm"
          variant="primary"
          icon={<Play className="h-3.5 w-3.5" />}
          onClick={() => {
            resetWizard({ datasetId: dataset.id, datasetReport: dataset.report });
            void wizardSelectDataset(dataset.id);
            navigate("new");
          }}
        >
          Use in a training run
        </Button>
      </div>
    </Panel>
  );
}

export function DatasetsPage() {
  const { datasets, busy } = useStore(appStore);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<DatasetEntry | null>(null);
  const [previewFor, setPreviewFor] = useState<DatasetEntry | null>(null);
  const [previewSamples, setPreviewSamples] = useState<string[]>([]);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [hubOpen, setHubOpen] = useState(false);
  const [hubId, setHubId] = useState("");
  const [hubSplit, setHubSplit] = useState("train");

  const addHub = async () => {
    const entry = await addHfDataset(hubId, hubSplit);
    if (!entry) return;
    setHubOpen(false);
    setHubId("");
    setSelectedId(entry.id);
  };

  const selected = datasets.find((dataset) => dataset.id === selectedId) ?? datasets[0] ?? null;
  const builtinCount = datasets.filter((dataset) => dataset.builtin).length;

  const openPreview = async (dataset: DatasetEntry) => {
    setPreviewFor(dataset);
    setPreviewLoading(true);
    setPreviewSamples([]);
    const samples = await previewDataset(dataset.id, 5);
    setPreviewSamples(samples ?? []);
    setPreviewLoading(false);
  };

  return (
    <>
      <PageHeader
        icon={<Database className="h-4 w-4" />}
        title="Datasets"
        subtitle={
          datasets.length
            ? `${datasets.length} available${builtinCount ? ` (${builtinCount} built-in)` : ""} · JSON, JSONL, CSV, TXT, Parquet, folders and the Hugging Face Hub`
            : "Import JSON, JSONL, CSV, TXT, Parquet, a folder of shards, or a Hub dataset"
        }
        actions={
          <>
            <Button
              size="sm"
              variant="quiet"
              icon={<RefreshCw className="h-3.5 w-3.5" />}
              onClick={() => void validateAllDatasets()}
              loading={busy.validateAll}
              disabled={!datasets.length}
            >
              Validate all
            </Button>
            <Button
              size="sm"
              variant="quiet"
              icon={<Cloud className="h-3.5 w-3.5" />}
              onClick={() => setHubOpen(true)}
            >
              Add from Hub
            </Button>
            <Button
              size="sm"
              variant="primary"
              icon={<Upload className="h-3.5 w-3.5" />}
              loading={busy.importDataset}
              onClick={() => void pickAndImportDatasets()}
            >
              Import dataset
            </Button>
          </>
        }
      />

      <PageBody wide>
        {datasets.length === 0 ? (
          <EmptyState
            icon={<Database className="h-7 w-7" />}
            title="No datasets imported"
            description="Datasets are referenced in place — nothing is copied, so importing a large file is instant, and a Hub dataset is just an id until the rows are actually needed. Validation then tells you exactly what the model would learn from."
            action={
              <div className="flex items-center gap-2">
                <Button
                  variant="quiet"
                  icon={<Cloud className="h-3.5 w-3.5" />}
                  onClick={() => setHubOpen(true)}
                >
                  Add from Hub
                </Button>
                <Button
                  variant="primary"
                  icon={<Upload className="h-3.5 w-3.5" />}
                  loading={busy.importDataset}
                  onClick={() => void pickAndImportDatasets()}
                >
                  Import a dataset
                </Button>
              </div>
            }
          />
        ) : (
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-[320px_minmax(0,1fr)]">
            <div className="space-y-2">
              {datasets.map((dataset) => (
                <button
                  key={dataset.id}
                  type="button"
                  onClick={() => setSelectedId(dataset.id)}
                  className={cn(
                    "flex w-full items-start gap-2.5 rounded-[12px] border p-3 text-left transition-colors",
                    selected?.id === dataset.id
                      ? "border-[var(--border)] bg-[var(--panel-2)]"
                      : "border-[var(--border-soft)] bg-[var(--panel)] hover:bg-[var(--hover)]",
                  )}
                >
                  <span className="mt-[3px] shrink-0 text-[var(--text-3)]">
                    {dataset.status === "ok" ? (
                      <CheckCircle2 className="h-3.5 w-3.5" style={{ color: "var(--green)" }} />
                    ) : dataset.status === "errors" ? (
                      <AlertTriangle className="h-3.5 w-3.5" style={{ color: "var(--red)" }} />
                    ) : (
                      <Database className="h-3.5 w-3.5" />
                    )}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[12.5px] font-medium">{dataset.name}</span>
                    <span className="zq-mono block truncate text-[10.5px] text-[var(--text-3)]">
                      {formatCount(dataset.usable)} usable / {formatCount(dataset.records)} ·{" "}
                      {formatBytes(dataset.sizeBytes)}
                    </span>
                  </span>
                  <span className="flex shrink-0 flex-col items-end gap-1">
                    <span className="flex items-center gap-1">
                      {dataset.builtin ? <Badge tone="info">built-in</Badge> : null}
                      <Badge tone={STATUS_TONE[dataset.status]}>
                        <Dot tone={dataset.status === "ok" ? "good" : dataset.status === "errors" ? "bad" : dataset.status === "warnings" ? "warn" : "neutral"} />
                        {dataset.status}
                      </Badge>
                    </span>
                    <span className="flex items-center gap-0.5">
                      <IconButton
                        title="Preview normalised samples"
                        className="h-6 w-6"
                        onClick={(event) => {
                          event.stopPropagation();
                          void openPreview(dataset);
                        }}
                      >
                        <Eye className="h-3 w-3" />
                      </IconButton>
                      {dataset.builtin ? null : (
                        <IconButton
                          title="Remove from the library"
                          className="h-6 w-6"
                          onClick={(event) => {
                            event.stopPropagation();
                            setPendingDelete(dataset);
                          }}
                        >
                          <Trash2 className="h-3 w-3" />
                        </IconButton>
                      )}
                    </span>
                  </span>
                </button>
              ))}
            </div>

            {selected ? <DatasetDetail dataset={selected} /> : null}
          </div>
        )}
      </PageBody>

      <Modal
        open={Boolean(previewFor)}
        onClose={() => setPreviewFor(null)}
        title={previewFor?.name ?? "Preview"}
        description="The first samples exactly as the trainer will see them, after field mapping and templating."
        width={720}
      >
        {previewLoading ? (
          <div className="space-y-2">
            {[0, 1, 2].map((index) => (
              <div key={index} className="zq-skeleton h-20" />
            ))}
          </div>
        ) : previewSamples.length ? (
          <div className="space-y-2">
            {previewSamples.map((sample, index) => (
              <pre
                key={index}
                className="zq-mono whitespace-pre-wrap rounded-[10px] border border-[var(--border-soft)] bg-[var(--code-bg)] p-3 text-[11.5px] leading-[18px] text-[var(--text-2)]"
              >
                {sample}
              </pre>
            ))}
          </div>
        ) : (
          <p className="text-[12.5px] text-[var(--text-3)]">No samples could be produced from this dataset.</p>
        )}
      </Modal>

      <Modal
        open={hubOpen}
        onClose={() => setHubOpen(false)}
        title="Add a dataset from the Hugging Face Hub"
        description="Nothing is downloaded until you validate or start a run — the id is all that is stored."
        width={560}
        footer={
          <div className="flex items-center justify-end gap-2">
            <Button size="sm" variant="ghost" onClick={() => setHubOpen(false)}>
              Cancel
            </Button>
            <Button
              size="sm"
              variant="primary"
              loading={busy.importDataset}
              disabled={!hubId.trim()}
              onClick={() => void addHub()}
            >
              Add and validate
            </Button>
          </div>
        }
      >
        <div className="space-y-3">
          <Field label="Dataset id" hint="For example tatsu-lab/alpaca, or a full huggingface.co/datasets/… URL.">
            <Input
              autoFocus
              value={hubId}
              onChange={(event) => setHubId(event.target.value)}
              placeholder="tatsu-lab/alpaca"
              spellCheck={false}
              onKeyDown={(event) => {
                if (event.key === "Enter" && hubId.trim()) void addHub();
              }}
            />
          </Field>
          <Field label="Split" hint="Validation and training both read this split.">
            <Input
              className="w-[160px]"
              value={hubSplit}
              onChange={(event) => setHubSplit(event.target.value)}
              placeholder="train"
              spellCheck={false}
            />
          </Field>
          <Note tone="info" title="Needs the opt-in packages">
            Downloading Hub datasets uses the <span className="zq-mono">datasets</span> package, which is part
            of the ML runtime in Settings → Environment. Without it, validation reports exactly what is
            missing instead of guessing.
          </Note>
        </div>
      </Modal>

      <ConfirmDialog
        open={Boolean(pendingDelete)}
        title="Remove this dataset from the library?"
        description="Only the library entry is removed. The file on disk is never touched."
        confirmLabel="Remove"
        onConfirm={() => {
          if (pendingDelete) void removeDataset(pendingDelete.id);
          setPendingDelete(null);
        }}
        onCancel={() => setPendingDelete(null)}
      />
    </>
  );
}
