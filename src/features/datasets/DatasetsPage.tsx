import { useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Cloud,
  Database,
  Download,
  ExternalLink,
  Eye,
  FolderOpen,
  HardDriveDownload,
  Play,
  RefreshCw,
  Trash2,
  Undo2,
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
  Select,
  Switch,
} from "@/components/ui/primitives";
import type { DatasetEntry, ValidationIssue } from "@/lib/types";
import { cn, formatBytes, formatCount } from "@/lib/utils";
import { useStore } from "@/state/store";
import {
  addHfDataset,
  appStore,
  chooseDatasetsFolder,
  exportDatasetToFile,
  navigate,
  openExternal,
  openPath,
  pickAndImportDatasets,
  previewDataset,
  refreshDatasetFormats,
  removeDataset,
  resetWizard,
  restoreDataset,
  revealPath,
  scanDatasets,
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

function originLabel(dataset: DatasetEntry): string {
  if (isHub(dataset)) return "Hugging Face Hub id";
  if (dataset.origin === "folder") return "found in the scanned folder";
  if (dataset.isDirectory) return "imported folder of shards";
  return "imported file";
}

/**
 * Single-file export.
 *
 * A dataset is often a folder of shards or a database; the export always writes
 * exactly one file, in a format the user picks, and the file name they choose
 * in the native dialog decides the actual format.
 */
function ExportDatasetDialog({ dataset, onClose }: { dataset: DatasetEntry; onClose: () => void }) {
  const { busy, datasetFormats } = useStore(appStore);
  const [format, setFormat] = useState("jsonl");
  const [raw, setRaw] = useState(false);
  const [written, setWritten] = useState<string | null>(null);

  const formats = datasetFormats?.export_formats ?? ["jsonl", "json", "csv", "tsv", "txt", "parquet"];

  const run = async () => {
    const output = await exportDatasetToFile(dataset.id, { format, raw });
    if (output) setWritten(output);
  };

  if (written) {
    return (
      <Modal
        open
        onClose={onClose}
        title="Exported as one file"
        description="The whole dataset was written to a single file — nothing else was created."
        width={560}
        footer={
          <div className="flex items-center justify-end gap-2">
            <Button size="sm" variant="quiet" icon={<FolderOpen className="h-3.5 w-3.5" />} onClick={() => void revealPath(written)}>
              Show in folder
            </Button>
            <Button size="sm" variant="primary" onClick={onClose}>
              Done
            </Button>
          </div>
        }
      >
        <Note tone="good" title="Written to">
          <span className="zq-mono break-all">{written}</span>
        </Note>
      </Modal>
    );
  }

  return (
    <Modal
      open
      onClose={onClose}
      title={`Export ${dataset.name} as one file`}
      description="One file, whatever the source was: a shard folder, a database, a spreadsheet or a Hub dataset."
      width={560}
      footer={
        <div className="flex items-center justify-end gap-2">
          <Button size="sm" variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button
            size="sm"
            variant="primary"
            icon={<Download className="h-3.5 w-3.5" />}
            loading={busy[`dataset-export:${dataset.id}`]}
            onClick={() => void run()}
          >
            Choose file and export
          </Button>
        </div>
      }
    >
      <div className="space-y-4">
        <Field
          label="Format"
          hint="The extension of the file you pick decides the real format; this only pre-selects it."
        >
          <Select value={format} onChange={(event) => setFormat(event.target.value)}>
            {formats.map((item) => (
              <option key={item} value={item}>
                {item === "jsonl"
                  ? "JSONL — one record per line (recommended)"
                  : item === "json"
                    ? "JSON — one array"
                    : item.toUpperCase()}
              </option>
            ))}
          </Select>
        </Field>

        <Switch
          checked={raw}
          onChange={setRaw}
          label="Keep the original records"
          description={
            raw
              ? "Every original field is written as it is; nothing is mapped or templated."
              : "Writes what the trainer would see: fields mapped, templates applied, chat rendered to text."
          }
        />

        <Note tone="info" title="What each format keeps">
          JSONL and JSON keep record boundaries and every field. CSV and TSV flatten nested values into JSON text.
          Plain text separates records with a blank line, so a sample that contains one is split when the file is
          read back. Parquet needs <span className="zq-mono">pyarrow</span> — if it is missing the export says so
          instead of writing a file that cannot be read.
        </Note>
      </div>
    </Modal>
  );
}

function DatasetDetail({ dataset }: { dataset: DatasetEntry }) {
  const report = dataset.report;
  const stats = report?.stats;
  const busy = useStore(appStore).busy;
  const [exportOpen, setExportOpen] = useState(false);

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
            <Button
              size="sm"
              variant="quiet"
              icon={<Download className="h-3.5 w-3.5" />}
              onClick={() => setExportOpen(true)}
            >
              Export as one file
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
        <KeyValue label="Source" value={originLabel(dataset)} />
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
        {report?.dataset?.compression ? (
          <KeyValue label="Compressed" value={report.dataset.compression} />
        ) : null}
        {report?.dataset?.items ? (
          <KeyValue label={dataset.format === "sqlite" ? "Table" : "Sheet"} value={report.dataset.items} />
        ) : null}
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
          value={formatCount(stats?.empty ?? 0)}
          tone={stats?.empty ? "warn" : undefined}
        />
      </div>

      {report?.dataset?.files?.length ? (
        <div className="mt-3 rounded-[10px] border border-[var(--border-soft)] bg-[var(--panel-2)] p-3">
          <div className="mb-1.5 flex items-center gap-2">
            <span className="text-[11px] font-medium uppercase tracking-[0.06em] text-[var(--text-3)]">
              Files read
            </span>
            <Badge>{report.dataset.files.length}</Badge>
          </div>
          <div className="zq-mono max-h-[120px] overflow-auto text-[11px] text-[var(--text-2)]">
            {report.dataset.files.map((file) => (
              <div key={file} className="truncate">
                {file}
              </div>
            ))}
          </div>
        </div>
      ) : null}

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

      {exportOpen ? <ExportDatasetDialog dataset={dataset} onClose={() => setExportOpen(false)} /> : null}
    </Panel>
  );
}

/** The folder the app scans, with the actions that operate on it. */
function FolderStrip() {
  const { datasetFolder, busy, hiddenDatasets } = useStore(appStore);

  return (
    <Panel>
      <PanelHeader
        title="Dataset folder"
        description={
          <span className="zq-mono truncate text-[11px]">{datasetFolder ?? "…"}</span>
        }
        icon={<HardDriveDownload className="h-4 w-4" />}
        actions={
          <>
            <Button
              size="sm"
              variant="quiet"
              icon={<RefreshCw className="h-3.5 w-3.5" />}
              loading={busy.scanDatasets}
              onClick={() => void scanDatasets()}
            >
              Scan now
            </Button>
            <Button size="sm" variant="quiet" icon={<FolderOpen className="h-3.5 w-3.5" />} onClick={() => void chooseDatasetsFolder()}>
              Change folder
            </Button>
            {datasetFolder ? (
              <IconButton title="Open the folder" onClick={() => void openPath(datasetFolder)}>
                <ExternalLink className="h-3.5 w-3.5" />
              </IconButton>
            ) : null}
          </>
        }
      />
      <p className="text-[12.5px] leading-[18px] text-[var(--text-2)]">
        Drop dataset files here and they appear in the list — every supported file is indexed, and each
        subfolder that contains dataset files is listed as one shard set. Nothing is copied: the files are read
        where they are.
      </p>
      {hiddenDatasets.length ? (
        <div className="mt-3 border-t border-[var(--border-soft)] pt-3">
          <div className="mb-1.5 flex items-center gap-2">
            <span className="text-[11px] font-medium uppercase tracking-[0.06em] text-[var(--text-3)]">
              Removed from the list
            </span>
            <Badge>{hiddenDatasets.length}</Badge>
          </div>
          <div className="space-y-1">
            {hiddenDatasets.map((dataset) => (
              <div key={dataset.id} className="flex items-center gap-2">
                <span className="zq-mono min-w-0 flex-1 truncate text-[11px] text-[var(--text-3)]">
                  {dataset.path}
                </span>
                <Button
                  size="sm"
                  variant="ghost"
                  icon={<Undo2 className="h-3 w-3" />}
                  onClick={() => void restoreDataset(dataset.id)}
                >
                  Restore
                </Button>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </Panel>
  );
}

/** Every dataset type the backend can read, straight from the backend. */
function SupportedTypes() {
  const { datasetFormats } = useStore(appStore);
  if (!datasetFormats) return null;

  return (
    <Panel>
      <PanelHeader
        title="Supported dataset types"
        description={`${datasetFormats.formats.length} types · read by the Python backend`}
        icon={<Database className="h-4 w-4" />}
        actions={
          <IconButton
            title="Re-check which readers are installed"
            onClick={() => void refreshDatasetFormats()}
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </IconButton>
        }
      />
      <div className="space-y-2">
        {datasetFormats.formats.map((format) => (
          <div key={format.id} className="flex items-start gap-2.5">
            <span className="mt-[5px] shrink-0">
              {format.available ? (
                <Dot tone="good" />
              ) : (
                <Dot tone="warn" />
              )}
            </span>
            <div className="min-w-0 flex-1">
              <p className="text-[12px]">
                <span className="font-medium">{format.extensions.join(" ")}</span>{" "}
                <span className="text-[var(--text-2)]">{format.label}</span>
              </p>
              {!format.available && format.hint ? (
                <p className="zq-mono mt-0.5 text-[11px] text-[var(--text-3)]">{format.hint}</p>
              ) : null}
            </div>
          </div>
        ))}
      </div>
      {datasetFormats.compression.length ? (
        <p className="mt-3 border-t border-[var(--border-soft)] pt-3 text-[11.5px] leading-[17px] text-[var(--text-3)]">
          Compressed files work too: <span className="zq-mono">{datasetFormats.compression.join(" ")}</span> —
          name them with the inner extension (<span className="zq-mono">shard.jsonl.gz</span>).
        </p>
      ) : null}
      {datasetFormats.hub_available === false ? (
        <Note tone="warn" title="Hugging Face Hub datasets need one extra package">
          Install <span className="zq-mono">datasets</span> in Settings → Environment to fetch Hub rows.
        </Note>
      ) : null}
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
  const [typesOpen, setTypesOpen] = useState(false);

  const addHub = async () => {
    const entry = await addHfDataset(hubId, hubSplit);
    if (!entry) return;
    setHubOpen(false);
    setHubId("");
    setSelectedId(entry.id);
  };

  const selected = datasets.find((dataset) => dataset.id === selectedId) ?? datasets[0] ?? null;

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
            ? `${datasets.length} in the library · scanned from the folder or imported from anywhere`
            : "No datasets yet — scan the folder, import a file, or add one from the Hub"
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
        <div className="space-y-4">
          <FolderStrip />

          <button
            type="button"
            onClick={() => setTypesOpen((open) => !open)}
            className="flex w-full items-center gap-2 text-left text-[11.5px] text-[var(--text-3)] transition-colors hover:text-[var(--text-2)]"
          >
            <Badge tone="info">JSON</Badge>
            <Badge tone="info">JSONL</Badge>
            <Badge tone="info">CSV</Badge>
            <Badge tone="info">TSV</Badge>
            <Badge tone="info">TXT</Badge>
            <Badge tone="info">Parquet</Badge>
            <Badge tone="info">Arrow</Badge>
            <Badge tone="info">ORC</Badge>
            <Badge tone="info">SQLite</Badge>
            <Badge tone="info">Excel</Badge>
            <Badge tone="info">YAML</Badge>
            <Badge tone="info">.gz</Badge>
            <span className="ml-1">{typesOpen ? "hide the full list" : "see every supported type and what it needs"}</span>
          </button>

          {typesOpen ? <SupportedTypes /> : null}

          {datasets.length === 0 ? (
            <EmptyState
              icon={<Database className="h-7 w-7" />}
              title="The library is empty — that is how the app ships"
              description="Nothing is bundled: every dataset in this list is one you brought. Scan the folder above, import a file or a shard folder from anywhere on disk, or add a dataset id from the Hugging Face Hub. Validation then tells you exactly what the model would learn from."
              action={
                <div className="flex items-center gap-2">
                  <Button
                    variant="quiet"
                    icon={<RefreshCw className="h-3.5 w-3.5" />}
                    loading={busy.scanDatasets}
                    onClick={() => void scanDatasets()}
                  >
                    Scan the dataset folder
                  </Button>
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
                        {isHub(dataset) ? "hub" : formatBytes(dataset.sizeBytes)}
                      </span>
                    </span>
                    <span className="flex shrink-0 flex-col items-end gap-1">
                      <span className="flex items-center gap-1">
                        {dataset.origin === "folder" ? <Badge tone="info">scanned</Badge> : null}
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
                        <IconButton
                          title="Remove from the list"
                          className="h-6 w-6"
                          onClick={(event) => {
                            event.stopPropagation();
                            setPendingDelete(dataset);
                          }}
                        >
                          <Trash2 className="h-3 w-3" />
                        </IconButton>
                      </span>
                    </span>
                  </button>
                ))}
              </div>

              {selected ? <DatasetDetail dataset={selected} /> : null}
            </div>
          )}
        </div>
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
          {/* Side by side, so the split does not read as a second datasets field. */}
          <div className="flex flex-col gap-3 sm:flex-row">
            <Field
              className="min-w-0 flex-1"
              label="Dataset id"
              hint="For example tatsu-lab/alpaca, or a full huggingface.co/datasets/… URL."
            >
              <Input
                autoFocus
                value={hubId}
                onChange={(event) => setHubId(event.target.value)}
                placeholder="owner/name — e.g. tatsu-lab/alpaca"
                spellCheck={false}
                aria-label="Hub dataset id"
                onKeyDown={(event) => {
                  if (event.key === "Enter" && hubId.trim()) void addHub();
                }}
              />
            </Field>
            <Field
              className="shrink-0 sm:w-[150px]"
              label="Split"
              hint="Read by validation and training."
            >
              <Input
                value={hubSplit}
                onChange={(event) => setHubSplit(event.target.value)}
                placeholder="train"
                spellCheck={false}
                aria-label="Hub split"
              />
            </Field>
          </div>
          <Note tone="info" title="Needs the opt-in packages">
            Downloading Hub datasets uses the <span className="zq-mono">datasets</span> package, which is part
            of the ML runtime in Settings → Environment. Without it, validation reports exactly what is
            missing instead of guessing.
          </Note>
        </div>
      </Modal>

      <ConfirmDialog
        open={Boolean(pendingDelete)}
        title="Remove this dataset from the list?"
        description={
          pendingDelete?.origin === "folder"
            ? "A folder-scanned dataset stays on disk and can be brought back from the Removed list. A manually imported entry is dropped from the library; its file is never touched."
            : "Only the library entry is removed. The file on disk is never touched."
        }
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
