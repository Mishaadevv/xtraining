import { useMemo, useState } from "react";
import { Brush, Database, FileDown, Scissors, Search } from "lucide-react";
import { BarChart } from "../components/charts";
import { api } from "../lib/api";
import { basename, number, truncate } from "../lib/format";
import { useApp } from "../state/app";
import { useRouter } from "../state/router";
import {
  Badge,
  Button,
  Callout,
  CodeBlock,
  EmptyState,
  Field,
  KeyValue,
  NumberInput,
  Panel,
  SectionHeader,
  Select,
  Stat,
  Table,
  Tab,
  Tabs,
  Td,
  TextInput,
  Th,
  Toggle,
} from "../components/ui";
import { ErrorPanel, Loading, NotFound, PathText, useEngine } from "./common";

type Mapping = Record<string, string>;

export function DatasetDetailPage({ id }: { id: string }) {
  const { registry, reportError, toast, settings } = useApp();
  const router = useRouter();
  const entry = (registry?.datasets ?? []).find((dataset) => dataset.id === id);
  const path = entry?.path ?? "";
  const [tab, setTab] = useState("overview");
  const [mapping, setMapping] = useState<Mapping>({});
  const [busy, setBusy] = useState(false);
  const [cleanResult, setCleanResult] = useState<any>(null);
  const [splitResult, setSplitResult] = useState<any>(null);
  const [tokenizerResult] = useState<any>(null);
  const [page, setPage] = useState(0);
  const [cleanOps, setCleanOps] = useState({
    drop_empty: true,
    trim_whitespace: true,
    drop_duplicates: true,
    strip_html: false,
    min_length: 0,
    max_length: 0,
    regex_pattern: "",
    regex_replacement: "",
  });
  const [splitConfig, setSplitConfig] = useState({ train: 0.9, validation: 0.05, test: 0.05, seed: 42 });
  const [tokenizerConfig, setTokenizerConfig] = useState({ vocab_size: 512, template: "chatml" });

  const report = useEngine<any>(
    "datasets.inspect",
    { path, mapping, sample_size: 4000 },
    { auto: Boolean(path), timeout: 900_000, deps: [path] },
  );
  const pageSize = 20;
  const rows = useEngine<any>(
    "datasets.preview",
    { path, mapping, limit: pageSize, offset: page * pageSize, template: tokenizerConfig.template },
    { auto: Boolean(path) && tab === "preview", timeout: 300_000, deps: [path, tab, page, JSON.stringify(mapping), tokenizerConfig.template] },
  );
  const tokenizerStats = useEngine<any>(
    "tokenizer.stats",
    { path, mapping, limit: 500, template: tokenizerConfig.template, tokenizer_path: tokenizerResult?.tokenizer ?? null },
    { auto: Boolean(path) && tab === "tokenizer", timeout: 300_000, deps: [path, tab, JSON.stringify(mapping), tokenizerResult?.tokenizer] },
  );

  const effectiveMapping = useMemo(() => {
    const merged: Mapping = { ...(report.data?.detected_mapping ?? {}), ...mapping };
    return merged;
  }, [report.data, mapping]);

  if (!entry) return <NotFound what="This dataset" onBack={() => router.navigate("/datasets")} />;

  const operations = useMemo(() => {
    const ops: any[] = [];
    if (cleanOps.drop_empty) ops.push({ type: "drop_empty" });
    if (cleanOps.trim_whitespace) ops.push({ type: "trim_whitespace" });
    if (cleanOps.strip_html) ops.push({ type: "strip_html" });
    if (cleanOps.drop_duplicates) ops.push({ type: "drop_duplicates" });
    if (cleanOps.min_length || cleanOps.max_length) {
      ops.push({ type: "length_filter", min: cleanOps.min_length, max: cleanOps.max_length });
    }
    if (cleanOps.regex_pattern) {
      ops.push({ type: "regex_replace", pattern: cleanOps.regex_pattern, replacement: cleanOps.regex_replacement });
    }
    return ops;
  }, [cleanOps]);

  const runCleaning = async (dryRun: boolean) => {
    setBusy(true);
    try {
      if (dryRun) {
        const result = await api.call<any>("datasets.plan_clean", { path, mapping: effectiveMapping, operations, preview_limit: 5 });
        setCleanResult({ dryRun: true, ...result });
      } else {
        const destination = `${path}.clean.jsonl`;
        const result = await api.call<any>("datasets.clean", {
          path,
          mapping: effectiveMapping,
          operations,
          destination,
          format: "jsonl",
        });
        setCleanResult({ dryRun: false, ...result });
        toast({
          title: `Cleaned dataset written`,
          body: `${number(result.output_records)} of ${number(result.input_records)} records kept → ${basename(result.output)}`,
          tone: "ok",
        });
      }
    } catch (error) {
      reportError(error, dryRun ? "Preview failed" : "Cleaning failed");
    } finally {
      setBusy(false);
    }
  };

  const runSplit = async () => {
    setBusy(true);
    try {
      const outputDir = `${basename(path)}-splits`;
      const result = await api.call<any>("datasets.split", {
        path,
        mapping: effectiveMapping,
        output_dir: `${settings?.workspace ?? "."}/exports/${outputDir}`,
        ratios: { train: splitConfig.train, validation: splitConfig.validation, test: splitConfig.test },
        seed: splitConfig.seed,
        format: "jsonl",
      });
      setSplitResult(result);
      toast({ title: "Split written", body: `${JSON.stringify(result.counts)}`, tone: "ok" });
    } catch (error) {
      reportError(error, "Split failed");
    } finally {
      setBusy(false);
    }
  };

  const runTokenizer = async () => {
    setBusy(true);
    try {
      const job = await api.jobs.start({
        kind: "tokenize",
        dataset_paths: [path],
        mapping: effectiveMapping,
        template: tokenizerConfig.template,
        vocab_size: tokenizerConfig.vocab_size,
      });
      toast({
        title: "Tokenizer training started",
        body: `Job ${job.jobId} is learning byte-pair merges from this dataset. Progress appears on the Jobs page.`,
        tone: "info",
      });
    } catch (error) {
      reportError(error, "Tokenizer training failed to start");
    } finally {
      setBusy(false);
    }
  };

  const exportDataset = async (format: string) => {
    setBusy(true);
    try {
      const destination = await api.dialog.saveFile({
        title: "Export dataset",
        defaultPath: `${settings?.workspace ?? "."}/exports/${basename(path)}.${format}`,
      });
      if (!destination) return;
      const result = await api.call<any>("datasets.export", { path, mapping: effectiveMapping, destination, format });
      toast({ title: `Exported as ${format}`, body: `${result.size_human} written to ${result.output}`, tone: "ok" });
    } catch (error) {
      reportError(error, "Export failed");
    } finally {
      setBusy(false);
    }
  };

  const reportData = report.data;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-2xs uppercase tracking-wider text-ink-3">dataset · {reportData?.kind ?? entry.kind ?? "file"}</div>
          <h1 className="text-lg font-semibold tracking-tight">{entry.name}</h1>
          <PathText value={path} />
        </div>
        <div className="flex flex-wrap gap-2">
          <Button onClick={() => api.shell.reveal(path)}>Reveal</Button>
          <Button onClick={() => void exportDataset("jsonl")} loading={busy} icon={<FileDown size={13} />}>
            Export JSONL
          </Button>
          <Button onClick={() => router.navigate(`/training/new?dataset=${encodeURIComponent(path)}`)}>Train on this</Button>
        </div>
      </div>

      {report.error ? <ErrorPanel error={report.error} onRetry={report.reload} /> : null}
      {report.loading && !reportData ? <Loading label="Reading the dataset…" lines={5} /> : null}
      {reportData?.empty ? (
        <Callout tone="warn" title="This dataset contains no readable records" hint="Check the file extension and that optional readers are installed." />
      ) : null}

      {reportData && !reportData.empty ? (
        <>
          <div className="grid gap-3 md:grid-cols-4">
            <Stat label="Records" value={number(reportData.record_count)} hint={`${reportData.size_human} on disk`} />
            <Stat label="Fields" value={String(reportData.field_names.length)} hint={reportData.field_names.slice(0, 3).join(", ")} />
            <Stat
              label="Duplicates / empty"
              value={`${reportData.duplicates} / ${reportData.empty_records}`}
              tone={reportData.duplicates > 0 ? "warn" : "muted"}
            />
            <Stat
              label="Tokens"
              value={reportData.token_estimate.total ? number(reportData.token_estimate.total) : "—"}
              hint={reportData.token_estimate.note}
              estimated={!tokenizerResult}
            />
          </div>

          <Tabs value={tab} onChange={setTab}>
            <Tab value="overview">Overview</Tab>
            <Tab value="preview">Preview</Tab>
            <Tab value="clean">Cleaning</Tab>
            <Tab value="split">Split</Tab>
            <Tab value="tokenizer">Tokenizer lab</Tab>
          </Tabs>

          {tab === "overview" ? (
            <div className="grid gap-3 lg:grid-cols-2">
              <Panel>
                <SectionHeader title="Structure" icon={<Database size={13} className="text-accent" />} />
                <Table>
                  <thead>
                    <tr>
                      <Th>Field</Th>
                      <Th>Type</Th>
                      <Th align="right">Missing</Th>
                    </tr>
                  </thead>
                  <tbody>
                    {reportData.fields.map((field: any) => (
                      <tr key={field.name}>
                        <Td>
                          <span className="font-mono text-2xs">{field.name}</span>
                        </Td>
                        <Td>{field.type}</Td>
                        <Td align="right">
                          {field.missing} {field.missing_percent !== null ? `(${field.missing_percent}%)` : ""}
                        </Td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
                <div className="mt-3">
                  <SectionHeader title="Length distribution" subtitle="Characters per record across the sampled records" />
                  <BarChart
                    bars={(reportData.length.histogram ?? []).map((bucket: any) => ({
                      label: `${bucket.from}`,
                      value: bucket.count,
                    }))}
                    height={90}
                  />
                </div>
              </Panel>

              <Panel>
                <SectionHeader title="Field mapping" subtitle="How raw fields become training text" />
                <div className="space-y-2">
                  {Object.entries(reportData.mapping_suggestion?.mapping ?? {}).map(([role, field]) => (
                    <div key={role} className="flex items-center gap-2">
                      <Badge tone="accent">{role}</Badge>
                      <Select
                        className="flex-1"
                        value={mapping[role] ?? String(field)}
                        onChange={(event) => setMapping({ ...mapping, [role]: event.target.value })}
                      >
                        {(reportData.field_names ?? []).map((name: string) => (
                          <option key={name} value={name}>
                            {name}
                          </option>
                        ))}
                      </Select>
                    </div>
                  ))}
                  {!Object.keys(reportData.mapping_suggestion?.mapping ?? {}).length ? (
                    <EmptyState title="No mapping was detected">
                      Set the fields manually below; the engine re-reads the dataset with your mapping.
                    </EmptyState>
                  ) : null}
                  <div className="flex flex-wrap gap-2 pt-1">
                    {["system", "user", "assistant", "text", "rejected", "label"].map((role) => (
                      <Button
                        key={role}
                        size="sm"
                        variant={mapping[role] ? "default" : "subtle"}
                        onClick={() =>
                          setMapping((current) => {
                            const next = { ...current };
                            if (next[role]) delete next[role];
                            else next[role] = reportData.field_names[0] ?? "";
                            return next;
                          })
                        }
                      >
                        {mapping[role] ? `unset ${role}` : `add ${role}`}
                      </Button>
                    ))}
                  </div>
                  <CodeBlock>
                    {JSON.stringify(
                      {
                        detected: reportData.detected_mapping,
                        active: effectiveMapping,
                        unmapped: reportData.mapping_suggestion?.unmapped ?? [],
                      },
                      null,
                      2,
                    )}
                  </CodeBlock>
                </div>
              </Panel>

              <Panel className="lg:col-span-2">
                <SectionHeader title="Quality signals" subtitle="Computed over a real sample of the dataset" />
                <div className="grid gap-3 md:grid-cols-3">
                  <KeyValue
                    columns={1}
                    items={[
                      ["Sampled records", number(reportData.sampled)],
                      ["Average length", `${reportData.length.average ?? "—"} chars`],
                      ["Max length", `${reportData.length.max ?? "—"} chars`],
                      ["p50 / p90 / p99", `${reportData.length.p50 ?? "—"} / ${reportData.length.p90 ?? "—"} / ${reportData.length.p99 ?? "—"}`],
                    ]}
                  />
                  <KeyValue
                    columns={1}
                    items={[
                      ["Exact duplicates", number(reportData.duplicates)],
                      ["Duplicate share", reportData.duplicate_percent !== null ? `${reportData.duplicate_percent}%` : "—"],
                      ["Empty records", number(reportData.empty_records)],
                      ["Media kind", reportData.media_kind ?? "text"],
                    ]}
                  />
                  <div className="rounded-md border border-line-soft bg-surface-2 p-2.5 text-2xs leading-relaxed text-ink-2">
                    Near-duplicate detection is off by default because it is expensive. Turn it on in the
                    Cleaning tab and the engine compares real shingle overlap instead of guessing.
                  </div>
                </div>
              </Panel>
            </div>
          ) : null}

          {tab === "preview" ? (
            <Panel>
              <SectionHeader
                title="Records"
                subtitle="Raw record, normalised training example and the exact text the trainer would see"
                icon={<Search size={13} className="text-accent" />}
                actions={
                  <div className="flex items-center gap-2">
                    <Button size="sm" variant="subtle" disabled={page === 0} onClick={() => setPage((value) => Math.max(0, value - 1))}>
                      Previous
                    </Button>
                    <span className="text-2xs text-ink-3">
                      {page * pageSize + 1}–{Math.min((page + 1) * pageSize, reportData.record_count)}
                    </span>
                    <Button
                      size="sm"
                      variant="subtle"
                      disabled={(page + 1) * pageSize >= reportData.record_count}
                      onClick={() => setPage((value) => value + 1)}
                    >
                      Next
                    </Button>
                  </div>
                }
              />
              {rows.error ? <ErrorPanel error={rows.error} onRetry={rows.reload} /> : null}
              {rows.loading ? <Loading lines={4} /> : null}
              <div className="space-y-2">
                {(rows.data?.rows ?? []).map((row: any) => (
                  <div key={row.index} className="rounded-md border border-line-soft bg-surface-2 p-2.5">
                    <div className="mb-1.5 flex items-center justify-between">
                      <Badge tone="muted">#{row.index}</Badge>
                      <Button size="sm" variant="ghost" onClick={() => api.registry.add("notes", { id: `${row.index}`, dataset: path, note: JSON.stringify(row.record) })}>
                        Bookmark
                      </Button>
                    </div>
                    <div className="grid gap-2 lg:grid-cols-3">
                      <div>
                        <div className="mb-0.5 text-2xs uppercase tracking-wide text-ink-3">Raw</div>
                        <CodeBlock max="max-h-40">{truncate(JSON.stringify(row.record), 600)}</CodeBlock>
                      </div>
                      <div>
                        <div className="mb-0.5 text-2xs uppercase tracking-wide text-ink-3">Normalised</div>
                        <CodeBlock max="max-h-40">{truncate(JSON.stringify(row.normalised), 600)}</CodeBlock>
                      </div>
                      <div>
                        <div className="mb-0.5 text-2xs uppercase tracking-wide text-ink-3">Training text</div>
                        <CodeBlock max="max-h-40">{truncate(row.text, 600)}</CodeBlock>
                      </div>
                    </div>
                  </div>
                ))}
                {rows.data && !rows.data.rows?.length ? <EmptyState title="No records on this page" /> : null}
              </div>
            </Panel>
          ) : null}

          {tab === "clean" ? (
            <div className="grid gap-3 lg:grid-cols-2">
              <Panel>
                <SectionHeader title="Operations" subtitle="Each change is previewed before it is written" icon={<Brush size={13} className="text-accent" />} />
                <div className="space-y-3">
                  <Toggle checked={cleanOps.drop_empty} onChange={(value) => setCleanOps({ ...cleanOps, drop_empty: value })} label="Drop empty records" hint="Removes records whose text is empty or whitespace only." />
                  <Toggle checked={cleanOps.trim_whitespace} onChange={(value) => setCleanOps({ ...cleanOps, trim_whitespace: value })} label="Normalise whitespace" hint="Collapses runs of spaces and tabs, trims the edges." />
                  <Toggle checked={cleanOps.drop_duplicates} onChange={(value) => setCleanOps({ ...cleanOps, drop_duplicates: value })} label="Drop exact duplicates" hint="Hashes the normalised record, so reordered keys still count as duplicates." />
                  <Toggle checked={cleanOps.strip_html} onChange={(value) => setCleanOps({ ...cleanOps, strip_html: value })} label="Strip HTML tags" hint="Only touches fields that actually contain markup." />
                  <div className="grid gap-3 sm:grid-cols-2">
                    <Field label="Minimum characters" help="Records shorter than this are removed. 0 disables the filter.">
                      <NumberInput value={cleanOps.min_length} onChange={(value) => setCleanOps({ ...cleanOps, min_length: Number(value) || 0 })} />
                    </Field>
                    <Field label="Maximum characters" help="Records longer than this are removed. 0 disables the filter.">
                      <NumberInput value={cleanOps.max_length} onChange={(value) => setCleanOps({ ...cleanOps, max_length: Number(value) || 0 })} />
                    </Field>
                  </div>
                  <Field label="Regex pattern" hint="Python regular expression applied to every string field, unless you scope it later.">
                    <TextInput value={cleanOps.regex_pattern} onChange={(event) => setCleanOps({ ...cleanOps, regex_pattern: event.target.value })} placeholder="\\s{2,}" />
                  </Field>
                  <Field label="Replacement">
                    <TextInput value={cleanOps.regex_replacement} onChange={(event) => setCleanOps({ ...cleanOps, regex_replacement: event.target.value })} placeholder=" " />
                  </Field>
                  <div className="flex gap-2">
                    <Button loading={busy} onClick={() => void runCleaning(true)}>
                      Preview changes
                    </Button>
                    <Button variant="primary" loading={busy} onClick={() => void runCleaning(false)}>
                      Write cleaned dataset
                    </Button>
                  </div>
                </div>
              </Panel>

              <Panel>
                <SectionHeader title="Result" subtitle={cleanResult ? (cleanResult.dryRun ? "Preview only — nothing written" : "Written to disk") : "Run a preview to see the effect"} />
                {cleanResult ? (
                  <div className="space-y-3">
                    <div className="grid gap-2 sm:grid-cols-3">
                      <Stat label="Input" value={number(cleanResult.input_records)} />
                      <Stat label="Output" value={number(cleanResult.output_records)} />
                      <Stat label="Dropped" value={number(cleanResult.dropped)} tone={cleanResult.dropped ? "warn" : "muted"} />
                    </div>
                    <CodeBlock>{JSON.stringify(cleanResult.operation_stats, null, 2)}</CodeBlock>
                    {cleanResult.preview?.length ? (
                      <div className="space-y-2">
                        <div className="text-2xs uppercase tracking-wide text-ink-3">First changes</div>
                        {cleanResult.preview.map((item: any, index: number) => (
                          <div key={index} className="grid gap-2 lg:grid-cols-2">
                            <CodeBlock max="max-h-32">{truncate(JSON.stringify(item.before), 400)}</CodeBlock>
                            <CodeBlock max="max-h-32">{item.after ? truncate(JSON.stringify(item.after), 400) : "dropped"}</CodeBlock>
                          </div>
                        ))}
                      </div>
                    ) : null}
                    {cleanResult.output ? (
                      <div className="flex items-center gap-2">
                        <PathText value={cleanResult.output} />
                        <Button size="sm" variant="ghost" onClick={() => api.shell.reveal(cleanResult.output)}>
                          Reveal
                        </Button>
                        <Button
                          size="sm"
                          variant="subtle"
                          onClick={async () => {
                            const report = await api.call<any>("datasets.inspect", { path: cleanResult.output, sample_size: 500 });
                            await api.registry.add("datasets", {
                              id: `${Date.now().toString(36)}`,
                              name: report.name,
                              path: cleanResult.output,
                              addedAt: new Date().toISOString(),
                              kind: report.kind,
                              summary: { records: report.record_count, size_bytes: report.size_bytes, fields: report.field_names, tokens: report.token_estimate?.total ?? null },
                            });
                            toast({ title: "Cleaned dataset added to the library", tone: "ok" });
                          }}
                        >
                          Add to library
                        </Button>
                      </div>
                    ) : null}
                  </div>
                ) : (
                  <EmptyState title="No cleaning run yet">
                    Preview first: the engine applies the operations in memory and shows exactly what would
                    change, including which records would be dropped.
                  </EmptyState>
                )}
              </Panel>
            </div>
          ) : null}

          {tab === "split" ? (
            <div className="grid gap-3 lg:grid-cols-2">
              <Panel>
                <SectionHeader title="Split configuration" subtitle="Deterministic: the same seed always produces the same split" icon={<Scissors size={13} className="text-accent" />} />
                <div className="space-y-3">
                  <div className="grid gap-3 sm:grid-cols-3">
                    <Field label="Train">
                      <NumberInput value={splitConfig.train} step={0.01} min={0} max={1} onChange={(value) => setSplitConfig({ ...splitConfig, train: Number(value) || 0 })} />
                    </Field>
                    <Field label="Validation">
                      <NumberInput value={splitConfig.validation} step={0.01} min={0} max={1} onChange={(value) => setSplitConfig({ ...splitConfig, validation: Number(value) || 0 })} />
                    </Field>
                    <Field label="Test">
                      <NumberInput value={splitConfig.test} step={0.01} min={0} max={1} onChange={(value) => setSplitConfig({ ...splitConfig, test: Number(value) || 0 })} />
                    </Field>
                  </div>
                  <Field label="Random seed" hint="Recorded in split.json next to the output files for reproducibility.">
                    <NumberInput value={splitConfig.seed} onChange={(value) => setSplitConfig({ ...splitConfig, seed: Number(value) || 0 })} />
                  </Field>
                  <div className="text-2xs text-ink-2">
                    Ratio total: {(splitConfig.train + splitConfig.validation + splitConfig.test).toFixed(2)} — the
                    engine uses exact record counts derived from these ratios, not roundings you cannot see.
                  </div>
                  <Button variant="primary" loading={busy} onClick={() => void runSplit()}>
                    Write split files
                  </Button>
                </div>
              </Panel>
              <Panel>
                <SectionHeader title="Split result" />
                {splitResult ? (
                  <div className="space-y-3">
                    <KeyValue
                      columns={1}
                      items={Object.entries(splitResult.counts ?? {}).map(([name, count]) => [name, number(Number(count))])}
                    />
                    <CodeBlock>{JSON.stringify(splitResult.files, null, 2)}</CodeBlock>
                  </div>
                ) : (
                  <EmptyState title="No split written yet">
                    Splits are written as real files under the workspace exports folder, together with a
                    split.json describing counts, seed and ratios.
                  </EmptyState>
                )}
              </Panel>
            </div>
          ) : null}

          {tab === "tokenizer" ? (
            <div className="grid gap-3 lg:grid-cols-2">
              <Panel>
                <SectionHeader title="Token statistics" subtitle={tokenizerStats.data?.exact ? "Measured with a real tokenizer" : "Estimated until a tokenizer is trained"} />
                {tokenizerStats.error ? <ErrorPanel error={tokenizerStats.error} onRetry={tokenizerStats.reload} /> : null}
                {tokenizerStats.data ? (
                  <>
                    <KeyValue
                      columns={1}
                      items={Object.entries(tokenizerStats.data.stats ?? {}).map(([key, value]) => [
                        key.replace(/_/g, " "),
                        typeof value === "number" ? number(value, 2) : String(value),
                      ])}
                    />
                    <div className="mt-2 flex items-center gap-2">
                      <Badge tone={tokenizerStats.data.exact ? "ok" : "warn"}>
                        {tokenizerStats.data.exact ? "exact" : "estimated (characters / 4)"}
                      </Badge>
                      {tokenizerStats.data.note ? <span className="text-2xs text-ink-2">{tokenizerStats.data.note}</span> : null}
                    </div>
                  </>
                ) : (
                  <Loading lines={4} />
                )}
              </Panel>
              <Panel>
                <SectionHeader title="Train a tokenizer" subtitle="Byte-level BPE learned from this dataset, in pure Python" />
                <div className="space-y-3">
                  <Field label="Vocabulary size" hint="256 byte symbols plus this many merges plus 4 chat special tokens.">
                    <NumberInput value={tokenizerConfig.vocab_size} step={64} min={256} onChange={(value) => setTokenizerConfig({ ...tokenizerConfig, vocab_size: Number(value) || 256 })} />
                  </Field>
                  <Field label="Chat template" hint="How multi-turn records are flattened for training.">
                    <Select value={tokenizerConfig.template} onChange={(event) => setTokenizerConfig({ ...tokenizerConfig, template: event.target.value })}>
                      <option value="chatml">ChatML (&lt;|role|&gt;)</option>
                      <option value="plain">role: content</option>
                      <option value="markdown">### Role:</option>
                    </Select>
                  </Field>
                  <Button variant="primary" loading={busy} onClick={() => void runTokenizer()}>
                    Start tokenizer training job
                  </Button>
                  {tokenizerResult ? <CodeBlock>{JSON.stringify(tokenizerResult, null, 2)}</CodeBlock> : null}
                  <Callout tone="info" title="Where the result lands">
                    The job writes tokenizer.json into its output folder. Pass that file to a training run to
                    reuse exactly the same vocabulary, which is what makes continuation training consistent.
                  </Callout>
                </div>
              </Panel>
            </div>
          ) : null}

          <Panel>
            <SectionHeader title="Exports" subtitle="Every export runs through the engine, not a browser download" />
            <div className="flex flex-wrap gap-2">
              {["jsonl", "json", "csv", "tsv", "txt", "parquet"].map((format) => (
                <Button key={format} loading={busy} onClick={() => void exportDataset(format)}>
                  {format}
                </Button>
              ))}
            </div>
          </Panel>
        </>
      ) : null}
    </div>
  );
}
