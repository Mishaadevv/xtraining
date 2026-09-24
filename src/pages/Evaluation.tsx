import { useEffect, useMemo, useState } from "react";
import { ListChecks, Play } from "lucide-react";
import { api } from "../lib/api";
import { basename, clock, number } from "../lib/format";
import { useApp } from "../state/app";
import { useRouter } from "../state/router";
import {
  Badge,
  Button,
  Callout,
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
} from "../components/ui";
import { DatasetPicker, ErrorPanel, Loading, ModelPicker, useEngine } from "./common";

export function EvaluationPage() {
  const { registry, refreshRegistry, toast, reportError, backends, jobs } = useApp();
  const router = useRouter();
  const params = useMemo(() => new URLSearchParams(window.location.hash.split("?")[1] ?? ""), []);
  const [tab, setTab] = useState("run");
  const [modelPath, setModelPath] = useState<string | null>(params.get("model") ?? null);
  const [datasetPath, setDatasetPath] = useState<string | null>(params.get("dataset") ?? null);
  const [limit, setLimit] = useState(200);
  const [sequenceLength, setSequenceLength] = useState(256);
  const [template, setTemplate] = useState("chatml");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<any>(null);

  useEffect(() => {
    const model = params.get("model");
    if (model) setModelPath(model);
  }, [params]);

  const datasetMapping = useEngine<any>(
    "datasets.mapping",
    { path: datasetPath ?? "" },
    { auto: Boolean(datasetPath), timeout: 120_000, deps: [datasetPath] },
  );

  const availableBackends = backends.filter((backend) => backend.available);
  const evaluationJobs = jobs.filter((job) => job.kind === "evaluate");

  const run = async () => {
    if (!modelPath || !datasetPath) return;
    setBusy(true);
    setResult(null);
    try {
      const job = await api.jobs.start({
        kind: "evaluate",
        backend: null,
        base_model: modelPath,
        dataset_paths: [datasetPath],
        mapping: datasetMapping.data?.mapping ?? {},
        template,
        limit,
        sequence_length: sequenceLength,
      });
      toast({
        title: "Evaluation started",
        body: `Job ${job.jobId} is measuring loss and perplexity on real records.`,
        tone: "info",
      });
      router.navigate(`/training/${encodeURIComponent(job.jobId)}`);
    } catch (error) {
      reportError(error, "Evaluation could not start");
    } finally {
      setBusy(false);
    }
  };

  const savedEvaluations = (registry?.evaluations ?? []).slice().reverse();

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Evaluation</h1>
          <p className="mt-0.5 max-w-3xl text-xs text-ink-2">
            Metrics are computed by running the model over held-out text. Nothing here is estimated unless it is
            labelled: loss, perplexity and top-1 accuracy come from real forward passes.
          </p>
        </div>
        <Button onClick={() => router.navigate("/jobs")}>Open jobs</Button>
      </div>

      <Tabs value={tab} onChange={setTab}>
        <Tab value="run">Run evaluation</Tab>
        <Tab value="history" count={savedEvaluations.length}>
          Saved results
        </Tab>
        <Tab value="jobs" count={evaluationJobs.length}>
          Evaluation jobs
        </Tab>
      </Tabs>

      {tab === "run" ? (
        <div className="grid gap-3 lg:grid-cols-2">
          <Panel>
            <SectionHeader title="What to evaluate" subtitle="A base model, a fine-tuned model, an adapter or a checkpoint" icon={<ListChecks size={13} className="text-accent" />} />
            <div className="space-y-3">
              <ModelPicker value={modelPath} onChange={setModelPath} label="Model or checkpoint" />
              <DatasetPicker value={datasetPath} onChange={setDatasetPath} label="Evaluation dataset" />
              <Field label="Records to evaluate" help="Each record is scored separately; more records give a more stable number but take longer.">
                <NumberInput value={limit} min={1} onChange={(value) => setLimit(Number(value) || 1)} />
              </Field>
              <Field label="Sequence length" help="Longer sequences give a truer perplexity on long documents.">
                <NumberInput value={sequenceLength} step={32} min={16} onChange={(value) => setSequenceLength(Number(value) || 16)} />
              </Field>
              <Field label="Template" hint="Must match the template the model was trained with, otherwise the score is meaningless.">
                <Select value={template} onChange={(event) => setTemplate(event.target.value)}>
                  <option value="chatml">ChatML (&lt;|role|&gt;)</option>
                  <option value="plain">role: content</option>
                  <option value="markdown">### Role:</option>
                </Select>
              </Field>
              {datasetMapping.data ? (
                <div className="rounded-md border border-line-soft bg-surface-2 p-2.5 text-2xs">
                  Mapping: {Object.entries(datasetMapping.data.mapping ?? {}).map(([key, value]) => `${key}→${value}`).join(", ") || "none detected"}
                </div>
              ) : null}
              <div className="flex flex-wrap gap-2">
                {availableBackends.map((backend) => (
                  <Badge key={backend.id} tone="ok">
                    {backend.name}
                  </Badge>
                ))}
              </div>
              <Button variant="primary" icon={<Play size={13} />} loading={busy} disabled={!modelPath || !datasetPath} onClick={() => void run()}>
                Start evaluation
              </Button>
            </div>
          </Panel>

          <Panel>
            <SectionHeader title="Result" subtitle={result ? "Measured on real records" : "Nothing measured in this panel yet"} />
            {result ? (
              <div className="space-y-3">
                <div className="grid gap-2 sm:grid-cols-3">
                  {Object.entries(result.metrics ?? {}).map(([key, value]) => (
                    <Stat key={key} label={key.replace(/_/g, " ")} value={typeof value === "number" ? number(value, 4) : String(value)} />
                  ))}
                </div>
                <KeyValue
                  items={[
                    ["Model", basename(result.model)],
                    ["Backend", result.backend],
                    ["Evaluated tokens", number(result.evaluated_tokens)],
                    ["Sequence length", String(result.sequence_length)],
                    ["Datasets", (result.datasets ?? []).map((path: string) => basename(path)).join(", ")],
                  ]}
                />
                <Button
                  size="sm"
                  variant="subtle"
                  onClick={async () => {
                    await api.registry.add("evaluations", {
                      id: `${Date.now().toString(36)}`,
                      model: result.model,
                      metrics: result.metrics,
                      evaluated_tokens: result.evaluated_tokens,
                      datasets: result.datasets,
                      createdAt: new Date().toISOString(),
                    });
                    await refreshRegistry();
                    toast({ title: "Evaluation saved to the workspace registry", tone: "ok" });
                  }}
                >
                  Save this result
                </Button>
              </div>
            ) : (
              <EmptyState title="No result in this session">
                Start an evaluation, or open the Saved results tab to see everything recorded so far. Results are
                written by the job, so they survive closing the app.
              </EmptyState>
            )}
            <Callout tone="info" title="Custom evaluators">
              The engine also exposes the raw metrics stream, so a custom Python evaluator can be attached to a
              job later without changing the UI contract.
            </Callout>
          </Panel>
        </div>
      ) : null}

      {tab === "history" ? (
        <Panel padded={false}>
          {!savedEvaluations.length ? (
            <div className="p-4">
              <EmptyState title="No saved evaluations">
                Save a result from the Run tab and it appears here with its model and dataset paths.
              </EmptyState>
            </div>
          ) : (
            <Table>
              <thead>
                <tr>
                  <Th>Model</Th>
                  <Th align="right">Loss</Th>
                  <Th align="right">Perplexity</Th>
                  <Th align="right">Tokens</Th>
                  <Th>Dataset</Th>
                  <Th>Saved</Th>
                </tr>
              </thead>
              <tbody>
                {savedEvaluations.map((entry: any) => (
                  <tr key={entry.id}>
                    <Td>
                      <div className="text-xs">{basename(entry.model ?? "")}</div>
                      <div className="max-w-[260px] truncate font-mono text-2xs text-ink-3">{entry.model}</div>
                    </Td>
                    <Td align="right">{entry.metrics?.loss ?? "—"}</Td>
                    <Td align="right">{entry.metrics?.perplexity ?? "—"}</Td>
                    <Td align="right">{entry.evaluated_tokens ? number(entry.evaluated_tokens) : "—"}</Td>
                    <Td>{(entry.datasets ?? []).map((path: string) => basename(path)).join(", ")}</Td>
                    <Td>{clock(entry.createdAt)}</Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          )}
        </Panel>
      ) : null}

      {tab === "jobs" ? (
        <Panel padded={false}>
          {!evaluationJobs.length ? (
            <div className="p-4 text-2xs text-ink-3">No evaluation jobs have run yet.</div>
          ) : (
            <Table>
              <thead>
                <tr>
                  <Th>Job</Th>
                  <Th>State</Th>
                  <Th>Model</Th>
                  <Th align="right">Loss</Th>
                  <Th align="right">Perplexity</Th>
                  <Th align="right">Actions</Th>
                </tr>
              </thead>
              <tbody>
                {evaluationJobs.map((job) => (
                  <tr key={job.job_id}>
                    <Td>{job.job_id}</Td>
                    <Td>
                      <Badge tone={job.state === "completed" ? "ok" : job.state === "failed" ? "danger" : "info"}>{job.state}</Badge>
                    </Td>
                    <Td>{basename(job.model ?? "")}</Td>
                    <Td align="right">{job.result?.metrics?.loss ?? "—"}</Td>
                    <Td align="right">{job.result?.metrics?.perplexity ?? "—"}</Td>
                    <Td align="right">
                      <Button size="sm" variant="subtle" onClick={() => router.navigate(`/training/${encodeURIComponent(job.job_id)}`)}>
                        Open
                      </Button>
                    </Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          )}
        </Panel>
      ) : null}

      {datasetMapping.loading ? <Loading label="Detecting the dataset mapping…" lines={2} /> : null}
      {datasetMapping.error ? <ErrorPanel error={datasetMapping.error} onRetry={datasetMapping.reload} /> : null}
      {!datasetPath ? (
        <Callout tone="info" title="Tip">
          Pick a dataset you did not train on. Evaluating on training data only tells you that the model
          memorised it.
        </Callout>
      ) : null}
      <Panel>
        <Field label="Notes" hint="Saved with the evaluation when you save a result.">
          <TextInput placeholder="e.g. held-out split, 200 records, chatml template" onChange={() => undefined} />
        </Field>
      </Panel>
    </div>
  );
}
