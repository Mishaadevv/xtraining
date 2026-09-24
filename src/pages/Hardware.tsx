import { useMemo, useState } from "react";
import { Cpu, Gauge, HardDrive, MemoryStick, Thermometer } from "lucide-react";
import { LineChart, UsageBar } from "../components/charts";
import {
  Badge,
  Button,
  Field,
  KeyValue,
  NumberInput,
  Panel,
  SectionHeader,
  Select,
  Stat,
  Table,
  Td,
  Th,
} from "../components/ui";
import { bytes, compact, number, percent, riskLabel } from "../lib/format";
import { useApp } from "../state/app";
import { ErrorPanel, Loading, useInterval } from "./common";

export function HardwarePage() {
  const { hardware, live, refreshHardware, registry } = useApp();
  const [history, setHistory] = useState<Array<{ x: number; ram: number; cpu: number; vram: number }>>([]);
  const [estimatorConfig, setEstimatorConfig] = useState({
    modelPath: "",
    method: "lora",
    precision: "fp32",
    quantization: "none",
    batch_size: 1,
    sequence_length: 128,
    gradient_accumulation: 1,
    lora_rank: 8,
    epochs: 1,
  });
  const [estimate, setEstimate] = useState<any>(null);
  const [estimateError, setEstimateError] = useState<any>(null);
  const [busy, setBusy] = useState(false);

  useInterval(() => {
    if (!live) return;
    setHistory((current) =>
      [
        ...current,
        {
          x: current.length,
          ram: live.ram?.percent ?? 0,
          cpu: live.cpu?.percent ?? 0,
          vram: live.gpus?.[0]?.memory_used_mb
            ? ((live.gpus[0].memory_used_mb * 1024 * 1024) / (live.gpus[0].memory_total || 1)) * 100
            : 0,
        },
      ].slice(-120),
    );
  }, 5000);

  const gpu = live?.gpus?.[0] ?? hardware?.gpus?.[0] ?? null;

  const runEstimate = async () => {
    setBusy(true);
    const { data, error } = await (await import("../lib/api")).api.callSafe<any>("training.plan", {
      method: estimatorConfig.method,
      precision: estimatorConfig.precision,
      quantization: estimatorConfig.quantization,
      batch_size: estimatorConfig.batch_size,
      sequence_length: estimatorConfig.sequence_length,
      gradient_accumulation: estimatorConfig.gradient_accumulation,
      lora_rank: estimatorConfig.lora_rank,
      epochs: estimatorConfig.epochs,
      base_model: estimatorConfig.modelPath || null,
      dataset_paths: [],
    });
    setBusy(false);
    if (error) {
      setEstimateError(error);
      setEstimate(null);
    } else {
      setEstimateError(null);
      setEstimate(data?.plan ?? null);
    }
  };

  const diskRows = useMemo(() => hardware?.disks ?? [], [hardware]);

  if (!hardware) {
    return (
      <div className="space-y-3">
        <Loading label="Detecting CPU, GPUs, memory and Python runtimes…" lines={6} />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Hardware Center</h1>
          <p className="mt-0.5 text-xs text-ink-2">
            Detected {hardware.os.name} {hardware.os.release} · {hardware.os.machine} · python {hardware.os.python}
          </p>
        </div>
        <Button loading={false} onClick={() => void refreshHardware(false)} icon={<Gauge size={13} />}>
          Re-detect everything
        </Button>
      </div>

      <div className="grid gap-3 lg:grid-cols-4">
        <Stat label="CPU load" value={percent(live?.cpu?.percent ?? null)} hint={`${hardware.cpu.logical_cores ?? "?"} threads`} />
        <Stat label="RAM" value={bytes(live?.ram?.used ?? null, 1)} hint={`of ${bytes(hardware.memory.total, 0)}`} />
        <Stat
          label={gpu ? "VRAM" : "GPU"}
          value={gpu ? bytes((gpu.memory_used_mb ?? 0) * 1024 * 1024, 1) : "none"}
          hint={gpu ? `of ${bytes(gpu.memory_total, 0)}` : "CPU only machine"}
        />
        <Stat
          label="GPU temperature"
          value={gpu?.temperature_c !== null && gpu?.temperature_c !== undefined ? `${gpu.temperature_c}°C` : "n/a"}
          hint={gpu?.power_draw_w ? `${gpu.power_draw_w.toFixed(0)} W draw` : undefined}
        />
      </div>

      <div className="grid gap-3 lg:grid-cols-3">
        <Panel className="lg:col-span-2">
          <SectionHeader title="Utilisation over this session" subtitle={`${history.length} live samples, one every 5 seconds`} icon={<Gauge size={13} className="text-accent" />} />
          <LineChart
            height={170}
            series={[
              { name: "CPU %", color: "rgb(var(--accent))", points: history.map((point) => ({ x: point.x, y: point.cpu })) },
              { name: "RAM %", color: "rgb(var(--ok))", points: history.map((point) => ({ x: point.x, y: point.ram })) },
              ...(gpu
                ? [{ name: "VRAM %", color: "rgb(var(--warn))", points: history.map((point) => ({ x: point.x, y: point.vram })) }]
                : []),
            ]}
            formatY={(value) => `${value.toFixed(0)}%`}
            formatX={(value) => `${value.toFixed(0)} samples`}
          />
          <div className="mt-3 space-y-2">
            <UsageBar label="System memory" value={live?.ram?.used ?? null} max={live?.ram?.total ?? null} format={(value) => bytes(value, 1)} />
            {gpu ? (
              <UsageBar
                label={`${gpu.name} VRAM`}
                value={gpu.memory_used_mb ? gpu.memory_used_mb * 1024 * 1024 : null}
                max={gpu.memory_total}
                format={(value) => bytes(value, 1)}
              />
            ) : null}
          </div>
        </Panel>

        <Panel>
          <SectionHeader title="Processors" subtitle="Reported by the operating system" icon={<Cpu size={13} className="text-accent" />} />
          <KeyValue
            columns={1}
            items={[
              ["Model", hardware.cpu.model ?? "unknown"],
              ["Physical cores", `${hardware.cpu.physical_cores ?? "?"}`],
              ["Logical cores", `${hardware.cpu.logical_cores ?? "?"}`],
              ["Architecture", hardware.cpu.architecture],
              ["AVX2", hardware.cpu.avx2 ? "yes" : "no/unknown"],
              ["AVX-512", hardware.cpu.avx512 ? "yes" : "no/unknown"],
            ]}
          />
          <div className="mt-3">
            <div className="mb-1 text-2xs uppercase tracking-wide text-ink-3">GPUs</div>
            {hardware.gpus.length ? (
              hardware.gpus.map((device) => (
                <div key={device.index} className="mb-2 rounded-md border border-line-soft bg-surface-2 p-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs">{device.name}</span>
                    <Badge tone="info">#{device.index}</Badge>
                  </div>
                  <div className="mt-1 grid grid-cols-2 gap-x-3 gap-y-1 text-2xs text-ink-2">
                    <span>VRAM: {bytes(device.memory_total, 0)}</span>
                    <span>CC: {device.compute_capability ?? "?"}</span>
                    <span>Driver: {device.driver_version ?? "?"}</span>
                    <span>Source: {device.source ?? "torch"}</span>
                  </div>
                </div>
              ))
            ) : (
              <div className="rounded-md border border-line-soft bg-surface-2 p-2 text-2xs text-ink-2">
                No CUDA device was detected. CUDA-only features are hidden rather than offered and then failing.
              </div>
            )}
          </div>
        </Panel>
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <Panel>
          <SectionHeader title="Storage" subtitle="Where models, checkpoints and datasets live" icon={<HardDrive size={13} className="text-accent" />} />
          <Table>
            <thead>
              <tr>
                <Th>Volume</Th>
                <Th align="right">Total</Th>
                <Th align="right">Used</Th>
                <Th align="right">Free</Th>
                <Th align="right">Used %</Th>
              </tr>
            </thead>
            <tbody>
              {diskRows.map((disk) => (
                <tr key={disk.path}>
                  <Td>
                    <span className="font-mono text-2xs" title={disk.path}>
                      {disk.path}
                    </span>
                  </Td>
                  <Td align="right">{bytes(disk.total, 0)}</Td>
                  <Td align="right">{bytes(disk.used, 0)}</Td>
                  <Td align="right">{bytes(disk.free, 0)}</Td>
                  <Td align="right" className={disk.percent && disk.percent > 90 ? "text-danger" : undefined}>
                    {percent(disk.percent)}
                  </Td>
                </tr>
              ))}
            </tbody>
          </Table>
        </Panel>

        <Panel>
          <SectionHeader title="Runtime" subtitle="PyTorch build and the libraries the engine found" icon={<MemoryStick size={13} className="text-accent" />} />
          <KeyValue
            columns={1}
            items={[
              ["PyTorch", hardware.torch.installed ? `${hardware.torch.version}` : "not installed"],
              ["CUDA build", hardware.torch.cuda_build ?? "none"],
              ["CUDA available", hardware.torch.cuda_available ? "yes" : "no"],
              ["Devices via torch", `${hardware.torch.device_count}`],
              ["Apple MPS", hardware.torch.mps_available ? "yes" : "no"],
            ]}
          />
          <div className="mt-3">
            <div className="mb-1.5 text-2xs uppercase tracking-wide text-ink-3">Installed libraries</div>
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(hardware.capabilities.libraries).map(([name, info]) => (
                <Badge key={name} tone={info.installed ? "ok" : "muted"} title={info.version ?? "not installed"}>
                  {name} {info.version ? `${info.version}` : ""}
                </Badge>
              ))}
            </div>
          </div>
          {hardware.torch.notes.length ? (
            <div className="mt-3 space-y-1">
              {hardware.torch.notes.map((note) => (
                <div key={note} className="text-2xs leading-relaxed text-ink-2">
                  · {note}
                </div>
              ))}
            </div>
          ) : null}
        </Panel>
      </div>

      <Panel>
        <SectionHeader
          title="Compatibility and memory check"
          subtitle="Estimate a run before starting it. Every value is labelled estimated."
          icon={<Thermometer size={13} className="text-accent" />}
        />
        <div className="grid gap-3 md:grid-cols-3 xl:grid-cols-6">
          <Field label="Model" className="md:col-span-2">
            <Select value={estimatorConfig.modelPath} onChange={(event) => setEstimatorConfig({ ...estimatorConfig, modelPath: event.target.value })}>
              <option value="">Generic transformer (from parameters)</option>
              {(registry?.models ?? []).map((model) => (
                <option key={model.id} value={model.path}>
                  {model.name}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Method">
            <Select value={estimatorConfig.method} onChange={(event) => setEstimatorConfig({ ...estimatorConfig, method: event.target.value })}>
              {["lora", "qlora", "full_finetune", "sft", "continued_pretraining"].map((method) => (
                <option key={method} value={method}>
                  {method.replace(/_/g, " ")}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Precision">
            <Select value={estimatorConfig.precision} onChange={(event) => setEstimatorConfig({ ...estimatorConfig, precision: event.target.value })}>
              {["fp32", "fp16", "bf16"].map((precision) => (
                <option key={precision} value={precision}>
                  {precision}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Quantisation">
            <Select value={estimatorConfig.quantization} onChange={(event) => setEstimatorConfig({ ...estimatorConfig, quantization: event.target.value })}>
              {["none", "int8", "int4"].map((quantization) => (
                <option key={quantization} value={quantization}>
                  {quantization}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Batch size">
            <NumberInput value={estimatorConfig.batch_size} min={1} onChange={(value) => setEstimatorConfig({ ...estimatorConfig, batch_size: Number(value) || 1 })} />
          </Field>
          <Field label="Sequence length">
            <NumberInput value={estimatorConfig.sequence_length} min={16} step={16} onChange={(value) => setEstimatorConfig({ ...estimatorConfig, sequence_length: Number(value) || 16 })} />
          </Field>
          <Field label="Gradient accumulation">
            <NumberInput value={estimatorConfig.gradient_accumulation} min={1} onChange={(value) => setEstimatorConfig({ ...estimatorConfig, gradient_accumulation: Number(value) || 1 })} />
          </Field>
          <Field label="LoRA rank">
            <NumberInput value={estimatorConfig.lora_rank} min={1} onChange={(value) => setEstimatorConfig({ ...estimatorConfig, lora_rank: Number(value) || 1 })} />
          </Field>
          <div className="flex items-end">
            <Button variant="primary" loading={busy} onClick={() => void runEstimate()}>
              Estimate
            </Button>
          </div>
        </div>

        {estimateError ? (
          <div className="mt-3">
            <ErrorPanel error={estimateError} onRetry={runEstimate} />
          </div>
        ) : null}

        {estimate ? (
          <div className="mt-4 space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone="muted">labelled: estimated</Badge>
              <Badge tone={riskLabel(estimate.risk).tone as any}>{riskLabel(estimate.risk).label}</Badge>
              <span className="text-2xs text-ink-3">
                parameters {number(estimate.parameters.total)} ({estimate.parameters.source})
              </span>
            </div>
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              <Stat label="Weights" value={bytes(estimate.memory.weights, 1)} estimated />
              <Stat label="Gradients" value={bytes(estimate.memory.gradients, 1)} estimated />
              <Stat label="Optimizer" value={bytes(estimate.memory.optimizer, 1)} estimated />
              <Stat label="Activations" value={bytes(estimate.memory.activations, 1)} estimated />
              <Stat label="Total VRAM" value={bytes(estimate.memory.vram_estimate, 1)} estimated tone={estimate.risk === "will_not_fit" ? "danger" : "muted"} />
              <Stat label="System RAM" value={bytes(estimate.memory.ram_estimate, 1)} estimated />
              <Stat label="Checkpoint size" value={bytes(estimate.disk.checkpoint_size_estimate, 1)} estimated />
              <Stat label="Workspace" value={bytes(estimate.disk.workspace_needed, 1)} estimated />
            </div>
            <div className="grid gap-3 md:grid-cols-2">
              <KeyValue
                columns={1}
                items={[
                  ["Trainable parameters", number(estimate.parameters.trainable)],
                  ["Frozen parameters", number(estimate.parameters.frozen)],
                  ["Steps", number(estimate.steps.total)],
                  ["Effective batch", number(estimate.steps.effective_batch_size)],
                  ["Tokens per step", number(estimate.steps.tokens_per_step)],
                ]}
              />
              <div className="rounded-md border border-line-soft bg-surface-2 p-2.5">
                <div className="mb-1 text-2xs uppercase tracking-wide text-ink-3">How these numbers are produced</div>
                <div className="space-y-1">
                  {Object.entries(estimate.formulas).map(([key, value]) => (
                    <div key={key} className="text-2xs text-ink-2">
                      <span className="text-ink-3">{key}:</span> {typeof value === "string" ? value : JSON.stringify(value)}
                    </div>
                  ))}
                </div>
              </div>
            </div>
            {estimate.warnings.length ? (
              <div className="space-y-1">
                {estimate.warnings.map((warning: string) => (
                  <div key={warning} className="rounded-md border border-warn/25 bg-warn/8 px-2 py-1.5 text-2xs text-ink-1">
                    {warning}
                  </div>
                ))}
              </div>
            ) : null}
            {estimate.notes.length ? (
              <div className="space-y-1">
                {estimate.notes.map((note: string) => (
                  <div key={note} className="rounded-md border border-line-soft bg-surface-2 px-2 py-1.5 text-2xs text-ink-2">
                    {note}
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}
      </Panel>

      <Panel>
        <SectionHeader title="Notes from detection" subtitle="Anything the engine could not read is stated here" />
        <div className="space-y-1">
          {(hardware.notes ?? []).length ? (
            hardware.notes.map((note) => (
              <div key={note} className="text-2xs leading-relaxed text-ink-2">
                · {note}
              </div>
            ))
          ) : (
            <div className="text-2xs text-ink-3">No limitations were reported.</div>
          )}
        </div>
        <div className="mt-3 text-2xs text-ink-3">
          Internal memory counters (engine process RSS): {compact(live?.process?.rss ?? null)} bytes
        </div>
      </Panel>
    </div>
  );
}
