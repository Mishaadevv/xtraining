import { useState } from "react";
import {
  Activity,
  CheckCircle2,
  Cpu,
  HardDrive,
  MemoryStick,
  Microchip,
  RefreshCw,
  Thermometer,
  XCircle,
  Zap,
} from "lucide-react";

import { PageBody, PageHeader } from "@/components/layout/Shell";
import { MeterBar } from "@/components/charts/Charts";
import { bridge } from "@/lib/bridge";
import {
  Badge,
  Button,
  Dot,
  EmptyState,
  KeyValue,
  Note,
  Panel,
  PanelHeader,
  Stat,
} from "@/components/ui/primitives";
import { formatBytes } from "@/lib/utils";
import { useStore } from "@/state/store";
import { appStore, navigate, refreshEnv, refreshHardware } from "@/state/appStore";

export function HardwarePage() {
  const { env, busy } = useStore(appStore);
  const hardware = env.hardware;
  const [sampling, setSampling] = useState(false);
  const [liveSample, setLiveSample] = useState<Record<string, any> | null>(null);

  const cudaReady = Boolean(hardware?.cuda_ready);
  const gpuAvailable = Boolean(hardware?.gpu?.available);

  const takeSample = async () => {
    setSampling(true);
    const result = await bridge.hardware.sample();
    setLiveSample((result.sample as Record<string, any>) ?? null);
    setSampling(false);
  };

  return (
    <>
      <PageHeader
        icon={<Cpu className="h-4 w-4" />}
        title="Hardware"
        subtitle="What this machine can actually do, read from nvidia-smi and PyTorch"
        actions={
          <>
            <Button
              size="sm"
              variant="quiet"
              icon={<Zap className="h-3.5 w-3.5" />}
              loading={sampling}
              onClick={() => void takeSample()}
            >
              Sample GPU
            </Button>
            <Button
              size="sm"
              variant="quiet"
              icon={<RefreshCw className="h-3.5 w-3.5" />}
              loading={env.loading || busy.refreshEnv}
              onClick={() => {
                void refreshHardware();
                void refreshEnv(true);
              }}
            >
              Re-detect
            </Button>
          </>
        }
      />

      <PageBody wide>
        <Panel className="mb-4">
          <div className="flex items-start gap-3">
            <span
              className="mt-[2px] shrink-0"
              style={{ color: cudaReady ? "var(--green)" : "var(--amber)" }}
            >
              {cudaReady ? <CheckCircle2 className="h-5 w-5" /> : <XCircle className="h-5 w-5" />}
            </span>
            <div className="min-w-0 flex-1">
              <h2 className="text-[14px] font-semibold">
                {cudaReady
                  ? "CUDA is ready — training will use the GPU"
                  : "GPU acceleration is not available — training will use the CPU"}
              </h2>
              {hardware?.cuda_blockers?.length ? (
                <ul className="mt-2 space-y-1">
                  {hardware.cuda_blockers.map((blocker, index) => (
                    <li key={index} className="flex items-start gap-2 text-[12px] leading-[18px] text-[var(--text-2)]">
                      <span className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-[var(--text-3)]" />
                      {blocker}
                    </li>
                  ))}
                </ul>
              ) : null}
              <div className="mt-3 flex flex-wrap gap-2">
                {!env.dependencies?.training_ready ? (
                  <Button size="sm" variant="primary" onClick={() => navigate("settings")}>
                    Install the ML runtime
                  </Button>
                ) : null}
                {gpuAvailable && !cudaReady ? (
                  <Button size="sm" variant="quiet" onClick={() => navigate("settings")}>
                    Fix the PyTorch build
                  </Button>
                ) : null}
              </div>
            </div>
            <Badge tone={cudaReady ? "good" : "warn"}>
              <Dot tone={cudaReady ? "good" : "warn"} />
              {hardware?.training_device?.toUpperCase() ?? "unknown"}
            </Badge>
          </div>
        </Panel>

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Panel>
            <PanelHeader
              icon={<Microchip className="h-4 w-4" />}
              title="NVIDIA GPUs"
              description="Read directly from nvidia-smi — available even without PyTorch."
            />
            {gpuAvailable && hardware?.gpu.gpus.length ? (
              <div className="space-y-3">
                {hardware.gpu.gpus.map((gpu) => (
                  <div key={gpu.index} className="rounded-[11px] border border-[var(--border-soft)] p-3">
                    <div className="mb-2 flex items-center justify-between gap-3">
                      <span className="truncate text-[13px] font-semibold">{gpu.name}</span>
                      <Badge tone="accent">{gpu.compute_capability ?? "sm?"}</Badge>
                    </div>
                    <MeterBar
                      value={gpu.memory_used_mb ?? 0}
                      max={gpu.memory_total_mb ?? 1}
                      label="VRAM in use"
                      display={`${formatBytes((gpu.memory_used_mb ?? 0) * 1024 ** 2)} / ${formatBytes((gpu.memory_total_mb ?? 0) * 1024 ** 2)}`}
                    />
                    <div className="mt-3 grid grid-cols-2 gap-x-6">
                      <KeyValue label="Utilisation" value={`${gpu.utilization_gpu ?? "—"}%`} />
                      <KeyValue label="Temperature" value={gpu.temperature_c != null ? `${gpu.temperature_c} °C` : "—"} />
                      <KeyValue label="Driver" value={gpu.driver_version ?? "—"} />
                      <KeyValue label="Free VRAM" value={formatBytes((gpu.memory_free_mb ?? 0) * 1024 ** 2)} />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <Note tone="warn" title="No NVIDIA GPU found">
                {hardware?.gpu.reason ?? "nvidia-smi was not found."}
              </Note>
            )}
          </Panel>

          <Panel>
            <PanelHeader
              icon={<Zap className="h-4 w-4" />}
              title="PyTorch and CUDA"
              description="What the installed PyTorch build can actually use."
            />
            {hardware?.cuda.torch_installed ? (
              <div>
                <div className="space-y-0">
                  <KeyValue label="PyTorch version" value={hardware.cuda.torch_version ?? "—"} />
                  <KeyValue
                    label="Built against CUDA"
                    value={hardware.cuda.cuda_build_version ?? "CPU-only build"}
                    tone={hardware.cuda.cuda_build_version ? undefined : "warn"}
                  />
                  <KeyValue label="cuDNN" value={hardware.cuda.cudnn_version ?? "—"} />
                  <KeyValue
                    label="CUDA devices"
                    value={String(hardware.cuda.device_count ?? 0)}
                    tone={hardware.cuda.available ? "good" : "warn"}
                  />
                  <KeyValue
                    label="bfloat16 support"
                    value={hardware.cuda.bf16_supported ? "yes" : "no"}
                    tone={hardware.cuda.bf16_supported ? "good" : undefined}
                  />
                  <KeyValue label="Local CUDA toolkit" value={hardware.cuda_toolkit ?? "not installed (not required)"} />
                </div>
                {hardware.cuda.devices.length ? (
                  <div className="mt-3 space-y-2 border-t border-[var(--border-soft)] pt-3">
                    {hardware.cuda.devices.map((device) => (
                      <div key={device.index} className="flex items-center justify-between gap-3">
                        <span className="truncate text-[12px]">{device.name ?? `cuda:${device.index}`}</span>
                        <span className="zq-mono shrink-0 text-[11.5px] text-[var(--text-2)]">
                          {device.error ? device.error : `${formatBytes((device.total_memory_mb ?? 0) * 1024 ** 2)} · sm_${device.compute_capability?.replace(".", "")}`}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : null}
                {hardware.cuda.reason ? (
                  <div className="mt-3">
                    <Note tone={hardware.cuda.available ? "good" : "warn"} title="Status">
                      {hardware.cuda.reason}
                    </Note>
                  </div>
                ) : null}
              </div>
            ) : (
              <Note tone="warn" title="PyTorch is not installed">
                {hardware?.cuda.reason ?? "Install the ML runtime from Settings → Environment."}
              </Note>
            )}
          </Panel>

          <Panel>
            <PanelHeader icon={<Cpu className="h-4 w-4" />} title="Processor" />
            <Stat label="Model" value={hardware?.cpu.model ?? env.system?.cpu.model ?? "—"} mono={false} />
            <div className="mt-3 space-y-0">
              <KeyValue label="Logical cores" value={hardware?.cpu.logical_cores ?? env.system?.cpu.logical_cores ?? "—"} />
              <KeyValue label="Architecture" value={hardware?.cpu.architecture ?? "—"} />
            </div>
          </Panel>

          <Panel>
            <PanelHeader icon={<MemoryStick className="h-4 w-4" />} title="Memory and system" />
            <Stat
              label="System RAM"
              value={
                hardware?.memory.total_mb
                  ? formatBytes(hardware.memory.total_mb * 1024 ** 2)
                  : env.system
                    ? formatBytes(env.system.memory.total_mb * 1024 ** 2)
                    : "—"
              }
              sub={
                env.system
                  ? `${formatBytes(env.system.memory.free_mb * 1024 ** 2)} free right now`
                  : undefined
              }
            />
            <div className="mt-3 space-y-0">
              <KeyValue label="Operating system" value={`${hardware?.os.system ?? env.system?.platform ?? "—"} ${hardware?.os.release ?? env.system?.release ?? ""}`} />
              <KeyValue label="Hostname" value={hardware?.os.hostname ?? env.system?.hostname ?? "—"} />
              <KeyValue label="Python" value={env.python?.version ?? "not found"} tone={env.python?.available ? undefined : "warn"} />
              <KeyValue label="Electron" value={env.system?.electron ?? "—"} />
            </div>
          </Panel>
        </div>

        {liveSample ? (
          <Panel className="mt-4">
            <PanelHeader
              icon={<Activity className="h-4 w-4" />}
              title="Live sample"
              description="A single nvidia-smi reading taken just now."
            />
            {liveSample.available && liveSample.gpu ? (
              <div className="grid grid-cols-2 gap-x-6 sm:grid-cols-4">
                <KeyValue label="GPU" value={liveSample.gpu.name} />
                <KeyValue label="Utilisation" value={`${liveSample.gpu.utilization_gpu ?? "—"}%`} />
                <KeyValue label="VRAM used" value={formatBytes((liveSample.gpu.memory_used_mb ?? 0) * 1024 ** 2)} />
                <KeyValue
                  label="Temperature"
                  value={liveSample.gpu.temperature_c != null ? `${liveSample.gpu.temperature_c} °C` : "—"}
                />
                <KeyValue label="Power" value={liveSample.gpu.power_draw_w != null ? `${liveSample.gpu.power_draw_w} W` : "—"} />
                <KeyValue label="Memory bus" value={liveSample.gpu.utilization_memory != null ? `${liveSample.gpu.utilization_memory}%` : "—"} />
                <KeyValue label="Driver" value={liveSample.gpu.driver_version ?? "—"} />
                <KeyValue label="Compute" value={liveSample.gpu.compute_capability ?? "—"} />
              </div>
            ) : (
              <Note tone="warn" title="No GPU to sample">
                {liveSample.reason ?? "nvidia-smi did not report any device."}
              </Note>
            )}
          </Panel>
        ) : null}

        {!hardware ? (
          <EmptyState
            className="mt-4"
            icon={<HardDrive className="h-7 w-7" />}
            title="No hardware snapshot yet"
            description="The Python backend could not be reached, so only Node-level information is available."
          />
        ) : null}

        <div className="mt-4 flex items-center gap-3 text-[11.5px] text-[var(--text-3)]">
          <Thermometer className="h-3.5 w-3.5" />
          Values are read live from the driver; nothing is estimated on this screen.
        </div>
      </PageBody>
    </>
  );
}
