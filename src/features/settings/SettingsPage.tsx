import { useEffect, useState } from "react";
import {
  AlertTriangle,
  Check,
  Copy,
  Database,
  Download,
  FolderOpen,
  HardDrive,
  KeyRound,
  Palette,
  RefreshCw,
  RotateCcw,
  Settings as SettingsIcon,
  Terminal,
  Wrench,
} from "lucide-react";

import { PageBody, PageHeader } from "@/components/layout/Shell";
import { ConfirmDialog } from "@/components/ui/Overlay";
import {
  Badge,
  Button,
  Dot,
  Field,
  IconButton,
  Input,
  KeyValue,
  Note,
  Panel,
  PanelHeader,
  Segmented,
  Select,
  Switch,
} from "@/components/ui/primitives";
import type { BackendCapability, DependencyInfo, InstallPlan } from "@/lib/types";
import { cn, formatBytes } from "@/lib/utils";
import { useStore } from "@/state/store";
import {
  appStore,
  discoverInterpreters,
  navigate,
  openPath,
  refreshEnv,
  saveHfToken,
  selectInterpreter,
  updateSettings,
} from "@/state/appStore";
import { bridge, isDesktop } from "@/lib/bridge";

interface InterpreterInfo {
  command: string;
  args?: string[];
  label: string;
  available: boolean;
  info?: { executable: string; version: string; implementation: string };
  reason?: string;
}

function SectionHeading({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="mb-2 mt-5 flex items-baseline gap-2 first:mt-0">
      <h2 className="text-[12px] font-semibold uppercase tracking-[0.14em] text-[var(--text-3)]">
        {title}
      </h2>
      {hint ? <span className="text-[11.5px] text-[var(--text-3)]">{hint}</span> : null}
    </div>
  );
}

export function SettingsPage() {
  const { settings, env, appInfo } = useStore(appStore);
  const dependencies = env.dependencies as DependencyInfo | null;
  const backends = (env.backends ?? []) as BackendCapability[];
  const plan = env.installPlan as InstallPlan | null;

  const [interpreters, setInterpreters] = useState<InterpreterInfo[]>([]);
  const [discovering, setDiscovering] = useState(false);
  const [tokenInput, setTokenInput] = useState("");
  const [tokenStatus, setTokenStatus] = useState<{ present: boolean; encrypted: boolean; insecure: boolean } | null>(null);
  const [copied, setCopied] = useState(false);
  const [confirmReset, setConfirmReset] = useState(false);

  useEffect(() => {
    void (async () => {
      if (!isDesktop) return;
      setDiscovering(true);
      const result = await discoverInterpreters(true);
      if (result && Array.isArray(result.interpreters)) {
        setInterpreters(result.interpreters as InterpreterInfo[]);
      }
      const status = await bridge.settings.tokenStatus();
      if (status.ok) setTokenStatus(status.storage as typeof tokenStatus);
      setDiscovering(false);
    })();
  }, []);

  const copyPlan = async () => {
    if (!plan) return;
    await navigator.clipboard?.writeText(plan.command);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const pythonAvailable = Boolean(env.python?.available);
  const trainingReady = Boolean(dependencies?.training_ready);

  return (
    <>
      <PageHeader
        icon={<SettingsIcon className="h-4 w-4" />}
        title="Settings"
        subtitle="Environment, storage and Hugging Face access"
        actions={
          <Button
            size="sm"
            variant="quiet"
            icon={<RefreshCw className="h-3.5 w-3.5" />}
            loading={env.loading}
            onClick={() => void refreshEnv(true)}
          >
            Re-detect environment
          </Button>
        }
      />

      <PageBody wide>
        {/* ------------------------------------------------------ environment */}
        {/* Messages elsewhere point here by name, so the section is labelled. */}
        <SectionHeading title="Environment" hint="Interpreter, packages and training backends" />
        <Panel className="mb-4">
          <PanelHeader
            icon={<Terminal className="h-4 w-4" />}
            title="Python interpreter"
            description="The backend runs as a subprocess of the interpreter selected here. A virtual environment is recommended."
            actions={
              <>
                <Button
                  size="sm"
                  variant="ghost"
                  loading={discovering}
                  onClick={() => {
                    void (async () => {
                      setDiscovering(true);
                      const result = await discoverInterpreters(true);
                      if (result && Array.isArray(result.interpreters)) {
                        setInterpreters(result.interpreters as InterpreterInfo[]);
                      }
                      setDiscovering(false);
                    })();
                  }}
                >
                  Find interpreters
                </Button>
                <Button
                  size="sm"
                  variant="quiet"
                  onClick={() => {
                    void (async () => {
                      const picked = await bridge.dialogs.pickPython();
                      const paths = (picked.paths as string[]) || [];
                      if (paths.length) await selectInterpreter(paths[0]);
                    })();
                  }}
                >
                  Browse…
                </Button>
              </>
            }
          />

          <div className="mb-3 flex flex-wrap items-center gap-2">
            <Badge tone={pythonAvailable ? "good" : "bad"}>
              <Dot tone={pythonAvailable ? "good" : "bad"} />
              {pythonAvailable ? `Python ${env.python?.version}` : "no interpreter found"}
            </Badge>
            {env.python?.executable ? (
              <span className="zq-mono truncate text-[11px] text-[var(--text-3)]">{env.python.executable}</span>
            ) : null}
            <span className="flex-1" />
            <Badge tone={trainingReady ? "good" : "warn"}>
              {trainingReady ? "ML runtime installed" : "ML runtime incomplete"}
            </Badge>
          </div>

          <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
            {interpreters.map((interpreter) => {
              const isActive = env.python?.executable === interpreter.info?.executable;
              return (
                <button
                  key={`${interpreter.command}_${(interpreter.args || []).join("_")}`}
                  type="button"
                  disabled={!interpreter.available}
                  onClick={() => interpreter.info?.executable && void selectInterpreter(interpreter.info.executable)}
                  className={cn(
                    "rounded-[11px] border p-3 text-left transition-colors",
                    isActive
                      ? "border-[var(--acc)] bg-[var(--acc-soft)]"
                      : "border-[var(--border-soft)] hover:bg-[var(--hover)]",
                    !interpreter.available && "cursor-not-allowed opacity-55 hover:bg-transparent",
                  )}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-[12.5px] font-medium">
                      {interpreter.info ? `Python ${interpreter.info.version}` : interpreter.label}
                    </span>
                    {isActive ? <Badge tone="accent">in use</Badge> : null}
                    {!interpreter.available ? <Badge tone="neutral">unavailable</Badge> : null}
                  </div>
                  <p className="zq-mono mt-0.5 truncate text-[10.5px] text-[var(--text-3)]">
                    {interpreter.info?.executable ?? interpreter.reason ?? interpreter.label}
                  </p>
                </button>
              );
            })}
          </div>
        </Panel>

        {/* ------------------------------------------------------- dependencies */}
        <Panel className="mb-4">
          <PanelHeader
            icon={<Download className="h-4 w-4" />}
            title="ML runtime packages"
            description="Training and inference need these. Diagnostics and dataset validation work without them."
          />

          {plan && !trainingReady ? (
            <div className="mb-4">
              <Note
                tone="warn"
                title="Install the ML runtime to enable training"
                actions={
                  <Button size="sm" variant="quiet" icon={copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />} onClick={() => void copyPlan()}>
                    {copied ? "Copied" : "Copy"}
                  </Button>
                }
              >
                <p className="mb-2">
                  Run this with the interpreter selected above. On an NVIDIA machine the CUDA wheel index
                  installs a GPU-enabled PyTorch build; a plain <code className="zq-mono">pip install torch</code>{" "}
                  is often CPU-only.
                </p>
                <pre className="zq-mono overflow-x-auto rounded-[8px] border border-[var(--border)] bg-[var(--code-bg)] p-2.5 text-[11px] leading-[17px]">
                  {plan.command}
                </pre>
              </Note>
            </div>
          ) : null}

          <div className="mb-3 flex flex-wrap items-center gap-2">
            <Field label="CUDA wheel tag" className="w-[180px]">
              <Select
                value={settings?.cudaWheelTag ?? "cu124"}
                onChange={(event) => void updateSettings({ cudaWheelTag: event.target.value })}
              >
                {["cu121", "cu124", "cu126", "cu128"].map((tag) => (
                  <option key={tag} value={tag}>
                    {tag}
                  </option>
                ))}
              </Select>
            </Field>
            <Button
              size="sm"
              variant="quiet"
              onClick={() => void bridge.env.installPlan(settings?.cudaWheelTag).then(() => refreshEnv())}
            >
              Rebuild command
            </Button>
            <span className="text-[11.5px] text-[var(--text-3)]">
              Pin the tag to the CUDA version your driver supports.
            </span>
          </div>

          {dependencies ? (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[560px] border-collapse">
                <thead>
                  <tr className="text-left text-[10.5px] uppercase tracking-[0.06em] text-[var(--text-3)]">
                    <th className="px-1 pb-2 font-medium">Package</th>
                    <th className="px-1 pb-2 font-medium">State</th>
                    <th className="px-1 pb-2 font-medium">Version</th>
                    <th className="px-1 pb-2 font-medium">Needed for</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(dependencies.packages).map(([name, info]) => (
                    <tr key={name} className="border-t border-[var(--border-soft)]">
                      <td className="zq-mono px-1 py-1.5 text-[12px]">{name}</td>
                      <td className="px-1 py-1.5">
                        {info.installed ? (
                          <Badge tone="good">
                            <Check className="h-3 w-3" />
                            installed
                          </Badge>
                        ) : (
                          <Badge tone={info.group === "core" ? "bad" : "neutral"}>missing</Badge>
                        )}
                      </td>
                      <td className="zq-mono px-1 py-1.5 text-[11.5px] text-[var(--text-2)]">
                        {info.version ?? "—"}
                      </td>
                      <td className="px-1 py-1.5 text-[11.5px] text-[var(--text-3)]">{info.purpose}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-[12.5px] text-[var(--text-3)]">
              {env.error?.message ?? "Package information is unavailable because the Python backend could not be reached."}
            </p>
          )}
        </Panel>

        {/* ---------------------------------------------------------- backends */}
        <Panel className="mb-4">
          <PanelHeader
            icon={<Wrench className="h-4 w-4" />}
            title="Training backends"
            description="The training engine is not tied to one framework. Each backend reports what it can run."
          />
          <div className="space-y-2">
            {backends.map((backend) => (
              <div
                key={backend.name}
                className="flex items-start justify-between gap-3 rounded-[11px] border border-[var(--border-soft)] p-3"
              >
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-[12.5px] font-medium">{backend.label}</span>
                    <Badge tone="accent">{backend.name}</Badge>
                  </div>
                  <div className="mt-1 flex flex-wrap gap-1.5">
                    {backend.methods.map((method) => (
                      <Badge key={method}>{method}</Badge>
                    ))}
                  </div>
                  {backend.missing.length ? (
                    <p className="mt-1.5 text-[11.5px] text-[var(--text-3)]">
                      Missing: {backend.missing.join(", ")}
                    </p>
                  ) : null}
                </div>
                <Badge tone={backend.available ? "good" : "warn"}>
                  <Dot tone={backend.available ? "good" : "warn"} />
                  {backend.available ? "ready" : "unavailable"}
                </Badge>
              </div>
            ))}
            {backends.length === 0 ? (
              <p className="text-[12.5px] text-[var(--text-3)]">No backend information available.</p>
            ) : null}
          </div>
        </Panel>

        <SectionHeading title="Data" hint="Model downloads and where files live" />
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {/* ---------------------------------------------------- hugging face */}
          <Panel>
            <PanelHeader
              icon={<KeyRound className="h-4 w-4" />}
              title="Hugging Face"
              description="Needed for gated repositories and faster downloads."
            />
            <Field
              label="Access token"
              hint={
                tokenStatus?.present
                  ? tokenStatus.encrypted
                    ? "A token is stored, encrypted with the operating system keychain."
                    : "Stored unencrypted — the OS keychain is unavailable on this system."
                  : "Stored only on this machine. Never sent anywhere except Hugging Face."
              }
            >
              <div className="flex gap-2">
                <Input
                  type="password"
                  value={tokenInput}
                  placeholder={tokenStatus?.present ? "•••••••• (stored)" : "hf_…"}
                  onChange={(event) => setTokenInput(event.target.value)}
                />
                <Button
                  variant="secondary"
                  disabled={!tokenInput.trim()}
                  onClick={() =>
                    void saveHfToken(tokenInput.trim()).then(() => {
                      setTokenInput("");
                      void bridge.settings.tokenStatus().then((status) => {
                        if (status.ok) setTokenStatus(status.storage as typeof tokenStatus);
                      });
                    })
                  }
                >
                  Save
                </Button>
                {tokenStatus?.present ? (
                  <Button
                    variant="ghost"
                    onClick={() =>
                      void saveHfToken(null).then(() =>
                        setTokenStatus({ present: false, encrypted: false, insecure: false }),
                      )
                    }
                  >
                    Remove
                  </Button>
                ) : null}
              </div>
            </Field>

            <div className="mt-3 border-t border-[var(--border-soft)] pt-3">
              <Field label="Model cache folder" hint="Where downloaded weights are stored.">
                <div className="flex gap-2">
                  <Input
                    value={settings?.hfCacheDir ?? (appInfo?.hfCacheDir as string) ?? ""}
                    placeholder={(appInfo?.hfCacheDir as string) ?? ""}
                    onChange={(event) => void updateSettings({ hfCacheDir: event.target.value || null })}
                  />
                  <Button
                    variant="secondary"
                    onClick={() => {
                      void (async () => {
                        const picked = await bridge.dialogs.pickDirectory("Select the model cache folder");
                        const paths = (picked.paths as string[]) || [];
                        if (paths.length) await updateSettings({ hfCacheDir: paths[0] });
                      })();
                    }}
                  >
                    Browse
                  </Button>
                  <IconButton
                    title="Open the cache folder"
                    onClick={() => void openPath((settings?.hfCacheDir as string) ?? (appInfo?.hfCacheDir as string))}
                  >
                    <FolderOpen className="h-3.5 w-3.5" />
                  </IconButton>
                </div>
              </Field>
            </div>
          </Panel>

          {/* --------------------------------------------------------- storage */}
          <Panel>
            <PanelHeader
              icon={<HardDrive className="h-4 w-4" />}
              title="Storage"
              description="Where runs, logs, checkpoints and trained models live."
            />
            <div className="space-y-2">
              {[
                { label: "App data", value: appInfo?.userData as string },
                { label: "Runs and checkpoints", value: appInfo?.runsDir as string },
                { label: "Trained models", value: appInfo?.modelsDir as string },
                { label: "Python backend", value: appInfo?.backendDir as string },
              ].map((entry) => (
                <div
                  key={entry.label}
                  className="flex items-center gap-2 rounded-[10px] border border-[var(--border-soft)] px-2.5 py-2"
                >
                  <span className="min-w-0 flex-1">
                    <span className="block text-[11px] text-[var(--text-3)]">{entry.label}</span>
                    <span className="zq-mono block truncate text-[11px]">{entry.value ?? "—"}</span>
                  </span>
                  <IconButton title="Open folder" onClick={() => void openPath(entry.value)}>
                    <FolderOpen className="h-3.5 w-3.5" />
                  </IconButton>
                </div>
              ))}
            </div>
            <p className="mt-3 flex items-start gap-2 text-[11.5px] leading-[17px] text-[var(--text-3)]">
              <Database className="mt-[3px] h-3 w-3 shrink-0" />
              Datasets are referenced in place and never copied, so importing a large file costs no disk space.
            </p>
          </Panel>

          {/* ------------------------------------------------------ appearance */}
          <Panel>
            <PanelHeader icon={<Palette className="h-4 w-4" />} title="Appearance" />
            <Field label="Theme" hint="Dark is the primary face of the Zeqou brand.">
              <Segmented
                value={(settings?.theme ?? "dark") as "dark" | "light"}
                onChange={(value) => void updateSettings({ theme: value })}
                options={[
                  { value: "dark", label: "Dark" },
                  { value: "light", label: "Light" },
                ]}
              />
            </Field>
          </Panel>

          {/* -------------------------------------------------------- advanced */}
          <Panel>
            <PanelHeader
              icon={<Wrench className="h-4 w-4" />}
              title="Advanced"
              description="How much the app decides for you. Messages that say “Settings → Advanced” bring you here."
            />
            <div className="space-y-4">
              <Switch
                checked={settings?.autoConfigure ?? true}
                onChange={(value) => void updateSettings({ autoConfigure: value })}
                label="Configure parameters automatically"
                description="Derive batch size, precision, context length and optimizer from the detected hardware, and explain each choice."
              />
              <Switch
                checked={settings?.simpleMode ?? true}
                onChange={(value) => void updateSettings({ simpleMode: value })}
                label="Start the wizard in Simple mode"
                description="Advanced mode reveals rank, scheduler, optimizer, retention and evaluation controls."
              />
              <Switch
                checked={settings?.advanced?.trustRemoteCode ?? false}
                onChange={(value) =>
                  void updateSettings({ advanced: { ...(settings?.advanced as any), trustRemoteCode: value } })
                }
                label="Allow custom model code"
                description="Some models ship Python that runs on load. Enable only for repositories you trust."
              />
            </div>
          </Panel>
        </div>

        <Panel className="mt-4">
          <PanelHeader
            icon={<AlertTriangle className="h-4 w-4" />}
            title="Reset"
            description="Restores default settings. Runs, models and datasets are never deleted."
            actions={
              <Button variant="danger" icon={<RotateCcw className="h-3.5 w-3.5" />} onClick={() => setConfirmReset(true)}>
                Reset settings
              </Button>
            }
          />
          <div className="space-y-0">
            <KeyValue label="Application" value={`${appInfo?.name ?? "ZeqouXTraining"} ${appInfo?.version ?? ""}`} />
            <KeyValue label="Electron" value={(appInfo?.electron as string) ?? "—"} />
            <KeyValue label="Node" value={(appInfo?.node as string) ?? "—"} />
            <KeyValue label="Chrome" value={(appInfo?.chrome as string) ?? "—"} />
            <KeyValue
              label="Secret storage"
              value={appInfo?.encryptionAvailable ? "OS keychain available" : "OS keychain unavailable"}
              tone={appInfo?.encryptionAvailable ? "good" : "warn"}
            />
          </div>
          <div className="mt-3 flex items-center gap-2">
            <Button
              size="sm"
              variant="ghost"
              onClick={() => navigate("hardware")}
            >
              Open hardware diagnostics
            </Button>
            {!isDesktop ? <Badge tone="warn">preview mode</Badge> : null}
          </div>
        </Panel>

        {!trainingReady && dependencies ? (
          <p className="mt-3 text-[11.5px] text-[var(--text-3)]">
            Core packages still missing: {dependencies.missing_core.join(", ") || "none"}. Estimated download
            size is roughly {formatBytes(300 * 1024 ** 2)} for a CPU build and {formatBytes(2.6 * 1024 ** 3)} for
            a CUDA build.
          </p>
        ) : null}
      </PageBody>

      <ConfirmDialog
        open={confirmReset}
        title="Reset all settings?"
        description="Settings return to their defaults. Interpreters, tokens, runs, datasets and trained models are not affected."
        confirmLabel="Reset settings"
        onConfirm={() => {
          void bridge.settings.reset().then(() => refreshEnv(true));
          setConfirmReset(false);
        }}
        onCancel={() => setConfirmReset(false)}
      />
    </>
  );
}
