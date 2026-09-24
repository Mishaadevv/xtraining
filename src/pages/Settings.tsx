import { useEffect, useState } from "react";
import {
  Bell,
  BookOpen,
  Cpu,
  Database,
  Download,
  FolderOpen,
  HardDrive,
  Keyboard,
  Monitor,
  RefreshCw,
  RotateCcw,
  Server,
  Shield,
  Sliders,
  Wrench,
} from "lucide-react";
import { api } from "../lib/api";
import { bytes, relative } from "../lib/format";
import { useApp } from "../state/app";
import { useRouter } from "../state/router";
import {
  Badge,
  Button,
  Callout,
  Field,
  Input,
  KeyValue,
  Panel,
  ProgressBar,
  SectionHeader,
  Select,
  Stat,
  Table,
  Td,
  Th,
  Toggle,
} from "../components/ui";
import { ErrorPanel, Loading, useEngine } from "./common";

const SECTIONS = [
  { id: "general", label: "General", icon: <Sliders size={12} /> },
  { id: "appearance", label: "Appearance", icon: <Monitor size={12} /> },
  { id: "hardware", label: "Hardware", icon: <Cpu size={12} /> },
  { id: "training", label: "Training", icon: <Wrench size={12} /> },
  { id: "models", label: "Models & datasets", icon: <Database size={12} /> },
  { id: "storage", label: "Storage", icon: <HardDrive size={12} /> },
  { id: "environment", label: "Environment", icon: <Server size={12} /> },
  { id: "security", label: "Security & privacy", icon: <Shield size={12} /> },
  { id: "notifications", label: "Notifications", icon: <Bell size={12} /> },
  { id: "updates", label: "Updates", icon: <RefreshCw size={12} /> },
  { id: "shortcuts", label: "Keyboard shortcuts", icon: <Keyboard size={12} /> },
  { id: "about", label: "About & docs", icon: <BookOpen size={12} /> },
];

const SHORTCUTS: Array<[string, string]> = [
  ["Ctrl/⌘ + K", "Command palette"],
  ["Esc", "Close the palette or a dialog"],
  ["Ctrl/⌘ + 1…9", "Jump to a sidebar section"],
  ["Alt + ←  /  Alt + →", "Navigate back / forward"],
  ["Ctrl/⌘ + Enter", "Send the prompt in the Playground"],
  ["Ctrl/⌘ + R", "Reload the current page data"],
];

export function SettingsPage() {
  const { settings, saveSettings, appInfo, hardware, environment, live, registry, updates, refreshUpdates, reportError, toast } =
    useApp();
  const router = useRouter();
  const [section, setSection] = useState("general");
  const [draft, setDraft] = useState<any>(settings);
  const storage = useEngine<any>("storage.report", {}, { timeout: 180_000 });
  const tools = useEngine<any>("tools.report", {}, { timeout: 120_000 });

  useEffect(() => setDraft(settings), [settings]);

  if (!draft) return <Loading label="Reading settings…" lines={4} />;

  const set = (patch: Record<string, unknown>) => setDraft((current: any) => ({ ...current, ...patch }));
  const commit = async (patch: Record<string, unknown>) => {
    set(patch);
    try {
      await saveSettings(patch);
    } catch (error) {
      reportError(error, "Settings were not saved");
    }
  };

  /** Run an update action and fold the answer back into the shared state. */
  const runUpdate = async (action: () => Promise<any>, success = "") => {
    try {
      await action();
      if (success) toast({ title: success, tone: "info" });
    } catch (error) {
      reportError(error, "The update action failed");
    } finally {
      await refreshUpdates();
    }
  };

  const pickWorkspace = async () => {
    const folder = await api.dialog.pickFolder({ title: "Choose the workspace folder", defaultPath: draft.workspace ?? undefined });
    if (folder) await commit({ workspace: folder });
  };

  const pickPython = async () => {
    const files = await api.dialog.pickFiles({ title: "Select a Python interpreter" });
    if (files?.[0]) await commit({ pythonPath: files[0] });
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Settings</h1>
          <p className="mt-0.5 max-w-3xl text-xs text-ink-2">
            Everything here is stored in the workspace (or the app data folder for theme and window state) as plain
            JSON. Nothing is synced anywhere.
          </p>
        </div>
        <Badge tone="muted">settings v{draft.version ?? 1}</Badge>
      </div>

      <div className="grid gap-3 lg:grid-cols-[220px_1fr]">
        <Panel padded={false}>
          <div className="space-y-0.5 p-2">
            {SECTIONS.map((entry) => (
              <button
                key={entry.id}
                onClick={() => setSection(entry.id)}
                className={`flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-xs transition-colors ${
                  section === entry.id ? "bg-accent/15 text-ink-0" : "text-ink-2 hover:bg-surface-3"
                }`}
              >
                {entry.icon}
                {entry.label}
              </button>
            ))}
          </div>
        </Panel>

        <div className="space-y-3">
          {section === "general" ? (
            <Panel>
              <SectionHeader title="General" subtitle="Workspace, interpreter and logging." />
              <div className="space-y-3">
                <Field
                  label="Workspace folder"
                  hint="Models, datasets, jobs, checkpoints and exports live here. It survives app updates."
                >
                  <div className="flex gap-2">
                    <Input value={draft.workspace ?? ""} readOnly placeholder="not chosen yet" />
                    <Button size="sm" icon={<FolderOpen size={12} />} onClick={() => void pickWorkspace()}>
                      Choose
                    </Button>
                    {draft.workspace ? (
                      <Button size="sm" variant="subtle" onClick={() => void api.shell.reveal(draft.workspace)}>
                        Reveal
                      </Button>
                    ) : null}
                  </div>
                </Field>

                <Field label="Python interpreter" hint="Leave empty to let the app pick the best interpreter it can find.">
                  <div className="flex gap-2">
                    <Input value={draft.pythonPath ?? ""} readOnly placeholder="auto-detect" />
                    <Button size="sm" icon={<Server size={12} />} onClick={() => void pickPython()}>
                      Choose
                    </Button>
                    <Button size="sm" variant="subtle" onClick={() => void commit({ pythonPath: null })}>
                      Auto
                    </Button>
                  </div>
                </Field>

                <Field label="Log level" hint="Controls how much the engine writes to job logs.">
                  <Select value={draft.logLevel ?? "info"} onChange={(event) => void commit({ logLevel: event.target.value })}>
                    <option value="debug">debug</option>
                    <option value="info">info</option>
                    <option value="warning">warning</option>
                    <option value="error">error</option>
                  </Select>
                </Field>

                <Toggle
                  checked={Boolean(draft.offlineMode)}
                  onChange={(value) => void commit({ offlineMode: value })}
                  label="Offline mode"
                  hint="Blocks any network access the engine would otherwise attempt (model downloads, Hub lookups, package installs)."
                />
              </div>
            </Panel>
          ) : null}

          {section === "appearance" ? (
            <Panel>
              <SectionHeader title="Appearance" subtitle="One visual language in both themes." />
              <div className="space-y-3">
                <Field label="Theme">
                  <Select value={draft.theme ?? "system"} onChange={(event) => void commit({ theme: event.target.value })}>
                    <option value="system">Follow the system</option>
                    <option value="dark">Dark</option>
                    <option value="light">Light</option>
                  </Select>
                </Field>
                <Field label="Language" hint="The interface ships in English; the setting is stored for future localisations.">
                  <Select value={draft.language ?? "en"} onChange={(event) => void commit({ language: event.target.value })}>
                    <option value="en">English</option>
                  </Select>
                </Field>
                <Toggle
                  checked={draft.reduceMotion ?? false}
                  onChange={(value) => void commit({ reduceMotion: value })}
                  label="Reduce motion"
                  hint="Disables page transitions and chart animations."
                />
              </div>
            </Panel>
          ) : null}

          {section === "hardware" ? (
            <Panel>
              <SectionHeader
                title="Hardware"
                subtitle="Detected once at startup and re-read live while the app is open."
                actions={
                  <Button size="sm" variant="subtle" onClick={() => router.navigate("/hardware")}>
                    Open Hardware Center
                  </Button>
                }
              />
              <div className="grid gap-3 sm:grid-cols-2">
                <Stat
                  label="CPU"
                  value={hardware?.cpu?.model ?? "—"}
                  hint={hardware?.cpu?.logical_cores ? `${hardware.cpu.logical_cores} logical cores` : undefined}
                />
                <Stat label="GPU" value={hardware?.gpus?.[0]?.name ?? "no CUDA device"} hint={hardware?.gpus?.[0]?.driver_version ? `driver ${hardware.gpus[0].driver_version}` : undefined} />
                <Stat label="RAM" value={bytes(hardware?.memory?.total ?? null)} hint={live?.ram ? `${bytes(live.ram.used)} in use now` : undefined} />
                <Stat label="VRAM" value={hardware?.gpus?.[0]?.memory_total ? bytes(hardware.gpus[0].memory_total) : "—"} />
              </div>
              <div className="mt-3">
                <Field label="Default device" hint="Where new jobs run unless a run overrides it.">
                  <Select value={draft.defaultDevice ?? "auto"} onChange={(event) => void commit({ defaultDevice: event.target.value })}>
                    <option value="auto">Auto</option>
                    <option value="cpu">CPU</option>
                    {hardware?.gpus?.map((gpu) => (
                      <option key={gpu.index} value={`cuda:${gpu.index}`}>
                        cuda:{gpu.index} — {gpu.name}
                      </option>
                    ))}
                  </Select>
                </Field>
              </div>
              <div className="mt-3">
                <Field label="Default training backend">
                  <Select value={draft.defaultBackend ?? "auto"} onChange={(event) => void commit({ defaultBackend: event.target.value === "auto" ? null : event.target.value })}>
                    <option value="auto">Auto — pick the best available</option>
                    <option value="tiny">Pure-Python tiny backend</option>
                    <option value="hf">Transformers + PEFT</option>
                  </Select>
                </Field>
              </div>
            </Panel>
          ) : null}

          {section === "training" ? (
            <Panel>
              <SectionHeader title="Training" subtitle="Defaults applied to every new run; each run can override them." />
              <div className="space-y-3">
                <Field label="Default method">
                  <Select value={draft.defaultMethod ?? "lora"} onChange={(event) => void commit({ defaultMethod: event.target.value })}>
                    {["lora", "qlora", "sft", "full_finetune", "continued_pretraining", "scratch"].map((method) => (
                      <option key={method} value={method}>
                        {method}
                      </option>
                    ))}
                  </Select>
                </Field>
                <Field label="Checkpoints to keep" hint="Oldest unprotected checkpoints are pruned beyond this number. Protected ones are never removed.">
                  <Input
                    type="number"
                    min={1}
                    max={50}
                    value={draft.checkpointPolicy?.keepLast ?? 3}
                    onChange={(event) => void commit({ checkpointPolicy: { ...draft.checkpointPolicy, keepLast: Number(event.target.value) } })}
                  />
                </Field>
                <Toggle
                  checked={Boolean(draft.checkpointPolicy?.protectBest)}
                  onChange={(value) => void commit({ checkpointPolicy: { ...draft.checkpointPolicy, protectBest: value } })}
                  label="Protect the best checkpoint"
                  hint="Marks the checkpoint with the lowest validation loss as protected automatically."
                />
                <Toggle
                  checked={draft.autoSave ?? true}
                  onChange={(value) => void commit({ autoSave: value })}
                  label="Auto-save run configuration"
                  hint="Writes the configuration to the job folder as soon as it is validated."
                />
                <Toggle
                  checked={draft.queueHeavyJobs ?? true}
                  onChange={(value) => void commit({ queueHeavyJobs: value })}
                  label="Queue heavy jobs"
                  hint="Prevents two training processes from competing for the same device."
                />
              </div>
            </Panel>
          ) : null}

          {section === "models" ? (
            <Panel>
              <SectionHeader title="Models & datasets" subtitle="Import behaviour and library defaults." />
              <div className="space-y-3">
                <Toggle
                  checked={draft.copyOnImport ?? false}
                  onChange={(value) => void commit({ copyOnImport: value })}
                  label="Copy imported models into the workspace"
                  hint="Off by default: the library references the original folder, so nothing is duplicated."
                />
                <Toggle
                  checked={draft.datasetCache ?? true}
                  onChange={(value) => void commit({ datasetCache: value })}
                  label="Cache dataset statistics"
                  hint="Saves record counts and token estimates next to the dataset to avoid rescanning large files."
                />
                <div className="rounded-md border border-line-soft bg-surface-2 p-3">
                  <KeyValue
                    items={[
                      ["Registered models", registry?.models?.length ?? 0],
                      ["Registered datasets", registry?.datasets?.length ?? 0],
                      ["Projects", registry?.projects?.length ?? 0],
                      ["Saved conversations", registry?.conversations?.length ?? 0],
                    ]}
                  />
                </div>
              </div>
            </Panel>
          ) : null}

          {section === "storage" ? (
            <Panel>
              <SectionHeader title="Storage" subtitle="Measured workspace usage." actions={<Button size="sm" variant="subtle" onClick={() => router.navigate("/files")}>Open Files</Button>} />
              {storage.error ? <ErrorPanel error={storage.error} onRetry={() => void storage.reload()} /> : null}
              {storage.data ? (
                <>
                  <div className="grid gap-3 sm:grid-cols-3">
                    <Stat label="Workspace" value={bytes(storage.data.total_bytes)} />
                    <Stat label="Free space" value={bytes(storage.data.disk?.free ?? null)} />
                    <Stat label="Protected artifacts" value={storage.data.protected?.length ?? 0} />
                  </div>
                  <div className="mt-3">
                    <Table>
                      <thead>
                        <tr>
                          <Th>Folder</Th>
                          <Th align="right">Files</Th>
                          <Th align="right">Size</Th>
                        </tr>
                      </thead>
                      <tbody>
                        {(storage.data.entries ?? []).map((entry: any) => (
                          <tr key={entry.name}>
                            <Td>{entry.name}</Td>
                            <Td align="right">{entry.files}</Td>
                            <Td align="right">{entry.human}</Td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  </div>
                  <div className="mt-3">
                    <Toggle
                      checked={draft.pruneOnStart ?? false}
                      onChange={(value) => void commit({ pruneOnStart: value })}
                      label="Offer a cleanup when the workspace passes 90% of the volume"
                      hint="The app always asks first; nothing is deleted automatically."
                    />
                  </div>
                </>
              ) : (
                <Loading lines={3} />
              )}
            </Panel>
          ) : null}

          {section === "environment" ? (
            <Panel>
              <SectionHeader title="Environment" subtitle="The interpreter and packages the engine actually uses." actions={<Button size="sm" variant="subtle" onClick={() => router.navigate("/environment")}>Open Environment</Button>} />
              {environment ? (
                <KeyValue
                  items={[
                    ["Engine interpreter", `${environment.engine_python?.executable ?? "?"} (python ${environment.engine_python?.version ?? "?"})`],
                    ["In virtual environment", environment.engine_python?.in_venv ? "yes" : "no"],
                    ["Workspace venv", environment.workspace_venv?.exists ? environment.workspace_venv.python : "not created"],
                    ["Transformers training", environment.ready?.transformers_training ? "ready" : "not ready"],
                    ["Adapter training", environment.ready?.peft ? "ready" : "not ready"],
                    ["Quantisation", environment.ready?.quantisation ? "ready" : "not ready"],
                    ["Preference training", environment.ready?.preference_training ? "ready" : "not ready"],
                  ]}
                  columns={1}
                />
              ) : (
                <Loading lines={3} />
              )}
              <div className="mt-3 space-y-2">
                <Toggle
                  checked={draft.allowInstall ?? true}
                  onChange={(value) => void commit({ allowInstall: value })}
                  label="Allow the app to create its own environment"
                  hint="Installations go into <workspace>/runtime/venv and never touch the system Python."
                />
                <Toggle
                  checked={draft.useWorkspaceVenv ?? true}
                  onChange={(value) => void commit({ useWorkspaceVenv: value })}
                  label="Prefer the workspace environment for jobs"
                  hint="Turn off to run jobs with the interpreter chosen above."
                />
              </div>
            </Panel>
          ) : null}

          {section === "security" ? (
            <Panel>
              <SectionHeader title="Security & privacy" subtitle="Local-first by default." />
              <div className="space-y-3">
                <Toggle
                  checked={Boolean(draft.offlineMode)}
                  onChange={(value) => void commit({ offlineMode: value })}
                  label="Offline mode"
                  hint="No downloads, no Hub calls, no update checks."
                />
                <Toggle
                  checked={draft.allowExternalCode ?? false}
                  onChange={(value) => void commit({ allowExternalCode: value })}
                  label="Allow loading models with custom code"
                  hint="Off by default. Custom architectures ship Python that would run inside the engine process; only enable this for a repository you trust."
                />
                <Toggle
                  checked={draft.localApiEnabled ?? false}
                  onChange={(value) => void commit({ localApiEnabled: value })}
                  label="Allow the local API to be bound beyond loopback"
                  hint="Off by default: the inference server binds to 127.0.0.1 only."
                />
                <Callout tone="info" title="What leaves this machine">
                  Nothing, unless you explicitly start a download. Datasets, models, checkpoints, logs and settings stay
                  on disk. The Deploy page serves a model on loopback for other local tools only.
                </Callout>
              </div>
            </Panel>
          ) : null}

          {section === "updates" ? (
            <Panel>
              <SectionHeader
                title="Updates"
                subtitle="The channel is checked when you ask; the download is verified before it is offered."
                actions={
                  <Button
                    size="sm"
                    icon={<RefreshCw size={12} />}
                    disabled={updates?.status === "checking" || updates?.status === "downloading"}
                    onClick={() => void runUpdate(api.updates.check)}
                  >
                    {updates?.status === "checking" ? "Checking…" : "Check for updates"}
                  </Button>
                }
              />

              {updates ? (
                <div className="space-y-3">
                  <div className="grid gap-3 sm:grid-cols-3">
                    <Stat label="Installed version" value={updates.currentVersion} />
                    <Stat
                      label="Channel"
                      value={updates.channel.provider ?? "—"}
                      hint={updates.channel.url ?? undefined}
                    />
                    <Stat
                      label="Last checked"
                      value={updates.lastCheckedAt ? relative(updates.lastCheckedAt) : "never"}
                      hint={updates.mode === "installed" ? "installed build" : updates.mode}
                    />
                  </div>

                  {!updates.supported ? (
                    <Callout tone="info" title="This build does not update itself">
                      {updates.reason}
                    </Callout>
                  ) : null}

                  {!updates.signatureVerification ? (
                    <Callout tone="warn" title="Update signature checks are disabled">
                      `ZEQOUX_UPDATE_SKIP_SIGNATURE=1` is set, so a downloaded update is not checked against the
                      publisher. Only do this while testing a self-signed build.
                    </Callout>
                  ) : null}

                  {updates.error ? (
                    <Callout tone={updates.error.code === "not_installable" ? "warn" : "danger"} title="Update check failed" hint={updates.error.hint}>
                      {updates.error.message}
                    </Callout>
                  ) : null}

                  {!updates.error && updates.backgroundError ? (
                    <div className="text-2xs text-ink-3">
                      The last automatic check could not reach the channel: {updates.backgroundError.message}
                    </div>
                  ) : null}

                  {updates.available ? (
                    <div className="rounded-md border border-line-soft bg-surface-2 p-3">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div>
                          <div className="text-xs font-medium text-ink-0">Version {updates.available.version} is available</div>
                          <div className="mt-0.5 text-2xs text-ink-3">
                            Installed {updates.currentVersion}
                            {updates.available.releaseDate ? ` · published ${relative(updates.available.releaseDate)}` : ""}
                            {updates.available.size ? ` · ${bytes(updates.available.size)}` : ""}
                          </div>
                        </div>
                        <div className="flex gap-2">
                          {updates.status === "downloaded" ? (
                            <Button
                              size="sm"
                              variant="primary"
                              icon={<RotateCcw size={12} />}
                              disabled={!updates.supported}
                              onClick={() => void runUpdate(api.updates.install, "The update will be installed on restart")}
                            >
                              Restart and install
                            </Button>
                          ) : (
                            <Button
                              size="sm"
                              variant="primary"
                              icon={<Download size={12} />}
                              disabled={updates.status === "downloading"}
                              onClick={() => void runUpdate(api.updates.download)}
                            >
                              {updates.status === "downloading" ? "Downloading…" : "Download"}
                            </Button>
                          )}
                        </div>
                      </div>

                      {updates.progress ? (
                        <div className="mt-3">
                          <ProgressBar
                            value={updates.progress.percent}
                            label={`${updates.progress.percent.toFixed(1)}% · ${bytes(
                              updates.progress.transferred,
                            )} of ${bytes(updates.progress.total)} · ${bytes(updates.progress.bytesPerSecond)}/s`}
                          />
                        </div>
                      ) : null}

                      {updates.available.releaseNotes && updates.status !== "downloading" ? (
                        <div className="mt-3 max-h-40 overflow-auto whitespace-pre-wrap rounded border border-line-soft bg-surface-1 p-2 text-2xs text-ink-2">
                          {updates.available.releaseNotes}
                        </div>
                      ) : null}
                    </div>
                  ) : null}

                  {updates.status === "up-to-date" ? (
                    <Callout tone="ok" title="Up to date">
                      {updates.currentVersion} is the newest version on this channel.
                    </Callout>
                  ) : null}
                </div>
              ) : (
                <Loading lines={2} />
              )}

              <div className="mt-4 space-y-3 border-t border-line-soft pt-3">
                <Field
                  label="Channel override"
                  hint="Leave empty to use the release channel this build shipped with. A URL here is treated as a generic channel serving latest.yml — the same mechanism the self-update test uses."
                >
                  <div className="flex gap-2">
                    <Input
                      value={draft.updateChannel ?? ""}
                      placeholder={updates?.channel.source === "release" ? "built-in release channel" : "https://…"}
                      onChange={(event) => set({ updateChannel: event.target.value })}
                      onBlur={() => void commit({ updateChannel: draft.updateChannel ? draft.updateChannel : null })}
                    />
                    {draft.updateChannel ? (
                      <Button size="sm" variant="subtle" onClick={() => void commit({ updateChannel: null })}>
                        Clear
                      </Button>
                    ) : null}
                  </div>
                </Field>

                <Toggle
                  checked={draft.checkForUpdatesOnStart !== false}
                  onChange={(value) => void commit({ checkForUpdatesOnStart: value })}
                  label="Check the channel 15 seconds after launch"
                  hint="A failed automatic check is written to the updater log and never interrupts you; a manual check reports what happened."
                />

                <KeyValue
                  items={[
                    ["Signature verification", updates?.signatureVerification ? "on" : "off"],
                    ["Channel source", updates?.channel.source ?? "—"],
                    ["Updater log", updates?.logFile ?? "—"],
                  ]}
                  columns={1}
                />
                {updates?.logFile ? (
                  <Button size="sm" variant="subtle" onClick={() => void api.shell.reveal(updates.logFile as string)}>
                    Reveal the updater log
                  </Button>
                ) : null}
              </div>
            </Panel>
          ) : null}

          {section === "notifications" ? (
            <Panel>
              <SectionHeader title="Notifications" subtitle="Only real events raise a notification." />
              <div className="space-y-3">
                <Toggle
                  checked={Boolean(draft.notifications?.jobFinished)}
                  onChange={(value) => void commit({ notifications: { ...draft.notifications, jobFinished: value } })}
                  label="Job finished"
                />
                <Toggle
                  checked={Boolean(draft.notifications?.jobFailed)}
                  onChange={(value) => void commit({ notifications: { ...draft.notifications, jobFailed: value } })}
                  label="Job failed"
                />
                <Toggle
                  checked={Boolean(draft.notifications?.serverStarted)}
                  onChange={(value) => void commit({ notifications: { ...draft.notifications, serverStarted: value } })}
                  label="Local server started or stopped"
                />
                <Toggle
                  checked={draft.notifications?.checkpoints ?? true}
                  onChange={(value) => void commit({ notifications: { ...draft.notifications, checkpoints: value } })}
                  label="Checkpoint saved"
                />
                <Toggle
                  checked={draft.notifications?.lowDisk ?? true}
                  onChange={(value) => void commit({ notifications: { ...draft.notifications, lowDisk: value } })}
                  label="Low disk space or high memory pressure"
                />
              </div>
            </Panel>
          ) : null}

          {section === "shortcuts" ? (
            <Panel>
              <SectionHeader title="Keyboard shortcuts" subtitle="The whole app is reachable from the keyboard." />
              <Table>
                <thead>
                  <tr>
                    <Th>Shortcut</Th>
                    <Th>Action</Th>
                  </tr>
                </thead>
                <tbody>
                  {SHORTCUTS.map(([keys, action]) => (
                    <tr key={keys}>
                      <Td>
                        <Badge tone="muted">{keys}</Badge>
                      </Td>
                      <Td>{action}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            </Panel>
          ) : null}

          {section === "about" ? (
            <div className="space-y-3">
              <Panel>
                <SectionHeader title="About" subtitle="Train. Tune. Test. Build." />
                <KeyValue
                  items={[
                    ["Application", `ZeqouXTraining ${appInfo?.version ?? ""}`],
                    ["Build", appInfo?.packaged ? "packaged" : "development"],
                    ["Electron", appInfo?.electron ?? "—"],
                    ["Node", appInfo?.node ?? "—"],
                    ["Platform", appInfo?.platform ?? "—"],
                    ["Engine folder", appInfo?.engineDir ?? "—"],
                    ["App data", appInfo?.userData ?? "—"],
                  ]}
                  columns={1}
                />
                <div className="mt-3 flex gap-2">
                  <Button size="sm" variant="subtle" onClick={() => router.navigate("/docs")} icon={<BookOpen size={12} />}>
                    Open documentation
                  </Button>
                  <Button size="sm" variant="subtle" onClick={() => void api.notify("ZeqouXTraining", "Notifications work on this system.")}>
                    Test a notification
                  </Button>
                </div>
              </Panel>

              <Panel>
                <SectionHeader title="Local API and extensions" subtitle="What this build can hand to other Zeqou tools." />
                {tools.error ? <ErrorPanel error={tools.error} onRetry={() => void tools.reload()} /> : null}
                {tools.data ? (
                  <KeyValue
                    items={[
                      ["Local API schema", tools.data.local_api?.schema ?? "—"],
                      ["Endpoint publisher", tools.data.local_api?.published ? JSON.stringify(tools.data.local_api.published) : "not published"],
                      ["Tool calling", tools.data.tool_calling?.reason ?? "—"],
                      ["Plugin folders", (tools.data.plugins?.roots ?? []).join(", ") || "none"],
                      ["Plugins found", (tools.data.plugins?.found ?? []).length],
                      ["llama.cpp converter", tools.data.packages?.gguf_converter?.available ? tools.data.packages.gguf_converter.path : "not found"],
                      ["bitsandbytes", tools.data.packages?.bitsandbytes?.available ? "installed" : "not installed"],
                    ]}
                    columns={1}
                  />
                ) : (
                  <Loading lines={3} />
                )}
                <div className="mt-3 text-2xs text-ink-3">{tools.data?.tool_calling?.note}</div>
              </Panel>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
