import { useState } from "react";
import {
  FlaskConical,
  Play,
  Send,
  Sparkles,
  Square,
  Trash2,
} from "lucide-react";

import { PageBody, PageHeader } from "@/components/layout/Shell";
import {
  Badge,
  Button,
  Dot,
  EmptyState,
  Field,
  Input,
  KeyValue,
  Note,
  Panel,
  PanelHeader,
  Textarea,
} from "@/components/ui/primitives";
import { cn } from "@/lib/utils";
import { useStore } from "@/state/store";
import {
  appStore,
  clearPlaygroundHistory,
  generateInPlayground,
  loadPlaygroundModel,
  setPlaygroundParams,
  unloadPlaygroundModel,
} from "@/state/appStore";

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  format?: (value: number) => string;
}) {
  return (
    <Field label={label} aside={<span className="zq-mono text-[11.5px] text-[var(--text-2)]">{format ? format(value) : value}</span>}>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number.parseFloat(event.target.value))}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-[var(--panel-2)] accent-[var(--acc)]"
      />
    </Field>
  );
}

export function PlaygroundPage() {
  const { models, playground, env } = useStore(appStore);
  const [prompt, setPrompt] = useState("Explain what a LoRA adapter is in two sentences.");
  const [system, setSystem] = useState("");

  const runnable = models.filter((model) => model.kind === "trained" || model.kind === "local");
  const torchReady = Boolean(env.dependencies?.capabilities?.inference?.ready);

  const streamed = playground.generating ? playground.streamed : playground.output;

  return (
    <>
      <PageHeader
        icon={<FlaskConical className="h-4 w-4" />}
        title="Playground"
        subtitle="Load a trained model and talk to it, without leaving the app"
        actions={
          playground.loaded ? (
            <Button
              size="sm"
              variant="quiet"
              icon={<Square className="h-3.5 w-3.5" />}
              onClick={() => void unloadPlaygroundModel()}
            >
              Unload
            </Button>
          ) : null
        }
      />

      <PageBody wide>
        {!torchReady ? (
          <div className="mb-4">
            <Note tone="warn" title="Inference needs the ML runtime">
              {env.dependencies?.capabilities?.inference?.missing?.length
                ? `Missing: ${env.dependencies.capabilities.inference.missing.join(", ")}.`
                : "The Python backend could not be reached."}{" "}
              Install it in Settings → Environment.
            </Note>
          </div>
        ) : null}

        <div className="grid grid-cols-1 gap-4 xl:grid-cols-[300px_minmax(0,1fr)]">
          <div className="space-y-4">
            <Panel>
              <PanelHeader
                icon={<Sparkles className="h-4 w-4" />}
                title="Model"
                description="Trained models are loaded with their base model and adapter."
              />
              {runnable.length === 0 ? (
                <p className="text-[12.5px] text-[var(--text-3)]">
                  No trained or local models yet. Train a model, or add a local model folder in the Models
                  screen.
                </p>
              ) : (
                <div className="space-y-1.5">
                  {runnable.map((model) => {
                    const active = playground.modelDir === model.path;
                    return (
                      <button
                        key={model.id}
                        type="button"
                        disabled={!model.path || playground.loading}
                        onClick={() => model.path && void loadPlaygroundModel(model.path, model.name)}
                        className={cn(
                          "flex w-full items-center gap-2.5 rounded-[10px] border p-2.5 text-left transition-colors",
                          active
                            ? "border-[var(--acc)] bg-[var(--acc-soft)]"
                            : "border-[var(--border-soft)] hover:bg-[var(--hover)]",
                          !model.path && "cursor-not-allowed opacity-50",
                        )}
                      >
                        <span className="min-w-0 flex-1">
                          <span className="block truncate text-[12px] font-medium">{model.name}</span>
                          <span className="zq-mono block truncate text-[10.5px] text-[var(--text-3)]">
                            {model.kind === "trained" ? `${model.method} adapter` : "local model"}
                          </span>
                        </span>
                        {active && playground.loaded ? (
                          <Badge tone="good">
                            <Dot tone="good" />
                            loaded
                          </Badge>
                        ) : null}
                      </button>
                    );
                  })}
                </div>
              )}

              {playground.loading ? (
                <p className="mt-3 text-[12px] text-[var(--text-3)]">Loading weights into memory…</p>
              ) : null}
              {playground.error ? (
                <div className="mt-3">
                  <Note tone="bad" title="Could not load the model">
                    {playground.error.message}
                    {playground.error.hint ? <p className="mt-1">{playground.error.hint}</p> : null}
                  </Note>
                </div>
              ) : null}
              {playground.loaded ? (
                <div className="mt-3 border-t border-[var(--border-soft)] pt-2">
                  <KeyValue label="Device" value={env.hardware?.training_device ?? "—"} />
                  <KeyValue label="Status" value="ready" tone="good" />
                </div>
              ) : null}
            </Panel>

            <Panel>
              <PanelHeader title="Sampling" description="Applies to the next generation." />
              <div className="space-y-3">
                <Slider
                  label="Max new tokens"
                  value={playground.params.maxNewTokens}
                  min={16}
                  max={1024}
                  step={16}
                  onChange={(value) => setPlaygroundParams({ maxNewTokens: value })}
                />
                <Slider
                  label="Temperature"
                  value={playground.params.temperature}
                  min={0}
                  max={1.5}
                  step={0.05}
                  format={(value) => value.toFixed(2)}
                  onChange={(value) => setPlaygroundParams({ temperature: value })}
                />
                <Slider
                  label="Top-p"
                  value={playground.params.topP}
                  min={0.1}
                  max={1}
                  step={0.05}
                  format={(value) => value.toFixed(2)}
                  onChange={(value) => setPlaygroundParams({ topP: value })}
                />
                <Slider
                  label="Repetition penalty"
                  value={playground.params.repetitionPenalty}
                  min={1}
                  max={1.5}
                  step={0.01}
                  format={(value) => value.toFixed(2)}
                  onChange={(value) => setPlaygroundParams({ repetitionPenalty: value })}
                />
                <p className="text-[11px] leading-[17px] text-[var(--text-3)]">
                  Temperature 0 makes generation deterministic (greedy), which is the quickest way to spot
                  whether training changed the model's behaviour at all.
                </p>
              </div>
            </Panel>
          </div>

          <div className="space-y-4">
            <Panel>
              <PanelHeader
                title="Prompt"
                description="The chat template of the base tokenizer is applied automatically when one exists."
              />
              <div className="space-y-3">
                <Field label="System message (optional)">
                  <Input
                    value={system}
                    placeholder="You are a concise technical assistant."
                    onChange={(event) => setSystem(event.target.value)}
                  />
                </Field>
                <Field label="Message">
                  <Textarea
                    rows={4}
                    value={prompt}
                    placeholder="Ask the model something…"
                    onChange={(event) => setPrompt(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                        event.preventDefault();
                        void generateInPlayground(prompt, system);
                      }
                    }}
                  />
                </Field>
                <div className="flex items-center gap-2">
                  <Button
                    variant="primary"
                    icon={<Send className="h-3.5 w-3.5" />}
                    loading={playground.generating}
                    disabled={!playground.loaded || !prompt.trim()}
                    onClick={() => void generateInPlayground(prompt, system)}
                  >
                    Generate
                  </Button>
                  <span className="text-[11.5px] text-[var(--text-3)]">⌘/Ctrl + Enter</span>
                </div>
              </div>
            </Panel>

            <Panel>
              <PanelHeader
                icon={<FlaskConical className="h-4 w-4" />}
                title="Output"
                description={
                  playground.generating
                    ? "Streaming tokens as they are produced"
                    : playground.history[0]?.tokensPerSecond
                      ? `${playground.history[0].tokensPerSecond} tokens/s · ${playground.history[0].seconds}s`
                      : undefined
                }
                actions={
                  playground.history.length ? (
                    <Button
                      size="sm"
                      variant="ghost"
                      icon={<Trash2 className="h-3.5 w-3.5" />}
                      onClick={() => clearPlaygroundHistory()}
                    >
                      Clear history
                    </Button>
                  ) : null
                }
              />
              {streamed ? (
                <pre className="zq-mono max-h-[320px] min-h-[120px] overflow-auto whitespace-pre-wrap rounded-[10px] border border-[var(--border-soft)] bg-[var(--code-bg)] p-3 text-[12px] leading-[19px] text-[var(--text)]">
                  {streamed}
                  {playground.generating ? <span className="animate-pulse">▍</span> : null}
                </pre>
              ) : playground.loaded ? (
                <EmptyState
                  icon={<Play className="h-6 w-6" />}
                  title="Ready when you are"
                  description="Send a prompt to see what the trained model actually learned."
                />
              ) : (
                <EmptyState
                  icon={<Sparkles className="h-6 w-6" />}
                  title="Load a model first"
                  description="Pick a trained adapter or a local model folder on the left."
                />
              )}
            </Panel>

            {playground.history.length > 0 ? (
              <Panel>
                <PanelHeader title="History" description="This session only — nothing is written to disk." />
                <div className="space-y-2">
                  {playground.history.slice(1).map((entry, index) => (
                    <div key={index} className="rounded-[10px] border border-[var(--border-soft)] p-2.5">
                      <p className="truncate text-[12px] font-medium">{entry.prompt}</p>
                      <p className="mt-1 line-clamp-3 text-[11.5px] leading-[17px] text-[var(--text-2)]">
                        {entry.output}
                      </p>
                    </div>
                  ))}
                </div>
              </Panel>
            ) : null}
          </div>
        </div>
      </PageBody>
    </>
  );
}
