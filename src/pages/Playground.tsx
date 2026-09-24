import { useEffect, useMemo, useRef, useState } from "react";
import { Bot, Eraser, RefreshCw, Send, Server, User } from "lucide-react";
import { api } from "../lib/api";
import { basename, number } from "../lib/format";
import { useApp } from "../state/app";
import {
  Badge,
  Button,
  Callout,
  Field,
  KeyValue,
  NumberInput,
  Panel,
  SectionHeader,
  Select,
  Stat,
  TextInput,
  Toggle,
  cx,
} from "../components/ui";
import { Loading, ModelPicker, useEngine } from "./common";

interface Message {
  role: "system" | "user" | "assistant";
  content: string;
  stats?: { tokens?: number; latency?: number; tokensPerSecond?: number };
}

export function PlaygroundPage() {
  const { toast, reportError, registry, refreshRegistry } = useApp();
  const params = useMemo(() => new URLSearchParams(window.location.hash.split("?")[1] ?? ""), []);
  const [modelPath, setModelPath] = useState<string | null>(params.get("model") ?? null);
  const [status, setStatus] = useState<any>(null);
  const [messages, setMessages] = useState<Message[]>([{ role: "user", content: "Hello! Introduce yourself in one sentence." }]);
  const [systemPrompt, setSystemPrompt] = useState("You are a helpful assistant.");
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamText, setStreamText] = useState("");
  const [loading, setLoading] = useState(false);
  const [lastResult, setLastResult] = useState<any>(null);
  const [paramsState, setParamsState] = useState({
    temperature: 0.8,
    top_p: 0.95,
    top_k: 0,
    max_tokens: 256,
    repetition_penalty: 1.0,
    seed: "" as string | number,
    stop: "",
  });
  const scrollRef = useRef<HTMLDivElement>(null);
  const backends = useEngine<any>("backends.list", {}, { timeout: 120_000 });

  useEffect(() => {
    void api.sidecar.status().then(setStatus).catch(() => setStatus(null));
  }, []);

  useEffect(
    () =>
      api.sidecar.onStream((event: any) => {
        if (event?.type === "token") setStreamText(String(event.text ?? ""));
        if (event?.type === "done" && event.result) setLastResult(event.result);
      }),
    [],
  );

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, streamText]);

  const loadModel = async () => {
    if (!modelPath) return;
    setLoading(true);
    try {
      const result = await api.sidecar.load(modelPath, undefined);
      setStatus({ state: "running", model: modelPath, ...result });
      toast({
        title: "Model loaded",
        body: `${basename(modelPath)} · backend ${result.backend}${result.parameters ? ` · ${number(result.parameters)} parameters` : ""}`,
        tone: "ok",
      });
    } catch (error: any) {
      reportError(error, "The model could not be loaded");
      setStatus(null);
    } finally {
      setLoading(false);
    }
  };

  const send = async () => {
    if (!input.trim() || streaming) return;
    const userMessage: Message = { role: "user", content: input.trim() };
    const conversation = [...messages, userMessage];
    setMessages(conversation);
    setInput("");
    setStreaming(true);
    setStreamText("");
    try {
      const payload = {
        messages: [
          ...(systemPrompt.trim() ? [{ role: "system", content: systemPrompt.trim() }] : []),
          ...conversation.map((message) => ({ role: message.role, content: message.content })),
        ],
        temperature: paramsState.temperature,
        top_p: paramsState.top_p,
        top_k: paramsState.top_k,
        max_tokens: paramsState.max_tokens,
        repetition_penalty: paramsState.repetition_penalty,
        seed: paramsState.seed === "" ? null : Number(paramsState.seed),
        stop: paramsState.stop ? paramsState.stop.split("\n").filter(Boolean) : null,
      };
      const result = await api.sidecar.generate(payload);
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: result.text ?? "",
          stats: {
            tokens: result.completion_tokens,
            latency: result.latency_seconds,
            tokensPerSecond: result.tokens_per_second,
          },
        },
      ]);
      setLastResult(result);
    } catch (error: any) {
      reportError(error, "Generation failed");
    } finally {
      setStreaming(false);
      setStreamText("");
    }
  };

  const saveConversation = async () => {
    try {
      await api.registry.add("conversations", {
        id: `${Date.now().toString(36)}`,
        model: modelPath,
        messages,
        params: paramsState,
        systemPrompt,
        createdAt: new Date().toISOString(),
      });
      await refreshRegistry();
      toast({ title: "Conversation saved", body: "Stored in the workspace registry, locally.", tone: "ok" });
    } catch (error) {
      reportError(error, "Could not save the conversation");
    }
  };

  const exportAsDataset = async () => {
    try {
      const destination = await api.dialog.saveFile({
        title: "Export conversation as training data",
        defaultPath: `${basename(modelPath ?? "chat")}-conversation.jsonl`,
        filters: [{ name: "JSONL", extensions: ["jsonl"] }],
      });
      if (!destination) return;
      const result = await api.call<any>("datasets.from_conversations", {
        destination,
        format: "jsonl",
        conversations: [{ messages }],
      });
      toast({
        title: "Conversation exported as training data",
        body: `${result.records} record(s) written to ${result.output}`,
        tone: "ok",
      });
      const report = await api.call<any>("datasets.inspect", { path: result.output, sample_size: 100 });
      await api.registry.add("datasets", {
        id: `${Date.now().toString(36)}`,
        name: basename(result.output),
        path: result.output,
        addedAt: new Date().toISOString(),
        kind: "jsonl",
        summary: { records: report.record_count, size_bytes: report.size_bytes, fields: report.field_names, tokens: report.token_estimate?.total ?? null },
      });
      await refreshRegistry();
    } catch (error) {
      reportError(error, "Export failed");
    }
  };

  const availableModels = registry?.models ?? [];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Playground</h1>
          <p className="mt-0.5 max-w-3xl text-xs text-ink-2">
            A model stays resident in a dedicated engine process, so responses stream token by token and the
            interface never blocks. Token counts and latency below are measured, not estimated.
          </p>
        </div>
        <div className="flex gap-2">
          <Button onClick={() => void loadModel()} loading={loading} disabled={!modelPath} icon={<Server size={13} />}>
            {status?.model ? "Reload model" : "Load model"}
          </Button>
          <Button
            onClick={async () => {
              await api.sidecar.unload();
              setStatus(null);
              toast({ title: "Model unloaded", body: "The inference process released the weights.", tone: "info" });
            }}
            disabled={!status}
          >
            Unload
          </Button>
        </div>
      </div>

      {!availableModels.length ? (
        <Loading label="Reading the model library…" lines={2} />
      ) : null}

      <div className="grid gap-3 lg:grid-cols-[1fr_320px]">
        <Panel padded={false} className="flex h-[560px] flex-col">
          <div className="flex items-center justify-between border-b border-line-soft p-3">
            <div className="flex items-center gap-2">
              <Badge tone={status ? "ok" : "warn"}>
                {status ? `loaded · ${status.backend ?? "backend"}` : "no model loaded"}
              </Badge>
              {status?.parameters ? <Badge tone="muted">{number(status.parameters)} parameters</Badge> : null}
              {status?.context_length ? <Badge tone="muted">context {status.context_length}</Badge> : null}
            </div>
            <div className="flex gap-2">
              <Button size="sm" variant="subtle" onClick={() => setMessages([])} icon={<Eraser size={11} />}>
                Clear
              </Button>
              <Button size="sm" variant="subtle" onClick={() => void saveConversation()} disabled={!messages.length}>
                Save conversation
              </Button>
              <Button size="sm" variant="subtle" onClick={() => void exportAsDataset()} disabled={!messages.length}>
                To dataset
              </Button>
            </div>
          </div>

          <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto p-3">
            {!messages.length ? (
              <Callout tone="info" title="Pick a model and send a message">
                Models produced by the tiny backend load in pure Python; Transformers models need the ML
                runtime installed. If a model cannot be loaded, the engine says exactly why.
              </Callout>
            ) : null}
            {messages.map((message, index) => (
              <div key={index} className={cx("flex gap-2", message.role === "user" ? "justify-end" : "justify-start")}>
                <div
                  className={cx(
                    "max-w-[80%] rounded-lg border px-3 py-2 text-xs leading-relaxed",
                    message.role === "user"
                      ? "border-accent/30 bg-accent/10 text-ink-0"
                      : "border-line-soft bg-surface-2 text-ink-1",
                  )}
                >
                  <div className="mb-1 flex items-center gap-1.5 text-2xs uppercase tracking-wide text-ink-3">
                    {message.role === "user" ? <User size={10} /> : <Bot size={10} />}
                    {message.role}
                    {message.stats?.tokens ? (
                      <span className="text-ink-3">
                        · {message.stats.tokens} tokens · {message.stats.latency?.toFixed(2)}s ·{" "}
                        {message.stats.tokensPerSecond?.toFixed(1)} tok/s
                      </span>
                    ) : null}
                  </div>
                  <div className="whitespace-pre-wrap">{message.content}</div>
                </div>
              </div>
            ))}
            {streaming ? (
              <div className="flex gap-2">
                <div className="max-w-[80%] rounded-lg border border-line-soft bg-surface-2 px-3 py-2 text-xs text-ink-1">
                  <div className="mb-1 flex items-center gap-1.5 text-2xs uppercase tracking-wide text-ink-3">
                    <RefreshCw size={10} className="animate-spin" /> streaming from the engine
                  </div>
                  <div className="whitespace-pre-wrap">{streamText || "…"}</div>
                </div>
              </div>
            ) : null}
          </div>

          <div className="border-t border-line-soft p-3">
            <div className="flex items-end gap-2">
              <textarea
                className="field min-h-[38px] flex-1 resize-y"
                rows={2}
                placeholder={status ? "Write a message… (Enter to send)" : "Load a model first"}
                value={input}
                disabled={!status || streaming}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    void send();
                  }
                }}
              />
              <Button variant="primary" icon={<Send size={13} />} disabled={!status || streaming || !input.trim()} onClick={() => void send()}>
                Send
              </Button>
            </div>
            <div className="mt-1 text-2xs text-ink-3">
              {lastResult
                ? `Last run: ${lastResult.completion_tokens} tokens in ${lastResult.latency_seconds}s (${lastResult.tokens_per_second} tok/s) on ${lastResult.backend}`
                : "No generation in this session yet."}
            </div>
          </div>
        </Panel>

        <div className="space-y-3">
          <Panel>
            <SectionHeader title="Model" />
            <div className="space-y-3">
              <ModelPicker value={modelPath} onChange={setModelPath} label="Model or checkpoint" />
              {status ? (
                <KeyValue
                  items={[
                    ["Loaded", basename(status.model ?? "")],
                    ["Backend", status.backend ?? "—"],
                    ["Context", String(status.context_length ?? "—")],
                    ["Status", status.state ?? "running"],
                  ]}
                />
              ) : null}
              {(backends.data?.backends ?? []).map((backend: any) => (
                <div key={backend.id} className="flex items-center justify-between text-2xs">
                  <span className="text-ink-2">{backend.name}</span>
                  <Badge tone={backend.available ? "ok" : "muted"}>{backend.available ? "ready" : "missing"}</Badge>
                </div>
              ))}
            </div>
          </Panel>

          <Panel>
            <SectionHeader title="Generation" subtitle="Sampling parameters passed straight to the backend" />
            <div className="space-y-3">
              <Field label="System prompt">
                <TextInput value={systemPrompt} onChange={(event) => setSystemPrompt(event.target.value)} />
              </Field>
              <div className="grid grid-cols-2 gap-2">
                <Field label="Temperature" help="0 is greedy decoding; higher values increase variety and risk.">
                  <NumberInput value={paramsState.temperature} step={0.05} min={0} max={2} onChange={(value) => setParamsState({ ...paramsState, temperature: Number(value) || 0 })} />
                </Field>
                <Field label="Top-p" help="Nucleus sampling: keep the smallest set of tokens whose probability sums to p.">
                  <NumberInput value={paramsState.top_p} step={0.01} min={0} max={1} onChange={(value) => setParamsState({ ...paramsState, top_p: Number(value) || 0 })} />
                </Field>
                <Field label="Top-k" help="0 disables top-k filtering.">
                  <NumberInput value={paramsState.top_k} min={0} onChange={(value) => setParamsState({ ...paramsState, top_k: Number(value) || 0 })} />
                </Field>
                <Field label="Max tokens">
                  <NumberInput value={paramsState.max_tokens} min={1} onChange={(value) => setParamsState({ ...paramsState, max_tokens: Number(value) || 1 })} />
                </Field>
                <Field label="Repetition penalty" help="Above 1 discourages repeating tokens that already appeared.">
                  <NumberInput value={paramsState.repetition_penalty} step={0.05} min={1} onChange={(value) => setParamsState({ ...paramsState, repetition_penalty: Number(value) || 1 })} />
                </Field>
                <Field label="Seed" help="Empty means a random seed each request.">
                  <TextInput value={String(paramsState.seed)} onChange={(event) => setParamsState({ ...paramsState, seed: event.target.value })} />
                </Field>
              </div>
              <Field label="Stop sequences" hint="One per line.">
                <TextInput value={paramsState.stop} onChange={(event) => setParamsState({ ...paramsState, stop: event.target.value })} />
              </Field>
            </div>
          </Panel>

          <Panel>
            <SectionHeader title="Session" />
            <div className="grid grid-cols-2 gap-2">
              <Stat label="Messages" value={String(messages.length)} />
              <Stat label="Saved chats" value={String((registry?.conversations ?? []).length)} />
            </div>
            <div className="mt-2">
              <Toggle
                checked={false}
                disabled
                onChange={() => undefined}
                label="Reasoning / thinking mode"
                hint="Not available: no installed backend for this model exposes a reasoning channel. The app never fabricates hidden reasoning."
              />
            </div>
            <div className="mt-2">
              <Select disabled value="none">
                <option value="none">Tool calling: not supported by this backend</option>
              </Select>
            </div>
          </Panel>

          {(registry?.conversations ?? []).length ? (
            <Panel>
              <SectionHeader title="Saved conversations" />
              <div className="space-y-1.5">
                {(registry?.conversations ?? []).slice(0, 6).map((conversation: any) => (
                  <button
                    key={conversation.id}
                    className="w-full rounded-md border border-line-soft bg-surface-2 px-2 py-1.5 text-left hover:bg-surface-3"
                    onClick={() => {
                      setMessages(conversation.messages ?? []);
                      setModelPath(conversation.model ?? modelPath);
                      setSystemPrompt(conversation.systemPrompt ?? "");
                    }}
                  >
                    <div className="truncate text-2xs">{basename(conversation.model ?? "")} · {conversation.messages?.length ?? 0} messages</div>
                    <div className="truncate text-2xs text-ink-3">
                      {(conversation.messages ?? [])[0]?.content?.slice(0, 60) ?? ""}
                    </div>
                  </button>
                ))}
              </div>
            </Panel>
          ) : null}
        </div>
      </div>
    </div>
  );
}
