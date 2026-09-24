import {
  Activity,
  Boxes,
  Brain,
  Cpu,
  Database,
  FileStack,
  FlaskConical,
  FolderTree,
  GitCompare,
  HardDrive,
  Layers,
  ListChecks,
  MessageSquare,
  Package,
  Rocket,
  Server,
  Settings,
  ShieldCheck,
  Terminal,
  BookOpen,
  Gauge,
} from "lucide-react";
import type { ReactNode } from "react";

export interface NavItem {
  id: string;
  label: string;
  path: string;
  icon: ReactNode;
  group: string;
  description: string;
  keywords?: string[];
}

export const NAV_ITEMS: NavItem[] = [
  {
    id: "dashboard",
    label: "Dashboard",
    path: "/dashboard",
    icon: <Gauge size={14} />,
    group: "Overview",
    description: "Live machine state, recent models, datasets and runs",
    keywords: ["home", "overview", "start"],
  },
  {
    id: "hardware",
    label: "Hardware",
    path: "/hardware",
    icon: <Cpu size={14} />,
    group: "Overview",
    description: "CPU, GPU, memory, discs and capability matrix",
    keywords: ["gpu", "vram", "cuda", "temperature", "monitor"],
  },
  {
    id: "environment",
    label: "Environment",
    path: "/environment",
    icon: <Package size={14} />,
    group: "Overview",
    description: "Python interpreters, installed packages, ML runtime install",
    keywords: ["python", "torch", "pip", "venv", "dependencies"],
  },
  {
    id: "models",
    label: "Models",
    path: "/models",
    icon: <Brain size={14} />,
    group: "Assets",
    description: "Model library with full architecture inspection",
    keywords: ["checkpoint", "inspect", "safetensors", "gguf", "lora"],
  },
  {
    id: "datasets",
    label: "Datasets",
    path: "/datasets",
    icon: <Database size={14} />,
    group: "Assets",
    description: "Import, validate, clean, split and export datasets",
    keywords: ["jsonl", "csv", "parquet", "clean", "split"],
  },
  {
    id: "files",
    label: "Files",
    path: "/files",
    icon: <FolderTree size={14} />,
    group: "Assets",
    description: "Workspace browser with real sizes and storage reporting",
    keywords: ["browser", "folders", "storage"],
  },
  {
    id: "projects",
    label: "Projects",
    path: "/projects",
    icon: <Boxes size={14} />,
    group: "Assets",
    description: "Group models, datasets and runs into projects",
    keywords: ["workspace", "organisation"],
  },
  {
    id: "training",
    label: "Training",
    path: "/training",
    icon: <Activity size={14} />,
    group: "Training",
    description: "Configure, launch and monitor real training runs",
    keywords: ["train", "finetune", "lora", "qlora", "sft", "resume"],
  },
  {
    id: "experiments",
    label: "Experiments",
    path: "/experiments",
    icon: <FlaskConical size={14} />,
    group: "Training",
    description: "Every run with its configuration, lineage and metrics",
    keywords: ["runs", "history", "lineage", "compare"],
  },
  {
    id: "evaluation",
    label: "Evaluation",
    path: "/evaluation",
    icon: <ListChecks size={14} />,
    group: "Training",
    description: "Measure real loss and perplexity on real data",
    keywords: ["perplexity", "loss", "benchmark", "score"],
  },
  {
    id: "playground",
    label: "Playground",
    path: "/playground",
    icon: <MessageSquare size={14} />,
    group: "Tools",
    description: "Chat with a trained or local model, token by token",
    keywords: ["chat", "inference", "test", "stream"],
  },
  {
    id: "compare",
    label: "Compare",
    path: "/compare",
    icon: <GitCompare size={14} />,
    group: "Tools",
    description: "Side by side generation with measured latency",
    keywords: ["diff", "before after", "blind"],
  },
  {
    id: "adapters",
    label: "Adapters",
    path: "/adapters",
    icon: <Layers size={14} />,
    group: "Tools",
    description: "LoRA/PEFT workspace: train, merge, test, export",
    keywords: ["lora", "peft", "merge"],
  },
  {
    id: "quantization",
    label: "Quantization",
    path: "/quantization",
    icon: <HardDrive size={14} />,
    group: "Tools",
    description: "Reduce precision with real conversion tools",
    keywords: ["int4", "int8", "gguf", "awq", "gptq"],
  },
  {
    id: "conversion",
    label: "Conversion",
    path: "/conversion",
    icon: <FileStack size={14} />,
    group: "Tools",
    description: "Format conversion with integrity checks",
    keywords: ["safetensors", "gguf", "export"],
  },
  {
    id: "deploy",
    label: "Deploy",
    path: "/deploy",
    icon: <Server size={14} />,
    group: "Tools",
    description: "Serve a model locally and inspect requests",
    keywords: ["server", "api", "openai", "endpoint"],
  },
  {
    id: "jobs",
    label: "Jobs",
    path: "/jobs",
    icon: <ShieldCheck size={14} />,
    group: "System",
    description: "Every engine process: state, logs, resources",
    keywords: ["processes", "queue", "logs"],
  },
  {
    id: "settings",
    label: "Settings",
    path: "/settings",
    icon: <Settings size={14} />,
    group: "System",
    description: "Workspace, interpreter, theme, notifications",
    keywords: ["preferences", "config", "theme"],
  },
  {
    id: "docs",
    label: "Documentation",
    path: "/docs",
    icon: <BookOpen size={14} />,
    group: "System",
    description: "In-app explanations of parameters and workflows",
    keywords: ["help", "guide", "parameters"],
  },
];

export const NAV_GROUPS = ["Overview", "Assets", "Training", "Tools", "System"];

export function navItemFor(path: string): NavItem {
  const id = path.split("/").filter(Boolean)[0] ?? "dashboard";
  return NAV_ITEMS.find((item) => item.id === id) ?? NAV_ITEMS[0];
}

export const QUICK_ACTIONS = [
  { label: "Import model", path: "/models?action=import", icon: <Brain size={13} />, keywords: ["model", "add"] },
  { label: "Import dataset", path: "/datasets?action=import", icon: <Database size={13} />, keywords: ["data", "add"] },
  { label: "New training run", path: "/training/new", icon: <Rocket size={13} />, keywords: ["train", "start"] },
  { label: "Resume training", path: "/training?filter=resumable", icon: <Activity size={13} />, keywords: ["resume"] },
  { label: "Open playground", path: "/playground", icon: <MessageSquare size={13} />, keywords: ["chat"] },
  { label: "Run evaluation", path: "/evaluation?action=run", icon: <ListChecks size={13} />, keywords: ["perplexity"] },
  { label: "Hardware center", path: "/hardware", icon: <Cpu size={13} />, keywords: ["gpu"] },
  { label: "Open terminal log", path: "/jobs", icon: <Terminal size={13} />, keywords: ["logs"] },
];
