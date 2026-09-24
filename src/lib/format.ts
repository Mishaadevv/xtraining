/** Formatting helpers. Every value shown here comes from real engine output. */

export function bytes(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = Math.abs(value);
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  const rounded = unit === 0 ? Math.round(size) : Number(size.toFixed(digits));
  return `${value < 0 ? "-" : ""}${rounded} ${units[unit]}`;
}

export function number(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

export function compact(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (Math.abs(value) < 1000) return number(value);
  const units = ["K", "M", "B", "T"];
  let size = value;
  let unit = -1;
  while (Math.abs(size) >= 1000 && unit < units.length - 1) {
    size /= 1000;
    unit += 1;
  }
  return `${size.toFixed(size < 10 ? 2 : 1)}${units[unit]}`;
}

export function percent(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${value.toFixed(digits)}%`;
}

export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds) || seconds < 0) return "—";
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  if (minutes < 60) return `${minutes}m ${rest}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

export function clock(iso: string | null | undefined): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function relative(iso: string | null | undefined): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const delta = (Date.now() - date.getTime()) / 1000;
  if (delta < 60) return "just now";
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  if (delta < 604800) return `${Math.floor(delta / 86400)}d ago`;
  return clock(iso);
}

export function truncate(value: string, max = 160): string {
  if (!value) return "";
  return value.length <= max ? value : `${value.slice(0, max - 1)}…`;
}

export function basename(target: string | null | undefined): string {
  if (!target) return "—";
  const parts = target.replace(/[\\/]+$/, "").split(/[\\/]/);
  return parts[parts.length - 1] || target;
}

export function jsonPreview(value: unknown, limit = 4000): string {
  try {
    const text = JSON.stringify(value, null, 2);
    return text.length > limit ? `${text.slice(0, limit)}\n…` : text;
  } catch {
    return String(value);
  }
}

export function stateTone(state: string | undefined): "ok" | "warn" | "danger" | "info" | "muted" {
  switch (state) {
    case "completed":
      return "ok";
    case "running":
    case "queued":
      return "info";
    case "paused":
    case "interrupted":
      return "warn";
    case "failed":
    case "cancelled":
      return "danger";
    default:
      return "muted";
  }
}

export function riskLabel(risk: string | undefined): { label: string; tone: "ok" | "warn" | "danger" | "muted" } {
  switch (risk) {
    case "fits":
    case "fits_cpu":
      return { label: risk === "fits_cpu" ? "Runs on CPU" : "Fits in memory", tone: "ok" };
    case "tight":
      return { label: "Tight — reduce batch size", tone: "warn" };
    case "will_not_fit":
      return { label: "Will not fit as configured", tone: "danger" };
    default:
      return { label: "Unknown", tone: "muted" };
  }
}
