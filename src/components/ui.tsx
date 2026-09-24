/** Shared UI primitives: compact, keyboard friendly, no decoration for its own sake. */
import clsx, { type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { AlertTriangle, Check, ChevronDown, Copy, Info, Loader2, X } from "lucide-react";
import {
  createContext,
  useContext,
  useEffect,
  useId,
  useRef,
  useState,
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
} from "react";

export function cx(...values: ClassValue[]): string {
  return twMerge(clsx(values));
}

/* ------------------------------------------------------------------ text --- */

export function Panel({
  children,
  className,
  quiet,
  padded = true,
  ...rest
}: { children: ReactNode; className?: string; quiet?: boolean; padded?: boolean } & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cx(quiet ? "panel-quiet" : "panel", padded && "p-4", "animate-fade-in", className)}
      {...rest}
    >
      {children}
    </div>
  );
}

export function SectionHeader({
  title,
  subtitle,
  actions,
  icon,
}: {
  title: string;
  subtitle?: ReactNode;
  actions?: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <div className="mb-3 flex items-start justify-between gap-4">
      <div className="min-w-0">
        <h2 className="flex items-center gap-2 text-[13px] font-semibold tracking-tight text-ink-0">
          {icon}
          {title}
        </h2>
        {subtitle ? <p className="mt-0.5 text-xs text-ink-2">{subtitle}</p> : null}
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
    </div>
  );
}

export function PageHeader({
  title,
  description,
  actions,
  breadcrumb,
}: {
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
  breadcrumb?: ReactNode;
}) {
  return (
    <header className="mb-4 flex flex-wrap items-end justify-between gap-3 border-b border-line-soft pb-3">
      <div className="min-w-0">
        {breadcrumb ? <div className="mb-1 text-2xs uppercase tracking-wider text-ink-3">{breadcrumb}</div> : null}
        <h1 className="text-lg font-semibold tracking-tight">{title}</h1>
        {description ? <p className="mt-1 max-w-3xl text-xs leading-relaxed text-ink-2">{description}</p> : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </header>
  );
}

export function Mono({ children, className }: { children: ReactNode; className?: string }) {
  return <span className={cx("font-mono text-xs", className)}>{children}</span>;
}

/* --------------------------------------------------------------- controls --- */

type ButtonVariant = "primary" | "default" | "ghost" | "danger" | "subtle";

export function Button({
  variant = "default",
  size = "md",
  loading,
  icon,
  children,
  className,
  ...rest
}: {
  variant?: ButtonVariant;
  size?: "sm" | "md";
  loading?: boolean;
  icon?: ReactNode;
} & ButtonHTMLAttributes<HTMLButtonElement>) {
  const variants: Record<ButtonVariant, string> = {
    primary: "bg-accent text-accent-ink hover:brightness-110 border border-transparent",
    default: "border border-line-soft bg-surface-2 text-ink-0 hover:bg-surface-3",
    ghost: "border border-transparent text-ink-1 hover:bg-surface-2 hover:text-ink-0",
    danger: "border border-danger/40 bg-danger/10 text-danger hover:bg-danger/20",
    subtle: "border border-line-soft bg-transparent text-ink-1 hover:bg-surface-2",
  };
  return (
    <button
      className={cx(
        "inline-flex items-center justify-center gap-1.5 rounded-md font-medium transition-all duration-150",
        "disabled:cursor-not-allowed disabled:opacity-45",
        size === "sm" ? "px-2 py-1 text-2xs" : "px-2.5 py-1.5 text-xs",
        variants[variant],
        className,
      )}
      disabled={rest.disabled || loading}
      {...rest}
    >
      {loading ? <Loader2 size={size === "sm" ? 11 : 13} className="animate-spin" /> : icon}
      {children}
    </button>
  );
}

export function IconButton({
  title,
  icon,
  className,
  ...rest
}: { title: string; icon: ReactNode } & ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      title={title}
      aria-label={title}
      className={cx(
        "inline-flex h-6 w-6 items-center justify-center rounded-md text-ink-2 transition-colors",
        "hover:bg-surface-3 hover:text-ink-0 disabled:opacity-40",
        className,
      )}
      {...rest}
    >
      {icon}
    </button>
  );
}

export function Badge({
  children,
  tone = "muted",
  className,
  title,
}: {
  children: ReactNode;
  tone?: "ok" | "warn" | "danger" | "info" | "accent" | "muted";
  className?: string;
  title?: string;
}) {
  const tones: Record<string, string> = {
    ok: "bg-ok/12 text-ok border-ok/25",
    warn: "bg-warn/12 text-warn border-warn/25",
    danger: "bg-danger/12 text-danger border-danger/25",
    info: "bg-info/12 text-info border-info/25",
    accent: "bg-accent/12 text-accent border-accent/30",
    muted: "bg-surface-3 text-ink-2 border-line-soft",
  };
  return (
    <span
      title={title}
      className={cx(
        "inline-flex items-center gap-1 whitespace-nowrap rounded border px-1.5 py-0.5 text-2xs font-medium",
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function Dot({ tone = "muted", pulse }: { tone?: "ok" | "warn" | "danger" | "info" | "muted"; pulse?: boolean }) {
  const tones: Record<string, string> = {
    ok: "bg-ok",
    warn: "bg-warn",
    danger: "bg-danger",
    info: "bg-info",
    muted: "bg-ink-3",
  };
  return <span className={cx("inline-block h-1.5 w-1.5 shrink-0 rounded-full", tones[tone], pulse && "animate-pulse-soft")} />;
}

export function Stat({
  label,
  value,
  hint,
  tone = "muted",
  estimated,
  className,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "ok" | "warn" | "danger" | "info" | "muted";
  estimated?: boolean;
  className?: string;
}) {
  return (
    <div className={cx("rounded-lg border border-line-soft bg-surface-2 px-3 py-2", className)}>
      <div className="flex items-center gap-1.5 text-2xs uppercase tracking-wide text-ink-3">
        {label}
        {estimated ? <Badge tone="muted" title="This value is an estimate, not a measurement">est</Badge> : null}
      </div>
      <div className="mt-1 flex items-baseline gap-1.5">
        <span className={cx("text-base font-semibold tabular-nums", tone === "danger" && "text-danger", tone === "warn" && "text-warn", tone === "ok" && "text-ok")}>
          {value}
        </span>
        {hint ? <span className="text-2xs text-ink-2">{hint}</span> : null}
      </div>
    </div>
  );
}

export function Field({
  label,
  hint,
  help,
  children,
  className,
  htmlFor,
}: {
  label: string;
  hint?: ReactNode;
  help?: ReactNode;
  children: ReactNode;
  className?: string;
  htmlFor?: string;
}) {
  return (
    <label className={cx("block", className)} htmlFor={htmlFor}>
      <div className="mb-1 flex items-center gap-1.5">
        <span className="text-2xs font-medium uppercase tracking-wide text-ink-2">{label}</span>
        {help ? <HelpTip content={help} /> : null}
      </div>
      {children}
      {hint ? <div className="mt-1 text-2xs leading-relaxed text-ink-3">{hint}</div> : null}
    </label>
  );
}

export function TextInput({ className, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={cx("field", className)} {...rest} />;
}

/** Alias kept for pages that predate the rename. */
export const Input = TextInput;

export function NumberInput({
  value,
  onChange,
  step = 1,
  min,
  max,
  className,
}: {
  value: number | "";
  onChange: (value: number | "") => void;
  step?: number;
  min?: number;
  max?: number;
  className?: string;
}) {
  return (
    <input
      type="number"
      className={cx("field tabular-nums", className)}
      value={value}
      step={step}
      min={min}
      max={max}
      onChange={(event) => {
        const raw = event.target.value;
        onChange(raw === "" ? "" : Number(raw));
      }}
    />
  );
}

export function Select({
  className,
  children,
  ...rest
}: SelectHTMLAttributes<HTMLSelectElement> & { children: ReactNode }) {
  return (
    <div className="relative">
      <select className={cx("field appearance-none pr-7", className)} {...rest}>
        {children}
      </select>
      <ChevronDown size={13} className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-ink-3" />
    </div>
  );
}

export function Toggle({
  checked,
  onChange,
  label,
  hint,
  disabled,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
  label: ReactNode;
  hint?: ReactNode;
  disabled?: boolean;
}) {
  return (
    <label className={cx("flex cursor-pointer items-start gap-2.5", disabled && "cursor-not-allowed opacity-50")}>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={cx(
          "mt-0.5 h-4 w-7 shrink-0 rounded-full border transition-colors",
          checked ? "border-accent/50 bg-accent/80" : "border-line-strong bg-surface-3",
        )}
      >
        <span
          className={cx(
            "block h-3 w-3 rounded-full bg-surface-1 transition-transform",
            checked ? "translate-x-3.5" : "translate-x-0.5",
          )}
        />
      </button>
      <span className="min-w-0">
        <span className="block text-xs text-ink-0">{label}</span>
        {hint ? <span className="mt-0.5 block text-2xs leading-relaxed text-ink-3">{hint}</span> : null}
      </span>
    </label>
  );
}

/* ------------------------------------------------------------------ tabs --- */

const TabsContext = createContext<{ value: string; setValue: (value: string) => void } | null>(null);

export function Tabs({ value, onChange, children }: { value: string; onChange: (value: string) => void; children: ReactNode }) {
  return (
    <TabsContext.Provider value={{ value, setValue: onChange }}>
      <div className="flex flex-wrap gap-1 border-b border-line-soft">{children}</div>
    </TabsContext.Provider>
  );
}

export function Tab({ value, children, count }: { value: string; children: ReactNode; count?: number }) {
  const context = useContext(TabsContext);
  const active = context?.value === value;
  return (
    <button
      type="button"
      onClick={() => context?.setValue(value)}
      className={cx(
        "-mb-px border-b-2 px-3 py-1.5 text-xs transition-colors",
        active ? "border-accent text-ink-0" : "border-transparent text-ink-2 hover:text-ink-0",
      )}
    >
      {children}
      {count !== undefined ? <span className="ml-1.5 text-2xs text-ink-3">{count}</span> : null}
    </button>
  );
}

/* ---------------------------------------------------------------- helpers --- */

export function Callout({
  tone = "info",
  title,
  children,
  hint,
  detail,
  onRetry,
}: {
  tone?: "info" | "ok" | "warn" | "danger";
  title: ReactNode;
  children?: ReactNode;
  hint?: ReactNode;
  detail?: string;
  onRetry?: () => void;
}) {
  const icons = {
    info: <Info size={14} />,
    ok: <Check size={14} />,
    warn: <AlertTriangle size={14} />,
    danger: <AlertTriangle size={14} />,
  } as const;
  const tones = {
    info: "border-info/30 bg-info/8 text-info",
    ok: "border-ok/30 bg-ok/8 text-ok",
    warn: "border-warn/30 bg-warn/8 text-warn",
    danger: "border-danger/30 bg-danger/8 text-danger",
  } as const;
  const [open, setOpen] = useState(false);
  return (
    <div className={cx("rounded-lg border px-3 py-2.5", tones[tone])}>
      <div className="flex items-start gap-2">
        <span className="mt-0.5">{icons[tone]}</span>
        <div className="min-w-0 flex-1">
          <div className="text-xs font-medium">{title}</div>
          {children ? <div className="mt-1 text-xs leading-relaxed text-ink-1">{children}</div> : null}
          {hint ? <div className="mt-1 text-xs leading-relaxed text-ink-2">What to do: {hint}</div> : null}
          <div className="mt-1.5 flex flex-wrap items-center gap-2">
            {onRetry ? (
              <Button size="sm" variant="subtle" onClick={onRetry}>
                Retry
              </Button>
            ) : null}
            {detail ? (
              <Button size="sm" variant="ghost" onClick={() => setOpen((value) => !value)}>
                {open ? "Hide details" : "Show details"}
              </Button>
            ) : null}
            {detail ? <CopyButton value={detail} label="Copy diagnostics" /> : null}
          </div>
          {open && detail ? (
            <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap rounded border border-line-soft bg-surface-1 p-2 font-mono text-2xs text-ink-1">
              {detail}
            </pre>
          ) : null}
        </div>
      </div>
    </div>
  );
}

export function CopyButton({ value, label = "Copy" }: { value: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Button
      size="sm"
      variant="ghost"
      icon={copied ? <Check size={11} /> : <Copy size={11} />}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(value);
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1600);
        } catch {
          setCopied(false);
        }
      }}
    >
      {copied ? "Copied" : label}
    </Button>
  );
}

export function HelpTip({ content, label = "Parameter help" }: { content: ReactNode; label?: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (event: MouseEvent) => {
      if (!ref.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);
  return (
    <span ref={ref} className="relative inline-flex">
      <button
        type="button"
        aria-label={label}
        onClick={() => setOpen((value) => !value)}
        className="inline-flex h-3.5 w-3.5 items-center justify-center rounded-full border border-line-strong text-[9px] text-ink-3 hover:border-accent/60 hover:text-accent"
      >
        ?
      </button>
      {open ? (
        <div className="absolute left-0 top-5 z-40 w-80 animate-scale-in rounded-lg border border-line-soft bg-surface-1 p-3 text-2xs leading-relaxed text-ink-1 shadow-xl">
          {content}
        </div>
      ) : null}
    </span>
  );
}

export function EmptyState({
  title,
  children,
  hint,
  action,
  icon,
}: {
  title: string;
  children?: ReactNode;
  /** Short line of guidance shown under the title. */
  hint?: ReactNode;
  action?: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-line-soft bg-surface-1/50 px-6 py-10 text-center">
      {icon ? <div className="text-ink-3">{icon}</div> : null}
      <div className="text-xs font-medium text-ink-1">{title}</div>
      {hint ? <div className="max-w-lg text-2xs leading-relaxed text-ink-2">{hint}</div> : null}
      {children ? <div className="max-w-lg text-xs leading-relaxed text-ink-3">{children}</div> : null}
      {action ? <div className="mt-1">{action}</div> : null}
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-xs text-ink-2">
      <Loader2 size={13} className="animate-spin" />
      {label}
    </span>
  );
}

export function SkeletonBlock({ lines = 3, className }: { lines?: number; className?: string }) {
  return (
    <div className={cx("space-y-2", className)}>
      {Array.from({ length: lines }).map((_, index) => (
        <div key={index} className="skeleton h-3" style={{ width: `${100 - index * 12}%` }} />
      ))}
    </div>
  );
}

export function ProgressBar({
  value,
  max = 100,
  tone = "accent",
  label,
  className,
}: {
  value: number;
  max?: number;
  tone?: "accent" | "ok" | "warn" | "danger";
  label?: ReactNode;
  className?: string;
}) {
  const ratio = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
  const tones = {
    accent: "bg-accent",
    ok: "bg-ok",
    warn: "bg-warn",
    danger: "bg-danger",
  } as const;
  return (
    <div className={className}>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-3">
        <div className={cx("h-full rounded-full transition-all duration-500", tones[tone])} style={{ width: `${ratio * 100}%` }} />
      </div>
      {label ? <div className="mt-1 text-2xs text-ink-2">{label}</div> : null}
    </div>
  );
}

export function Table({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className="overflow-auto rounded-lg border border-line-soft">
      <table className={cx("w-full border-collapse text-xs", className)}>{children}</table>
    </div>
  );
}

export function Th({ children, className, align = "left" }: { children?: ReactNode; className?: string; align?: "left" | "right" | "center" }) {
  return (
    <th
      className={cx(
        "sticky top-0 z-10 whitespace-nowrap border-b border-line-soft bg-surface-2 px-2.5 py-1.5 text-2xs font-medium uppercase tracking-wide text-ink-3",
        align === "right" && "text-right",
        align === "center" && "text-center",
        align === "left" && "text-left",
        className,
      )}
    >
      {children}
    </th>
  );
}

export function Td({
  children,
  className,
  align = "left",
  colSpan,
  title,
}: {
  children?: ReactNode;
  className?: string;
  align?: "left" | "right" | "center";
  colSpan?: number;
  title?: string;
}) {
  return (
    <td
      colSpan={colSpan}
      title={title}
      className={cx(
        "border-b border-line-soft/60 px-2.5 py-1.5 align-middle",
        align === "right" && "text-right tabular-nums",
        align === "center" && "text-center",
        className,
      )}
    >
      {children}
    </td>
  );
}

export function KeyValue({ items, columns = 2 }: { items: Array<[ReactNode, ReactNode]>; columns?: number }) {
  return (
    <div className={cx("grid gap-x-6 gap-y-1.5", columns === 2 ? "grid-cols-1 md:grid-cols-2" : "grid-cols-1")}>
      {items.map(([key, value], index) => (
        <div key={index} className="flex items-baseline justify-between gap-3 border-b border-line-soft/50 py-1">
          <span className="text-2xs uppercase tracking-wide text-ink-3">{key}</span>
          <span className="min-w-0 truncate text-right text-xs text-ink-0" title={typeof value === "string" ? value : undefined}>
            {value}
          </span>
        </div>
      ))}
    </div>
  );
}

export function CodeBlock({ children, className, max = "max-h-80" }: { children: ReactNode; className?: string; max?: string }) {
  return (
    <pre className={cx("overflow-auto rounded-lg border border-line-soft bg-surface-2 p-2.5 font-mono text-2xs leading-relaxed text-ink-1", max, className)}>
      {children}
    </pre>
  );
}

export function Modal({
  open,
  onClose,
  title,
  children,
  footer,
  width = "max-w-2xl",
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  width?: string;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/45 p-6 backdrop-blur-[2px] animate-fade-in" onMouseDown={onClose}>
      <div
        className={cx("mt-10 w-full animate-slide-up rounded-xl border border-line-soft bg-surface-1 shadow-2xl", width)}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-line-soft px-4 py-2.5">
          <div className="text-xs font-semibold">{title}</div>
          <IconButton title="Close" icon={<X size={13} />} onClick={onClose} />
        </div>
        <div className="max-h-[70vh] overflow-auto p-4">{children}</div>
        {footer ? <div className="flex items-center justify-end gap-2 border-t border-line-soft px-4 py-2.5">{footer}</div> : null}
      </div>
    </div>
  );
}

export function useDebounced<T>(value: T, delay = 300): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delay);
    return () => window.clearTimeout(timer);
  }, [value, delay]);
  return debounced;
}

export function useIdSafe(prefix: string): string {
  const id = useId();
  return `${prefix}-${id.replace(/[:]/g, "")}`;
}
