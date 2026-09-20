import type {
  ButtonHTMLAttributes,
  InputHTMLAttributes,
  ReactNode,
  SelectHTMLAttributes,
  TextareaHTMLAttributes,
} from "react";
import { AlertTriangle, Check, Info, Loader2, X } from "lucide-react";

import { cn } from "@/lib/utils";

/* ------------------------------------------------------------------ Button */

type ButtonVariant = "primary" | "secondary" | "ghost" | "quiet" | "danger";
type ButtonSize = "sm" | "md" | "lg";

const VARIANTS: Record<ButtonVariant, string> = {
  primary:
    "bg-[var(--text)] text-[var(--bg)] border border-transparent hover:opacity-90 active:opacity-80",
  secondary:
    "bg-[var(--panel-2)] text-[var(--text)] border border-[var(--border)] hover:bg-[var(--hover)]",
  ghost:
    "bg-transparent text-[var(--text-2)] border border-transparent hover:text-[var(--text)] hover:bg-[var(--hover)]",
  quiet:
    "bg-transparent text-[var(--text-2)] border border-[var(--border)] hover:border-[var(--border-strong)] hover:text-[var(--text)]",
  danger:
    "bg-transparent text-[var(--red)] border border-[rgba(224,108,117,0.4)] hover:bg-[rgba(224,108,117,0.12)]",
};

const SIZES: Record<ButtonSize, string> = {
  sm: "h-7 px-2.5 text-[12px] gap-1.5 rounded-[8px]",
  md: "h-9 px-3.5 text-[13px] gap-2 rounded-[10px]",
  lg: "h-11 px-5 text-[14px] gap-2 rounded-[12px]",
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  busy?: boolean;
  /** Alias of `busy` — reads better at call sites. */
  loading?: boolean;
  icon?: ReactNode;
}

export function Button({
  variant = "secondary",
  size = "md",
  busy = false,
  loading,
  icon,
  className,
  children,
  disabled,
  ...rest
}: ButtonProps) {
  return (
    <button
      type="button"
      disabled={disabled || busy || loading}
      className={cn(
        "inline-flex select-none items-center justify-center whitespace-nowrap font-medium",
        "transition-all duration-[130ms] ease-[cubic-bezier(0.2,0.7,0.25,1)]",
        "disabled:cursor-not-allowed disabled:opacity-50",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--acc-soft)]",
        SIZES[size],
        VARIANTS[variant],
        className,
      )}
      {...rest}
    >
      {busy || loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : icon}
      {children}
    </button>
  );
}

export function IconButton({
  className,
  children,
  title,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      title={title}
      className={cn(
        "inline-flex h-8 w-8 items-center justify-center rounded-[8px] border border-transparent",
        "text-[var(--text-2)] transition-colors duration-[130ms]",
        "hover:bg-[var(--hover)] hover:text-[var(--text)]",
        "disabled:cursor-not-allowed disabled:opacity-40",
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}

/* ------------------------------------------------------------------- Badge */

type BadgeTone = "neutral" | "good" | "warn" | "bad" | "accent" | "info";

const BADGE_TONES: Record<BadgeTone, string> = {
  neutral: "text-[var(--text-2)] border-[var(--border)]",
  good: "text-[var(--green)] border-[rgba(95,191,143,0.4)]",
  warn: "text-[var(--amber)] border-[rgba(217,164,91,0.4)]",
  bad: "text-[var(--red)] border-[rgba(224,108,117,0.4)]",
  accent: "text-[var(--acc)] border-[var(--acc-soft)]",
  info: "text-[var(--blue)] border-[rgba(122,165,248,0.4)]",
};

export function Badge({
  tone = "neutral",
  className,
  children,
}: {
  tone?: BadgeTone;
  className?: string;
  children: ReactNode;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border bg-transparent px-2 py-[2px]",
        "text-[11px] font-medium leading-[16px]",
        BADGE_TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function Dot({
  tone = "neutral",
}: {
  tone?: "good" | "warn" | "bad" | "neutral" | "accent";
}) {
  const colors: Record<string, string> = {
    good: "bg-[var(--green)]",
    warn: "bg-[var(--amber)]",
    bad: "bg-[var(--red)]",
    accent: "bg-[var(--acc)]",
    neutral: "bg-[var(--text-3)]",
  };
  return <span className={cn("inline-block h-1.5 w-1.5 shrink-0 rounded-full", colors[tone])} />;
}

/* ------------------------------------------------------------------ Panels */

export function Panel({
  className,
  children,
  padded = true,
}: {
  className?: string;
  children: ReactNode;
  padded?: boolean;
}) {
  return (
    <section
      className={cn(
        "rounded-[14px] border border-[var(--border-soft)] bg-[var(--panel)]",
        padded && "p-4",
        className,
      )}
    >
      {children}
    </section>
  );
}

export function PanelHeader({
  title,
  description,
  icon,
  actions,
  className,
}: {
  title: ReactNode;
  description?: ReactNode;
  icon?: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <header className={cn("mb-3 flex items-start justify-between gap-3", className)}>
      <div className="flex min-w-0 items-start gap-2.5">
        {icon ? <span className="mt-[2px] shrink-0 text-[var(--text-3)]">{icon}</span> : null}
        <div className="min-w-0">
          <h3 className="truncate text-[13px] font-semibold leading-5">{title}</h3>
          {description ? (
            <p className="mt-0.5 text-[12px] leading-[18px] text-[var(--text-2)]">{description}</p>
          ) : null}
        </div>
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-1.5">{actions}</div> : null}
    </header>
  );
}

/* ------------------------------------------------------------------ Fields */

export function Field({
  label,
  hint,
  aside,
  children,
  className,
}: {
  label: ReactNode;
  hint?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <label className={cn("block", className)}>
      <span className="mb-1.5 flex items-baseline justify-between gap-2">
        <span className="text-[12px] font-medium text-[var(--text-2)]">{label}</span>
        {aside}
      </span>
      {children}
      {hint ? (
        <span className="mt-1 block text-[11px] leading-4 text-[var(--text-3)]">{hint}</span>
      ) : null}
    </label>
  );
}

export function Input({ className, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={cn("zq-input", className)} {...rest} />;
}

export function Textarea({ className, ...rest }: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea className={cn("zq-input", className)} {...rest} />;
}

export function Select({ className, children, ...rest }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select className={cn("zq-input", className)} {...rest}>
      {children}
    </select>
  );
}

export function Switch({
  checked,
  onChange,
  label,
  description,
  disabled,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
  label?: ReactNode;
  description?: ReactNode;
  disabled?: boolean;
}) {
  return (
    <div className={cn("flex items-start gap-3", disabled && "opacity-55")}>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={cn(
          "relative mt-[3px] h-[18px] w-[32px] shrink-0 rounded-full border transition-colors duration-200",
          checked
            ? "border-[var(--acc)] bg-[var(--acc-soft)]"
            : "border-[var(--border)] bg-[var(--panel-2)]",
        )}
      >
        <span
          className={cn(
            "absolute top-[2px] h-[12px] w-[12px] rounded-full transition-all duration-200 ease-[var(--spring)]",
            checked ? "left-[16px] bg-[var(--acc)]" : "left-[2px] bg-[var(--text-3)]",
          )}
        />
      </button>
      {label ? (
        <div className="min-w-0">
          <div className="text-[13px] leading-5">{label}</div>
          {description ? (
            <div className="text-[12px] leading-[18px] text-[var(--text-2)]">{description}</div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

/* --------------------------------------------------------------- Segmented */

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  className,
  size = "md",
}: {
  value: T;
  options: { value: T; label: ReactNode; hint?: string }[];
  onChange: (value: T) => void;
  className?: string;
  size?: "sm" | "md";
}) {
  return (
    <div
      role="tablist"
      className={cn(
        "inline-flex items-center gap-0.5 rounded-[10px] border border-[var(--border-soft)] bg-[var(--panel-2)] p-0.5",
        className,
      )}
    >
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="tab"
            aria-selected={active}
            title={option.hint}
            onClick={() => onChange(option.value)}
            className={cn(
              "rounded-[8px] font-medium transition-all duration-[160ms] ease-[cubic-bezier(0.2,0.7,0.25,1)]",
              size === "sm" ? "px-2.5 py-[3px] text-[11.5px]" : "px-3 py-1.5 text-[12.5px]",
              active
                ? "bg-[var(--panel)] text-[var(--text)] shadow-[0_1px_2px_rgba(0,0,0,0.25)]"
                : "text-[var(--text-3)] hover:text-[var(--text-2)]",
            )}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

/* ---------------------------------------------------------------- Progress */

export function ProgressBar({
  value,
  className,
  tone = "accent",
  height = 6,
}: {
  value: number;
  className?: string;
  tone?: "accent" | "good" | "warn" | "bad";
  height?: number;
}) {
  const colors: Record<string, string> = {
    accent: "bg-[var(--acc)]",
    good: "bg-[var(--green)]",
    warn: "bg-[var(--amber)]",
    bad: "bg-[var(--red)]",
  };
  const bounded = Math.max(0, Math.min(100, Number.isFinite(value) ? value : 0));
  return (
    <div
      role="progressbar"
      aria-valuenow={Math.round(bounded)}
      aria-valuemin={0}
      aria-valuemax={100}
      className={cn("w-full overflow-hidden rounded-full bg-[var(--panel-2)]", className)}
      style={{ height }}
    >
      <div
        className={cn("h-full rounded-full transition-[width] duration-[400ms] ease-out", colors[tone])}
        style={{ width: `${bounded}%` }}
      />
    </div>
  );
}

/* -------------------------------------------------------------------- Stat */

export function Stat({
  label,
  value,
  sub,
  tone,
  mono = true,
}: {
  label: ReactNode;
  value: ReactNode;
  sub?: ReactNode;
  tone?: "good" | "warn" | "bad" | "accent";
  mono?: boolean;
}) {
  const toneClass =
    tone === "good"
      ? "text-[var(--green)]"
      : tone === "warn"
        ? "text-[var(--amber)]"
        : tone === "bad"
          ? "text-[var(--red)]"
          : tone === "accent"
            ? "text-[var(--acc)]"
            : "";
  return (
    <div className="min-w-0">
      <div className="text-[11px] font-medium uppercase tracking-[0.06em] text-[var(--text-3)]">
        {label}
      </div>
      <div className={cn("mt-0.5 truncate text-[15px] font-semibold leading-6", mono && "zq-mono", toneClass)}>
        {value}
      </div>
      {sub ? <div className="mt-0.5 truncate text-[11.5px] text-[var(--text-3)]">{sub}</div> : null}
    </div>
  );
}

/* -------------------------------------------------------------- EmptyState */

export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
}: {
  icon?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center rounded-[14px] border border-dashed border-[var(--border)]",
        "px-6 py-12 text-center",
        className,
      )}
    >
      {icon ? <div className="mb-3 text-[var(--text-3)]">{icon}</div> : null}
      <p className="text-[13.5px] font-medium">{title}</p>
      {description ? (
        <p className="mt-1 max-w-[52ch] text-[12.5px] leading-[19px] text-[var(--text-2)]">
          {description}
        </p>
      ) : null}
      {action ? <div className="mt-4">{action}</div> : null}
    </div>
  );
}

/* -------------------------------------------------------------------- Note */

export function Note({
  tone = "info",
  title,
  children,
  actions,
  className,
}: {
  tone?: "info" | "good" | "warn" | "bad";
  title?: ReactNode;
  children?: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  const map = {
    info: { color: "var(--blue)", icon: <Info className="h-3.5 w-3.5" /> },
    good: { color: "var(--green)", icon: <Check className="h-3.5 w-3.5" /> },
    warn: { color: "var(--amber)", icon: <AlertTriangle className="h-3.5 w-3.5" /> },
    bad: { color: "var(--red)", icon: <AlertTriangle className="h-3.5 w-3.5" /> },
  } as const;

  return (
    <div
      className={cn("rounded-[12px] border p-3", className)}
      style={{
        borderColor: `color-mix(in srgb, ${map[tone].color} 34%, transparent)`,
        background: `color-mix(in srgb, ${map[tone].color} 8%, transparent)`,
      }}
    >
      <div className="flex items-start gap-2.5">
        <span className="mt-[3px] shrink-0" style={{ color: map[tone].color }}>
          {map[tone].icon}
        </span>
        <div className="min-w-0 flex-1">
          {title ? <p className="text-[12.5px] font-semibold leading-5">{title}</p> : null}
          {children ? (
            <div className="mt-0.5 text-[12px] leading-[18px] text-[var(--text-2)]">{children}</div>
          ) : null}
        </div>
        {actions ? <div className="shrink-0">{actions}</div> : null}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------- Misc */

export function Spinner({ className }: { className?: string }) {
  return <Loader2 className={cn("h-3.5 w-3.5 animate-spin text-[var(--text-3)]", className)} />;
}

export function KeyValue({
  label,
  value,
  tone,
  mono = true,
  className,
}: {
  label: ReactNode;
  value: ReactNode;
  tone?: "good" | "warn" | "bad";
  mono?: boolean;
  className?: string;
}) {
  const toneClass =
    tone === "good"
      ? "text-[var(--green)]"
      : tone === "warn"
        ? "text-[var(--amber)]"
        : tone === "bad"
          ? "text-[var(--red)]"
          : "";
  return (
    <div className={cn("flex items-baseline justify-between gap-3 py-[5px]", className)}>
      <span className="shrink-0 text-[12px] text-[var(--text-3)]">{label}</span>
      <span className={cn("min-w-0 truncate text-right text-[12.5px]", mono && "zq-mono", toneClass)}>
        {value}
      </span>
    </div>
  );
}

export function CloseButton({ onClick, title = "Close" }: { onClick: () => void; title?: string }) {
  return (
    <IconButton onClick={onClick} title={title}>
      <X className="h-3.5 w-3.5" />
    </IconButton>
  );
}
