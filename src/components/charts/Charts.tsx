import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { cn } from "@/lib/utils";

/** Track an element's width so charts render at real pixel size (no distortion). */
export function useElementWidth<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const element = ref.current;
    if (!element) return undefined;
    setWidth(element.clientWidth);
    if (typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) setWidth(Math.round(entry.contentRect.width));
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return { ref, width };
}

export interface ChartPoint {
  x: number;
  y: number;
  meta?: Record<string, string | number | null>;
}

interface LineChartProps {
  points: ChartPoint[];
  height?: number;
  color?: string;
  secondary?: ChartPoint[];
  secondaryColor?: string;
  formatY?: (value: number) => string;
  formatX?: (value: number) => string;
  labelY?: string;
  labelX?: string;
  emptyMessage?: string;
  yPadRatio?: number;
  tooltipRows?: (index: number) => { label: string; value: string }[];
}

/**
 * A real line chart: computed scales, gridlines, axis labels, and a hover
 * crosshair that reports the nearest sample.
 */
export function LineChart({
  points,
  height = 200,
  color = "var(--acc)",
  secondary,
  secondaryColor = "var(--blue)",
  formatY = (value) => value.toFixed(3),
  formatX = (value) => String(Math.round(value)),
  labelY,
  labelX,
  emptyMessage = "No data yet",
  yPadRatio = 0.08,
  tooltipRows,
}: LineChartProps) {
  const { ref, width } = useElementWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);

  const pad = { top: 12, right: 14, bottom: 22, left: 46 };
  const innerWidth = Math.max(10, width - pad.left - pad.right);
  const innerHeight = Math.max(10, height - pad.top - pad.bottom);

  const scale = useMemo(() => {
    if (!points.length) return null;
    const xs = points.map((point) => point.x);
    let ys = points.map((point) => point.y);
    if (secondary && secondary.length) ys = ys.concat(secondary.map((point) => point.y));

    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    let yMin = Math.min(...ys);
    let yMax = Math.max(...ys);
    const span = yMax - yMin;
    const padY = span > 0 ? span * yPadRatio : Math.max(0.05, Math.abs(yMax) * 0.1 || 0.05);
    yMin -= padY;
    yMax += padY;

    const spanX = xMax - xMin || 1;
    const spanY = yMax - yMin || 1;

    return {
      xMin,
      xMax,
      yMin,
      yMax,
      toX: (value: number) => pad.left + ((value - xMin) / spanX) * innerWidth,
      toY: (value: number) => pad.top + innerHeight - ((value - yMin) / spanY) * innerHeight,
      ticks: [yMin, yMin + spanY / 3, yMin + (spanY * 2) / 3, yMax],
    };
  }, [points, secondary, innerWidth, innerHeight, pad.left, pad.top, yPadRatio]);

  if (!points.length || !scale) {
    return (
      <div
        ref={ref}
        className="flex items-center justify-center rounded-[10px] border border-dashed border-[var(--border)] text-[12px] text-[var(--text-3)]"
        style={{ height }}
      >
        {emptyMessage}
      </div>
    );
  }

  const linePath = points
    .map((point, index) => `${index === 0 ? "M" : "L"}${scale.toX(point.x).toFixed(2)},${scale.toY(point.y).toFixed(2)}`)
    .join(" ");

  const areaPath = `${linePath} L${scale.toX(points[points.length - 1].x).toFixed(2)},${(pad.top + innerHeight).toFixed(2)} L${scale.toX(points[0].x).toFixed(2)},${(pad.top + innerHeight).toFixed(2)} Z`;

  const secondaryPath = secondary && secondary.length
    ? secondary
        .map((point, index) => `${index === 0 ? "M" : "L"}${scale.toX(point.x).toFixed(2)},${scale.toY(point.y).toFixed(2)}`)
        .join(" ")
    : null;

  const activeIndex =
    hover == null ? null : Math.max(0, Math.min(points.length - 1, hover));
  const active = activeIndex == null ? null : points[activeIndex];
  const gradientId = `zqgrad_${Math.abs(points[0].x)}_${points.length}`;

  return (
    <div ref={ref} className="relative w-full" style={{ height }}>
      <svg
        width="100%"
        height={height}
        onMouseLeave={() => setHover(null)}
        onMouseMove={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          const x = event.clientX - rect.left;
          const ratio = (x - pad.left) / Math.max(1, innerWidth);
          setHover(Math.round(Math.max(0, Math.min(1, ratio)) * (points.length - 1)));
        }}
      >
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.22" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>

        {scale.ticks.map((tick, index) => {
          const y = scale.toY(tick);
          return (
            <g key={`tick_${index}`}>
              <line
                x1={pad.left}
                x2={pad.left + innerWidth}
                y1={y}
                y2={y}
                stroke="var(--border-soft)"
                strokeDasharray={index === 0 ? undefined : "3 4"}
              />
              <text
                x={pad.left - 8}
                y={y + 3}
                textAnchor="end"
                fontSize="10"
                fill="var(--text-3)"
                fontFamily="var(--mono)"
              >
                {formatY(tick)}
              </text>
            </g>
          );
        })}

        <path d={areaPath} fill={`url(#${gradientId})`} />
        {secondaryPath ? (
          <path d={secondaryPath} fill="none" stroke={secondaryColor} strokeWidth="1.4" strokeDasharray="4 3" opacity="0.85" />
        ) : null}
        <path d={linePath} fill="none" stroke={color} strokeWidth="1.7" strokeLinejoin="round" strokeLinecap="round" />

        {active ? (
          <g>
            <line
              x1={scale.toX(active.x)}
              x2={scale.toX(active.x)}
              y1={pad.top}
              y2={pad.top + innerHeight}
              stroke="var(--text-3)"
              strokeDasharray="3 3"
            />
            <circle cx={scale.toX(active.x)} cy={scale.toY(active.y)} r="3.2" fill={color} />
          </g>
        ) : null}

        <text x={pad.left} y={height - 6} fontSize="10" fill="var(--text-3)" fontFamily="var(--mono)">
          {formatX(points[0].x)}
        </text>
        <text
          x={pad.left + innerWidth}
          y={height - 6}
          fontSize="10"
          fill="var(--text-3)"
          textAnchor="end"
          fontFamily="var(--mono)"
        >
          {formatX(points[points.length - 1].x)}
        </text>
      </svg>

      {labelY ? (
        <span className="pointer-events-none absolute left-1 top-0 text-[10px] uppercase tracking-[0.06em] text-[var(--text-3)]">
          {labelY}
        </span>
      ) : null}
      {labelX ? (
        <span className="pointer-events-none absolute bottom-0 right-0 text-[10px] uppercase tracking-[0.06em] text-[var(--text-3)]">
          {labelX}
        </span>
      ) : null}

      {active ? (
        <div
          className="pointer-events-none absolute top-1 z-10 min-w-[132px] rounded-[8px] border border-[var(--border)] bg-[var(--panel-2)] px-2.5 py-1.5 shadow-[var(--shadow)]"
          style={{
            left: Math.max(0, Math.min(scale.toX(active.x) + 12, Math.max(0, width - 150))),
          }}
        >
          {(tooltipRows ? tooltipRows(activeIndex ?? 0) : [
            { label: labelX ?? "x", value: formatX(active.x) },
            { label: labelY ?? "value", value: formatY(active.y) },
          ]).map((row) => (
            <div key={row.label} className="flex items-baseline justify-between gap-3">
              <span className="text-[10.5px] text-[var(--text-3)]">{row.label}</span>
              <span className="zq-mono text-[11px]">{row.value}</span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export function Sparkline({
  values,
  height = 32,
  color = "var(--acc)",
  className,
}: {
  values: number[];
  height?: number;
  color?: string;
  className?: string;
}) {
  const clean = values.filter((value) => Number.isFinite(value));
  if (clean.length < 2) {
    return <div className={cn("h-8 rounded bg-[var(--panel-2)]", className)} />;
  }
  const min = Math.min(...clean);
  const max = Math.max(...clean);
  const span = max - min || 1;
  const path = clean
    .map((value, index) => {
      const x = (index / (clean.length - 1)) * 100;
      const y = height - ((value - min) / span) * (height - 4) - 2;
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");

  return (
    <svg viewBox={`0 0 100 ${height}`} preserveAspectRatio="none" className={cn("w-full", className)} style={{ height }}>
      <path d={path} fill="none" stroke={color} strokeWidth="1.4" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

export function MeterBar({
  value,
  max = 100,
  label,
  display,
  tone,
  className,
  height = 6,
}: {
  value: number | null;
  max?: number;
  label?: ReactNode;
  display?: ReactNode;
  tone?: "accent" | "good" | "warn" | "bad";
  className?: string;
  height?: number;
}) {
  const ratio = value == null || !Number.isFinite(value) ? 0 : Math.max(0, Math.min(1, value / (max || 1)));
  const autoTone: typeof tone =
    tone ?? (ratio > 0.92 ? "bad" : ratio > 0.78 ? "warn" : "accent");
  const colors: Record<string, string> = {
    accent: "var(--acc)",
    good: "var(--green)",
    warn: "var(--amber)",
    bad: "var(--red)",
  };

  return (
    <div className={cn("min-w-0", className)}>
      {label || display ? (
        <div className="mb-1 flex items-baseline justify-between gap-2">
          <span className="text-[11.5px] text-[var(--text-2)]">{label}</span>
          <span className="zq-mono text-[11.5px]">{display}</span>
        </div>
      ) : null}
      <div className="w-full overflow-hidden rounded-full bg-[var(--panel-2)]" style={{ height }}>
        <div
          className="h-full rounded-full transition-[width] duration-500 ease-out"
          style={{ width: `${ratio * 100}%`, background: colors[autoTone ?? "accent"] }}
        />
      </div>
    </div>
  );
}
