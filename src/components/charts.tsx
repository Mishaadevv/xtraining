/** Charts drawn from real metric series streamed by the engine. No sample data. */
import { useMemo, useState } from "react";
import { cx } from "./ui";

export interface Series {
  name: string;
  color: string;
  points: Array<{ x: number; y: number }>;
}

interface LineChartProps {
  series: Series[];
  height?: number;
  formatY?: (value: number) => string;
  formatX?: (value: number) => string;
  yLogarithmic?: boolean;
  emptyLabel?: string;
}

export function LineChart({
  series,
  height = 180,
  formatY = (value) => value.toFixed(3),
  formatX = (value) => value.toFixed(0),
  yLogarithmic = false,
  emptyLabel = "No data yet",
}: LineChartProps) {
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null);
  const active = series.filter((item) => item.points.length > 0);

  const bounds = useMemo(() => {
    if (!active.length) return null;
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const item of active) {
      for (const point of item.points) {
        if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) continue;
        minX = Math.min(minX, point.x);
        maxX = Math.max(maxX, point.x);
        minY = Math.min(minY, point.y);
        maxY = Math.max(maxY, point.y);
      }
    }
    if (!Number.isFinite(minX)) return null;
    if (minX === maxX) maxX = minX + 1;
    if (minY === maxY) {
      minY -= 1;
      maxY += 1;
    }
    if (yLogarithmic && minY > 0) {
      return { minX, maxX, minY: Math.log10(minY), maxY: Math.log10(maxY), log: true as const, rawMinY: minY, rawMaxY: maxY };
    }
    return { minX, maxX, minY, maxY, log: false as const, rawMinY: minY, rawMaxY: maxY };
  }, [active, yLogarithmic]);

  if (!bounds) {
    return (
      <div className="flex h-40 items-center justify-center rounded-lg border border-dashed border-line-soft text-2xs text-ink-3">
        {emptyLabel}
      </div>
    );
  }

  const padding = { left: 46, right: 12, top: 10, bottom: 20 };
  const width = 720;
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const scaleX = (value: number) => padding.left + ((value - bounds.minX) / (bounds.maxX - bounds.minX)) * innerWidth;
  const scaleY = (value: number) => {
    const transformed = bounds.log ? Math.log10(Math.max(value, 1e-9)) : value;
    return padding.top + innerHeight - ((transformed - bounds.minY) / (bounds.maxY - bounds.minY)) * innerHeight;
  };

  const gridValues = 4;
  const gridLines = Array.from({ length: gridValues + 1 }, (_, index) => {
    const ratio = index / gridValues;
    const transformed = bounds.maxY - ratio * (bounds.maxY - bounds.minY);
    return {
      y: padding.top + ratio * innerHeight,
      raw: bounds.log ? 10 ** transformed : transformed,
    };
  });

  const hovered = hover
    ? active
        .map((item) => {
          let nearest = item.points[0];
          let best = Infinity;
          const targetX = bounds.minX + ((hover.x - padding.left) / innerWidth) * (bounds.maxX - bounds.minX);
          for (const point of item.points) {
            const distance = Math.abs(point.x - targetX);
            if (distance < best) {
              best = distance;
              nearest = point;
            }
          }
          return { name: item.name, color: item.color, point: nearest };
        })
        .filter(Boolean)
    : [];

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
        style={{ height }}
        preserveAspectRatio="none"
        onMouseMove={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          setHover({
            x: ((event.clientX - rect.left) / rect.width) * width,
            y: ((event.clientY - rect.top) / rect.height) * height,
          });
        }}
        onMouseLeave={() => setHover(null)}
      >
        {gridLines.map((line, index) => (
          <g key={index}>
            <line x1={padding.left} x2={width - padding.right} y1={line.y} y2={line.y} stroke="rgb(var(--line-soft))" strokeWidth={0.5} />
            <text x={padding.left - 6} y={line.y + 3} textAnchor="end" className="fill-ink-3" style={{ fontSize: 8.5 }}>
              {formatY(line.raw)}
            </text>
          </g>
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
          const value = bounds.minX + ratio * (bounds.maxX - bounds.minX);
          return (
            <text
              key={ratio}
              x={scaleX(value)}
              y={height - 6}
              textAnchor="middle"
              className="fill-ink-3"
              style={{ fontSize: 8.5 }}
            >
              {formatX(value)}
            </text>
          );
        })}
        {active.map((item) => (
          <polyline
            key={item.name}
            fill="none"
            stroke={item.color}
            strokeWidth={1.4}
            strokeLinejoin="round"
            strokeLinecap="round"
            points={item.points
              .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
              .map((point) => `${scaleX(point.x)},${scaleY(point.y)}`)
              .join(" ")}
          />
        ))}
        {hover ? (
          <line
            x1={hover.x}
            x2={hover.x}
            y1={padding.top}
            y2={padding.top + innerHeight}
            stroke="rgb(var(--accent))"
            strokeDasharray="3 3"
            strokeWidth={0.75}
            opacity={0.7}
          />
        ) : null}
      </svg>
      {hover && hovered.length ? (
        <div
          className="pointer-events-none absolute top-2 z-20 rounded-md border border-line-soft bg-surface-1/95 px-2 py-1.5 text-2xs shadow-lg backdrop-blur"
          style={{ left: `${Math.min(70, (hover.x / width) * 100)}%` }}
        >
          {hovered.map((entry) => (
            <div key={entry.name} className="flex items-center gap-2 whitespace-nowrap">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: entry.color }} />
              <span className="text-ink-2">{entry.name}</span>
              <span className="font-mono tabular-nums text-ink-0">
                {formatY(entry.point.y)} @ {formatX(entry.point.x)}
              </span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export function Sparkline({
  values,
  className,
  tone = "accent",
  height = 26,
}: {
  values: number[];
  className?: string;
  tone?: "accent" | "ok" | "warn" | "danger";
  height?: number;
}) {
  const clean = values.filter((value) => Number.isFinite(value));
  if (clean.length < 2) {
    return <div className={cx("h-[26px] rounded bg-surface-3/40", className)} style={{ height }} />;
  }
  const min = Math.min(...clean);
  const max = Math.max(...clean);
  const range = max - min || 1;
  const points = clean
    .map((value, index) => `${(index / (clean.length - 1)) * 100},${28 - ((value - min) / range) * 26}`)
    .join(" ");
  const colors = { accent: "rgb(var(--accent))", ok: "rgb(var(--ok))", warn: "rgb(var(--warn))", danger: "rgb(var(--danger))" };
  return (
    <svg viewBox="0 0 100 30" preserveAspectRatio="none" className={cx("w-full", className)} style={{ height }}>
      <polyline points={points} fill="none" stroke={colors[tone]} strokeWidth={0.9} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

export function BarChart({
  bars,
  formatLabel = (value) => String(value),
  height = 80,
}: {
  bars: Array<{ label: string; value: number }>;
  formatLabel?: (value: string | number) => string;
  height?: number;
}) {
  const max = Math.max(1, ...bars.map((bar) => bar.value));
  if (!bars.length) return null;
  return (
    <div className="flex items-end gap-1" style={{ height }}>
      {bars.map((bar, index) => (
        <div key={index} className="group flex flex-1 flex-col items-center justify-end gap-1" title={`${formatLabel(bar.label)}: ${bar.value}`}>
          <div
            className="w-full rounded-t bg-accent/70 transition-all group-hover:bg-accent"
            style={{ height: `${(bar.value / max) * (height - 18)}px` }}
          />
          <span className="truncate text-[8.5px] text-ink-3">{formatLabel(bar.label)}</span>
        </div>
      ))}
    </div>
  );
}

export function UsageBar({
  label,
  value,
  max,
  format,
  tone,
}: {
  label: string;
  value: number | null | undefined;
  max: number | null | undefined;
  format: (value: number) => string;
  tone?: "accent" | "warn" | "danger" | "ok";
}) {
  const ratio = value && max ? Math.min(1, value / max) : null;
  const resolvedTone = tone ?? (ratio === null ? "accent" : ratio > 0.9 ? "danger" : ratio > 0.75 ? "warn" : "accent");
  const colors = { accent: "bg-accent", ok: "bg-ok", warn: "bg-warn", danger: "bg-danger" };
  return (
    <div>
      <div className="flex items-baseline justify-between text-2xs">
        <span className="uppercase tracking-wide text-ink-3">{label}</span>
        <span className="tabular-nums text-ink-1">
          {value === null || value === undefined ? "—" : format(value)}
          {max ? <span className="text-ink-3"> / {format(max)}</span> : null}
        </span>
      </div>
      <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-surface-3">
        <div className={cx("h-full rounded-full transition-all duration-700", colors[resolvedTone])} style={{ width: `${(ratio ?? 0) * 100}%` }} />
      </div>
    </div>
  );
}
