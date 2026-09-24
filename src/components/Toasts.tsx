import { AlertTriangle, Check, Info, X } from "lucide-react";
import { useApp } from "../state/app";
import { Badge, Button, CopyButton, IconButton, cx } from "./ui";

export function Toasts() {
  const { toasts, dismiss } = useApp();
  if (!toasts.length) return null;
  return (
    <div className="pointer-events-none fixed bottom-8 right-4 z-50 flex w-96 flex-col gap-2">
      {toasts.map((toast) => {
        const tone = {
          info: "border-info/40",
          ok: "border-ok/40",
          warn: "border-warn/40",
          danger: "border-danger/40",
        }[toast.tone];
        const icon = {
          info: <Info size={13} className="text-info" />,
          ok: <Check size={13} className="text-ok" />,
          warn: <AlertTriangle size={13} className="text-warn" />,
          danger: <AlertTriangle size={13} className="text-danger" />,
        }[toast.tone];
        return (
          <div
            key={toast.id}
            className={cx(
              "pointer-events-auto animate-slide-up rounded-lg border bg-surface-1/97 p-3 shadow-2xl backdrop-blur",
              tone,
            )}
          >
            <div className="flex items-start gap-2">
              <span className="mt-0.5">{icon}</span>
              <div className="min-w-0 flex-1">
                <div className="text-xs font-medium text-ink-0">{toast.title}</div>
                {toast.body ? <div className="mt-0.5 break-words text-xs text-ink-1">{toast.body}</div> : null}
                {toast.hint ? <div className="mt-1 text-2xs leading-relaxed text-ink-2">{toast.hint}</div> : null}
                {toast.detail ? (
                  <details className="mt-1.5">
                    <summary className="cursor-pointer text-2xs text-ink-3">Details</summary>
                    <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap rounded border border-line-soft bg-surface-2 p-1.5 font-mono text-2xs text-ink-2">
                      {toast.detail}
                    </pre>
                    <div className="mt-1">
                      <CopyButton value={toast.detail} label="Copy diagnostics" />
                    </div>
                  </details>
                ) : null}
              </div>
              <IconButton title="Dismiss" icon={<X size={12} />} onClick={() => dismiss(toast.id)} />
            </div>
          </div>
        );
      })}
      <div className="pointer-events-none flex justify-end">
        <Badge tone="muted" className="pointer-events-auto">
          <Button size="sm" variant="ghost" onClick={() => toasts.forEach((toast) => dismiss(toast.id))}>
            Clear all
          </Button>
        </Badge>
      </div>
    </div>
  );
}
