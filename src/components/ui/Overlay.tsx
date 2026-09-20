import { useEffect, type ReactNode } from "react";
import { AlertTriangle, Check, Info, X } from "lucide-react";

import { cn } from "@/lib/utils";
import type { Toast } from "@/lib/types";
import { Button, IconButton } from "./primitives";
import { dismissToast } from "@/state/appStore";

/* ------------------------------------------------------------------- Modal */

export function Modal({
  open,
  onClose,
  title,
  description,
  children,
  footer,
  width = 560,
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  description?: ReactNode;
  children?: ReactNode;
  footer?: ReactNode;
  width?: number;
}) {
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="zq-fade fixed inset-0 z-50 flex items-center justify-center p-6"
      style={{ background: "rgba(0,0,0,0.55)", backdropFilter: "blur(3px)" }}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        className="zq-rise flex max-h-[85vh] w-full flex-col overflow-hidden rounded-[16px] border border-[var(--border)] bg-[var(--panel)] shadow-[var(--shadow)]"
        style={{ maxWidth: width }}
      >
        <header className="flex items-start justify-between gap-4 border-b border-[var(--border-soft)] px-4 py-3">
          <div className="min-w-0">
            <h2 className="truncate text-[14px] font-semibold">{title}</h2>
            {description ? (
              <p className="mt-0.5 text-[12px] leading-[18px] text-[var(--text-2)]">{description}</p>
            ) : null}
          </div>
          <IconButton onClick={onClose} title="Close">
            <X className="h-4 w-4" />
          </IconButton>
        </header>
        <div className="zq-scroll min-h-0 flex-1 px-4 py-4">{children}</div>
        {footer ? (
          <footer className="flex items-center justify-end gap-2 border-t border-[var(--border-soft)] px-4 py-3">
            {footer}
          </footer>
        ) : null}
      </div>
    </div>
  );
}

/* ----------------------------------------------------------- ConfirmDialog */

export function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel = "Confirm",
  tone = "danger",
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  description?: ReactNode;
  confirmLabel?: string;
  tone?: "danger" | "primary";
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <Modal
      open={open}
      onClose={onCancel}
      title={title}
      width={440}
      footer={
        <>
          <Button variant="ghost" onClick={onCancel}>
            Cancel
          </Button>
          <Button variant={tone === "danger" ? "danger" : "primary"} onClick={onConfirm}>
            {confirmLabel}
          </Button>
        </>
      }
    >
      <div className="text-[12.5px] leading-[19px] text-[var(--text-2)]">{description}</div>
    </Modal>
  );
}

/* -------------------------------------------------------------- ToastStack */

const TOAST_ICONS = {
  info: <Info className="h-3.5 w-3.5" style={{ color: "var(--blue)" }} />,
  good: <Check className="h-3.5 w-3.5" style={{ color: "var(--green)" }} />,
  warn: <AlertTriangle className="h-3.5 w-3.5" style={{ color: "var(--amber)" }} />,
  bad: <AlertTriangle className="h-3.5 w-3.5" style={{ color: "var(--red)" }} />,
};

export function ToastStack({ toasts }: { toasts: Toast[] }) {
  if (!toasts.length) return null;
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-[60] flex w-[380px] flex-col gap-2">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={cn(
            "zq-rise pointer-events-auto flex items-start gap-2.5 rounded-[12px] border p-3",
            "border-[var(--border)] bg-[var(--panel-2)] shadow-[var(--shadow)]",
          )}
        >
          <span className="mt-[3px] shrink-0">{TOAST_ICONS[toast.tone]}</span>
          <div className="min-w-0 flex-1">
            <p className="text-[12.5px] font-semibold leading-5">{toast.title}</p>
            {toast.message ? (
              <p className="mt-0.5 break-words text-[11.5px] leading-[17px] text-[var(--text-2)]">
                {toast.message}
              </p>
            ) : null}
          </div>
          <IconButton
            onClick={() => dismissToast(toast.id)}
            title="Dismiss"
            className="h-6 w-6"
          >
            <X className="h-3 w-3" />
          </IconButton>
        </div>
      ))}
    </div>
  );
}
