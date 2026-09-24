/** A small internal router: paths, segments and a back stack. No fake pages. */
import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";

interface RouteState {
  path: string;
  segments: string[];
  history: string[];
  navigate: (path: string, options?: { replace?: boolean }) => void;
  back: () => void;
  forward: () => void;
  canGoBack: boolean;
  canGoForward: boolean;
}

const RouterContext = createContext<RouteState | null>(null);

export function RouterProvider({ children, initial = "/dashboard" }: { children: ReactNode; initial?: string }) {
  const [history, setHistory] = useState<string[]>([initial]);
  const [index, setIndex] = useState(0);
  const path = history[index] ?? initial;

  const navigate = useCallback(
    (target: string, options?: { replace?: boolean }) => {
      setHistory((current) => {
        const base = current.slice(0, index + 1);
        if (options?.replace) {
          const next = [...base.slice(0, -1), target];
          setIndex(next.length - 1);
          return next;
        }
        if (base[base.length - 1] === target) return current;
        const next = [...base, target];
        setIndex(next.length - 1);
        return next.slice(-60);
      });
    },
    [index],
  );

  const back = useCallback(() => setIndex((current) => Math.max(0, current - 1)), []);
  const forward = useCallback(() => setIndex((current) => Math.min(history.length - 1, current + 1)), [history.length]);

  const value = useMemo<RouteState>(
    () => ({
      path,
      segments: path.split("/").filter(Boolean),
      history,
      navigate,
      back,
      forward,
      canGoBack: index > 0,
      canGoForward: index < history.length - 1,
    }),
    [path, history, navigate, back, forward, index],
  );

  return <RouterContext.Provider value={value}>{children}</RouterContext.Provider>;
}

export function useRouter(): RouteState {
  const context = useContext(RouterContext);
  if (!context) throw new Error("useRouter must be used inside RouterProvider");
  return context;
}

export function sectionOf(path: string): string {
  const first = path.split("/").filter(Boolean)[0] ?? "dashboard";
  return first;
}

export function detailId(path: string): string | null {
  const parts = path.split("/").filter(Boolean);
  return parts.length > 1 ? decodeURIComponent(parts.slice(1).join("/")) : null;
}
