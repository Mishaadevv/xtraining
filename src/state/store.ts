import { useCallback, useRef, useSyncExternalStore } from "react";

export type Listener = () => void;

/**
 * A deliberately small external store.
 *
 * The app has exactly one store, so a state library would be overhead. Updates
 * are pushed from the main process (training progress, GPU samples), and
 * useSyncExternalStore keeps React in step without any extra machinery.
 */
export class Store<T extends object> {
  private state: T;
  private listeners = new Set<Listener>();

  constructor(initial: T) {
    this.state = initial;
  }

  get = (): T => this.state;

  set = (patch: Partial<T> | ((state: T) => Partial<T>)): void => {
    const next = typeof patch === "function" ? patch(this.state) : patch;
    let changed = false;
    for (const key of Object.keys(next) as (keyof T)[]) {
      if (!Object.is(this.state[key], next[key])) {
        changed = true;
        break;
      }
    }
    if (!changed) return;
    this.state = { ...this.state, ...next };
    this.emit();
  };

  /** Replace state without change detection — used on bootstrap. */
  reset = (next: T): void => {
    this.state = next;
    this.emit();
  };

  subscribe = (listener: Listener): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  emit = (): void => {
    for (const listener of [...this.listeners]) {
      try {
        listener();
      } catch (error) {
        console.error("[zeqou] store listener failed", error);
      }
    }
  };
}

export function useStore<T extends object>(store: Store<T>): T {
  return useSyncExternalStore(store.subscribe, store.get, store.get);
}

/**
 * Subscribe to a derived slice. The selector result is cached and compared so a
 * fast-moving field (GPU samples arriving every second) does not re-render an
 * unrelated page.
 */
export function useSelect<T extends object, S>(
  store: Store<T>,
  selector: (state: T) => S,
  isEqual: (a: S, b: S) => boolean = Object.is,
): S {
  const cache = useRef<{ value: S } | null>(null);
  const selectorRef = useRef(selector);
  selectorRef.current = selector;

  const getSnapshot = useCallback(() => {
    const next = selectorRef.current(store.get());
    if (cache.current && isEqual(cache.current.value, next)) return cache.current.value;
    cache.current = { value: next };
    return next;
  }, [store, isEqual]);

  return useSyncExternalStore(store.subscribe, getSnapshot, getSnapshot);
}
