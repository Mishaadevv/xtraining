import { useEffect, useState } from "react";

import { Sidebar } from "@/components/layout/Shell";
import { ToastStack } from "@/components/ui/Overlay";
import { Button, Note } from "@/components/ui/primitives";
import { isDesktop } from "@/lib/bridge";
import { useStore } from "@/state/store";
import { appStore, bootstrap, refreshEnv, startAutoRefresh, subscribeToEvents } from "@/state/appStore";

import { ProjectsPage } from "@/features/projects/ProjectsPage";
import { NewTrainingPage } from "@/features/new-training/NewTrainingPage";
import { TrainingPage } from "@/features/training/TrainingPage";
import { ModelsPage } from "@/features/models/ModelsPage";
import { DatasetsPage } from "@/features/datasets/DatasetsPage";
import { PlaygroundPage } from "@/features/playground/PlaygroundPage";
import { HardwarePage } from "@/features/hardware/HardwarePage";
import { SettingsPage } from "@/features/settings/SettingsPage";

export default function App() {
  const state = useStore(appStore);
  const [desktopNoticeDismissed, setDesktopNoticeDismissed] = useState(false);

  useEffect(() => {
    let unsubscribe: (() => void) | undefined;
    let stopAutoRefresh: (() => void) | undefined;
    void (async () => {
      await bootstrap();
      unsubscribe = subscribeToEvents();
      // Event pushes only cover a live run; this keeps the lists, the GPU badge
      // and the environment snapshot current without a manual Refresh.
      stopAutoRefresh = startAutoRefresh();
    })();
    return () => {
      unsubscribe?.();
      stopAutoRefresh?.();
    };
  }, []);

  // The status strip and badge should refresh when the window regains focus,
  // since the user may have installed a GPU driver or a Python package meanwhile.
  useEffect(() => {
    const onFocus = () => void refreshEnv();
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, []);

  if (!state.booted) {
    return (
      <div className="flex h-full items-center justify-center bg-[var(--bg)]">
        <div className="flex flex-col items-center gap-3">
          <img src="./ico.png" alt="Zeqou" className="h-10 w-10 rounded-[10px] opacity-90" />
          <div className="h-1 w-[140px] overflow-hidden rounded-full bg-[var(--panel-2)]">
            <div className="h-full w-1/2 animate-pulse rounded-full bg-[var(--acc)]" />
          </div>
          <p className="text-[12px] text-[var(--text-3)]">Starting ZeqouXTraining…</p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative flex h-full bg-[var(--bg)] text-[var(--text)]">
      <div className="zq-aurora" aria-hidden />
      <Sidebar />

      <main className="relative z-[1] flex min-w-0 flex-1 flex-col">
        {!isDesktop && !desktopNoticeDismissed ? (
          <div className="px-6 pt-4">
            <Note
              tone="warn"
              title="Interface preview only"
              actions={
                <Button size="sm" variant="ghost" onClick={() => setDesktopNoticeDismissed(true)}>
                  Dismiss
                </Button>
              }
            >
              The Electron shell is not connected, so hardware detection, dataset validation and
              training are unavailable. Run <code className="zq-mono">npm run dev</code> to use the
              real application.
            </Note>
          </div>
        ) : null}

        {state.page === "projects" ? <ProjectsPage /> : null}
        {state.page === "new" ? <NewTrainingPage /> : null}
        {state.page === "training" ? <TrainingPage /> : null}
        {state.page === "models" ? <ModelsPage /> : null}
        {state.page === "datasets" ? <DatasetsPage /> : null}
        {state.page === "playground" ? <PlaygroundPage /> : null}
        {state.page === "hardware" ? <HardwarePage /> : null}
        {state.page === "settings" ? <SettingsPage /> : null}
      </main>

      <ToastStack toasts={state.toasts} />
    </div>
  );
}
