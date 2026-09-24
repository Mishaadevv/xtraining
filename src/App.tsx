import { useEffect, useState } from "react";
import { CommandPalette } from "./components/CommandPalette";
import { Shell } from "./components/Shell";
import { Toasts } from "./components/Toasts";
import { Button, Callout, Panel, Spinner } from "./components/ui";
import { AppProvider, useApp } from "./state/app";
import { RouterProvider, detailId, useRouter } from "./state/router";
import { DashboardPage } from "./pages/Dashboard";
import { HardwarePage } from "./pages/Hardware";
import { EnvironmentPage } from "./pages/Environment";
import { ModelsPage } from "./pages/Models";
import { ModelDetailPage } from "./pages/ModelDetail";
import { DatasetsPage } from "./pages/Datasets";
import { DatasetDetailPage } from "./pages/DatasetDetail";
import { TrainingPage } from "./pages/Training";
import { NewTrainingPage } from "./pages/NewTraining";
import { TrainingRunPage } from "./pages/TrainingRun";
import { ExperimentsPage } from "./pages/Experiments";
import { EvaluationPage } from "./pages/Evaluation";
import { PlaygroundPage } from "./pages/Playground";
import { ComparePage } from "./pages/Compare";
import { AdaptersPage } from "./pages/Adapters";
import { QuantizationPage } from "./pages/Quantization";
import { ConversionPage } from "./pages/Conversion";
import { DeployPage } from "./pages/Deploy";
import { FilesPage } from "./pages/Files";
import { ProjectsPage } from "./pages/Projects";
import { JobsPage } from "./pages/Jobs";
import { SettingsPage } from "./pages/Settings";
import { DocsPage } from "./pages/Docs";

function ThemeSync() {
  const { settings } = useApp();
  useEffect(() => {
    const theme = settings?.theme ?? "system";
    const dark =
      theme === "dark" ||
      (theme === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.classList.toggle("dark", dark);
    const listener = (event: MediaQueryListEvent) => {
      if ((settings?.theme ?? "system") === "system") {
        document.documentElement.classList.toggle("dark", event.matches);
      }
    };
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    query.addEventListener("change", listener);
    return () => query.removeEventListener("change", listener);
  }, [settings?.theme]);
  return null;
}

function Routes() {
  const router = useRouter();
  const [section, id] = [router.path.split("/").filter(Boolean)[0] ?? "dashboard", detailId(router.path)];

  switch (section) {
    case "dashboard":
      return <DashboardPage />;
    case "hardware":
      return <HardwarePage />;
    case "environment":
      return <EnvironmentPage />;
    case "models":
      return id ? <ModelDetailPage id={id} /> : <ModelsPage />;
    case "datasets":
      return id ? <DatasetDetailPage id={id} /> : <DatasetsPage />;
    case "training":
      if (id === "new") return <NewTrainingPage />;
      return id ? <TrainingRunPage jobId={id} /> : <TrainingPage />;
    case "experiments":
      return <ExperimentsPage />;
    case "evaluation":
      return <EvaluationPage />;
    case "playground":
      return <PlaygroundPage />;
    case "compare":
      return <ComparePage />;
    case "adapters":
      return <AdaptersPage />;
    case "quantization":
      return <QuantizationPage />;
    case "conversion":
      return <ConversionPage />;
    case "deploy":
      return <DeployPage />;
    case "files":
      return <FilesPage />;
    case "projects":
      return <ProjectsPage />;
    case "jobs":
      return <JobsPage />;
    case "settings":
      return <SettingsPage />;
    case "docs":
      return <DocsPage />;
    default:
      return (
        <Panel>
          <Callout tone="warn" title={`No page for “${router.path}”`}>
            The route does not exist. Use the command palette (Ctrl+K) to open a real page.
          </Callout>
          <div className="mt-3">
            <Button onClick={() => router.navigate("/dashboard")}>Back to dashboard</Button>
          </div>
        </Panel>
      );
  }
}

function Boot() {
  const { ready, bootError, appInfo } = useApp();
  const router = useRouter();
  const [paletteOpen, setPaletteOpen] = useState(false);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPaletteOpen(true);
      }
      if (event.key === "Escape") setPaletteOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  if (!ready) {
    return (
      <div className="flex h-full items-center justify-center bg-surface-0">
        <Spinner label="Starting the engine and reading hardware…" />
      </div>
    );
  }

  if (bootError) {
    return (
      <div className="flex h-full items-center justify-center bg-surface-0 p-8">
        <div className="w-full max-w-2xl">
          <Callout tone="danger" title={bootError.message} hint={bootError.hint} detail={bootError.detail}>
            The app could not reach its Python engine ({bootError.code}). Nothing is simulated: the
            interface stays unpopulated until the engine answers.
          </Callout>
          <div className="mt-3 flex gap-2">
            <Button variant="primary" onClick={() => window.location.reload()}>
              Restart the interface
            </Button>
            <Button onClick={() => router.navigate("/environment")}>Open Environment</Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <Shell onOpenPalette={() => setPaletteOpen(true)}>
      <Routes />
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
      <Toasts />
      {appInfo?.packaged ? null : (
        <div className="pointer-events-none fixed bottom-9 right-4 z-30 max-w-xs text-right text-[10px] text-ink-3">
          development build · engine at {appInfo?.engineDir?.split(/[\\/]/).slice(-1)[0]}
        </div>
      )}
    </Shell>
  );
}

export default function App() {
  return (
    <AppProvider>
      <ThemeSync />
      <RouterProvider>
        <Boot />
      </RouterProvider>
    </AppProvider>
  );
}
