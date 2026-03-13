import { useState } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { LeftSidebar } from "../components/LeftSidebar";
import { RightPanel } from "../components/RightPanel";
import { TopBar } from "../components/TopBar";
import { LayoutContext, type ArtifactItem, type CreateRunPanelState, type RunArtifact } from "./LayoutContext";

export function AppLayout() {
  const location = useLocation();
  const [runArtifact, setRunArtifact] = useState<RunArtifact | null>(null);
  const [focusedItem, setFocusedItem] = useState<ArtifactItem | null>(null);
  const [createRunPanelState, setCreateRunPanelState] = useState<CreateRunPanelState | null>(null);
  const [runsRefresh, setRunsRefresh] = useState<(() => void) | undefined>(undefined);

  const showRightPanel = location.pathname === "/runs"
    || location.pathname.startsWith("/runs/")
    || location.pathname.startsWith("/items/");

  return (
    <LayoutContext.Provider
      value={{
        runArtifact,
        focusedItem,
        createRunPanelState,
        setRunInspectorState: ({ artifact, focusedItem: f }) => {
          setRunArtifact(artifact);
          setFocusedItem(f);
        },
        setCreateRunPanelState,
      }}
    >
      <div className="flex h-screen overflow-hidden bg-slate-50 text-slate-900">
        <LeftSidebar />
        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          <TopBar />
          <main className="min-h-0 flex-1 overflow-auto p-6">
            <Outlet context={{ setRunsRefresh }} />
          </main>
        </div>
        {showRightPanel ? <RightPanel onRefreshRuns={runsRefresh} /> : null}
      </div>
    </LayoutContext.Provider>
  );
}
