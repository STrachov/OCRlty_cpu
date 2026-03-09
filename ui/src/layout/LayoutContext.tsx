import { createContext, useContext } from "react";
import type { GroundTruthView, JobView } from "../api/types";

export type ArtifactItem = Record<string, unknown>;
export type RunArtifact = Record<string, unknown>;

export type CreateRunPanelState = {
  selectedTaskId: string;
  selectedTaskDescription: string;
  files: Array<{ name: string; size: number }>;
  totalSize: number;
  gtMode: "none" | "existing" | "upload";
  selectedGroundTruth: GroundTruthView | null;
  job: JobView | null;
  jobId: string | null;
  runId: string | null;
  submitError: unknown | null;
  gtError: unknown | null;
  jobFetchError: unknown | null;
  isSubmitting: boolean;
  isUploadingGt: boolean;
};

type LayoutContextValue = {
  runArtifact: RunArtifact | null;
  focusedItem: ArtifactItem | null;
  createRunPanelState: CreateRunPanelState | null;
  setRunInspectorState: (state: { artifact: RunArtifact | null; focusedItem: ArtifactItem | null }) => void;
  setCreateRunPanelState: (state: CreateRunPanelState | null) => void;
};

export const LayoutContext = createContext<LayoutContextValue | null>(null);

export function useLayoutContext(): LayoutContextValue {
  const ctx = useContext(LayoutContext);
  if (!ctx) {
    throw new Error("useLayoutContext must be used within AppLayout");
  }
  return ctx;
}
