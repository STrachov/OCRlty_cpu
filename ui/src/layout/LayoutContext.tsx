import { createContext, useContext } from "react";

export type ArtifactItem = Record<string, unknown>;
export type RunArtifact = Record<string, unknown>;

type LayoutContextValue = {
  runArtifact: RunArtifact | null;
  focusedItem: ArtifactItem | null;
  setRunInspectorState: (state: { artifact: RunArtifact | null; focusedItem: ArtifactItem | null }) => void;
};

export const LayoutContext = createContext<LayoutContextValue | null>(null);

export function useLayoutContext(): LayoutContextValue {
  const ctx = useContext(LayoutContext);
  if (!ctx) {
    throw new Error("useLayoutContext must be used within AppLayout");
  }
  return ctx;
}
