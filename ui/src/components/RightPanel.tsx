import { useMemo } from "react";
import { useLocation, useParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { CopyButton } from "./CopyButton";
import { ItemInspector } from "./ItemInspector";
import { downloadJson } from "../utils/downloadJson";
import { useLayoutContext } from "../layout/LayoutContext";

type RightPanelProps = {
  onRefreshRuns?: () => void;
};

export function RightPanel({ onRefreshRuns }: RightPanelProps) {
  const location = useLocation();
  const { run_id, request_id } = useParams<{ run_id?: string; request_id?: string }>();
  const queryClient = useQueryClient();
  const { runArtifact, focusedItem } = useLayoutContext();

  const runData = useMemo(() => {
    if (!run_id) {
      return null;
    }
    return (queryClient.getQueryData(["run", run_id]) ?? runArtifact) as Record<string, unknown> | null;
  }, [queryClient, runArtifact, run_id]);

  const itemData = useMemo(() => {
    if (!request_id) {
      return null;
    }
    return queryClient.getQueryData(["item", request_id]) as Record<string, unknown> | null;
  }, [queryClient, request_id]);

  if (location.pathname === "/runs") {
    return (
      <aside className="w-[360px] shrink-0 border-l border-slate-200 bg-white p-4">
        <div className="rounded border border-slate-200 p-3 text-sm">
          <p className="font-medium">Sorted: newest first</p>
          <button
            type="button"
            onClick={onRefreshRuns}
            className="mt-3 rounded border border-slate-300 px-3 py-2 text-sm hover:bg-slate-100"
          >
            Refresh
          </button>
        </div>
      </aside>
    );
  }

  if (location.pathname.startsWith("/runs/") && run_id) {
    return (
      <aside className="w-[360px] shrink-0 space-y-4 overflow-auto border-l border-slate-200 bg-white p-4">
        {/* <div className="rounded border border-slate-200 p-3 text-sm">
          <p><span className="font-medium">run_id:</span> {String(runData?.run_id ?? run_id)}</p>
          <p><span className="font-medium">task_id:</span> {String(runData?.task_id ?? "-")}</p>
          <p><span className="font-medium">created_at:</span> {String(runData?.created_at ?? "-")}</p>
          <p><span className="font-medium">ok_count:</span> {String(runData?.ok_count ?? "-")}</p>
          <p><span className="font-medium">error_count:</span> {String(runData?.error_count ?? "-")}</p>
          
        </div>
        <div className="rounded border border-slate-200 p-3 text-xs text-slate-600">
          ↑/↓ navigate, Enter open
        </div> */}
        <div className="rounded border border-slate-200 p-3">
          <h3 className="mb-2 text-sm font-semibold">Item Inspector</h3>
          <ItemInspector
            item={focusedItem}
            runId={run_id}
            evalByRequestId={
              (runData?.eval as { by_request_id?: Record<string, { gt_ok: boolean; pred_ok: boolean; mismatches_count: number }> } | undefined)?.by_request_id
            }
          />
        </div>
        <button
            type="button"
            onClick={() => downloadJson(`run-${run_id}.json`, runData ?? {})}
            className="mt-3 rounded border border-slate-300 px-3 py-2 text-xs hover:bg-slate-100"
          >
            Download Run JSON
          </button>
      </aside>
    );
  }

  if (location.pathname.startsWith("/items/") && request_id) {
    const displayId = typeof itemData?.request_id === "string" ? itemData.request_id : request_id;
    return (
      <aside className="w-[360px] shrink-0 border-l border-slate-200 bg-white p-4">
        <div className="rounded border border-slate-200 p-3 text-sm">
          <p className="mb-2 font-medium">Item Context</p>
          <div className="mb-3 flex items-center gap-2">
            <span className="font-mono text-xs">{displayId}</span>
            <CopyButton text={displayId} />
          </div>
          <button
            type="button"
            onClick={() => downloadJson(`item-${displayId}.json`, itemData ?? {})}
            className="rounded border border-slate-300 px-3 py-2 text-sm hover:bg-slate-100"
          >
            Download JSON
          </button>
        </div>
      </aside>
    );
  }

  return null;
}
