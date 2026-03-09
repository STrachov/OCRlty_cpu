import { useMemo } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { CopyButton } from "./CopyButton";
import { ErrorPanel } from "./ErrorPanel";
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
  const { runArtifact, focusedItem, createRunPanelState } = useLayoutContext();

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

  if (location.pathname === "/runs/new") {
    const progress = createRunPanelState?.job?.progress;
    const jobStatus = createRunPanelState?.job?.status ?? (createRunPanelState?.jobId ? "queued" : null);

    return (
      <aside className="w-[360px] shrink-0 space-y-4 overflow-auto border-l border-slate-200 bg-white p-4">
        <div className="rounded border border-slate-200 p-3 text-sm">
          <h3 className="mb-2 text-sm font-semibold">Run summary</h3>
          <p><span className="font-medium">task:</span> {createRunPanelState?.selectedTaskId || "-"}</p>
          {createRunPanelState?.selectedTaskDescription ? (
            <p className="mt-1 text-xs text-slate-600">{createRunPanelState.selectedTaskDescription}</p>
          ) : null}
          <p className="mt-2"><span className="font-medium">files:</span> {String(createRunPanelState?.files.length ?? 0)}</p>
          <p><span className="font-medium">total size:</span> {String(createRunPanelState?.totalSize ?? 0)} bytes</p>
          <p>
            <span className="font-medium">ground truth:</span>{" "}
            {createRunPanelState?.selectedGroundTruth
              ? `${createRunPanelState.selectedGroundTruth.name} (${createRunPanelState.selectedGroundTruth.gt_id})`
              : "none"}
          </p>
        </div>

        <div className="rounded border border-slate-200 p-3 text-sm">
          <h3 className="mb-2 text-sm font-semibold">Launch status</h3>
          <p><span className="font-medium">submit:</span> {createRunPanelState?.isSubmitting ? "starting" : "idle"}</p>
          <p><span className="font-medium">GT upload:</span> {createRunPanelState?.isUploadingGt ? "uploading" : "idle"}</p>
          <p><span className="font-medium">job status:</span> {jobStatus ?? "-"}</p>
          {createRunPanelState?.jobId ? (
            <div className="mt-2 flex items-center gap-2">
              <span className="font-medium">job_id:</span>
              <span className="font-mono text-xs">{createRunPanelState.jobId}</span>
              <CopyButton text={createRunPanelState.jobId} />
            </div>
          ) : null}
          {createRunPanelState?.runId ? (
            <div className="mt-2">
              <div className="flex items-center gap-2">
                <span className="font-medium">run_id:</span>
                <span className="font-mono text-xs">{createRunPanelState.runId}</span>
                <CopyButton text={createRunPanelState.runId} />
              </div>
              <Link
                to={`/runs/${encodeURIComponent(createRunPanelState.runId)}`}
                className="mt-2 inline-block rounded border border-slate-300 px-3 py-2 text-xs hover:bg-slate-100"
              >
                Open now
              </Link>
            </div>
          ) : null}
        </div>

        {progress && Object.keys(progress).length > 0 ? (
          <div className="rounded border border-slate-200 p-3 text-sm">
            <h3 className="mb-2 text-sm font-semibold">Progress</h3>
            <table className="w-full text-xs">
              <tbody>
                {Object.entries(progress).map(([key, value]) => (
                  <tr key={key} className="border-t border-slate-100">
                    <td className="py-1 pr-2 font-mono">{key}</td>
                    <td className="py-1">{typeof value === "string" ? value : JSON.stringify(value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}

        {createRunPanelState?.job?.error ? (
          <div className="rounded border border-rose-200 bg-rose-50 p-3 text-sm text-rose-900">
            <h3 className="mb-2 text-sm font-semibold">Job failed</h3>
            <p><span className="font-medium">type:</span> {String(createRunPanelState.job.error.type ?? "-")}</p>
            <p><span className="font-medium">message:</span> {String(createRunPanelState.job.error.message ?? "-")}</p>
          </div>
        ) : null}

        {createRunPanelState?.submitError ? <ErrorPanel error={createRunPanelState.submitError} /> : null}
        {createRunPanelState?.gtError ? <ErrorPanel error={createRunPanelState.gtError} /> : null}
        {createRunPanelState?.jobFetchError ? <ErrorPanel error={createRunPanelState.jobFetchError} /> : null}
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
