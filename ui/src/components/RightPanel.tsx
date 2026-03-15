import { useEffect, useMemo, useState } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { CopyButton } from "./CopyButton";
import { ErrorPanel } from "./ErrorPanel";
import { ItemInspector } from "./ItemInspector";
import { downloadJson } from "../utils/downloadJson";
import { useLayoutContext } from "../layout/LayoutContext";
import { fetchJson } from "../api/client";
import { getItem } from "../api/runs";
import { resolveReceiptImageUrl } from "../utils/resolveReceiptImageUrl";

type RightPanelProps = {
  onRefreshRuns?: () => void;
};

type EvalFieldRow = {
  path: string;
  matched: number;
  mismatch: number;
  pred_missing: number;
  gt_missing: number;
  canon_error: number;
  accuracy: number | null;
  total: number;
};

function normalizeEvalFields(value: unknown): EvalFieldRow[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((row) => {
      if (!row || typeof row !== "object") {
        return null;
      }
      const record = row as Record<string, unknown>;
      const path = typeof record.path === "string" ? record.path : "";
      if (!path) {
        return null;
      }
      const matched = Number(record.matched ?? 0) || 0;
      const mismatch = Number(record.mismatch ?? 0) || 0;
      const pred_missing = Number(record.pred_missing ?? 0) || 0;
      const gt_missing = Number(record.gt_missing ?? 0) || 0;
      const canon_error = Number(record.canon_error ?? 0) || 0;
      const total = Number(record.total ?? matched + mismatch + pred_missing + gt_missing + canon_error) || 0;
      const accuracyRaw = record.accuracy;
      const accuracy = typeof accuracyRaw === "number" ? accuracyRaw : null;
      return {
        path,
        matched,
        mismatch,
        pred_missing,
        gt_missing,
        canon_error,
        accuracy,
        total,
      };
    })
    .filter((row): row is EvalFieldRow => row !== null);
}

function ItemImagePreview({ itemData }: { itemData: Record<string, unknown> | null }) {
  const [imgSrc, setImgSrc] = useState<string | null>(null);
  const [imgFailed, setImgFailed] = useState(false);
  const [viewerOpen, setViewerOpen] = useState(false);

  const seedImage = useMemo(() => resolveReceiptImageUrl(itemData), [itemData]);

  useEffect(() => {
    setImgSrc(null);
    setImgFailed(false);
    if (!seedImage) {
      return;
    }
    if (!seedImage.includes("/v1/inputs/presign?")) {
      setImgSrc(seedImage);
      return;
    }

    let path = "";
    try {
      const u = new URL(seedImage);
      path = `${u.pathname}${u.search}`;
    } catch {
      setImgFailed(true);
      return;
    }

    void fetchJson<{ url: string }>(path)
      .then((res) => {
        if (res.data?.url) {
          setImgSrc(res.data.url);
        } else {
          setImgFailed(true);
        }
      })
      .catch(() => {
        setImgFailed(true);
      });
  }, [seedImage]);

  return (
    <>
      <div className="rounded border border-slate-200 p-3 text-sm">
        <p className="mb-2 font-medium">Image</p>
        {imgSrc && !imgFailed ? (
          <button
            type="button"
            onClick={() => setViewerOpen(true)}
            className="block w-full rounded border border-slate-200 p-1 hover:border-slate-300"
          >
            <img
              src={imgSrc}
              alt="Receipt preview"
              className="max-h-[400px] w-full rounded object-contain"
              onError={() => setImgFailed(true)}
            />
          </button>
        ) : (
          <p className="rounded border border-dashed border-slate-300 p-3 text-xs text-slate-500">
            Image not available
          </p>
        )}
      </div>

      {viewerOpen && imgSrc ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-6" onClick={() => setViewerOpen(false)}>
          <button
            type="button"
            onClick={() => setViewerOpen(false)}
            className="absolute right-6 top-6 rounded bg-white/90 px-3 py-2 text-sm font-medium text-slate-900 hover:bg-white"
          >
            Close
          </button>
          <img
            src={imgSrc}
            alt="Receipt full size"
            className="max-h-full max-w-full rounded bg-white p-2 object-contain"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      ) : null}
    </>
  );
}

export function RightPanel({ onRefreshRuns }: RightPanelProps) {
  const location = useLocation();
  const { run_id, request_id } = useParams<{ run_id?: string; request_id?: string }>();
  const queryClient = useQueryClient();
  const { runArtifact, focusedItem, focusedRunSummary, createRunPanelState } = useLayoutContext();

  const runData = useMemo(() => {
    if (!run_id) {
      return null;
    }
    return (queryClient.getQueryData(["run", run_id]) ?? runArtifact) as Record<string, unknown> | null;
  }, [queryClient, runArtifact, run_id]);

  const itemQuery = useQuery({
    queryKey: ["item", request_id],
    queryFn: () => getItem(request_id ?? ""),
    enabled: Boolean(request_id),
  });
  const itemData = (itemQuery.data ?? null) as Record<string, unknown> | null;

  if (location.pathname === "/runs") {
    const evalSummary =
      focusedRunSummary && typeof focusedRunSummary.eval_summary === "object" && focusedRunSummary.eval_summary !== null
        ? focusedRunSummary.eval_summary
        : null;
    const problematicFields = normalizeEvalFields(evalSummary?.fields)
      .filter((row) => row.mismatch > 0 || row.pred_missing > 0 || row.gt_missing > 0 || row.canon_error > 0)
      .sort((a, b) => {
        const severityA = a.mismatch * 4 + a.pred_missing * 3 + a.gt_missing * 3 + a.canon_error * 2;
        const severityB = b.mismatch * 4 + b.pred_missing * 3 + b.gt_missing * 3 + b.canon_error * 2;
        if (severityA !== severityB) {
          return severityB - severityA;
        }
        return a.path.localeCompare(b.path);
      })
      .slice(0, 8);
    return (
      <aside className="w-[360px] shrink-0 space-y-4 overflow-y-auto border-l border-slate-200 bg-white p-4">
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
        {/* <div className="rounded border border-slate-200 p-3 text-sm">
          <h3 className="mb-2 text-sm font-semibold">Run Preview</h3>
          {focusedRunSummary ? (
            <>
              <p><span className="font-medium">run_id:</span> {String(focusedRunSummary.run_id ?? "-")}</p>
              <p><span className="font-medium">task_id:</span> {String(focusedRunSummary.task_id ?? "-")}</p>
              <p><span className="font-medium">created_at:</span> {String(focusedRunSummary.created_at ?? "-")}</p>
              <p><span className="font-medium">item_count:</span> {String(focusedRunSummary.item_count ?? "-")}</p>
              <p><span className="font-medium">ok_count:</span> {String(focusedRunSummary.ok_count ?? "-")}</p>
              <p><span className="font-medium">error_count:</span> {String(focusedRunSummary.error_count ?? "-")}</p>
            </>
          ) : (
            <p className="text-slate-500">Select a run to preview it here.</p>
          )}
        </div> */}
        {focusedRunSummary ? (
          <div className="rounded border border-slate-200 p-3 text-sm">
            <h3 className="mb-2 text-sm font-semibold">Run Settings</h3>
            <p><span className="font-medium">model_id:</span> {String(focusedRunSummary.model_id ?? "-")}</p>
            <p><span className="font-medium">temperature:</span> {String(focusedRunSummary.temperature ?? "-")}</p>
            <p><span className="font-medium">max_tokens:</span> {String(focusedRunSummary.max_tokens ?? "-")}</p>
            <p><span className="font-medium">code_version:</span> {String(focusedRunSummary.code_version ?? "-")}</p>
            <p><span className="font-medium">prompt_sha256:</span> {String(focusedRunSummary.prompt_sha256 ?? "-")}</p>
            <p><span className="font-medium">schema_sha256:</span> {String(focusedRunSummary.schema_sha256 ?? "-")}</p>
          </div>
        ) : null}
        {focusedRunSummary ? (
          <div className="rounded border border-slate-200 p-3 text-sm">
            <h3 className="mb-2 text-sm font-semibold">Evaluation</h3>
            {evalSummary ? (
              <>
                <p><span className="font-medium">items:</span> {String(evalSummary.items ?? "-")}</p>
                <p><span className="font-medium">gt_found:</span> {String(evalSummary.gt_found ?? "-")}</p>
                <p><span className="font-medium">gt_missing:</span> {String(evalSummary.gt_missing ?? "-")}</p>
                <p><span className="font-medium">pred_found:</span> {String(evalSummary.pred_found ?? "-")}</p>
                <p><span className="font-medium">pred_missing:</span> {String(evalSummary.pred_missing ?? "-")}</p>
                <p><span className="font-medium">mismatched:</span> {String(evalSummary.mismatched ?? "-")}</p>
                <p><span className="font-medium">str_mode:</span> {String(evalSummary.str_mode ?? "-")}</p>
                <p><span className="font-medium">decimal_sep:</span> {String(evalSummary.decimal_sep ?? "-")}</p>
              </>
            ) : (
              <p className="text-slate-500">No evaluation for this run.</p>
            )}
          </div>
        ) : null}
        {focusedRunSummary && evalSummary ? (
          <div className="rounded border border-slate-200 p-3 text-sm">
            <h3 className="mb-2 text-sm font-semibold">Field Quality</h3>
            {problematicFields.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead className="bg-slate-100 text-left">
                    <tr>
                      <th className="px-2 py-1">path</th>
                      <th className="px-2 py-1">mismatch</th>
                      <th className="px-2 py-1">pred-</th>
                      <th className="px-2 py-1">gt-</th>
                      <th className="px-2 py-1">canon</th>
                      <th className="px-2 py-1">acc</th>
                    </tr>
                  </thead>
                  <tbody>
                    {problematicFields.map((row) => (
                      <tr key={row.path} className="border-t border-slate-100 align-top">
                        <td className="px-2 py-1 font-mono text-[11px] text-slate-700">{row.path}</td>
                        <td className="px-2 py-1">{row.mismatch}</td>
                        <td className="px-2 py-1">{row.pred_missing}</td>
                        <td className="px-2 py-1">{row.gt_missing}</td>
                        <td className="px-2 py-1">{row.canon_error}</td>
                        <td className="px-2 py-1">
                          {row.accuracy === null ? "-" : `${(row.accuracy * 100).toFixed(0)}%`}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-slate-500">No problematic fields for this run.</p>
            )}
          </div>
        ) : null}
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
      <aside className="w-[360px] shrink-0 overflow-y-auto border-l border-slate-200 bg-white p-4">
        <div className="space-y-4">
          <ItemImagePreview itemData={itemData} />
          <div className="rounded border border-slate-200 p-3 text-sm">
            <p className="mb-2 font-medium">Item Context</p>
            <p className="mb-1 font-medium text-slate-700">request_id</p>
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
        </div>
      </aside>
    );
  }

  return null;
}
