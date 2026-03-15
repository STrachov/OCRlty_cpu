import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate, useParams } from "react-router-dom";
import { deleteRun, getMe, getRun } from "../api/runs";
import type { BatchArtifact } from "../api/types";
import { ErrorPanel } from "../components/ErrorPanel";
import { useLayoutContext } from "../layout/LayoutContext";

// type ArtifactItem = Record<string, unknown>;
type ArtifactItem = {
  request_id?: string;
  file?: string;
  schema_valid?: boolean | null;
  error?: unknown;
  timings_ms?: {
    total?: number; // или number | null, если бывает null
    [k: string]: unknown;
  } | null;
  [k: string]: unknown; // чтобы не потерять другие поля
};

type EvalByRequestItem = {
  gt_ok: boolean;
  pred_ok: boolean;
  mismatches_count: number;
};

type BatchEval = {
  summary?: Record<string, unknown>;
  by_request_id?: Record<string, EvalByRequestItem>;
  eval_artifact_rel?: string;
};

function hasEvalFail(ev?: EvalByRequestItem): boolean {
  if (!ev) {
    return false;
  }
  return ev.mismatches_count > 0 || ev.gt_ok === false || ev.pred_ok === false;
}

function compactError(value: unknown): string {
  if (value == null) {
    return "-";
  }
  if (typeof value === "string") {
    return value;
  }
  try {
    const s = JSON.stringify(value);
    return s.length > 120 ? `${s.slice(0, 117)}...` : s;
  } catch {
    return String(value);
  }
}

export function RunDetailsPage() {
  const { run_id } = useParams<{ run_id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { setRunInspectorState } = useLayoutContext();
  const [onlySchemaInvalid, setOnlySchemaInvalid] = useState(false);
  const [onlyErrors, setOnlyErrors] = useState(false);
  const [onlyEvalMismatches, setOnlyEvalMismatches] = useState(false);
  const [searchFile, setSearchFile] = useState("");
  const [focusedIndex, setFocusedIndex] = useState(0);

  const meQuery = useQuery({
    queryKey: ["me"],
    queryFn: getMe,
    staleTime: 5 * 60 * 1000,
  });
  const runQuery = useQuery({
    queryKey: ["run", run_id],
    queryFn: () => getRun(run_id ?? ""),
    enabled: Boolean(run_id),
  });
  const deleteRunMutation = useMutation({
    mutationFn: () => deleteRun(run_id ?? ""),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["runs"] });
      await queryClient.removeQueries({ queryKey: ["run", run_id] });
      navigate("/runs", { replace: true });
    },
  });

  const artifact = (runQuery.data ?? {}) as BatchArtifact;
  const items = (Array.isArray(artifact.items) ? artifact.items : []) as ArtifactItem[];
  const batchEval = (artifact.eval ?? null) as BatchEval | null;
  const evalByRequestId = batchEval?.by_request_id ?? {};
  const evalSummary = batchEval?.summary ?? null;
  const mismatchedSamples = useMemo(
    () => Object.values(evalByRequestId).filter((ev) => ev.mismatches_count > 0).length,
    [evalByRequestId]
  );

  const filteredItems = useMemo(() => {
    return items.filter((item) => {
      const schemaValid = item.schema_valid as boolean | null | undefined;
      const hasError = item.error !== null && item.error !== undefined;
      const file = typeof item.file === "string" ? item.file : "";
      const requestId = typeof item.request_id === "string" ? item.request_id : "";
      const ev = requestId ? evalByRequestId[requestId] : undefined;

      if (onlySchemaInvalid && schemaValid !== false) {
        return false;
      }
      if (onlyErrors && !hasError) {
        return false;
      }
      if (searchFile && !file.toLowerCase().includes(searchFile.toLowerCase())) {
        return false;
      }
      if (onlyEvalMismatches && !hasEvalFail(ev)) {
        return false;
      }
      return true;
    });
  }, [evalByRequestId, items, onlyErrors, onlyEvalMismatches, onlySchemaInvalid, searchFile]);

  useEffect(() => {
    if (focusedIndex >= filteredItems.length) {
      setFocusedIndex(Math.max(filteredItems.length - 1, 0));
    }
  }, [filteredItems.length, focusedIndex]);

  useEffect(() => {
    setRunInspectorState({
      artifact: artifact as Record<string, unknown>,
      focusedItem: filteredItems[focusedIndex] ?? null,
    });
  }, [artifact, filteredItems, focusedIndex, setRunInspectorState]);

  useEffect(() => {
    return () => {
      setRunInspectorState({ artifact: null, focusedItem: null });
    };
  }, [setRunInspectorState]);

  const topRunId = (artifact.run_id as string | undefined) ?? run_id ?? "-";
  const topTaskId = (artifact.task_id as string | undefined) ?? "-";
  const topCreatedAt = (artifact.created_at as string | null | undefined) ?? "-";
  const topItemCount = (artifact.item_count as number | null | undefined) ?? items.length;
  const topOkCount = (artifact.ok_count as number | null | undefined) ?? "-";
  const topErrorCount = (artifact.error_count as number | null | undefined) ?? "-";
  const topArtifactRel = (artifact.artifact_rel as string | null | undefined) ?? "-";
  const evalArtifactRel = typeof batchEval?.eval_artifact_rel === "string" ? batchEval.eval_artifact_rel : null;
  const canDeleteRun = meQuery.data?.scopes.includes("runs:delete") ?? false;
  const getItemHref = (requestId: string, fileName?: string) => {
    const query = new URLSearchParams();
    if (evalArtifactRel) {
      query.set("eval_artifact_rel", evalArtifactRel);
    }
    if (fileName) {
      query.set("file_name", fileName);
    }
    return `/items/${encodeURIComponent(requestId)}${query.toString() ? `?${query.toString()}` : ""}`;
  };

  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold">Run Details</h2>

      {runQuery.isLoading ? <p className="text-sm text-slate-600">Loading run artifact...</p> : null}
      {runQuery.isError ? <ErrorPanel error={runQuery.error} /> : null}

      {!runQuery.isError && !runQuery.isLoading ? (
        <>
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-slate-200 bg-white p-4">
            <p className="text-sm text-slate-600">
              Full delete removes the run artifact, related item/eval artifacts, and related terminal jobs.
            </p>
            {canDeleteRun ? (
              <button
                type="button"
                onClick={() => {
                  if (!run_id) {
                    return;
                  }
                  if (!window.confirm(`Delete run ${run_id}? This cannot be undone.`)) {
                    return;
                  }
                  deleteRunMutation.mutate();
                }}
                disabled={deleteRunMutation.isPending}
                className="rounded border border-rose-300 bg-rose-50 px-4 py-2 text-sm font-medium text-rose-900 hover:bg-rose-100 disabled:opacity-60"
              >
                {deleteRunMutation.isPending ? "Deleting..." : "Delete run"}
              </button>
            ) : null}
          </div>

          {deleteRunMutation.isError ? <ErrorPanel error={deleteRunMutation.error} /> : null}

          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr),340px]">
            <div className="rounded-md border border-slate-200 bg-white p-4 text-sm">
              <p><span className="font-medium">run_id:</span> {topRunId}</p>
              <p><span className="font-medium">task_id:</span> {topTaskId}</p>
              <p><span className="font-medium">created_at:</span> {topCreatedAt}</p>
              <p><span className="font-medium">item_count:</span> {String(topItemCount)}</p>
              <p><span className="font-medium">ok_count:</span> {String(topOkCount)}</p>
              <p><span className="font-medium">error_count:</span> {String(topErrorCount)}</p>
              <p><span className="font-medium">artifact_rel:</span> {topArtifactRel}</p>
            </div>
            {evalSummary ? (
              <div className="rounded-md border border-slate-200 bg-white p-4 text-sm">
                <p className="mb-2 text-base font-semibold">Evaluation</p>
                <p><span className="font-medium">items:</span> {String(evalSummary.items ?? "-")}</p>
                <p><span className="font-medium">gt_found:</span> {String(evalSummary.gt_found ?? "-")}</p>
                <p><span className="font-medium">gt_missing:</span> {String(evalSummary.gt_missing ?? "-")}</p>
                <p><span className="font-medium">pred_found:</span> {String(evalSummary.pred_found ?? "-")}</p>
                <p><span className="font-medium">pred_missing:</span> {String(evalSummary.pred_missing ?? "-")}</p>
                <p><span className="font-medium">str_mode:</span> {String(evalSummary.str_mode ?? "-")}</p>
                <p><span className="font-medium">decimal_sep:</span> {String(evalSummary.decimal_sep ?? "-")}</p>
                <p><span className="font-medium">mismatched_samples:</span> {String(mismatchedSamples)}</p>
              </div>
            ) : null}
          </div>

          <div className="flex flex-wrap items-center gap-4 rounded-md border border-slate-200 bg-white p-3 text-sm">
            <label className="inline-flex items-center gap-2">
              <input
                type="checkbox"
                checked={onlySchemaInvalid}
                onChange={(e) => setOnlySchemaInvalid(e.target.checked)}
              />
              Only schema invalid
            </label>

            <label className="inline-flex items-center gap-2">
              <input type="checkbox" checked={onlyErrors} onChange={(e) => setOnlyErrors(e.target.checked)} />
              Only errors
            </label>

            <label className="inline-flex items-center gap-2">
              <input
                type="checkbox"
                checked={onlyEvalMismatches}
                onChange={(e) => setOnlyEvalMismatches(e.target.checked)}
              />
              Only eval mismatches
            </label>

            <label className="inline-flex items-center gap-2">
              <span>File search</span>
              <input
                type="text"
                value={searchFile}
                onChange={(e) => setSearchFile(e.target.value)}
                className="rounded border border-slate-300 px-2 py-1"
                placeholder="substring"
              />
            </label>
          </div>

          <div
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "ArrowDown") {
                e.preventDefault();
                setFocusedIndex((v) => Math.min(v + 1, Math.max(0, filteredItems.length - 1)));
              } else if (e.key === "ArrowUp") {
                e.preventDefault();
                setFocusedIndex((v) => Math.max(v - 1, 0));
              } else if (e.key === "Enter") {
                const current = filteredItems[focusedIndex];
                const requestId = typeof current?.request_id === "string" ? current.request_id : "";
                const fileName = typeof current?.file === "string" ? current.file : undefined;
                if (requestId) {
                  navigate(getItemHref(requestId, fileName), { state: { runId: run_id } });
                }
              }
            }}
            className="overflow-x-auto rounded-md border border-slate-200 bg-white outline-none focus:ring-2 focus:ring-slate-300"
          >
            <table className="min-w-full text-sm">
              <thead className="bg-slate-100 text-left">
                <tr>
                  <th className="px-3 py-2">file</th>
                  <th className="px-3 py-2">schema_valid</th>
                  <th className="px-3 py-2">error</th>
                  <th className="px-3 py-2">total time</th>
                  <th className="px-3 py-2">Mismatches</th>
                </tr>
              </thead>
              <tbody>
                {filteredItems.map((item, idx) => {
                  const requestId = typeof item.request_id === "string" ? item.request_id : "";
                  const fileName = typeof item.file === "string" ? item.file : undefined;
                  const isFocused = idx === focusedIndex;
                  const ev = requestId ? evalByRequestId[requestId] : undefined;
                  const evalFail = hasEvalFail(ev);
                  return (
                    <tr
                      key={`${requestId || "req"}-${idx}`}
                      className={`border-t border-slate-100 ${isFocused ? "bg-slate-100" : ""} ${evalFail ? "border-l-2 border-l-amber-300" : ""}`}
                      onClick={() => setFocusedIndex(idx)}
                    >
                      <td className="px-3 py-2">
                        {requestId ? (
                          <Link
                            className="text-blue-700 hover:underline"
                            to={getItemHref(requestId, fileName)}
                            state={{ runId: run_id }}
                          >
                            {typeof item.file === "string" ? item.file : "-"}
                          </Link>
                        ) : (
                          typeof item.file === "string" ? item.file : "-"
                        )}
                      </td>
                      <td className="px-3 py-2">{String((item.schema_valid as boolean | null | undefined) ?? "-")}</td>
                      <td className="max-w-[360px] truncate px-3 py-2" title={compactError(item.error)}>
                        {compactError(item.error)}
                      </td>
                      <td className="px-3 py-2">{item.timings_ms?.total ?? "-"}</td>
                      <td className="px-3 py-2">{ev ? ev.mismatches_count : "\u2014"}</td>
                    </tr>
                  );
                })}
                {filteredItems.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="px-3 py-6 text-center text-slate-500">
                      No items matching filters.
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </>
      ) : null}
    </section>
  );
}
