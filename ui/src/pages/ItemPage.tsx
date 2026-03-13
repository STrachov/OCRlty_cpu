import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useParams, useSearchParams } from "react-router-dom";
import { getEvalArtifact, getItem } from "../api/runs";
import type { EvalArtifact, EvalMismatch, EvalSample } from "../api/types";
import { Collapsible } from "../components/Collapsible";
import { CopyButton } from "../components/CopyButton";
import { ErrorPanel } from "../components/ErrorPanel";
import { JsonView } from "../components/JsonView";
import { downloadJson } from "../utils/downloadJson";

type FlatValueRow = {
  path: string;
  value: unknown;
};

type ComparisonRow = {
  path: string;
  pred: unknown;
  gt: unknown;
  reason: string | null;
  mismatch: EvalMismatch | null;
};

function formatValue(value: unknown): string {
  if (value == null) {
    return "-";
  }
  if (typeof value === "string") {
    return value;
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function flattenObject(value: unknown, path = ""): FlatValueRow[] {
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return [{ path: path || "$", value }];
    }
    return value.flatMap((item, index) => flattenObject(item, `${path}[${index}]`));
  }
  if (value && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 0) {
      return [{ path: path || "$", value }];
    }
    return entries.flatMap(([key, nested]) => flattenObject(nested, path ? `${path}.${key}` : key));
  }
  return [{ path: path || "$", value }];
}

function evalStatus(sample: EvalSample | null): { label: string; className: string } {
  if (!sample) {
    return { label: "Not available", className: "bg-slate-100 text-slate-700" };
  }
  if (sample.gt_ok === false) {
    return { label: "GT missing", className: "bg-amber-100 text-amber-900" };
  }
  if (sample.pred_ok === false) {
    return { label: "Prediction missing", className: "bg-amber-100 text-amber-900" };
  }
  if ((sample.mismatches_count ?? 0) > 0) {
    return { label: "FAIL", className: "bg-rose-100 text-rose-900" };
  }
  return { label: "OK", className: "bg-emerald-100 text-emerald-900" };
}

export function ItemPage() {
  const { request_id } = useParams<{ request_id: string }>();
  const [searchParams] = useSearchParams();
  const evalArtifactRel = searchParams.get("eval_artifact_rel");

  const itemQuery = useQuery({
    queryKey: ["item", request_id],
    queryFn: () => getItem(request_id ?? ""),
    enabled: Boolean(request_id),
  });
  const evalQuery = useQuery({
    queryKey: ["eval", evalArtifactRel],
    queryFn: () => getEvalArtifact(evalArtifactRel ?? ""),
    enabled: Boolean(evalArtifactRel),
  });

  const artifact = itemQuery.data ?? {};
  const displayRequestId = (artifact.request_id as string | undefined) ?? request_id ?? "-";
  const timings = artifact.timings_ms && typeof artifact.timings_ms === "object"
    ? (artifact.timings_ms as Record<string, unknown>)
    : null;
  const schemaErrors = artifact.schema_errors;
  const errorHistory = artifact.error_history;
  const rawIsNull = Object.prototype.hasOwnProperty.call(artifact, "raw") && artifact.raw === null;
  const evalArtifact = (evalQuery.data ?? null) as EvalArtifact | null;
  const evalSummary = evalArtifact?.summary ?? null;
  const evalSample = useMemo(() => {
    const samples = Array.isArray(evalArtifact?.samples) ? evalArtifact.samples : [];
    return (samples.find((sample) => sample.request_id === request_id) ?? null) as EvalSample | null;
  }, [evalArtifact?.samples, request_id]);
  const evalBadge = evalStatus(evalSample);
  const mismatches = Array.isArray(evalSample?.mismatches) ? evalSample.mismatches : [];
  const comparisonRows = useMemo(() => {
    if (!evalSample) {
      return [] as ComparisonRow[];
    }
    const predRows = flattenObject(evalSample.pred ?? null);
    const gtRows = flattenObject(evalSample.gt ?? null);
    const predMap = new Map(predRows.map((row) => [row.path, row.value]));
    const gtMap = new Map(gtRows.map((row) => [row.path, row.value]));
    const mismatchMap = new Map<string, EvalMismatch>();

    mismatches.forEach((mismatch) => {
      if (typeof mismatch?.path === "string" && mismatch.path) {
        mismatchMap.set(mismatch.path, mismatch);
      }
    });

    const paths = Array.from(new Set([...predMap.keys(), ...mismatchMap.keys()])).sort((a, b) => a.localeCompare(b));
    return paths.map((path) => {
      const mismatch = mismatchMap.get(path) ?? null;
      return {
        path,
        pred: predMap.has(path) ? predMap.get(path) : undefined,
        gt: gtMap.has(path) ? gtMap.get(path) : undefined,
        reason: mismatch?.reason ?? null,
        mismatch,
      };
    });
  }, [evalSample, mismatches]);

  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold">Item Details</h2>

      {itemQuery.isLoading ? <p className="text-sm text-slate-600">Loading extract artifact...</p> : null}
      {itemQuery.isError ? <ErrorPanel error={itemQuery.error} /> : null}

      {!itemQuery.isError && !itemQuery.isLoading ? (
        <>
          <div className="rounded-md border border-slate-200 bg-white p-4 text-sm">
            <div className="flex items-center gap-2">
              <span className="font-medium">request_id:</span>
              <span className="font-mono text-xs">{displayRequestId}</span>
              {displayRequestId ? <CopyButton text={displayRequestId} /> : null}
              <button
                type="button"
                onClick={() => downloadJson(`item-${displayRequestId}.json`, artifact)}
                className="rounded border border-slate-300 px-3 py-1 text-xs hover:bg-slate-100"
              >
                Download JSON
              </button>
            </div>
          </div>

          {evalArtifactRel ? (
            <div className="rounded-md border border-slate-200 bg-white p-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <h3 className="text-base font-semibold">Evaluation</h3>
                <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${evalBadge.className}`}>
                  {evalBadge.label}
                </span>
              </div>

              {evalQuery.isLoading ? <p className="mt-3 text-sm text-slate-600">Loading eval artifact...</p> : null}
              {evalQuery.isError ? <div className="mt-3"><ErrorPanel error={evalQuery.error} /></div> : null}

              {!evalQuery.isLoading && !evalQuery.isError ? (
                <div className="mt-3 space-y-3 text-sm">
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="rounded border border-slate-200 p-3">
                      <p><span className="font-medium">mismatches_count:</span> {String(evalSample?.mismatches_count ?? "-")}</p>
                      <p><span className="font-medium">gt_ok:</span> {String(evalSample?.gt_ok ?? "-")}</p>
                      <p><span className="font-medium">pred_ok:</span> {String(evalSample?.pred_ok ?? "-")}</p>
                      <p><span className="font-medium">gt_id:</span> {String(evalArtifact?.gt_id ?? "-")}</p>
                      <p><span className="font-medium">gt_name:</span> {String(evalArtifact?.gt_name ?? "-")}</p>
                    </div>
                    <div className="rounded border border-slate-200 p-3">
                      <p><span className="font-medium">str_mode:</span> {String(evalSummary?.str_mode ?? "-")}</p>
                      <p><span className="font-medium">decimal_sep:</span> {String(evalSummary?.decimal_sep ?? "-")}</p>
                      <p><span className="font-medium">batch_artifact_rel:</span> {String(evalArtifact?.batch_artifact_rel ?? "-")}</p>
                      <p><span className="font-medium">eval_id:</span> {String(evalArtifact?.eval_id ?? "-")}</p>
                    </div>
                  </div>

                  {evalSample ? (
                    <div>
                      <h4 className="mb-2 text-sm font-semibold">Comparison</h4>
                      <div className="overflow-x-auto rounded border border-slate-200">
                        <table className="min-w-full text-sm">
                          <thead className="bg-slate-100 text-left">
                            <tr>
                              <th className="px-3 py-2">path</th>
                              <th className="px-3 py-2">reason</th>
                              <th className="px-3 py-2">pred</th>
                              <th className="px-3 py-2">gt</th>
                            </tr>
                          </thead>
                          <tbody>
                            {comparisonRows.map((row) => (
                              <tr
                                key={row.path}
                                className={`border-t border-slate-100 align-top ${row.reason ? "bg-rose-50" : ""}`}
                              >
                                <td className="px-3 py-2 font-mono text-xs">{row.path}</td>
                                <td className="px-3 py-2">{row.reason ?? "-"}</td>
                                <td className="max-w-[280px] px-3 py-2">
                                  <p className="break-words">{formatValue(row.pred)}</p>
                                  {row.mismatch?.pred_canon !== undefined ? (
                                    <p className="mt-1 text-xs text-slate-500">canon: {formatValue(row.mismatch.pred_canon)}</p>
                                  ) : null}
                                  {row.mismatch?.pred_err ? <p className="mt-1 text-xs text-rose-700">error: {row.mismatch.pred_err}</p> : null}
                                </td>
                                <td className="max-w-[280px] px-3 py-2">
                                  <p className="break-words">{formatValue(row.gt)}</p>
                                  {row.mismatch?.gt_canon !== undefined ? (
                                    <p className="mt-1 text-xs text-slate-500">canon: {formatValue(row.mismatch.gt_canon)}</p>
                                  ) : null}
                                  {row.mismatch?.gt_err ? <p className="mt-1 text-xs text-rose-700">error: {row.mismatch.gt_err}</p> : null}
                                </td>
                              </tr>
                            ))}
                            {comparisonRows.length === 0 ? (
                              <tr className="border-t border-slate-100">
                                <td className="px-3 py-3 text-slate-500" colSpan={4}>No comparison rows available.</td>
                              </tr>
                            ) : null}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  ) : (
                    <p className="rounded border border-dashed border-slate-300 p-3 text-sm text-slate-500">
                      Current request_id was not found in eval.samples.
                    </p>
                  )}

                  <Collapsible title="eval_summary" defaultOpen={false}>
                    <JsonView data={evalSummary ?? null} />
                  </Collapsible>

                  <Collapsible title="eval_sample" defaultOpen={false}>
                    <JsonView data={evalSample ?? null} />
                  </Collapsible>
                </div>
              ) : null}
            </div>
          ) : null}

          {timings ? (
            <div className="rounded-md border border-slate-200 bg-white p-4">
              <h3 className="mb-2 text-base font-semibold">timings_ms</h3>
              <table className="min-w-[320px] text-sm">
                <thead>
                  <tr>
                    <th className="px-2 py-1 text-left">key</th>
                    <th className="px-2 py-1 text-left">ms</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(timings).map(([k, v]) => (
                    <tr key={k} className="border-t border-slate-100">
                      <td className="px-2 py-1 font-mono text-xs">{k}</td>
                      <td className="px-2 py-1">{String(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}

          <Collapsible title="schema_errors" defaultOpen={false}>
            <JsonView data={schemaErrors ?? null} />
          </Collapsible>

          <Collapsible title="error_history" defaultOpen={false}>
            <JsonView data={errorHistory ?? null} />
          </Collapsible>

          {rawIsNull ? (
            <div className="rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
              raw hidden (requires scope debug:read_raw)
            </div>
          ) : null}

          <div>
            <h3 className="mb-2 text-base font-semibold">Full Artifact JSON</h3>
            <JsonView data={artifact} />
          </div>
        </>
      ) : null}
    </section>
  );
}
