import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { getRun } from "../api/runs";
import { CopyButton } from "../components/CopyButton";
import { ErrorPanel } from "../components/ErrorPanel";

type ArtifactItem = Record<string, unknown>;

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
  const [onlySchemaInvalid, setOnlySchemaInvalid] = useState(false);
  const [onlyErrors, setOnlyErrors] = useState(false);
  const [searchFile, setSearchFile] = useState("");

  const runQuery = useQuery({
    queryKey: ["run", run_id],
    queryFn: () => getRun(run_id ?? ""),
    enabled: Boolean(run_id),
  });

  const artifact = runQuery.data ?? {};
  const items = (Array.isArray(artifact.items) ? artifact.items : []) as ArtifactItem[];

  const filteredItems = useMemo(() => {
    return items.filter((item) => {
      const schemaValid = item.schema_valid as boolean | null | undefined;
      const hasError = item.error !== null && item.error !== undefined;
      const file = typeof item.file === "string" ? item.file : "";

      if (onlySchemaInvalid && schemaValid !== false) {
        return false;
      }
      if (onlyErrors && !hasError) {
        return false;
      }
      if (searchFile && !file.toLowerCase().includes(searchFile.toLowerCase())) {
        return false;
      }
      return true;
    });
  }, [items, onlyErrors, onlySchemaInvalid, searchFile]);

  const topRunId = (artifact.run_id as string | undefined) ?? run_id ?? "-";
  const topTaskId = (artifact.task_id as string | undefined) ?? "-";
  const topCreatedAt = (artifact.created_at as string | null | undefined) ?? "-";
  const topItemCount = (artifact.item_count as number | null | undefined) ?? items.length;
  const topOkCount = (artifact.ok_count as number | null | undefined) ?? "-";
  const topErrorCount = (artifact.error_count as number | null | undefined) ?? "-";
  const topArtifactRel = (artifact.artifact_rel as string | null | undefined) ?? "-";

  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold">Run Details</h2>

      {runQuery.isLoading ? <p className="text-sm text-slate-600">Loading run artifact...</p> : null}
      {runQuery.isError ? <ErrorPanel error={runQuery.error} /> : null}

      {!runQuery.isError && !runQuery.isLoading ? (
        <>
          <div className="rounded-md border border-slate-200 bg-white p-4 text-sm">
            <p><span className="font-medium">run_id:</span> {topRunId}</p>
            <p><span className="font-medium">task_id:</span> {topTaskId}</p>
            <p><span className="font-medium">created_at:</span> {topCreatedAt}</p>
            <p><span className="font-medium">item_count:</span> {String(topItemCount)}</p>
            <p><span className="font-medium">ok_count:</span> {String(topOkCount)}</p>
            <p><span className="font-medium">error_count:</span> {String(topErrorCount)}</p>
            <p><span className="font-medium">artifact_rel:</span> {topArtifactRel}</p>
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

          <div className="overflow-x-auto rounded-md border border-slate-200 bg-white">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-100 text-left">
                <tr>
                  <th className="px-3 py-2">file</th>
                  <th className="px-3 py-2">schema_valid</th>
                  <th className="px-3 py-2">error</th>
                  <th className="px-3 py-2">parsed</th>
                  
                </tr>
              </thead>
              <tbody>
                {filteredItems.map((item, idx) => {
                  const requestId = typeof item.request_id === "string" ? item.request_id : "";
                  return (
                    <tr key={`${requestId || "req"}-${idx}`} className="border-t border-slate-100">
                      <td className="px-3 py-2"> 
                        {requestId ? (
                          <Link className="text-blue-700 hover:underline" to={`/items/${encodeURIComponent(requestId)}`}>
                            {typeof item.file === "string" ? item.file : "-"}
                          </Link>
                        ) : (
                          "-"
                        )}
                        </td>
                      <td className="px-3 py-2">{String((item.schema_valid as boolean | null | undefined) ?? "-")}</td>
                      <td className="max-w-[360px] truncate px-3 py-2" title={compactError(item.error)}>
                        {compactError(item.error)}
                      </td>
                      <td className="px-3 py-2">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-xs">
                            {item.parsed ? JSON.stringify(item.parsed) : "-"}
                          </span>
                          
                        </div>
                      </td>
                      
                    </tr>
                  );
                })}
                {filteredItems.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="px-3 py-6 text-center text-slate-500">
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
