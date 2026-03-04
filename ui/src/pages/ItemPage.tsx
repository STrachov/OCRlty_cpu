import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import { getItem } from "../api/runs";
import { Collapsible } from "../components/Collapsible";
import { CopyButton } from "../components/CopyButton";
import { ErrorPanel } from "../components/ErrorPanel";
import { JsonView } from "../components/JsonView";

export function ItemPage() {
  const { request_id } = useParams<{ request_id: string }>();

  const itemQuery = useQuery({
    queryKey: ["item", request_id],
    queryFn: () => getItem(request_id ?? ""),
    enabled: Boolean(request_id),
  });

  const artifact = itemQuery.data ?? {};
  const displayRequestId = (artifact.request_id as string | undefined) ?? request_id ?? "-";
  const timings = artifact.timings_ms && typeof artifact.timings_ms === "object"
    ? (artifact.timings_ms as Record<string, unknown>)
    : null;
  const schemaErrors = artifact.schema_errors;
  const errorHistory = artifact.error_history;
  const rawIsNull = Object.prototype.hasOwnProperty.call(artifact, "raw") && artifact.raw === null;

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
            </div>
          </div>

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
