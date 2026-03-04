import { useInfiniteQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { listRuns } from "../api/runs";
import { CopyButton } from "../components/CopyButton";
import { ErrorPanel } from "../components/ErrorPanel";

export function RunsPage() {
  const runsQuery = useInfiniteQuery({
    queryKey: ["runs", 50],
    queryFn: ({ pageParam }) => listRuns(50, pageParam),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
  });

  const rows = runsQuery.data?.pages.flatMap((page) => page.items ?? []) ?? [];

  return (
    <section className="space-y-4">
      <div>
        <h2 className="text-xl font-semibold">Runs</h2>
        <p className="text-sm text-slate-600">Recent batch runs from `/v1/runs`.</p>
      </div>

      {runsQuery.isLoading ? <p className="text-sm text-slate-600">Loading runs...</p> : null}

      {runsQuery.isError ? <ErrorPanel error={runsQuery.error} /> : null}

      {!runsQuery.isError ? (
        <div className="overflow-x-auto rounded-md border border-slate-200 bg-white">
          <table className="min-w-full text-sm">
            <thead className="bg-slate-100 text-left">
              <tr>
                
                <th className="px-3 py-2">run_id</th>
                <th className="px-3 py-2">task_id</th>
                <th className="px-3 py-2">item_count</th>
                <th className="px-3 py-2">ok_count</th>
                <th className="px-3 py-2">error_count</th>
                
                <th className="px-3 py-2">created_at</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, idx) => (
                <tr key={`${row.run_id ?? "run"}-${idx}`} className="border-t border-slate-100">
                  
                  <td className="px-3 py-2">
                    <div className="flex items-center gap-2">
                      {row.run_id ? (
                      <Link className="text-blue-700 hover:underline" to={`/runs/${encodeURIComponent(String(row.run_id))}`}>
                        {String(row.run_id)}
                      </Link>
                    ) : (
                      "-"
                    )}
                    {row.run_id ? <CopyButton text={String(row.run_id)} /> : null}
                    </div>
                  </td>
                  <td className="px-3 py-2">{String(row.task_id ?? "-")}</td>
                  <td className="px-3 py-2">{String(row.item_count ?? "-")}</td>
                  <td className="px-3 py-2">{String(row.ok_count ?? "-")}</td>
                  <td className="px-3 py-2">{String(row.error_count ?? "-")}</td>
                  
                  <td className="px-3 py-2">{String(row.created_at ?? "-")}</td>
                </tr>
              ))}
              {rows.length === 0 && !runsQuery.isLoading ? (
                <tr>
                  <td colSpan={7} className="px-3 py-6 text-center text-slate-500">
                    No runs found.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      ) : null}

      {runsQuery.hasNextPage ? (
        <button
          type="button"
          onClick={() => runsQuery.fetchNextPage()}
          disabled={runsQuery.isFetchingNextPage}
          className="rounded border border-slate-300 bg-white px-4 py-2 text-sm hover:bg-slate-100 disabled:opacity-60"
        >
          {runsQuery.isFetchingNextPage ? "Loading..." : "Load more"}
        </button>
      ) : null}
    </section>
  );
}
