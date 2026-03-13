import { useEffect, useMemo, useState } from "react";
import { useInfiniteQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useOutletContext } from "react-router-dom";
import { listRuns } from "../api/runs";
import { CopyButton } from "../components/CopyButton";
import { ErrorPanel } from "../components/ErrorPanel";
import { useLayoutContext } from "../layout/LayoutContext";

type RunsOutletContext = {
  setRunsRefresh: (fn: (() => void) | undefined) => void;
};

function parseCreatedAt(v: unknown): number {
  if (typeof v !== "string" || !v.trim()) {
    return Number.NEGATIVE_INFINITY;
  }
  const t = Date.parse(v);
  return Number.isNaN(t) ? Number.NEGATIVE_INFINITY : t;
}

export function RunsPage() {
  const queryClient = useQueryClient();
  const { setRunsRefresh } = useOutletContext<RunsOutletContext>();
  const { setRunsListState } = useLayoutContext();
  const [focusedIndex, setFocusedIndex] = useState(0);

  const runsQuery = useInfiniteQuery({
    queryKey: ["runs", 50],
    queryFn: ({ pageParam }) => listRuns(50, pageParam),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
  });

  const rows = useMemo(() => {
    const flat = runsQuery.data?.pages.flatMap((page) => page.items ?? []) ?? [];
    return [...flat].sort((a, b) => {
      const ta = parseCreatedAt(a.created_at);
      const tb = parseCreatedAt(b.created_at);
      if (ta !== tb) {
        return tb - ta;
      }
      return String(b.run_id ?? "").localeCompare(String(a.run_id ?? ""));
    });
  }, [runsQuery.data]);

  useEffect(() => {
    if (focusedIndex >= rows.length) {
      setFocusedIndex(Math.max(rows.length - 1, 0));
    }
  }, [focusedIndex, rows.length]);

  useEffect(() => {
    setRunsListState({ focusedRunSummary: rows[focusedIndex] ?? null });
  }, [focusedIndex, rows, setRunsListState]);

  useEffect(() => {
    setRunsRefresh(() => () => {
      void queryClient.invalidateQueries({ queryKey: ["runs"] });
    });
    return () => {
      setRunsRefresh(undefined);
      setRunsListState({ focusedRunSummary: null });
    };
  }, [queryClient, setRunsListState, setRunsRefresh]);

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold">Runs</h2>
        </div>
        <Link
          to="/runs/new"
          className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700"
        >
          New run
        </Link>
      </div>

      {runsQuery.isLoading ? <p className="text-sm text-slate-600">Loading runs...</p> : null}
      {runsQuery.isError ? <ErrorPanel error={runsQuery.error} /> : null}

      {!runsQuery.isError ? (
        <div
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") {
              e.preventDefault();
              setFocusedIndex((v) => Math.min(v + 1, Math.max(0, rows.length - 1)));
            } else if (e.key === "ArrowUp") {
              e.preventDefault();
              setFocusedIndex((v) => Math.max(v - 1, 0));
            }
          }}
          className="overflow-x-auto rounded-md border border-slate-200 bg-white outline-none focus:ring-2 focus:ring-slate-300"
        >
          <table className="min-w-full text-sm">
            <thead className="bg-slate-100 text-left">
              <tr>
                <th className="px-3 py-2">created_at</th>
                <th className="px-3 py-2">run_id</th>
                <th className="px-3 py-2">task_id</th>
                <th className="px-3 py-2">item_count</th>
                <th className="px-3 py-2">ok_count</th>
                <th className="px-3 py-2">error_count</th>
                <th className="px-3 py-2">eval</th>

              </tr>
            </thead>
            <tbody>
              {rows.map((row, idx) => (
                <tr
                  key={`${row.run_id ?? "run"}-${idx}`}
                  className={`border-t border-slate-100 ${idx === focusedIndex ? "bg-slate-100" : ""}`}
                  onClick={() => setFocusedIndex(idx)}
                  onMouseEnter={() => setFocusedIndex(idx)}
                >
                  <td className="px-3 py-2">{String(row.created_at ?? "-")}</td>
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
                  <td className="px-3 py-2">{row.eval_summary ? "yes" : "no"}</td>
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
