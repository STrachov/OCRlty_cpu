import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useOutletContext, useSearchParams } from "react-router-dom";
import { listRuns } from "../api/runs";
import { CopyButton } from "../components/CopyButton";
import { ErrorPanel } from "../components/ErrorPanel";
import { useLayoutContext } from "../layout/LayoutContext";

type RunsOutletContext = {
  setRunsRefresh: (fn: (() => void) | undefined) => void;
};

const PAGE_SIZE = 10;

export function RunsPage() {
  const queryClient = useQueryClient();
  const { setRunsRefresh } = useOutletContext<RunsOutletContext>();
  const { setRunsListState } = useLayoutContext();
  const [searchParams, setSearchParams] = useSearchParams();
  const [focusedIndex, setFocusedIndex] = useState(0);
  const [pageIndex, setPageIndex] = useState(0);
  const [cursorStack, setCursorStack] = useState<(string | undefined)[]>([undefined]);
  const taskFilter = searchParams.get("task_id")?.trim() ?? "";
  const cursor = cursorStack[pageIndex];

  const runsQuery = useQuery({
    queryKey: ["runs", PAGE_SIZE, cursor ?? null, taskFilter],
    queryFn: () => listRuns(PAGE_SIZE, cursor, taskFilter || undefined),
    placeholderData: (previousData) => previousData,
  });

  const rows = runsQuery.data?.items ?? [];
  const nextCursor = runsQuery.data?.next_cursor ?? null;
  const canGoPrev = pageIndex > 0;
  const canGoNext = Boolean(nextCursor);

  function updateTaskFilter(value: string) {
    const params = new URLSearchParams(searchParams);
    const nextValue = value.trim();
    if (nextValue) {
      params.set("task_id", nextValue);
    } else {
      params.delete("task_id");
    }
    setSearchParams(params, { replace: true });
    setPageIndex(0);
    setCursorStack([undefined]);
    setFocusedIndex(0);
  }

  useEffect(() => {
    setFocusedIndex(0);
  }, [cursor, taskFilter]);

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

  function handlePrevPage() {
    if (!canGoPrev) {
      return;
    }
    setPageIndex((value) => value - 1);
  }

  function handleNextPage() {
    if (!nextCursor) {
      return;
    }
    setCursorStack((value) => {
      if (value[pageIndex + 1] === nextCursor) {
        return value;
      }
      return [...value.slice(0, pageIndex + 1), nextCursor];
    });
    setPageIndex((value) => value + 1);
  }

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

      <div className="flex flex-wrap items-end gap-3 rounded-md border border-slate-200 bg-white p-4">
        <label className="flex min-w-[280px] flex-1 flex-col gap-1 text-sm">
          <span className="font-medium text-slate-700">task_id</span>
          <input
            type="text"
            value={taskFilter}
            onChange={(e) => updateTaskFilter(e.target.value)}
            placeholder="Exact task_id"
            className="rounded border border-slate-300 px-3 py-2 outline-none focus:border-slate-400"
          />
        </label>
        {taskFilter ? (
          <button
            type="button"
            onClick={() => updateTaskFilter("")}
            className="rounded border border-slate-300 bg-white px-3 py-2 text-sm hover:bg-slate-100"
          >
            Clear filter
          </button>
        ) : null}
      </div>

      {runsQuery.isLoading ? <p className="text-sm text-slate-600">Loading runs...</p> : null}
      {runsQuery.isError ? <ErrorPanel error={runsQuery.error} /> : null}

      {!runsQuery.isError ? (
        <div
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") {
              e.preventDefault();
              setFocusedIndex((value) => Math.min(value + 1, Math.max(0, rows.length - 1)));
            } else if (e.key === "ArrowUp") {
              e.preventDefault();
              setFocusedIndex((value) => Math.max(value - 1, 0));
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
                <th className="px-3 py-2">time</th>
                <th className="px-3 py-2">mismatched</th>
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
                  <td className="px-3 py-2">{String(row.total_time ?? "-")}</td>
                  <td className="px-3 py-2">{String(row.eval_summary?.mismatched ?? "n/a")}</td>
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

      {!runsQuery.isError ? (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-slate-200 bg-white px-4 py-3">
          <p className="text-sm text-slate-600">
            Page {pageIndex + 1}
            {runsQuery.isFetching ? " · Loading..." : ""}
          </p>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handlePrevPage}
              disabled={!canGoPrev || runsQuery.isFetching}
              className="rounded border border-slate-300 bg-white px-4 py-2 text-sm hover:bg-slate-100 disabled:opacity-60"
            >
              Previous
            </button>
            <button
              type="button"
              onClick={handleNextPage}
              disabled={!canGoNext || runsQuery.isFetching}
              className="rounded border border-slate-300 bg-white px-4 py-2 text-sm hover:bg-slate-100 disabled:opacity-60"
            >
              Next
            </button>
          </div>
        </div>
      ) : null}
    </section>
  );
}
