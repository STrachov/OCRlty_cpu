import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useOutletContext, useSearchParams } from "react-router-dom";
import { deleteRun, getMe, listRuns } from "../api/runs";
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
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]);
  const taskFilter = searchParams.get("task_id")?.trim() ?? "";
  const cursor = cursorStack[pageIndex];

  const meQuery = useQuery({
    queryKey: ["me"],
    queryFn: getMe,
    staleTime: 5 * 60 * 1000,
  });
  const runsQuery = useQuery({
    queryKey: ["runs", PAGE_SIZE, cursor ?? null, taskFilter],
    queryFn: () => listRuns(PAGE_SIZE, cursor, taskFilter || undefined),
    placeholderData: (previousData) => previousData,
  });
  const deleteRunsMutation = useMutation({
    mutationFn: async (runIds: string[]) => {
      const deleted: string[] = [];
      const failed: Array<{ runId: string; message: string }> = [];
      for (const runId of runIds) {
        try {
          await deleteRun(runId);
          deleted.push(runId);
        } catch (error) {
          const message = error instanceof Error ? error.message : "Delete failed";
          failed.push({ runId, message });
        }
      }
      return { deleted, failed };
    },
    onSuccess: async ({ deleted }) => {
      if (deleted.length > 0) {
        setSelectedRunIds((current) => current.filter((runId) => !deleted.includes(runId)));
        await queryClient.invalidateQueries({ queryKey: ["runs"] });
      }
    },
  });

  const rows = runsQuery.data?.items ?? [];
  const nextCursor = runsQuery.data?.next_cursor ?? null;
  const canGoPrev = pageIndex > 0;
  const canGoNext = Boolean(nextCursor);
  const canDeleteRuns = meQuery.data?.scopes.includes("runs:delete") ?? false;
  const pageRunIds = useMemo(
    () => rows.map((row) => String(row.run_id ?? "")).filter((runId) => runId),
    [rows]
  );
  const selectedOnPageCount = pageRunIds.filter((runId) => selectedRunIds.includes(runId)).length;
  const allPageRowsSelected = pageRunIds.length > 0 && selectedOnPageCount === pageRunIds.length;

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

  function toggleRunSelection(runId: string, checked: boolean) {
    setSelectedRunIds((current) => {
      if (checked) {
        if (current.includes(runId)) {
          return current;
        }
        return [...current, runId];
      }
      return current.filter((value) => value !== runId);
    });
  }

  function toggleSelectPage(checked: boolean) {
    setSelectedRunIds((current) => {
      if (checked) {
        const next = new Set(current);
        pageRunIds.forEach((runId) => next.add(runId));
        return [...next];
      }
      return current.filter((runId) => !pageRunIds.includes(runId));
    });
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

  function handleDeleteSelected() {
    if (selectedRunIds.length === 0) {
      return;
    }
    const preview = selectedRunIds.slice(0, 5).join(", ");
    const suffix = selectedRunIds.length > 5 ? ` and ${selectedRunIds.length - 5} more` : "";
    if (!window.confirm(`Delete ${selectedRunIds.length} run(s): ${preview}${suffix}? This cannot be undone.`)) {
      return;
    }
    deleteRunsMutation.mutate(selectedRunIds);
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

      {canDeleteRuns ? (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-slate-200 bg-white px-4 py-3">
          <div className="text-sm text-slate-600">
            Selected: {selectedRunIds.length}
            {deleteRunsMutation.isSuccess ? (
              <span>
                {" · "}Deleted {deleteRunsMutation.data.deleted.length}
                {deleteRunsMutation.data.failed.length > 0 ? ` · Failed ${deleteRunsMutation.data.failed.length}` : ""}
              </span>
            ) : null}
          </div>
          <button
            type="button"
            onClick={handleDeleteSelected}
            disabled={selectedRunIds.length === 0 || deleteRunsMutation.isPending}
            className="rounded border border-rose-300 bg-rose-50 px-4 py-2 text-sm font-medium text-rose-900 hover:bg-rose-100 disabled:opacity-60"
          >
            {deleteRunsMutation.isPending ? "Deleting..." : "Delete selected"}
          </button>
        </div>
      ) : null}

      {runsQuery.isLoading ? <p className="text-sm text-slate-600">Loading runs...</p> : null}
      {runsQuery.isError ? <ErrorPanel error={runsQuery.error} /> : null}
      {deleteRunsMutation.isError ? <ErrorPanel error={deleteRunsMutation.error} /> : null}
      {deleteRunsMutation.data?.failed.length ? (
        <div className="rounded-md border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
          <p className="font-semibold">Some runs were not deleted.</p>
          <ul className="mt-2 space-y-1">
            {deleteRunsMutation.data.failed.map((item) => (
              <li key={item.runId}>
                <span className="font-mono">{item.runId}</span>: {item.message}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

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
                {canDeleteRuns ? (
                  <th className="px-3 py-2">
                    <input
                      type="checkbox"
                      checked={allPageRowsSelected}
                      onChange={(e) => toggleSelectPage(e.target.checked)}
                      aria-label="Select current page runs"
                    />
                  </th>
                ) : null}
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
                  {canDeleteRuns ? (
                    <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}>
                      {row.run_id ? (
                        <input
                          type="checkbox"
                          checked={selectedRunIds.includes(String(row.run_id))}
                          onChange={(e) => toggleRunSelection(String(row.run_id), e.target.checked)}
                          aria-label={`Select run ${String(row.run_id)}`}
                        />
                      ) : null}
                    </td>
                  ) : null}
                  <td className="px-3 py-2">{String(row.created_at ?? "-")}</td>
                  <td className="px-3 py-2">
                    {row.run_id ? (
                      <Link className="text-blue-700 hover:underline" to={`/runs/${encodeURIComponent(String(row.run_id))}`}>
                        {String(row.run_id)}
                      </Link>
                    ) : (
                      "-"
                    )}
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
                  <td colSpan={canDeleteRuns ? 8 : 7} className="px-3 py-6 text-center text-slate-500">
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
