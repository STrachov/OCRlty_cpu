import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { createGroundTruthFromRun, getMe, listRunsCatalog } from "../api/runs";
import { ErrorPanel } from "../components/ErrorPanel";

const PAGE_SIZE = 10;

export function CreateGroundTruthPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [searchParams] = useSearchParams();
  const [pageIndex, setPageIndex] = useState(0);
  const [cursorStack, setCursorStack] = useState<(string | undefined)[]>([undefined]);
  const [selectedRunId, setSelectedRunId] = useState(searchParams.get("source_run_id")?.trim() ?? "");
  const [name, setName] = useState("");
  const cursor = cursorStack[pageIndex];

  const meQuery = useQuery({
    queryKey: ["me"],
    queryFn: getMe,
    staleTime: 5 * 60 * 1000,
  });
  const canUseGroundTruths = meQuery.data?.scopes.includes("debug:run") ?? false;
  const catalogQuery = useQuery({
    queryKey: ["runs-catalog", PAGE_SIZE, cursor ?? null],
    queryFn: () => listRunsCatalog(PAGE_SIZE, cursor),
    enabled: canUseGroundTruths,
    placeholderData: (previousData) => previousData,
  });
  const createMutation = useMutation({
    mutationFn: () => createGroundTruthFromRun({ run_id: selectedRunId, name: name.trim() || undefined }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["ground_truths"] });
      navigate("/ground-truths", { replace: true });
    },
  });

  useEffect(() => {
    if (searchParams.get("source_run_id")?.trim()) {
      setSelectedRunId(searchParams.get("source_run_id")?.trim() ?? "");
    }
  }, [searchParams]);

  const rows = catalogQuery.data?.items ?? [];
  const nextCursor = catalogQuery.data?.next_cursor ?? null;
  const canGoPrev = pageIndex > 0;
  const canGoNext = Boolean(nextCursor);

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
          <h2 className="text-xl font-semibold">New Ground Truth</h2>
          <p className="text-sm text-slate-600">Create a reusable GT file from an existing run.</p>
        </div>
        <Link
          to="/ground-truths"
          className="rounded border border-slate-300 bg-white px-4 py-2 text-sm hover:bg-slate-100"
        >
          Back to list
        </Link>
      </div>

      {meQuery.isError ? <ErrorPanel error={meQuery.error} /> : null}
      {catalogQuery.isError ? <ErrorPanel error={catalogQuery.error} /> : null}
      {createMutation.isError ? <ErrorPanel error={createMutation.error} /> : null}

      {!canUseGroundTruths && meQuery.data ? (
        <div className="rounded-md border border-slate-200 bg-white p-4 text-sm text-slate-600">
          Ground truth creation requires the `debug:run` scope.
        </div>
      ) : null}

      {canUseGroundTruths ? (
        <>
          <div className="rounded-md border border-slate-200 bg-white p-4">
            <label className="block text-sm font-medium text-slate-800">
              Name
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="mt-1 w-full rounded border border-slate-300 px-3 py-2"
                placeholder={selectedRunId ? `${selectedRunId}.json` : "ground_truth.json"}
              />
            </label>
            <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
              <p className="text-sm text-slate-600">
                Selected run: <span className="font-mono">{selectedRunId || "-"}</span>
              </p>
              <button
                type="button"
                onClick={() => createMutation.mutate()}
                disabled={!selectedRunId || createMutation.isPending}
                className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-60"
              >
                {createMutation.isPending ? "Creating..." : "Create ground truth"}
              </button>
            </div>
          </div>

          <div className="overflow-x-auto rounded-md border border-slate-200 bg-white">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-100 text-left">
                <tr>
                  <th className="px-3 py-2"></th>
                  <th className="px-3 py-2">created_at</th>
                  <th className="px-3 py-2">run_id</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr
                    key={row.run_id}
                    className={`border-t border-slate-100 ${selectedRunId === row.run_id ? "bg-slate-100" : ""}`}
                    onClick={() => setSelectedRunId(row.run_id)}
                  >
                    <td className="px-3 py-2">
                      <input
                        type="radio"
                        name="source_run_id"
                        checked={selectedRunId === row.run_id}
                        onChange={() => setSelectedRunId(row.run_id)}
                      />
                    </td>
                    <td className="px-3 py-2">{String(row.created_at ?? "-")}</td>
                    <td className="px-3 py-2 font-mono text-xs">{row.run_id}</td>
                  </tr>
                ))}
                {rows.length === 0 && !catalogQuery.isLoading ? (
                  <tr>
                    <td colSpan={3} className="px-3 py-6 text-center text-slate-500">
                      No runs found.
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-slate-200 bg-white px-4 py-3">
            <p className="text-sm text-slate-600">
              Page {pageIndex + 1}
              {catalogQuery.isFetching ? " · Loading..." : ""}
            </p>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handlePrevPage}
                disabled={!canGoPrev || catalogQuery.isFetching}
                className="rounded border border-slate-300 bg-white px-4 py-2 text-sm hover:bg-slate-100 disabled:opacity-60"
              >
                Previous
              </button>
              <button
                type="button"
                onClick={handleNextPage}
                disabled={!canGoNext || catalogQuery.isFetching}
                className="rounded border border-slate-300 bg-white px-4 py-2 text-sm hover:bg-slate-100 disabled:opacity-60"
              >
                Next
              </button>
            </div>
          </div>
        </>
      ) : null}
    </section>
  );
}
