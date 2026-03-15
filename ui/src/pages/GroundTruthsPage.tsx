import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { deleteGroundTruth, getMe, listGroundTruths } from "../api/runs";
import { ErrorPanel } from "../components/ErrorPanel";

export function GroundTruthsPage() {
  const queryClient = useQueryClient();
  const meQuery = useQuery({
    queryKey: ["me"],
    queryFn: getMe,
    staleTime: 5 * 60 * 1000,
  });
  const canUseGroundTruths = meQuery.data?.scopes.includes("debug:run") ?? false;
  const groundTruthsQuery = useQuery({
    queryKey: ["ground_truths", 100, 0],
    queryFn: () => listGroundTruths(100, 0),
    enabled: canUseGroundTruths,
  });
  const deleteMutation = useMutation({
    mutationFn: deleteGroundTruth,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["ground_truths"] });
    },
  });

  function handleDelete(gtId: string) {
    if (deleteMutation.isPending) {
      return;
    }
    const confirmed = window.confirm(`Delete ground truth ${gtId} permanently?`);
    if (!confirmed) {
      return;
    }
    deleteMutation.mutate(gtId);
  }

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold">Ground Truths</h2>
          <p className="text-sm text-slate-600">Manage reusable GT files created from existing runs.</p>
        </div>
        {canUseGroundTruths ? (
          <Link
            to="/ground-truths/new"
            className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700"
          >
            New ground truth
          </Link>
        ) : null}
      </div>

      {meQuery.isError ? <ErrorPanel error={meQuery.error} /> : null}
      {groundTruthsQuery.isError ? <ErrorPanel error={groundTruthsQuery.error} /> : null}
      {deleteMutation.isError ? <ErrorPanel error={deleteMutation.error} /> : null}

      {!canUseGroundTruths && meQuery.data ? (
        <div className="rounded-md border border-slate-200 bg-white p-4 text-sm text-slate-600">
          Ground truths are available only for users with the `debug:run` scope.
        </div>
      ) : null}

      {canUseGroundTruths ? (
        <div className="overflow-x-auto rounded-md border border-slate-200 bg-white">
          <table className="min-w-full text-sm">
            <thead className="bg-slate-100 text-left">
              <tr>
                <th className="px-3 py-2">created_at</th>
                <th className="px-3 py-2">gt_id</th>
                <th className="px-3 py-2">name</th>
                <th className="px-3 py-2">task_id</th>
                <th className="px-3 py-2">source</th>
                <th className="px-3 py-2">source_run_id</th>
                <th className="px-3 py-2">updated_at</th>
                <th className="px-3 py-2 text-right">actions</th>
              </tr>
            </thead>
            <tbody>
              {(groundTruthsQuery.data?.items ?? []).map((item) => (
                <tr key={item.gt_id} className="border-t border-slate-100">
                  <td className="px-3 py-2">{item.created_at}</td>
                  <td className="px-3 py-2 font-mono text-xs">
                    <Link className="text-blue-700 hover:underline" to={`/ground-truths/${encodeURIComponent(item.gt_id)}`}>
                      {item.gt_id}
                    </Link>
                  </td>
                  <td className="px-3 py-2">
                    <Link className="text-blue-700 hover:underline" to={`/ground-truths/${encodeURIComponent(item.gt_id)}`}>
                      {item.name}
                    </Link>
                  </td>
                  <td className="px-3 py-2">{String(item.task_id ?? "-")}</td>
                  <td className="px-3 py-2">{String(item.source_type ?? "-")}</td>
                  <td className="px-3 py-2 font-mono text-xs">{String(item.source_run_id ?? "-")}</td>
                  <td className="px-3 py-2">{String(item.updated_at ?? "-")}</td>
                  <td className="px-3 py-2 text-right">
                    <button
                      type="button"
                      onClick={() => handleDelete(item.gt_id)}
                      disabled={deleteMutation.isPending}
                      className="rounded border border-red-200 px-3 py-1 text-xs font-medium text-red-700 hover:bg-red-50 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {deleteMutation.isPending ? "Deleting..." : "Delete"}
                    </button>
                  </td>
                </tr>
              ))}
              {(groundTruthsQuery.data?.items ?? []).length === 0 && !groundTruthsQuery.isLoading ? (
                <tr>
                  <td colSpan={8} className="px-3 py-6 text-center text-slate-500">
                    No ground truths found.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      ) : null}
    </section>
  );
}
