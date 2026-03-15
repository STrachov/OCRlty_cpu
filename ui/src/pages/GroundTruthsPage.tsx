import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { getMe, listGroundTruths } from "../api/runs";
import { ErrorPanel } from "../components/ErrorPanel";

export function GroundTruthsPage() {
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
              </tr>
            </thead>
            <tbody>
              {(groundTruthsQuery.data?.items ?? []).map((item) => (
                <tr key={item.gt_id} className="border-t border-slate-100">
                  <td className="px-3 py-2">{item.created_at}</td>
                  <td className="px-3 py-2 font-mono text-xs">{item.gt_id}</td>
                  <td className="px-3 py-2">{item.name}</td>
                  <td className="px-3 py-2">{String(item.task_id ?? "-")}</td>
                  <td className="px-3 py-2">{String(item.source_type ?? "-")}</td>
                  <td className="px-3 py-2 font-mono text-xs">{String(item.source_run_id ?? "-")}</td>
                  <td className="px-3 py-2">{String(item.updated_at ?? "-")}</td>
                </tr>
              ))}
              {(groundTruthsQuery.data?.items ?? []).length === 0 && !groundTruthsQuery.isLoading ? (
                <tr>
                  <td colSpan={7} className="px-3 py-6 text-center text-slate-500">
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
