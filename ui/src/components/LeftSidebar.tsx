import { Link, useLocation } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getMe } from "../api/runs";
import { getApiBaseUrl } from "../auth/storage";
import { CopyButton } from "./CopyButton";

export function LeftSidebar() {
  const location = useLocation();
  const apiBaseUrl = getApiBaseUrl() ?? "";
  const meQuery = useQuery({
    queryKey: ["me"],
    queryFn: getMe,
    staleTime: 5 * 60 * 1000,
  });

  const isRuns = location.pathname.startsWith("/runs");

  return (
    <aside className="w-[260px] shrink-0 border-r border-slate-200 bg-white p-4">
      <nav className="mb-6">
        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Navigation</p>
        <Link
          to="/runs"
          className={`block rounded px-3 py-2 text-sm ${isRuns ? "bg-slate-900 text-white" : "text-slate-700 hover:bg-slate-100"}`}
        >
          Runs
        </Link>
      </nav>

      <section className="space-y-3 text-sm">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Connection</p>
        <div className="rounded border border-slate-200 p-3">
          <p className="mb-1 text-xs text-slate-500">API base URL</p>
          <p className="break-all font-mono text-xs text-slate-700">{apiBaseUrl || "n/a"}</p>
          {apiBaseUrl ? <div className="mt-2"><CopyButton text={apiBaseUrl} /></div> : null}
        </div>

        <div className="rounded border border-slate-200 p-3">
          <p className="mb-1 text-xs text-slate-500">User</p>
          <p className="text-slate-700">
            {meQuery.data ? `${meQuery.data.key_id} (${meQuery.data.role})` : meQuery.isLoading ? "Loading..." : "Unknown"}
          </p>
        </div>

        <div className="rounded border border-slate-200 p-3">
          <p className="mb-1 text-xs text-slate-500">Links</p>
          <div className="flex flex-col gap-1">
            <a
              href={apiBaseUrl ? `${apiBaseUrl}/docs` : "#"}
              target="_blank"
              rel="noreferrer"
              className="text-blue-700 hover:underline"
            >
              API docs
            </a>
            <a
              href={apiBaseUrl ? `${apiBaseUrl}/openapi.json` : "#"}
              target="_blank"
              rel="noreferrer"
              className="text-blue-700 hover:underline"
            >
              OpenAPI
            </a>
          </div>
        </div>
      </section>
    </aside>
  );
}
