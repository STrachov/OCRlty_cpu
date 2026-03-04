import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { getMe } from "../api/runs";
import { clearAuth, getApiBaseUrl } from "../auth/storage";
import { Breadcrumbs } from "./Breadcrumbs";

export function TopBar() {
  const navigate = useNavigate();
  const apiBaseUrl = getApiBaseUrl();
  const meQuery = useQuery({
    queryKey: ["me"],
    queryFn: getMe,
    staleTime: 5 * 60 * 1000,
  });

  const userLabel = meQuery.data
    ? `${meQuery.data.key_id} (${meQuery.data.role})`
    : meQuery.isLoading
      ? "Loading user..."
      : "Unknown user";

  return (
    <header className="border-b border-slate-200 bg-white px-6 py-3">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-lg font-semibold">OCRlty UI</h1>
          <Breadcrumbs />
          <p className="text-sm text-slate-600">API: {apiBaseUrl || "n/a"}</p>
          <p className="text-sm text-slate-600">User: {userLabel}</p>
        </div>
        <button
          type="button"
          onClick={() => {
            clearAuth();
            navigate("/login", { replace: true });
          }}
          className="rounded border border-slate-300 px-3 py-2 text-sm hover:bg-slate-100"
        >
          Logout
        </button>
      </div>
    </header>
  );
}
