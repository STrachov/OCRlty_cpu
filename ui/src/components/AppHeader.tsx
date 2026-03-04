import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { getMe } from "../api/runs";
import { clearAuth, getApiBaseUrl } from "../auth/storage";

export function AppHeader() {
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
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex w-full max-w-7xl items-center justify-between gap-4 px-4 py-3 sm:px-6 lg:px-8">
        <div>
          <h1 className="text-lg font-semibold">OCRlty UI</h1>
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
