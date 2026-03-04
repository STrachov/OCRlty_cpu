import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { getMe } from "../api/runs";
import { clearAuth, getApiBaseUrl, setApiBaseUrl, setApiKey } from "../auth/storage";
import { ErrorPanel } from "../components/ErrorPanel";

export function LoginPage() {
  const navigate = useNavigate();
  const [apiBaseUrl, setApiBaseUrlInput] = useState(getApiBaseUrl() ?? "http://127.0.0.1:8080");
  const [apiKey, setApiKeyInput] = useState("");

  const meMutation = useMutation({
    mutationFn: async () => {
      setApiBaseUrl(apiBaseUrl);
      setApiKey(apiKey);
      return getMe();
    },
    onSuccess: () => {
      navigate("/runs", { replace: true });
    },
  });

  return (
    <div className="min-h-screen bg-slate-50 px-4 py-10">
      <div className="mx-auto max-w-xl rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
        <h1 className="text-xl font-semibold">OCRlty UI Login</h1>
        <p className="mt-2 text-sm text-slate-600">Connect to your OCRlty Orchestrator API.</p>

        <form
          className="mt-6 space-y-4"
          onSubmit={(e) => {
            e.preventDefault();
            meMutation.mutate();
          }}
        >
          <label className="block text-sm font-medium text-slate-800">
            API Base URL
            <input
              type="text"
              value={apiBaseUrl}
              onChange={(e) => setApiBaseUrlInput(e.target.value)}
              className="mt-1 w-full rounded border border-slate-300 px-3 py-2"
              placeholder="http://127.0.0.1:8080"
              required
            />
          </label>

          <label className="block text-sm font-medium text-slate-800">
            API Key
            <input
              type="text"
              value={apiKey}
              onChange={(e) => setApiKeyInput(e.target.value)}
              className="mt-1 w-full rounded border border-slate-300 px-3 py-2"
              placeholder="Paste API key"
              required
            />
          </label>

          <div className="flex flex-wrap gap-3">
            <button
              type="submit"
              disabled={meMutation.isPending}
              className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-60"
            >
              {meMutation.isPending ? "Checking..." : "Log in"}
            </button>
            <button
              type="button"
              onClick={() => {
                clearAuth();
                setApiKeyInput("");
              }}
              className="rounded border border-slate-300 px-4 py-2 text-sm hover:bg-slate-100"
            >
              Logout
            </button>
          </div>
        </form>

        {meMutation.isSuccess ? (
          <div className="mt-6 rounded-md border border-emerald-200 bg-emerald-50 p-4 text-sm">
            <p>
              <span className="font-medium">key_id:</span> {meMutation.data.key_id}
            </p>
            <p>
              <span className="font-medium">role:</span> {meMutation.data.role}
            </p>
            <div className="mt-2 flex flex-wrap gap-2">
              {meMutation.data.scopes.map((scope) => (
                <span key={scope} className="rounded bg-emerald-100 px-2 py-1 text-xs text-emerald-800">
                  {scope}
                </span>
              ))}
            </div>
          </div>
        ) : null}

        {meMutation.isError ? (
          <div className="mt-6">
            <ErrorPanel error={meMutation.error} />
          </div>
        ) : null}
      </div>
    </div>
  );
}
