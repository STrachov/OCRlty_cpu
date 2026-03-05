import { useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import type { MeResponse } from "../api/types";
import { clearAuth, getApiBaseUrl, getApiKey, setApiBaseUrl, setApiKey } from "../auth/storage";
import { ErrorPanel } from "../components/ErrorPanel";

type TestState = {
  loading: boolean;
  result: MeResponse | null;
  error: unknown;
};

function makeClientRequestId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now().toString(16)}${Math.random().toString(16).slice(2)}`;
}

function asApiError(params: {
  message: string;
  httpStatus: number;
  code?: string;
  request_id?: string | null;
  details?: Record<string, unknown> | null;
  xRequestId?: string;
}) {
  const err = new Error(params.message) as Error & {
    httpStatus: number;
    code?: string;
    request_id?: string | null;
    details?: Record<string, unknown> | null;
    xRequestId?: string;
  };
  err.httpStatus = params.httpStatus;
  err.code = params.code;
  err.request_id = params.request_id;
  err.details = params.details;
  err.xRequestId = params.xRequestId;
  return err;
}

async function testMeWithCredentials(baseUrl: string, apiKey: string): Promise<MeResponse> {
  const normalizedBase = baseUrl.trim().replace(/\/+$/, "");
  const response = await fetch(`${normalizedBase}/v1/me`, {
    method: "GET",
    headers: {
      Authorization: `Bearer ${apiKey.trim()}`,
      "X-Request-ID": makeClientRequestId(),
    },
  });

  const xRequestId = response.headers.get("x-request-id") ?? undefined;
  if (response.ok) {
    return (await response.json()) as MeResponse;
  }

  try {
    const parsed = (await response.json()) as {
      error?: {
        code?: string;
        message?: string;
        request_id?: string | null;
        details?: Record<string, unknown> | null;
      };
    };
    const payload = parsed?.error;
    throw asApiError({
      message: payload?.message || response.statusText || "Request failed",
      httpStatus: response.status,
      code: payload?.code,
      request_id: payload?.request_id,
      details: payload?.details ?? null,
      xRequestId,
    });
  } catch (e) {
    if (typeof e === "object" && e && "httpStatus" in e) {
      throw e;
    }
    throw asApiError({
      message: response.statusText || "Request failed",
      httpStatus: response.status,
      xRequestId,
    });
  }
}

export function SettingsPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [apiBaseUrl, setApiBaseUrlInput] = useState(getApiBaseUrl() ?? "http://127.0.0.1:8080");
  const [apiKey, setApiKeyInput] = useState(getApiKey() ?? "");
  const [showSaved, setShowSaved] = useState(false);
  const [testState, setTestState] = useState<TestState>({ loading: false, result: null, error: null });

  const normalizedBase = useMemo(() => apiBaseUrl.trim().replace(/\/+$/, ""), [apiBaseUrl]);

  return (
    <section className="space-y-4">
      <div>
        <h2 className="text-xl font-semibold">Settings</h2>
        <p className="text-sm text-slate-600">Manage API connection and switch user credentials.</p>
      </div>

      <div className="max-w-2xl rounded-md border border-slate-200 bg-white p-4">
        <div className="space-y-4">
          <label className="block text-sm font-medium text-slate-800">
            API Base URL
            <input
              type="text"
              value={apiBaseUrl}
              onChange={(e) => {
                setApiBaseUrlInput(e.target.value);
                setShowSaved(false);
              }}
              className="mt-1 w-full rounded border border-slate-300 px-3 py-2"
              placeholder="http://127.0.0.1:8080"
            />
          </label>

          <label className="block text-sm font-medium text-slate-800">
            API Key
            <input
              type="text"
              value={apiKey}
              onChange={(e) => {
                setApiKeyInput(e.target.value);
                setShowSaved(false);
              }}
              className="mt-1 w-full rounded border border-slate-300 px-3 py-2"
              placeholder="Paste API key"
            />
          </label>

          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              disabled={testState.loading}
              onClick={async () => {
                setShowSaved(false);
                setTestState({ loading: true, result: null, error: null });
                try {
                  const me = await testMeWithCredentials(apiBaseUrl, apiKey);
                  setTestState({ loading: false, result: me, error: null });
                } catch (e) {
                  setTestState({ loading: false, result: null, error: e });
                }
              }}
              className="rounded border border-slate-300 px-4 py-2 text-sm hover:bg-slate-100 disabled:opacity-60"
            >
              {testState.loading ? "Testing..." : "Test"}
            </button>

            <button
              type="button"
              onClick={async () => {
                setApiBaseUrl(apiBaseUrl);
                setApiKey(apiKey);
                setShowSaved(true);
                await queryClient.invalidateQueries({ queryKey: ["me"] });
                await queryClient.refetchQueries({ queryKey: ["me"], type: "active" });
              }}
              className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700"
            >
              Save
            </button>

            <button
              type="button"
              onClick={() => {
                clearAuth();
                navigate("/login", { replace: true });
              }}
              className="rounded border border-slate-300 px-4 py-2 text-sm hover:bg-slate-100"
            >
              Logout
            </button>
          </div>

          {showSaved ? <p className="text-sm text-emerald-700">Saved</p> : null}

          {testState.result ? (
            <div className="rounded-md border border-emerald-200 bg-emerald-50 p-4 text-sm">
              <p>
                <span className="font-medium">key_id:</span> {testState.result.key_id}
              </p>
              <p>
                <span className="font-medium">role:</span> {testState.result.role}
              </p>
              <div className="mt-2 flex flex-wrap gap-2">
                {testState.result.scopes.map((scope) => (
                  <span key={scope} className="rounded bg-emerald-100 px-2 py-1 text-xs text-emerald-800">
                    {scope}
                  </span>
                ))}
              </div>
            </div>
          ) : null}

          {testState.error ? <ErrorPanel error={testState.error} /> : null}
        </div>
      </div>

      <div className="max-w-2xl rounded-md border border-slate-200 bg-white p-4 text-sm">
        <p className="mb-2 font-medium">Links</p>
        <div className="flex flex-col gap-1">
          <a
            href={normalizedBase ? `${normalizedBase}/docs` : "#"}
            target="_blank"
            rel="noreferrer"
            className="text-blue-700 hover:underline"
          >
            API docs
          </a>
          <a
            href={normalizedBase ? `${normalizedBase}/openapi.json` : "#"}
            target="_blank"
            rel="noreferrer"
            className="text-blue-700 hover:underline"
          >
            OpenAPI
          </a>
        </div>
      </div>
    </section>
  );
}
