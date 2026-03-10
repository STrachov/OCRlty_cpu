import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { getMe, listRuntimeSettings, updateRuntimeSetting } from "../api/runs";
import type { MeResponse, RuntimeSettingItem, RuntimeSettingKey } from "../api/types";
import { clearAuth, getApiBaseUrl, getApiKey, setApiBaseUrl, setApiKey } from "../auth/storage";
import { ErrorPanel } from "../components/ErrorPanel";

type TestState = {
  loading: boolean;
  result: MeResponse | null;
  error: unknown;
};

type RuntimeEditorState = Record<RuntimeSettingKey, string>;

const RUNTIME_SETTING_ORDER: RuntimeSettingKey[] = [
  "VLLM_BASE_URL",
  "VLLM_API_KEY",
  "VLLM_MODEL",
  "INFERENCE_BACKEND",
  "DEBUG_MODE",
  "S3_PRESIGN_TTL_S",
];

const RUNTIME_SETTING_LABELS: Record<RuntimeSettingKey, string> = {
  VLLM_BASE_URL: "vLLM Base URL",
  VLLM_API_KEY: "vLLM API Key",
  VLLM_MODEL: "vLLM Model",
  INFERENCE_BACKEND: "Inference Backend",
  DEBUG_MODE: "Debug Mode",
  S3_PRESIGN_TTL_S: "S3 Presign TTL",
};

const RUNTIME_SETTING_HINTS: Record<RuntimeSettingKey, string> = {
  VLLM_BASE_URL: "HTTP endpoint used by the backend for vLLM requests.",
  VLLM_API_KEY: "Stored as a DB override. Existing secret values stay masked.",
  VLLM_MODEL: "Model identifier used for inference requests.",
  INFERENCE_BACKEND: "Switch between runtime backends without restart.",
  DEBUG_MODE: "Controls debug behavior on the backend.",
  S3_PRESIGN_TTL_S: "Lifetime in seconds for generated presigned URLs.",
};

function normalizeEditableValue(item: RuntimeSettingItem): string {
  if (item.is_secret) {
    return "";
  }
  const candidate = item.value ?? item.effective_value;
  if (candidate == null) {
    return "";
  }
  return String(candidate);
}

function formatDisplayValue(value: RuntimeSettingItem["effective_value"]): string {
  if (value == null) {
    return "not set";
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  return String(value);
}

function formatUpdatedAt(value: string | null): string {
  if (!value) {
    return "Never";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString();
}

function validateRuntimeSetting(key: RuntimeSettingKey, value: string): string | null {
  const trimmed = value.trim();
  if (key === "INFERENCE_BACKEND" && trimmed !== "vllm" && trimmed !== "mock") {
    return "Allowed values: vllm or mock.";
  }
  if (key === "DEBUG_MODE" && !["1", "0", "true", "false", "yes", "no", "on", "off"].includes(trimmed.toLowerCase())) {
    return "Enter a boolean-like value such as true, false, 1, or 0.";
  }
  if (key === "S3_PRESIGN_TTL_S") {
    if (!/^\d+$/.test(trimmed)) {
      return "Enter an integer value.";
    }
    const parsed = Number(trimmed);
    if (parsed < 1 || parsed > 86400) {
      return "Allowed range: 1..86400.";
    }
  }
  if ((key === "VLLM_BASE_URL" || key === "VLLM_MODEL") && trimmed.length === 0) {
    return "Value cannot be empty.";
  }
  return null;
}

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
  const [runtimeEditor, setRuntimeEditor] = useState<RuntimeEditorState>({
    VLLM_BASE_URL: "",
    VLLM_API_KEY: "",
    VLLM_MODEL: "",
    INFERENCE_BACKEND: "vllm",
    DEBUG_MODE: "false",
    S3_PRESIGN_TTL_S: "",
  });
  const [runtimeValidationErrors, setRuntimeValidationErrors] = useState<Partial<Record<RuntimeSettingKey, string>>>({});

  const normalizedBase = useMemo(() => apiBaseUrl.trim().replace(/\/+$/, ""), [apiBaseUrl]);
  const meQuery = useQuery({
    queryKey: ["me"],
    queryFn: getMe,
    staleTime: 5 * 60 * 1000,
  });
  const isAdmin = meQuery.data?.role === "admin";
  const runtimeSettingsQuery = useQuery({
    queryKey: ["runtime-settings"],
    queryFn: listRuntimeSettings,
    enabled: isAdmin,
    staleTime: 0,
  });
  const updateRuntimeSettingMutation = useMutation({
    mutationFn: ({ key, value }: { key: RuntimeSettingKey; value: string }) => updateRuntimeSetting(key, { value }),
    onSuccess: async () => {
      await runtimeSettingsQuery.refetch();
    },
  });

  useEffect(() => {
    if (!runtimeSettingsQuery.data?.items) {
      return;
    }
    setRuntimeEditor((current) => {
      const next = { ...current };
      for (const item of runtimeSettingsQuery.data.items) {
        next[item.key] = normalizeEditableValue(item);
      }
      return next;
    });
  }, [runtimeSettingsQuery.data]);

  const runtimeItems = useMemo(() => {
    const byKey = new Map(runtimeSettingsQuery.data?.items.map((item) => [item.key, item]) ?? []);
    return RUNTIME_SETTING_ORDER.map((key) => byKey.get(key)).filter((item): item is RuntimeSettingItem => Boolean(item));
  }, [runtimeSettingsQuery.data]);

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

      {isAdmin ? (
        <div className="max-w-4xl rounded-md border border-slate-200 bg-white p-4">
          <div className="mb-4">
            <h3 className="text-lg font-semibold">Runtime Settings</h3>
            <p className="text-sm text-slate-600">
              Backend reads these values at runtime. Changes apply automatically after a short cache TTL.
            </p>
          </div>

          {runtimeSettingsQuery.isLoading ? <p className="text-sm text-slate-600">Loading runtime settings...</p> : null}
          {runtimeSettingsQuery.isError ? <ErrorPanel error={runtimeSettingsQuery.error} /> : null}
          {updateRuntimeSettingMutation.isError ? <ErrorPanel error={updateRuntimeSettingMutation.error} /> : null}

          {runtimeItems.length > 0 ? (
            <div className="space-y-4">
              {runtimeItems.map((item) => {
                const validationError = runtimeValidationErrors[item.key];
                const isSaving = updateRuntimeSettingMutation.isPending && updateRuntimeSettingMutation.variables?.key === item.key;
                const sourceClasses =
                  item.source === "db"
                    ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                    : "border-amber-200 bg-amber-50 text-amber-700";

                return (
                  <div key={item.key} className="rounded-md border border-slate-200 p-4">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div>
                        <div className="flex flex-wrap items-center gap-2">
                          <h4 className="text-sm font-semibold text-slate-900">{RUNTIME_SETTING_LABELS[item.key]}</h4>
                          <span className="rounded border border-slate-200 bg-slate-50 px-2 py-0.5 font-mono text-xs text-slate-600">
                            {item.key}
                          </span>
                          <span className={`rounded border px-2 py-0.5 text-xs font-medium ${sourceClasses}`}>
                            source: {item.source}
                          </span>
                          {item.is_secret ? (
                            <span className="rounded border border-slate-200 bg-slate-50 px-2 py-0.5 text-xs text-slate-600">
                              secret
                            </span>
                          ) : null}
                        </div>
                        <p className="mt-1 text-sm text-slate-600">{RUNTIME_SETTING_HINTS[item.key]}</p>
                      </div>
                      <div className="text-right text-xs text-slate-500">
                        <p>Effective: {item.is_secret ? "***" : formatDisplayValue(item.effective_value)}</p>
                        <p>Updated: {formatUpdatedAt(item.updated_at)}</p>
                      </div>
                    </div>

                    <div className="mt-4 space-y-3">
                      {item.key === "INFERENCE_BACKEND" ? (
                        <label className="block text-sm font-medium text-slate-800">
                          Value
                          <select
                            value={runtimeEditor[item.key]}
                            onChange={(e) => {
                              const value = e.target.value;
                              setRuntimeEditor((current) => ({ ...current, [item.key]: value }));
                              setRuntimeValidationErrors((current) => ({ ...current, [item.key]: undefined }));
                            }}
                            className="mt-1 w-full rounded border border-slate-300 px-3 py-2"
                          >
                            <option value="vllm">vllm</option>
                            <option value="mock">mock</option>
                          </select>
                        </label>
                      ) : item.key === "DEBUG_MODE" ? (
                        <label className="block text-sm font-medium text-slate-800">
                          Value
                          <select
                            value={runtimeEditor[item.key]}
                            onChange={(e) => {
                              const value = e.target.value;
                              setRuntimeEditor((current) => ({ ...current, [item.key]: value }));
                              setRuntimeValidationErrors((current) => ({ ...current, [item.key]: undefined }));
                            }}
                            className="mt-1 w-full rounded border border-slate-300 px-3 py-2"
                          >
                            <option value="true">true</option>
                            <option value="false">false</option>
                          </select>
                        </label>
                      ) : (
                        <label className="block text-sm font-medium text-slate-800">
                          Value
                          <input
                            type={item.key === "VLLM_API_KEY" ? "password" : item.key === "S3_PRESIGN_TTL_S" ? "number" : "text"}
                            inputMode={item.key === "S3_PRESIGN_TTL_S" ? "numeric" : undefined}
                            min={item.key === "S3_PRESIGN_TTL_S" ? 1 : undefined}
                            max={item.key === "S3_PRESIGN_TTL_S" ? 86400 : undefined}
                            value={runtimeEditor[item.key]}
                            onChange={(e) => {
                              const value = e.target.value;
                              setRuntimeEditor((current) => ({ ...current, [item.key]: value }));
                              setRuntimeValidationErrors((current) => ({ ...current, [item.key]: undefined }));
                            }}
                            className="mt-1 w-full rounded border border-slate-300 px-3 py-2"
                            placeholder={item.is_secret ? "Enter new secret value" : undefined}
                          />
                        </label>
                      )}

                      <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-slate-500">
                        <p>
                          Override: {item.is_set ? (item.is_secret ? "***" : formatDisplayValue(item.value)) : "not set"}
                        </p>
                        <button
                          type="button"
                          disabled={isSaving}
                          onClick={() => {
                            const value = runtimeEditor[item.key];
                            const validation = validateRuntimeSetting(item.key, value);
                            if (validation) {
                              setRuntimeValidationErrors((current) => ({ ...current, [item.key]: validation }));
                              return;
                            }
                            setRuntimeValidationErrors((current) => ({ ...current, [item.key]: undefined }));
                            updateRuntimeSettingMutation.mutate({ key: item.key, value });
                          }}
                          className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-60"
                        >
                          {isSaving ? "Saving..." : "Save"}
                        </button>
                      </div>

                      {validationError ? <p className="text-sm text-rose-700">{validationError}</p> : null}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : null}
        </div>
      ) : meQuery.data ? (
        <div className="max-w-2xl rounded-md border border-slate-200 bg-white p-4 text-sm text-slate-600">
          <p className="font-medium text-slate-900">Runtime Settings</p>
          <p className="mt-1">This section is available only for users with the `admin` role.</p>
        </div>
      ) : null}
    </section>
  );
}
