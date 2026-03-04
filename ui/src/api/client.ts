import { getApiBaseUrl, getApiKey } from "../auth/storage";
import type { ApiClientError, ApiErrorResponse } from "./types";

function createClientRequestId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  const bytes = new Uint8Array(16);
  if (typeof crypto !== "undefined" && typeof crypto.getRandomValues === "function") {
    crypto.getRandomValues(bytes);
  } else {
    for (let i = 0; i < bytes.length; i += 1) {
      bytes[i] = Math.floor(Math.random() * 256);
    }
  }
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function asApiError(params: {
  message: string;
  httpStatus: number;
  code?: string;
  request_id?: string | null;
  details?: Record<string, unknown> | null;
  xRequestId?: string;
}): ApiClientError {
  const err = new Error(params.message) as ApiClientError;
  err.name = "ApiClientError";
  err.httpStatus = params.httpStatus;
  err.code = params.code;
  err.request_id = params.request_id;
  err.details = params.details;
  err.xRequestId = params.xRequestId;
  return err;
}

export async function fetchJson<T>(
  path: string,
  init?: RequestInit
): Promise<{ data: T; meta: { httpStatus: number; xRequestId?: string } }> {
  const baseUrl = getApiBaseUrl();
  if (!baseUrl) {
    throw asApiError({ message: "API base URL is not configured", httpStatus: 0 });
  }

  const apiKey = getApiKey();
  const headers = new Headers(init?.headers ?? {});

  if (apiKey) {
    headers.set("Authorization", `Bearer ${apiKey}`);
  }
  headers.set("X-Request-ID", createClientRequestId());
  if (init?.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${baseUrl}${path}`, {
    ...init,
    headers,
  });

  const xRequestId = response.headers.get("X-Request-ID") ?? undefined;

  if (response.ok) {
    const data = (await response.json()) as T;
    return { data, meta: { httpStatus: response.status, xRequestId } };
  }

  let parsed: ApiErrorResponse | undefined;
  try {
    parsed = (await response.json()) as ApiErrorResponse;
  } catch {
    throw asApiError({
      message: response.statusText || "Request failed",
      httpStatus: response.status,
      xRequestId,
    });
  }

  const payload = parsed?.error;
  throw asApiError({
    message: payload?.message || response.statusText || "Request failed",
    httpStatus: response.status,
    code: payload?.code,
    request_id: payload?.request_id,
    details: payload?.details ?? null,
    xRequestId,
  });
}
