export type MeResponse = {
  key_id: string;
  role: string;
  scopes: string[];
};

export type RunSummary = {
  run_id?: string;
  created_at?: string | null;
  task_id?: string | null;
  item_count?: number | null;
  ok_count?: number | null;
  error_count?: number | null;
  artifact_rel?: string | null;
  [key: string]: unknown;
};

export type RunsListResponse = {
  items: RunSummary[];
  limit: number;
  next_cursor: string | null;
};

export type ApiErrorPayload = {
  code?: string;
  message?: string;
  request_id?: string | null;
  details?: Record<string, unknown> | null;
};

export type ApiErrorResponse = {
  error?: ApiErrorPayload;
};

export type BatchArtifact = Record<string, unknown> & {
  items?: Record<string, unknown>[];
};

export type ExtractArtifact = Record<string, unknown>;

export type ApiClientError = Error & {
  httpStatus: number;
  code?: string;
  request_id?: string | null;
  details?: Record<string, unknown> | null;
  xRequestId?: string;
};
