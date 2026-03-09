export type MeResponse = {
  key_id: string;
  role: string;
  scopes: string[];
};

export type TaskSummary = {
  task_id: string;
  description: string;
};

export type TaskListResponse = {
  items: TaskSummary[];
};

export type GroundTruthView = {
  gt_id: string;
  name: string;
  created_at: string;
};

export type GroundTruthListResponse = {
  items: GroundTruthView[];
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

export type JobCreateResponse = {
  job_id: string;
  status: string;
  poll_url: string;
};

export type JobView = {
  job_id: string;
  kind: string;
  status: string;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  owner_key_id?: string | null;
  owner_role?: string | null;
  cancel_requested: boolean;
  request: Record<string, unknown>;
  progress?: Record<string, unknown> | null;
  result_rel?: string | null;
  result_meta?: Record<string, unknown> | null;
  result_bytes?: number | null;
  result_sha256?: string | null;
  result?: Record<string, unknown> | null;
  error_rel?: string | null;
  error?: Record<string, unknown> | null;
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
  eval?: {
    summary?: Record<string, unknown>;
    by_request_id?: Record<string, {
      gt_ok: boolean;
      pred_ok: boolean;
      mismatches_count: number;
    }>;
    eval_artifact_rel?: string;
  };
};

export type ExtractArtifact = Record<string, unknown>;

export type ApiClientError = Error & {
  httpStatus: number;
  code?: string;
  request_id?: string | null;
  details?: Record<string, unknown> | null;
  xRequestId?: string;
};
