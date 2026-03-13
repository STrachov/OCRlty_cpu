export type MeResponse = {
  key_id: string;
  role: string;
  scopes: string[];
};

export type RuntimeSettingKey =
  | "VLLM_BASE_URL"
  | "VLLM_API_KEY"
  | "VLLM_MODEL"
  | "INFERENCE_BACKEND"
  | "DEBUG_MODE"
  | "S3_PRESIGN_TTL_S";

export type RuntimeSettingItem = {
  key: RuntimeSettingKey;
  value: string | number | boolean | null;
  effective_value: string | number | boolean | null;
  source: "db" | "env";
  updated_at: string | null;
  is_secret: boolean;
  is_set: boolean;
};

export type RuntimeSettingsResponse = {
  items: RuntimeSettingItem[];
};

export type UpdateRuntimeSettingRequest = {
  value: string;
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

export type EvalSummary = {
  items?: number;
  gt_found?: number;
  gt_missing?: number;
  pred_found?: number;
  pred_missing?: number;
  str_mode?: string;
  decimal_sep?: string;
  mismatched?: number;
  [key: string]: unknown;
};

export type RunSummary = {
  run_id?: string;
  created_at?: string | null;
  task_id?: string | null;
  item_count?: number | null;
  ok_count?: number | null;
  error_count?: number | null;
  artifact_rel?: string | null;
  eval_summary?: EvalSummary | null;
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
    summary?: EvalSummary;
    by_request_id?: Record<string, {
      gt_ok: boolean;
      pred_ok: boolean;
      mismatches_count: number;
    }>;
    eval_artifact_rel?: string;
  };
};

export type ExtractArtifact = Record<string, unknown>;

export type EvalMismatch = {
  path?: string;
  reason?: string;
  pred?: unknown;
  gt?: unknown;
  pred_canon?: unknown;
  gt_canon?: unknown;
  pred_err?: string;
  gt_err?: string;
};

export type EvalSample = {
  image_base?: string;
  request_id?: string;
  gt_ok?: boolean;
  pred_ok?: boolean;
  pred?: unknown;
  gt?: unknown;
  mismatches_count?: number;
  mismatches?: EvalMismatch[];
};

export type EvalArtifact = Record<string, unknown> & {
  eval_id?: string;
  created_at?: string;
  run_id?: string;
  gt_id?: string;
  gt_name?: string | null;
  batch_artifact_rel?: string;
  summary?: EvalSummary;
  fields?: Array<Record<string, unknown>>;
  samples?: EvalSample[];
};

export type ApiClientError = Error & {
  httpStatus: number;
  code?: string;
  request_id?: string | null;
  details?: Record<string, unknown> | null;
  xRequestId?: string;
};
