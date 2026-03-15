import { fetchJson } from "./client";
import type {
  BatchArtifact,
  DeleteRunResponse,
  EvalArtifact,
  ExtractArtifact,
  GroundTruthFromRunRequest,
  GroundTruthListResponse,
  GroundTruthUpdateRequest,
  GroundTruthView,
  JobCreateResponse,
  JobView,
  MeResponse,
  RunCatalogResponse,
  RuntimeSettingItem,
  RuntimeSettingsResponse,
  RunsListResponse,
  TaskListResponse,
  UpdateRuntimeSettingRequest,
} from "./types";

export async function getMe(): Promise<MeResponse> {
  const { data } = await fetchJson<MeResponse>("/v1/me");
  return data;
}

export async function listRuntimeSettings(): Promise<RuntimeSettingsResponse> {
  const { data } = await fetchJson<RuntimeSettingsResponse>("/v1/admin/runtime_settings");
  return data;
}

export async function updateRuntimeSetting(
  key: string,
  payload: UpdateRuntimeSettingRequest
): Promise<RuntimeSettingItem> {
  const { data } = await fetchJson<RuntimeSettingItem>(`/v1/admin/runtime_settings/${encodeURIComponent(key)}`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
  return data;
}

export async function listRuns(limit: number, cursor?: string, taskId?: string): Promise<RunsListResponse> {
  const query = new URLSearchParams({ limit: String(limit) });
  if (cursor) {
    query.set("cursor", cursor);
  }
  if (taskId) {
    query.set("task_id", taskId);
  }
  const { data } = await fetchJson<RunsListResponse>(`/v1/runs?${query.toString()}`);
  return data;
}

export async function listRunsCatalog(limit: number, cursor?: string): Promise<RunCatalogResponse> {
  const query = new URLSearchParams({ limit: String(limit) });
  if (cursor) {
    query.set("cursor", cursor);
  }
  const { data } = await fetchJson<RunCatalogResponse>(`/v1/runs/catalog?${query.toString()}`);
  return data;
}

export async function getRun(runId: string): Promise<BatchArtifact> {
  const { data } = await fetchJson<BatchArtifact>(`/v1/runs/${encodeURIComponent(runId)}`);
  return data;
}

export async function deleteRun(runId: string): Promise<DeleteRunResponse> {
  const { data } = await fetchJson<DeleteRunResponse>(`/v1/runs/${encodeURIComponent(runId)}`, {
    method: "DELETE",
  });
  return data;
}

export async function getItem(requestId: string): Promise<ExtractArtifact> {
  const { data } = await fetchJson<ExtractArtifact>(`/v1/runs/item/${encodeURIComponent(requestId)}`);
  return data;
}

export async function getEvalArtifact(artifactRel: string): Promise<EvalArtifact> {
  const query = new URLSearchParams({ artifact_rel: artifactRel });
  const { data } = await fetchJson<EvalArtifact>(`/v1/runs/eval?${query.toString()}`);
  return data;
}

export async function getTasks(): Promise<TaskListResponse> {
  const { data } = await fetchJson<TaskListResponse>("/v1/tasks");
  return data;
}

export async function listGroundTruths(limit: number, offset: number): Promise<GroundTruthListResponse> {
  const query = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  const { data } = await fetchJson<GroundTruthListResponse>(`/v1/ground_truths?${query.toString()}`);
  return data;
}

export async function uploadGroundTruth(file: File): Promise<GroundTruthView> {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await fetchJson<GroundTruthView>("/v1/ground_truths/upload", {
    method: "POST",
    body: formData,
  });
  return data;
}

export async function createGroundTruthFromRun(payload: GroundTruthFromRunRequest): Promise<GroundTruthView> {
  const { data } = await fetchJson<GroundTruthView>("/v1/ground_truths/from_run", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  return data;
}

export async function createGroundTruth(payload: GroundTruthUpdateRequest): Promise<GroundTruthView> {
  const { data } = await fetchJson<GroundTruthView>("/v1/ground_truths", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  return data;
}

export async function createGroundTruthDraftFromRun(payload: GroundTruthFromRunRequest): Promise<unknown> {
  const { data } = await fetchJson<unknown>("/v1/ground_truths/from_run/draft", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  return data;
}

export async function getGroundTruthContent(gtId: string): Promise<unknown> {
  const { data } = await fetchJson<unknown>(`/v1/ground_truths/${encodeURIComponent(gtId)}/content`);
  return data;
}

export async function updateGroundTruth(gtId: string, payload: GroundTruthUpdateRequest): Promise<GroundTruthView> {
  const { data } = await fetchJson<GroundTruthView>(`/v1/ground_truths/${encodeURIComponent(gtId)}`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
  return data;
}

export async function createRunAsync(formData: FormData): Promise<JobCreateResponse> {
  const { data } = await fetchJson<JobCreateResponse>("/v1/batch_extract_upload_async", {
    method: "POST",
    body: formData,
  });
  return data;
}

export async function getJob(jobId: string): Promise<JobView> {
  const { data } = await fetchJson<JobView>(`/v1/jobs/${encodeURIComponent(jobId)}`);
  return data;
}
