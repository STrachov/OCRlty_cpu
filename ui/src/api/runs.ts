import { fetchJson } from "./client";
import type {
  BatchArtifact,
  ExtractArtifact,
  GroundTruthListResponse,
  GroundTruthView,
  JobCreateResponse,
  JobView,
  MeResponse,
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

export async function listRuns(limit: number, cursor?: string): Promise<RunsListResponse> {
  const query = new URLSearchParams({ limit: String(limit) });
  if (cursor) {
    query.set("cursor", cursor);
  }
  const { data } = await fetchJson<RunsListResponse>(`/v1/runs?${query.toString()}`);
  return data;
}

export async function getRun(runId: string): Promise<BatchArtifact> {
  const { data } = await fetchJson<BatchArtifact>(`/v1/runs/${encodeURIComponent(runId)}`);
  return data;
}

export async function getItem(requestId: string): Promise<ExtractArtifact> {
  const { data } = await fetchJson<ExtractArtifact>(`/v1/runs/item/${encodeURIComponent(requestId)}`);
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
