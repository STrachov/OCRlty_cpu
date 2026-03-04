import { fetchJson } from "./client";
import type { BatchArtifact, ExtractArtifact, MeResponse, RunsListResponse } from "./types";

export async function getMe(): Promise<MeResponse> {
  const { data } = await fetchJson<MeResponse>("/v1/me");
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
