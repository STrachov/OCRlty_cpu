import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";
import {
  createRunAsync,
  getJob,
  getMe,
  getTasks,
  listGroundTruths,
  uploadGroundTruth,
} from "../api/runs";
import type { GroundTruthView, JobView, TaskSummary } from "../api/types";
import { ErrorPanel } from "../components/ErrorPanel";
import { useLayoutContext } from "../layout/LayoutContext";

type GroundTruthMode = "none" | "existing" | "upload";

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value >= 10 || unitIndex === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unitIndex]}`;
}

function parseDateValue(value: string): number {
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? Number.NEGATIVE_INFINITY : parsed;
}

function extractRunId(job: JobView | null | undefined): string | null {
  if (!job) {
    return null;
  }
  const metaRunId = job.result_meta?.run_id;
  if (typeof metaRunId === "string" && metaRunId) {
    return metaRunId;
  }
  const resultRunId = job.result?.run_id;
  if (typeof resultRunId === "string" && resultRunId) {
    return resultRunId;
  }
  return null;
}

export function CreateRunPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const { setCreateRunPanelState } = useLayoutContext();
  const [selectedTaskId, setSelectedTaskId] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [gtMode, setGtMode] = useState<GroundTruthMode>("none");
  const [selectedGtId, setSelectedGtId] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [autoOpenQueued, setAutoOpenQueued] = useState(false);

  const meQuery = useQuery({
    queryKey: ["me"],
    queryFn: getMe,
    staleTime: 5 * 60 * 1000,
  });
  const tasksQuery = useQuery({
    queryKey: ["tasks"],
    queryFn: getTasks,
  });

  const canUseGroundTruths = meQuery.data?.scopes.includes("debug:run") ?? false;
  const groundTruthsQuery = useQuery({
    queryKey: ["ground_truths", 50, 0],
    queryFn: () => listGroundTruths(50, 0),
    enabled: canUseGroundTruths,
  });

  const taskItems = tasksQuery.data?.items ?? [];

  useEffect(() => {
    if (!selectedTaskId && taskItems.length > 0) {
      setSelectedTaskId(taskItems[0].task_id);
    }
  }, [selectedTaskId, taskItems]);

  const selectedTask = useMemo<TaskSummary | null>(
    () => taskItems.find((task) => task.task_id === selectedTaskId) ?? null,
    [selectedTaskId, taskItems]
  );

  const sortedGroundTruths = useMemo(() => {
    const items = groundTruthsQuery.data?.items ?? [];
    return [...items].sort((a, b) => {
      const ta = parseDateValue(a.created_at);
      const tb = parseDateValue(b.created_at);
      if (ta !== tb) {
        return tb - ta;
      }
      return b.gt_id.localeCompare(a.gt_id);
    });
  }, [groundTruthsQuery.data]);

  const selectedGroundTruth = useMemo<GroundTruthView | null>(
    () => sortedGroundTruths.find((item) => item.gt_id === selectedGtId) ?? null,
    [selectedGtId, sortedGroundTruths]
  );

  useEffect(() => {
    if (!canUseGroundTruths && gtMode !== "none") {
      setGtMode("none");
      setSelectedGtId("");
    }
  }, [canUseGroundTruths, gtMode]);

  const uploadGroundTruthMutation = useMutation({
    mutationFn: uploadGroundTruth,
    onSuccess: async (uploaded) => {
      const nextItems = await queryClient.fetchQuery({
        queryKey: ["ground_truths", 50, 0],
        queryFn: () => listGroundTruths(50, 0),
      });
      const existsInList = nextItems.items.some((item) => item.gt_id === uploaded.gt_id);
      if (!existsInList) {
        void queryClient.setQueryData(["ground_truths", 50, 0], {
          items: [uploaded, ...nextItems.items],
        });
      }
      setGtMode("upload");
      setSelectedGtId(uploaded.gt_id);
    },
  });

  const submitMutation = useMutation({
    mutationFn: createRunAsync,
    onSuccess: async (job) => {
      setJobId(job.job_id);
      setAutoOpenQueued(true);
      await queryClient.invalidateQueries({ queryKey: ["runs"] });
    },
  });

  const jobQuery = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJob(jobId ?? ""),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const status = (query.state.data as JobView | undefined)?.status;
      return status === "queued" || status === "running" ? 1500 : false;
    },
  });

  const jobData = jobQuery.data ?? null;
  const runId = extractRunId(jobData);
  const totalSize = useMemo(() => files.reduce((sum, file) => sum + file.size, 0), [files]);
  const effectiveGtId = gtMode === "none" ? "" : selectedGtId;
  const isFormValid =
    Boolean(selectedTaskId) &&
    files.length > 0 &&
    (gtMode === "none" || Boolean(effectiveGtId));

  useEffect(() => {
    if (autoOpenQueued && jobData?.status === "succeeded" && runId) {
      const timer = window.setTimeout(() => {
        navigate(`/runs/${encodeURIComponent(runId)}`);
      }, 1200);
      return () => window.clearTimeout(timer);
    }
    return undefined;
  }, [autoOpenQueued, jobData?.status, navigate, runId]);

  useEffect(() => {
    setCreateRunPanelState({
      selectedTaskId,
      selectedTaskDescription: selectedTask?.description ?? "",
      files: files.map((file) => ({ name: file.name, size: file.size })),
      totalSize,
      gtMode,
      selectedGroundTruth: effectiveGtId ? selectedGroundTruth : null,
      job: jobData,
      jobId,
      runId,
      submitError: submitMutation.error ?? null,
      gtError: uploadGroundTruthMutation.error ?? null,
      jobFetchError: jobQuery.error ?? null,
      isSubmitting: submitMutation.isPending,
      isUploadingGt: uploadGroundTruthMutation.isPending,
    });
  }, [
    effectiveGtId,
    files,
    gtMode,
    jobData,
    jobId,
    jobQuery.error,
    runId,
    selectedGroundTruth,
    selectedTask?.description,
    selectedTaskId,
    setCreateRunPanelState,
    submitMutation.error,
    submitMutation.isPending,
    totalSize,
    uploadGroundTruthMutation.error,
    uploadGroundTruthMutation.isPending,
  ]);

  useEffect(() => () => {
    setCreateRunPanelState(null);
  }, [setCreateRunPanelState]);

  function appendFiles(nextFiles: FileList | File[]) {
    const incoming = Array.from(nextFiles);
    if (incoming.length === 0) {
      return;
    }
    setFiles((current) => [...current, ...incoming]);
  }

  function removeFile(index: number) {
    setFiles((current) => current.filter((_, currentIndex) => currentIndex !== index));
  }

  function handleRunSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!isFormValid || submitMutation.isPending) {
      return;
    }

    const formData = new FormData();
    formData.append("task_id", selectedTaskId);
    for (const file of files) {
      formData.append("files", file);
    }
    if (effectiveGtId) {
      formData.append("gt_id", effectiveGtId);
    }

    submitMutation.mutate(formData);
  }

  return (
    <section className="mx-auto max-w-4xl space-y-5">
      <div>
        <h2 className="text-xl font-semibold">New run</h2>
        <p className="text-sm text-slate-600">Create an async batch run from uploaded files.</p>
      </div>

      {tasksQuery.isLoading ? <p className="text-sm text-slate-600">Loading tasks...</p> : null}
      {tasksQuery.isError ? <ErrorPanel error={tasksQuery.error} /> : null}
      {meQuery.isError ? <ErrorPanel error={meQuery.error} /> : null}

      {!tasksQuery.isLoading && !tasksQuery.isError ? (
        <form className="space-y-5" onSubmit={handleRunSubmit}>
          <div className="rounded-md border border-slate-200 bg-white p-5">
            <h3 className="text-base font-semibold">Task</h3>
            <div className="mt-4">
              <label className="block text-sm font-medium text-slate-800">
                task_id
                <select
                  className="mt-1 w-full rounded border border-slate-300 px-3 py-2"
                  value={selectedTaskId}
                  onChange={(event) => setSelectedTaskId(event.target.value)}
                  disabled={taskItems.length === 0}
                >
                  {taskItems.length === 0 ? <option value="">No tasks available</option> : null}
                  {taskItems.map((task) => (
                    <option key={task.task_id} value={task.task_id}>
                      {task.task_id} - {task.description || "No description"}
                    </option>
                  ))}
                </select>
              </label>
              {selectedTask?.description ? (
                <p className="mt-2 text-sm text-slate-600">{selectedTask.description}</p>
              ) : null}
              {taskItems.length === 0 ? (
                <div className="mt-3 rounded border border-dashed border-slate-300 bg-slate-50 p-4 text-sm text-slate-600">
                  No tasks available
                </div>
              ) : null}
            </div>
          </div>

          <div className="rounded-md border border-slate-200 bg-white p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="text-base font-semibold">Files</h3>
                <p className="text-sm text-slate-600">Add one or more receipt images for this batch run.</p>
              </div>
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="rounded border border-slate-300 px-4 py-2 text-sm hover:bg-slate-100"
              >
                Select files
              </button>
            </div>

            <div
              className="mt-4 rounded-lg border border-dashed border-slate-300 bg-slate-50 p-8 text-center"
              onDragOver={(event) => {
                event.preventDefault();
              }}
              onDrop={(event) => {
                event.preventDefault();
                appendFiles(event.dataTransfer.files);
              }}
            >
              <p className="text-sm font-medium text-slate-700">Drag and drop files here</p>
              <p className="mt-1 text-sm text-slate-500">or use the file picker to add them manually</p>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              multiple
              className="hidden"
              onChange={(event) => {
                if (event.target.files) {
                  appendFiles(event.target.files);
                  event.target.value = "";
                }
              }}
            />

            {files.length > 0 ? (
              <div className="mt-4 overflow-hidden rounded-md border border-slate-200">
                <table className="min-w-full text-sm">
                  <thead className="bg-slate-100 text-left">
                    <tr>
                      <th className="px-3 py-2">Filename</th>
                      <th className="px-3 py-2">Size</th>
                      <th className="px-3 py-2 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {files.map((file, index) => (
                      <tr key={`${file.name}-${file.size}-${index}`} className="border-t border-slate-100">
                        <td className="px-3 py-2">{file.name}</td>
                        <td className="px-3 py-2">{formatBytes(file.size)}</td>
                        <td className="px-3 py-2 text-right">
                          <button
                            type="button"
                            onClick={() => removeFile(index)}
                            className="rounded border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-100"
                          >
                            Remove
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="mt-4 text-sm text-slate-500">No files selected yet.</p>
            )}
          </div>

          {canUseGroundTruths ? (
            <div className="rounded-md border border-slate-200 bg-white p-5">
              <div>
                <h3 className="text-base font-semibold">Ground truths</h3>
                <p className="text-sm text-slate-600">Optional auto-eval source for this run.</p>
              </div>

              <div className="mt-4 flex flex-wrap gap-3">
                {([
                  ["none", "None"],
                  ["existing", "Select existing"],
                  ["upload", "Upload new"],
                ] as Array<[GroundTruthMode, string]>).map(([mode, label]) => (
                  <label
                    key={mode}
                    className={`rounded border px-3 py-2 text-sm ${gtMode === mode ? "border-slate-900 bg-slate-900 text-white" : "border-slate-300 bg-white text-slate-900"}`}
                  >
                    <input
                      type="radio"
                      name="gt_mode"
                      value={mode}
                      checked={gtMode === mode}
                      onChange={() => setGtMode(mode)}
                      className="sr-only"
                    />
                    {label}
                  </label>
                ))}
              </div>

              {gtMode === "existing" ? (
                <div className="mt-4 space-y-3">
                  {groundTruthsQuery.isLoading ? <p className="text-sm text-slate-600">Loading ground truths...</p> : null}
                  {groundTruthsQuery.isError ? <ErrorPanel error={groundTruthsQuery.error} /> : null}
                  {!groundTruthsQuery.isLoading && !groundTruthsQuery.isError ? (
                    sortedGroundTruths.length > 0 ? (
                      <label className="block text-sm font-medium text-slate-800">
                        gt_id
                        <select
                          className="mt-1 w-full rounded border border-slate-300 px-3 py-2"
                          value={selectedGtId}
                          onChange={(event) => setSelectedGtId(event.target.value)}
                        >
                          <option value="">Select ground truth</option>
                          {sortedGroundTruths.map((item) => (
                            <option key={item.gt_id} value={item.gt_id}>
                              {item.name} - {item.created_at}
                            </option>
                          ))}
                        </select>
                      </label>
                    ) : (
                      <div className="rounded border border-dashed border-slate-300 bg-slate-50 p-4 text-sm text-slate-600">
                        No ground truths uploaded yet
                      </div>
                    )
                  ) : null}
                </div>
              ) : null}

              {gtMode === "upload" ? (
                <div className="mt-4 space-y-3">
                  <div className="flex flex-wrap items-center gap-3">
                    <input
                      type="file"
                      accept="application/json,.json"
                      onChange={(event) => {
                        const nextFile = event.target.files?.[0];
                        if (nextFile) {
                          uploadGroundTruthMutation.mutate(nextFile);
                          event.target.value = "";
                        }
                      }}
                      disabled={uploadGroundTruthMutation.isPending}
                      className="block text-sm"
                    />
                    {uploadGroundTruthMutation.isPending ? (
                      <span className="text-sm text-slate-600">Uploading ground truth...</span>
                    ) : null}
                  </div>
                  {uploadGroundTruthMutation.isError ? <ErrorPanel error={uploadGroundTruthMutation.error} /> : null}
                  {selectedGroundTruth ? (
                    <div className="rounded border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-900">
                      Using GT `{selectedGroundTruth.name}` ({selectedGroundTruth.gt_id})
                    </div>
                  ) : (
                    <p className="text-sm text-slate-500">Upload a UTF-8 JSON file to create and select a new GT.</p>
                  )}
                </div>
              ) : null}
            </div>
          ) : null}

          {submitMutation.isError ? <ErrorPanel error={submitMutation.error} /> : null}
          {jobQuery.isError ? <ErrorPanel error={jobQuery.error} /> : null}

          {runId && jobData?.status === "succeeded" ? (
            <div className="rounded-md border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-900">
              <p className="font-medium">Run created successfully.</p>
              <div className="mt-3 flex flex-wrap gap-3">
                <Link
                  to={`/runs/${encodeURIComponent(runId)}`}
                  className="rounded bg-emerald-700 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-600"
                >
                  Open now
                </Link>
                <span className="self-center text-sm">Auto-opening run details...</span>
              </div>
            </div>
          ) : null}

          <div className="flex flex-wrap items-center gap-3">
            <button
              type="submit"
              disabled={!isFormValid || submitMutation.isPending || tasksQuery.isLoading || taskItems.length === 0}
              className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-60"
            >
              {submitMutation.isPending ? "Starting..." : "Start"}
            </button>
            <Link
              to="/runs"
              className="rounded border border-slate-300 px-4 py-2 text-sm hover:bg-slate-100"
            >
              Cancel
            </Link>
            {!selectedTaskId ? <span className="text-sm text-slate-500">Select a task to continue.</span> : null}
            {selectedTaskId && files.length === 0 ? <span className="text-sm text-slate-500">Add at least one file.</span> : null}
          </div>
        </form>
      ) : null}
    </section>
  );
}
