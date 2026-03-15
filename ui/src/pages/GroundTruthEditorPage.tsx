import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate, useParams } from "react-router-dom";
import { getGroundTruthContent, getMe, listGroundTruths, updateGroundTruth } from "../api/runs";
import { ErrorPanel } from "../components/ErrorPanel";

function prettyJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

export function GroundTruthEditorPage() {
  const { gt_id } = useParams<{ gt_id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [name, setName] = useState("");
  const [contentText, setContentText] = useState("");
  const [validationError, setValidationError] = useState<string | null>(null);

  const meQuery = useQuery({
    queryKey: ["me"],
    queryFn: getMe,
    staleTime: 5 * 60 * 1000,
  });
  const metaQuery = useQuery({
    queryKey: ["ground_truths", 100, 0],
    queryFn: () => listGroundTruths(100, 0),
    enabled: Boolean(gt_id),
  });
  const contentQuery = useQuery({
    queryKey: ["ground_truth-content", gt_id],
    queryFn: () => getGroundTruthContent(gt_id ?? ""),
    enabled: Boolean(gt_id),
  });
  const updateMutation = useMutation({
    mutationFn: async () => {
      const parsed = JSON.parse(contentText);
      return updateGroundTruth(gt_id ?? "", { name: name.trim() || "ground_truth.json", content: parsed });
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["ground_truths"] });
      await queryClient.invalidateQueries({ queryKey: ["ground_truth-content", gt_id] });
      navigate("/ground-truths", { replace: true });
    },
  });

  useEffect(() => {
    const item = (metaQuery.data?.items ?? []).find((row) => row.gt_id === gt_id);
    if (item && !name) {
      setName(item.name);
    }
  }, [gt_id, metaQuery.data, name]);

  useEffect(() => {
    if (contentQuery.data !== undefined && !contentText) {
      setContentText(prettyJson(contentQuery.data));
    }
  }, [contentQuery.data, contentText]);

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold">Edit Ground Truth</h2>
          <p className="text-sm text-slate-600">Edit an existing ground truth document and save it back to storage.</p>
        </div>
        <Link
          to="/ground-truths"
          className="rounded border border-slate-300 bg-white px-4 py-2 text-sm hover:bg-slate-100"
        >
          Back to list
        </Link>
      </div>

      {meQuery.isError ? <ErrorPanel error={meQuery.error} /> : null}
      {metaQuery.isError ? <ErrorPanel error={metaQuery.error} /> : null}
      {contentQuery.isError ? <ErrorPanel error={contentQuery.error} /> : null}
      {updateMutation.isError ? <ErrorPanel error={updateMutation.error} /> : null}

      <div className="rounded-md border border-slate-200 bg-white p-4">
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr),auto,auto]">
          <label className="block text-sm font-medium text-slate-800">
            Name
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="mt-1 w-full rounded border border-slate-300 px-3 py-2"
            />
          </label>
          <div className="flex items-end">
            <button
              type="button"
              onClick={() => {
                try {
                  JSON.parse(contentText);
                  setValidationError(null);
                } catch (error) {
                  setValidationError(error instanceof Error ? error.message : "Invalid JSON");
                }
              }}
              className="rounded border border-slate-300 bg-white px-4 py-2 text-sm hover:bg-slate-100"
            >
              Validate JSON
            </button>
          </div>
          <div className="flex items-end">
            <button
              type="button"
              onClick={() => updateMutation.mutate()}
              disabled={!gt_id || updateMutation.isPending}
              className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-60"
            >
              {updateMutation.isPending ? "Saving..." : "Save"}
            </button>
          </div>
        </div>
      </div>

      <div className="rounded-md border border-slate-200 bg-white p-4">
        <textarea
          value={contentText}
          onChange={(e) => setContentText(e.target.value)}
          className="min-h-[640px] w-full rounded border border-slate-300 px-3 py-2 font-mono text-xs"
          placeholder="Ground truth JSON"
        />
        {validationError ? <p className="mt-2 text-sm text-rose-700">{validationError}</p> : null}
      </div>
    </section>
  );
}
