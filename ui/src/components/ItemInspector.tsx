import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Collapsible } from "./Collapsible";
import { CopyButton } from "./CopyButton";
import { JsonView } from "./JsonView";
import { resolveReceiptImageUrl } from "../utils/resolveReceiptImageUrl";
import { fetchJson } from "../api/client";

type ItemInspectorProps = {
  item: Record<string, unknown> | null;
  runId?: string;
};

export function ItemInspector({ item, runId }: ItemInspectorProps) {
  const [imgSrc, setImgSrc] = useState<string | null>(null);
  const [imgFailed, setImgFailed] = useState(false);

  const requestId = typeof item?.request_id === "string" ? item.request_id : "";
  const seedImage = useMemo(() => resolveReceiptImageUrl(item ?? null), [item]);

  useEffect(() => {
    setImgSrc(null);
    setImgFailed(false);
    if (!seedImage) {
      return;
    }
    if (!seedImage.includes("/v1/inputs/presign?")) {
      setImgSrc(seedImage);
      return;
    }

    let path = "";
    try {
      const u = new URL(seedImage);
      path = `${u.pathname}${u.search}`;
    } catch {
      setImgFailed(true);
      return;
    }

    void fetchJson<{ url: string }>(path)
      .then((res) => {
        if (res.data?.url) {
          setImgSrc(res.data.url);
        } else {
          setImgFailed(true);
        }
      })
      .catch(() => {
        setImgFailed(true);
      });
  }, [seedImage]);

  if (!item) {
    return <p className="text-sm text-slate-500">No row selected.</p>;
  }

  return (
    <div className="space-y-3 text-sm">
      
      <div>
        <p className="mb-2 text-xs text-slate-500">Receipt image preview</p>
        {imgSrc && !imgFailed ? (
          <img
            src={imgSrc}
            alt="Receipt preview"
            className="max-h-100 w-full rounded border border-slate-200 object-contain"
            onError={() => setImgFailed(true)}
          />
        ) : (
          <p className="rounded border border-dashed border-slate-300 p-3 text-xs text-slate-500">Receipt image not available</p>
        )}
      </div>

      <Collapsible title="parsed" defaultOpen>
        <JsonView data={item.parsed ?? null} />
      </Collapsible>


      <div>
        <p className="text-xs text-slate-500">file</p>
        <p className="break-all">{typeof item.file === "string" ? item.file : "-"}</p>
      </div>

      <div>
        <p className="text-xs text-slate-500">request_id</p>
        <div className="flex items-center gap-2">
          <span className="break-all font-mono text-xs">{requestId || "-"}</span>
          {requestId ? <CopyButton text={requestId} /> : null}
        </div>
      </div>

      <p><span className="text-xs text-slate-500">schema_valid:</span> {String(item.schema_valid ?? "-")}</p>
      <p><span className="text-xs text-slate-500">error:</span> {item.error == null ? "-" : JSON.stringify(item.error)}</p>

      {item.timings_ms && typeof item.timings_ms === "object" ? (
        <div>
          <p className="mb-1 text-xs text-slate-500">timings_ms</p>
          <table className="w-full text-xs">
            <tbody>
              {Object.entries(item.timings_ms as Record<string, unknown>).map(([k, v]) => (
                <tr key={k} className="border-t border-slate-100">
                  <td className="py-1 pr-2 font-mono">{k}</td>
                  <td className="py-1">{String(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      <Collapsible title="schema_errors">
        <JsonView data={item.schema_errors ?? null} />
      </Collapsible>

      <Collapsible title="error_history">
        <JsonView data={item.error_history ?? null} />
      </Collapsible>

      

      {/* {requestId ? (
        <Link
          to={`/items/${encodeURIComponent(requestId)}`}
          state={runId ? { runId } : undefined}
          className="inline-block rounded bg-slate-900 px-3 py-2 text-xs font-medium text-white hover:bg-slate-700"
        >
          Open item page
        </Link>
      ) : null} */}
    </div>
  );
}
