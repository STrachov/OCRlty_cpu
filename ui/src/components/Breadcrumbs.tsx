import { useMemo } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Link, useLocation, useParams } from "react-router-dom";

function CrumbLink({ to, label }: { to: string; label: string }) {
  return (
    <Link className="text-blue-700 hover:underline" to={to}>
      {label}
    </Link>
  );
}

export function Breadcrumbs() {
  const location = useLocation();
  const { run_id, request_id } = useParams<{ run_id?: string; request_id?: string }>();
  const queryClient = useQueryClient();

  const itemRunId = useMemo(() => {
    if (!request_id) {
      return null;
    }
    const stateRunId = (location.state as { runId?: unknown } | null)?.runId;
    if (typeof stateRunId === "string" && stateRunId) {
      return stateRunId;
    }
    const itemData = queryClient.getQueryData(["item", request_id]) as Record<string, unknown> | undefined;
    return itemData && typeof itemData.run_id === "string" ? itemData.run_id : null;
  }, [location.state, queryClient, request_id]);

  if (location.pathname === "/runs") {
    return <span className="text-sm text-slate-700">Runs</span>;
  }

  if (location.pathname.startsWith("/runs/") && run_id) {
    return (
      <div className="flex items-center gap-2 text-sm text-slate-700">
        <CrumbLink to="/runs" label="Runs" />
        <span>/</span>
        <span className="font-mono text-xs">{run_id}</span>
      </div>
    );
  }

  if (location.pathname.startsWith("/items/") && request_id) {
    return (
      <div className="flex items-center gap-2 text-sm text-slate-700">
        <CrumbLink to="/runs" label="Runs" />
        <span>/</span>
        {itemRunId ? (
          <CrumbLink to={`/runs/${encodeURIComponent(itemRunId)}`} label="Run" />
        ) : (
          <span>Item</span>
        )}
        <span>/</span>
        <span className="font-mono text-xs">{request_id}</span>
      </div>
    );
  }

  return null;
}
