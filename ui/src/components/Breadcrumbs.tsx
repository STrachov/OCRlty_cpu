import { useMemo } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Link, useLocation, useParams } from "react-router-dom";

function CrumbLink({ to, label }: { to: string; label: string }) {
  return (
    <Link
      className="rounded-md border border-slate-200 bg-white px-2.5 py-1 text-xs font-medium text-slate-700 transition hover:border-slate-300 hover:bg-slate-50"
      to={to}
      title={label}
    >
      {label}
    </Link>
  );
}

function CrumbCurrent({ label, mono = false }: { label: string; mono?: boolean }) {
  return (
    <span
      className={`max-w-[340px] truncate rounded-md border border-slate-300 bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-900 ${mono ? "font-mono" : ""}`}
      title={label}
    >
      {label}
    </span>
  );
}

function Sep() {
  return <span className="text-slate-400">/</span>;
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
    return (
      <div className="flex items-center gap-2">
        <CrumbCurrent label="Runs" />
      </div>
    );
  }

  if (location.pathname === "/settings") {
    return (
      <div className="flex items-center gap-2">
        <CrumbCurrent label="Settings" />
      </div>
    );
  }

  if (location.pathname.startsWith("/runs/") && run_id) {
    return (
      <div className="flex items-center gap-2">
        <CrumbLink to="/runs" label="Runs" />
        <Sep />
        <CrumbCurrent label={run_id} mono />
      </div>
    );
  }

  if (location.pathname.startsWith("/items/") && request_id) {
    return (
      <div className="flex items-center gap-2">
        <CrumbLink to="/runs" label="Runs" />
        <Sep />
        {itemRunId ? (
          <CrumbLink to={`/runs/${encodeURIComponent(itemRunId)}`} label={itemRunId} />
        ) : (
          <CrumbCurrent label="Item" />
        )}
        <Sep />
        <CrumbCurrent label={request_id} mono />
      </div>
    );
  }

  return null;
}
