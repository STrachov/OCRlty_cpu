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
        <span>Item</span>
        <span>/</span>
        <span className="font-mono text-xs">{request_id}</span>
      </div>
    );
  }

  return null;
}
