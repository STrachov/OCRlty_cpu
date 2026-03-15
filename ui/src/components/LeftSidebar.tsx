import { Link, useLocation } from "react-router-dom";

export function LeftSidebar() {
  const location = useLocation();
  const isRuns = location.pathname.startsWith("/runs");
  const isGroundTruths = location.pathname.startsWith("/ground-truths");
  const isSettings = location.pathname.startsWith("/settings");

  const baseItem = "block rounded px-3 py-2 text-sm";
  const activeItem = "bg-slate-900 text-white";
  const idleItem = "text-slate-700 hover:bg-slate-100";

  return (
    <aside className="w-[260px] shrink-0 overflow-y-auto border-r border-slate-200 bg-white p-4">
      <nav>
        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Navigation</p>
        <div className="space-y-1">
          <Link to="/runs" className={`${baseItem} ${isRuns ? activeItem : idleItem}`}>
            Runs
          </Link>
          <Link to="/ground-truths" className={`${baseItem} ${isGroundTruths ? activeItem : idleItem}`}>
            Ground Truths
          </Link>
          <Link to="/settings" className={`${baseItem} ${isSettings ? activeItem : idleItem}`}>
            Settings
          </Link>
        </div>
      </nav>
    </aside>
  );
}
