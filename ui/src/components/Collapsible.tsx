import type { ReactNode } from "react";

type CollapsibleProps = {
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
};

export function Collapsible({ title, children, defaultOpen = false }: CollapsibleProps) {
  return (
    <details
      open={defaultOpen}
      className="rounded-md border border-slate-200 bg-white p-3"
    >
      <summary className="cursor-pointer font-medium text-slate-800">{title}</summary>
      <div className="mt-3">{children}</div>
    </details>
  );
}
