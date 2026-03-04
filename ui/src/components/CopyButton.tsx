import { useEffect, useState } from "react";

type CopyButtonProps = {
  text: string;
};

export function CopyButton({ text }: CopyButtonProps) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) {
      return;
    }
    const t = window.setTimeout(() => setCopied(false), 1200);
    return () => window.clearTimeout(t);
  }, [copied]);

  return (
    <span className="inline-flex items-center gap-2">
      <button
        type="button"
        onClick={async () => {
          if (!text) {
            return;
          }
          await navigator.clipboard.writeText(text);
          setCopied(true);
        }}
        className="rounded border border-slate-300 px-2 py-1 text-xs text-slate-700 hover:bg-slate-100"
      >
        Copy
      </button>
      {copied ? <span className="text-xs text-emerald-700">Copied</span> : null}
    </span>
  );
}
