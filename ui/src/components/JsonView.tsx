type JsonViewProps = {
  data: unknown;
};

export function JsonView({ data }: JsonViewProps) {
  return (
    <pre className="max-h-[60vh] overflow-auto rounded-md border border-slate-200 bg-slate-900 p-4 text-xs text-slate-100">
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}
