import { CopyButton } from "./CopyButton";

type ErrorPanelProps = {
  error: unknown;
};

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : null;
}

export function ErrorPanel({ error }: ErrorPanelProps) {
  const e = asRecord(error);
  const details = asRecord(e?.details);
  const detailsErrors = Array.isArray(details?.errors) ? (details?.errors as unknown[]) : null;

  const message =
    typeof e?.message === "string" && e.message.length > 0 ? e.message : "Request failed";
  const code = typeof e?.code === "string" ? e.code : undefined;
  const httpStatus =
    typeof e?.httpStatus === "number" ? e.httpStatus : typeof e?.status === "number" ? e.status : undefined;
  const requestId = typeof e?.request_id === "string" ? e.request_id : undefined;
  const xRequestId = typeof e?.xRequestId === "string" ? e.xRequestId : undefined;

  return (
    <div className="rounded-md border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900">
      <h3 className="mb-2 text-base font-semibold">Request Error</h3>
      <div className="space-y-1">
        <p>
          <span className="font-medium">HTTP status:</span> {httpStatus ?? "n/a"}
        </p>
        <p>
          <span className="font-medium">Message:</span> {message}
        </p>
        <p>
          <span className="font-medium">Code:</span> {code ?? "n/a"}
        </p>
      </div>

      <div className="mt-3 space-y-2">
        <div className="flex items-center gap-2">
          <span className="font-medium">error.request_id:</span>
          <span className="font-mono text-xs">{requestId ?? "n/a"}</span>
          {requestId ? <CopyButton text={requestId} /> : null}
        </div>
        <div className="flex items-center gap-2">
          <span className="font-medium">X-Request-ID:</span>
          <span className="font-mono text-xs">{xRequestId ?? "n/a"}</span>
          {xRequestId ? <CopyButton text={xRequestId} /> : null}
        </div>
      </div>

      {detailsErrors ? (
        <div className="mt-3">
          <p className="font-medium">Validation details:</p>
          <ul className="ml-5 list-disc space-y-1">
            {detailsErrors.map((item, idx) => (
              <li key={idx} className="font-mono text-xs">
                {typeof item === "string" ? item : JSON.stringify(item)}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
