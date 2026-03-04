import { getApiBaseUrl } from "../auth/storage";

export function resolveReceiptImageUrl(data: Record<string, unknown> | null | undefined): string | null {
  if (!data || typeof data !== "object") {
    return null;
  }

  const imageUrl = typeof data.image_url === "string" ? data.image_url : null;
  if (imageUrl) {
    return imageUrl;
  }

  const inputUrl = typeof data.input_url === "string" ? data.input_url : null;
  if (inputUrl) {
    return inputUrl;
  }

  const inputRef = typeof data.input_ref === "string" ? data.input_ref : null;
  const base = getApiBaseUrl();
  if (inputRef && base) {
    const rel = inputRef.replace(/\\/g, "/").replace(/^\/+/, "");
    return `${base}/v1/inputs/presign?input_ref=${encodeURIComponent(rel)}`;
  }

  return null;
}
