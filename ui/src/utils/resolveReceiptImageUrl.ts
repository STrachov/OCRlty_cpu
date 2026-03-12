import { getApiBaseUrl } from "../auth/storage";

function getStringValue(source: Record<string, unknown> | null | undefined, key: string): string | null {
  if (!source) {
    return null;
  }
  const value = source[key];
  return typeof value === "string" && value ? value : null;
}

export function resolveReceiptImageUrl(data: Record<string, unknown> | null | undefined): string | null {
  if (!data || typeof data !== "object") {
    return null;
  }

  const input = data.input && typeof data.input === "object"
    ? data.input as Record<string, unknown>
    : null;

  const imageUrl = getStringValue(data, "image_url") ?? getStringValue(input, "image_url");
  if (imageUrl) {
    return imageUrl;
  }

  const inputUrl = getStringValue(data, "input_url") ?? getStringValue(input, "input_url");
  if (inputUrl) {
    return inputUrl;
  }

  const inputRef = getStringValue(data, "input_ref")
    ?? getStringValue(data, "image_ref")
    ?? getStringValue(input, "input_ref")
    ?? getStringValue(input, "image_ref");
  const base = getApiBaseUrl();
  if (inputRef && base) {
    const rel = inputRef.replace(/\\/g, "/").replace(/^\/+/, "");
    return `${base}/v1/inputs/presign?input_ref=${encodeURIComponent(rel)}`;
  }

  return null;
}
