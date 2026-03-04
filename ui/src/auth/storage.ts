export const API_BASE_URL_KEY = "ocrlty_api_base_url";
export const API_KEY_KEY = "ocrlty_api_key";

export function getApiBaseUrl(): string | null {
  return localStorage.getItem(API_BASE_URL_KEY);
}

export function setApiBaseUrl(v: string): void {
  localStorage.setItem(API_BASE_URL_KEY, v.trim());
}

export function getApiKey(): string | null {
  return localStorage.getItem(API_KEY_KEY);
}

export function setApiKey(v: string): void {
  localStorage.setItem(API_KEY_KEY, v.trim());
}

export function clearAuth(): void {
  localStorage.removeItem(API_BASE_URL_KEY);
  localStorage.removeItem(API_KEY_KEY);
}
