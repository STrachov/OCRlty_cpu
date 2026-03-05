# UI_SPEC.md — OCRlty UI (living spec)

This is the **living UI spec** for the OCRlty frontend under `./ui/`.
It captures current behavior, UX decisions, and near-term roadmap.

> Stable workflow/rules live in `AGENTS.md`.

## Product intent

A fast, readable UI for operators to:
- validate API key (`/v1/me`)
- list runs (newest first)
- open a run and inspect items quickly (filters + keyboard navigation + inspector)
- open a specific item and view full extract JSON and diagnostics

## Routes

### `/login`
Purpose: entry gate when auth config missing.

- Inputs: API base URL + API key
- Action: **Log in** → `GET /v1/me`
- On success: persist config (localStorage) and go to `/runs`

### `/settings`
Purpose: change API base URL and API key (switch user) without leaving the app.

- Inputs: API base URL + API key (prefilled from localStorage)
- Actions:
  - **Test** → `GET /v1/me` show `key_id/role/scopes`
  - **Save** → persist to localStorage and refresh cached `me`
  - **Logout** → clear localStorage and go to `/login`
- Links:
  - `{API_BASE}/docs`
  - `{API_BASE}/openapi.json`

### `/runs`
Purpose: list runs and enter run details.

- Shows table sorted newest-first:
  - primary: `created_at` desc (invalid/missing last)
  - fallback: `run_id` desc
- `run_id` is a link to `/runs/:run_id`

### `/runs/:run_id`
Purpose: inspect one run (batch artifact) with fast drill-down.

Main area:
- run summary (best effort)
- filter controls:
  - Only schema invalid
  - Only errors
  - Search by file substring
- items table:
  - columns currently expected: `file`, `schema_valid`, `error`, `parsed` (preview)
  - selecting a row updates the Inspector

Right panel (contextual):
- run summary card (run_id/task_id/created_at/ok/error)
- keyboard hint: `↑/↓ navigate, Enter open`
- **Item Inspector** for selected row:
  - file
  - request_id (copy)
  - schema_valid
  - error (full)
  - timings_ms (if present)
  - schema_errors (collapsible)
  - error_history (collapsible)
  - parsed (pretty JSON)
  - “Open item page” → `/items/:request_id` (if request_id exists)
  - receipt image preview (best effort)

Keyboard navigation:
- table container is focusable (`tabIndex=0`)
- `ArrowUp/ArrowDown` moves selection through **visible (filtered)** rows
- `Enter` navigates to item page if `request_id` exists
- selection change scrolls focused row into view
- typing inside inputs (search) must not be overridden by arrow navigation

### `/items/:request_id`
Purpose: full item view (extract artifact).

- Shows:
  - receipt image (bigger than inspector preview) — best effort
  - timings_ms
  - schema_errors (collapsible)
  - error_history (collapsible)
  - full JSON view of the extract artifact
- Right panel:
  - request_id (copy)
  - download JSON button

## Receipt image (best-effort)

Backend may not expose a dedicated image-by-ref endpoint today.

Implementation recommendation:
- `resolveReceiptImageUrl(artifactOrItem)`:
  - if `image_url` exists → use it
  - else if `input_url` exists → use it
  - else if `input_ref` exists → optional attempt: `${API_BASE}/v1/inputs?ref=...` (may not exist yet)
- UI behavior:
  - if URL missing or image fails to load → show placeholder “Receipt image not available”
  - do not show global error alerts for missing images

## Error handling

- Normalize errors in `src/api/client.ts`:
  - show `httpStatus`, `error.code`, `error.message`, `error.request_id`, plus header `X-Request-ID`
- `ErrorPanel` renders details safely; for validation errors (422), show list from `details.errors` if present.

## Download JSON

Buttons that download already-fetched JSON:
- Run: `run_<run_id>.json`
- Item: `item_<request_id>.json`

## UI layout

- Left sidebar (permanent): navigation only
  - Runs
  - Settings
- Top bar: breadcrumbs + global header info (title/API/user/logout)
- Right panel: contextual actions + inspector

## Roadmap

### P0 (current/near-term)
- Stable layout (sidebar/topbar/right panel)
- Runs sorting newest-first
- Run inspector + keyboard navigation
- Settings page for switching API base/key
- Best-effort receipt image display (preview + full)

### P1 (next)
- Better `parsed` preview in table (truncate + “expand”)
- Better filtering (e.g., by `schema_valid`, by “has_error”)
- Optional: virtualized table for large runs

### P2 (later)
- Jobs page (polling job progress) if/when needed
- Server-provided receipt image endpoint (then remove best-effort hacks)

## Decisions log

- Left sidebar contains only permanent nav; page context goes to right panel.
- Runs are displayed newest-first (created_at desc, run_id fallback).
- Run details table shows a compact `parsed` preview; full details live in Inspector and Item page.
- Keyboard navigation controls the selection when table container is focused.
- Receipt image shown in Inspector (preview) and Item page (full), best-effort with graceful fallback.
