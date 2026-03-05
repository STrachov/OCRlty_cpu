# AGENTS.md — OCRlty UI

This folder contains the **frontend UI** for the OCRlty project (Vite + React + TypeScript).

**Scope:** UI work only. The backend (FastAPI) lives outside `./ui/` and is **out of scope** unless explicitly requested.

> This file is intentionally **stable** (workflow + rules).  
> For fast-changing UI requirements and decisions, see **`UI_SPEC.md`**.

## Goals

The UI is a **lightweight operator console**:

- authenticate via API key (stored locally)
- browse recent runs
- inspect a run (items table + filters + inspector)
- inspect an item (full JSON + diagnostics + receipt image when available)
- operator UX: keyboard navigation, inspector panel, breadcrumbs

The UI prioritizes **clarity** and **speed of debugging** over visual polish.

## Non-goals

- Do not change or require backend changes.
- No SSR, no SEO.
- No complex global state management beyond React + TanStack Query.
- Avoid adding heavy UI frameworks unless needed.

## Tech stack

- React 18, TypeScript
- Vite
- React Router
- TanStack Query (data fetching + cache + refetch)
- Styling: simple CSS/Tailwind (keep minimal and readable)

## Running locally

From repo root:

```bash
cd ui
npm install
npm run dev
```

UI runs on: `http://localhost:5173`

Backend is expected at: `http://127.0.0.1:8080` (configurable in Settings / Login).

Build:

```bash
cd ui
npm run build
npm run preview
```

## Authentication & storage

We use API key auth.

### Request header

Requests use header:

```text
Authorization: Bearer <api_key>
```

### Local storage

API base URL and API key are stored in `localStorage`.

Keys:

- `ocrlty_api_base_url`
- `ocrlty_api_key`

All access to `localStorage` must go through:

- `src/auth/storage.ts`

### Pages for auth/config

- `/login` — entry gate when auth config is missing
- `/settings` — operator can change API base URL and API key and run `/v1/me` test

## Backend API (UI contracts)

Primary endpoints used by UI:

- `GET /v1/me` — validate key, show `key_id / role / scopes`
- `GET /v1/runs?limit=&cursor=` — runs list
- `GET /v1/runs/{run_id}` — run (batch artifact)
- `GET /v1/runs/item/{request_id}` — item (extract artifact)

### Error format

```json
{
  "error": {
    "code": "string",
    "message": "string",
    "request_id": "string|null",
    "details": {}
  }
}
```

Also read response header:

- `X-Request-ID`

UI must display request IDs in `ErrorPanel` and allow copy.

## Layout conventions

- **Left sidebar** is **permanent navigation only** (e.g., Runs, Settings).
- **Top bar** includes breadcrumbs and global header info.
- **Right panel** is **contextual** (run summary, inspector, page-specific actions).

## Folder structure (expected)

```
src/
  api/
    client.ts        # fetch wrapper: base URL, auth header, error normalization, X-Request-ID
    types.ts         # TS types (loose types ok for artifacts)
    runs.ts          # endpoint functions (getMe, listRuns, getRun, getItem)
  auth/
    storage.ts       # localStorage getters/setters
    RequireAuth.tsx  # protected routes
  components/
    CopyButton.tsx
    ErrorPanel.tsx
    JsonView.tsx
    Breadcrumbs.tsx
    LeftSidebar.tsx
    RightPanel.tsx
    ItemInspector.tsx
  layout/
    AppLayout.tsx    # left sidebar + topbar + outlet (+ right panel)
  pages/
    RunsPage.tsx
    RunDetailsPage.tsx
    ItemPage.tsx
    LoginPage.tsx
    SettingsPage.tsx
```

## UX requirements (high-level)

### Runs list
- Always show newest-first ordering in UI:
  1) `created_at` desc (missing/invalid last)
  2) fallback: `run_id` desc

### Run details
- Items table supports filters:
  - only schema invalid
  - only errors
  - file substring search
- Right inspector panel shows details for selected row.
- Keyboard navigation:
  - table container is focusable (`tabIndex=0`)
  - `ArrowUp` / `ArrowDown` moves selection through **visible (filtered)** rows
  - `Enter` opens item page if `request_id` exists
  - selection change scrolls row into view
  - must not break typing inside inputs (navigation only when table container is focused)

### Item page
- Show full JSON and diagnostic blocks:
  - `schema_errors`, `error_history`, `timings_ms`
- If receipt image is available via URL fields, show it; otherwise show placeholder.

## Coding rules

- Keep components small; prefer composition over prop drilling.
- Avoid introducing additional global stores unless necessary.
- All network calls go through `src/api/client.ts`.
- Any new route must be added to router and protected when needed.
- Prefer accessible interactions:
  - keyboard navigation should not break text inputs
  - use `button` elements for actions

## Quality checks (before finishing a change)

### Typecheck / build

```bash
cd ui
npm run build
```

### Manual smoke test

- Login/Settings: change base URL/key, test `/v1/me`
- Runs page loads and sorted newest-first
- Run details: filters work, inspector updates, keyboard navigation works
- Item page loads, JSON renders

## What to do when uncertain

- Prefer UI-only best-effort behavior (placeholders) over adding backend dependencies.
- If a feature needs backend support (e.g., receipt image bytes), implement graceful fallback and document the assumption in `UI_SPEC.md`.
- Ask for explicit approval before changing API contracts or storage keys.
