#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# Defaults
# -----------------------
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8080}"

# Where persistent data/volumes live inside the container (VPS-friendly)
export DATA_DIR="${DATA_DIR:-/data}"
export TMP_DIR="${TMP_DIR:-${DATA_DIR}/tmp}"

# SQLite for auth (persist via volume)
export AUTH_DB_PATH="${AUTH_DB_PATH:-${DATA_DIR}/auth/auth.db}"

# Logging defaults (your app emits JSON logs itself)
export LOG_FORMAT="${LOG_FORMAT:-json}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

# FastAPI app import path
APP_MODULE="${APP_MODULE:-app.main:app}"

mkdir -p "$(dirname "$AUTH_DB_PATH")"
mkdir -p "$TMP_DIR"

# -----------------------
# Minimal config checks
# -----------------------
# Auth
if [[ "${AUTH_ENABLED:-1}" != "0" ]]; then
  if [[ -z "${API_KEY_PEPPER:-}" ]]; then
    echo "ERROR: API_KEY_PEPPER is required when AUTH_ENABLED=1" >&2
    exit 1
  fi
fi

# vLLM (GPU endpoint)
if [[ -z "${VLLM_BASE_URL:-}" ]]; then
  echo "ERROR: VLLM_BASE_URL is required (e.g. https://POD_ID-8000.proxy.runpod.net)" >&2
  exit 1
fi
if [[ -z "${VLLM_API_KEY:-}" ]]; then
  echo "ERROR: VLLM_API_KEY is required to call vLLM" >&2
  exit 1
fi

# S3/R2 artifacts (enabled when S3_BUCKET is set)
if [[ -n "${S3_BUCKET:-}" ]]; then
  if [[ -z "${S3_ENDPOINT_URL:-}" ]]; then
    echo "ERROR: S3_ENDPOINT_URL is required when S3_BUCKET is set (R2: https://<ACCOUNT_ID>.r2.cloudflarestorage.com)" >&2
    exit 1
  fi
  if [[ -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
    echo "ERROR: AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY are required for S3/R2 access" >&2
    exit 1
  fi
fi

# -----------------------
# Runner selection
# -----------------------
RUNNER="${RUNNER:-uvicorn}"

if [[ "$RUNNER" == "gunicorn" ]]; then
  # For your 1 vCPU / 1GB RAM VPS: keep 1 worker unless you upgrade.
  WORKERS="${GUNICORN_WORKERS:-1}"
  TIMEOUT="${GUNICORN_TIMEOUT:-300}"
  GRACE="${GUNICORN_GRACEFUL_TIMEOUT:-30}"
  KEEPALIVE="${GUNICORN_KEEPALIVE:-5}"

  exec gunicorn "$APP_MODULE" \
    -k uvicorn.workers.UvicornWorker \
    -w "$WORKERS" \
    -b "${HOST}:${PORT}" \
    --timeout "$TIMEOUT" \
    --graceful-timeout "$GRACE" \
    --keep-alive "$KEEPALIVE"
else
  # Uvicorn defaults: no access log (because it’s not JSON), but can be enabled via env.
  ACCESS_LOG="${UVICORN_ACCESS_LOG:-0}"       # 0/1
  KEEP_ALIVE="${UVICORN_TIMEOUT_KEEP_ALIVE:-5}"

  ARGS=(--host "$HOST" --port "$PORT" --proxy-headers --forwarded-allow-ips "*" --timeout-keep-alive "$KEEP_ALIVE")

  if [[ "$ACCESS_LOG" == "0" ]]; then
    ARGS+=(--no-access-log)
  fi

  exec uvicorn "$APP_MODULE" "${ARGS[@]}"
fi
