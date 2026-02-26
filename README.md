# OCRlty CPU Orchestrator (FastAPI)

CPU‑оркестратор для OCR/IE пайплайна: принимает изображения чеков, дергает inference backend (vLLM / mock), валидирует ответ по JSON‑schema, сохраняет артефакты (local или S3/R2), умеет батчи и async‑джобы (202 + polling).

---

## 1) Архитектура (очень кратко)

- **CPU Orchestrator (этот репозиторий)**: FastAPI, auth, jobs, artifacts (local/S3), batch‑run, eval/debug.
- **Inference backend**:
  - `INFERENCE_BACKEND=vllm` — отправка на OpenAI‑совместимый endpoint vLLM (`/v1/chat/completions`).
  - `INFERENCE_BACKEND=mock` — возвращает “псевдо‑ответ” (полезно для smoke‑тестов без GPU).

---

## 2) Быстрый старт
### 2.1 Локально (venv)
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
source .venv/bin/activate

python -m pip install -U "pip<26" "pip-tools==7.5.2"
pip install -r requirements.txt
```

> Приложение читает `.env` (через настройки/dotenv). Запускай из корня проекта, чтобы `.env` находился рядом.

Запуск:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload --env-file .env
```

Проверка:
```bash
curl -fsS http://127.0.0.1:8080/v1/health
```

### 2.2 Docker (VPS / локально)
`docker-compose.yml` (пример):
```yaml
services:
  orchestrator:
    image: ghcr.io/<OWNER>/ocrlty-orchestrator:latest
    restart: always
    env_file: [.env]
    ports: ["8080:8080"]
    volumes:
      - ./data:/data
```

Запуск:
```bash
docker compose up -d
docker compose logs -f
curl -fsS http://127.0.0.1:8080/v1/health
```

---

## 3) Конфигурация (.env)

Минимально (mock):
```env
INFERENCE_BACKEND=mock

AUTH_ENABLED=1
API_KEY_PEPPER=change-me
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/ocrlty

# куда писать артефакты локально (если не S3)
ARTIFACTS_DIR=/data/artifacts
DATA_ROOT=/data
```

vLLM:
```env
INFERENCE_BACKEND=vllm
VLLM_BASE_URL=https://POD_ID-8000.proxy.runpod.net/v1
VLLM_API_KEY=...
VLLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
```

S3/R2 (если хочешь хранить артефакты в облаке):
```env
S3_BUCKET=ocrlty
S3_ENDPOINT_URL=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
S3_REGION=auto
S3_PREFIX=prod
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

Логи:
```env
LOG_LEVEL=INFO
LOG_FORMAT=json
```

Apply migrations before starting app/worker:
```bash
alembic upgrade head
```

---

## 4) Аутентификация (PostgreSQL auth DB)

Ключ передаётся:
- `Authorization: Bearer <api_key>` (канонично)
- или `X-API-Key: <api_key>`

### Инициализация БД
```bash
python -m app.auth_cli init-db
# или явно:
python -m app.auth_cli --database-url "postgresql+psycopg://postgres:postgres@localhost:5432/ocrlty" init-db
```

### Создать ключ
```bash
export API_KEY_PEPPER="your-secret-pepper"
python -m app.auth_cli create-key --key-id client-1 --role client
```

Опционально scopes:
```bash
python -m app.auth_cli create-key --key-id dbg-1 --role debugger --scopes "extract:run,debug:run,debug:read_raw"
# или:
python -m app.auth_cli create-key --key-id dbg-1 --role debugger --scopes '["extract:run","debug:run"]'
```

> Сгенерированный `api_key` показывается **только один раз** в stdout.

---

## 5) API (эндпоинты)

### 5.1 Core
- `GET /v1/health` — healthcheck
- `GET /v1/me` — кто я (проверка auth)
- `GET /docs` — Swagger UI

### 5.2 Sync
- `POST /v1/extract` — 1 изображение → 1 extract‑артефакт  
  Body: `task_id`, и **одно из**: `image_base64` **или** `image_ref`; + `mime_type`, `request_id`.
- `POST /v1/batch_extract_upload` — multipart upload (несколько файлов) → batch  
  Важно: `persist_inputs=true`, если хочешь потом `batch_extract_rerun`.
- `POST /v1/batch_extract` — batch по директории на сервере (**images_dir должен быть под DATA_ROOT**)
- `POST /v1/batch_extract_rerun` — повторный прогон по сохранённым input_ref из прошлого upload/run

### 5.3 Async (202 + polling)
- `POST /v1/extract_async`
- `POST /v1/batch_extract_async`
- `POST /v1/batch_extract_upload_async`
- `POST /v1/batch_extract_rerun_async`
- `GET /v1/jobs/{job_id}`
- `POST /v1/jobs/{job_id}/cancel` (best-effort)

### 5.4 Runs
- `GET /v1/runs?limit=...&cursor=...` ? ?????? batch run-?? (cursor = `<created_at>|<run_id>`)
- `GET /v1/runs/{run_id}` ? ?????? batch artifact

### 5.5 Debug (только если `DEBUG_MODE=1` и scopes)
- `GET /v1/debug/artifacts?limit=...`
- `GET /v1/debug/artifacts/{request_id}`
- `GET /v1/debug/artifacts/{request_id}/raw`
- `GET /v1/debug/artifacts/backup.tar.gz`

---

## 6) Важные детали про ответы

- Hard ?????? (HTTP 4xx/5xx) ?????? ? ??????? `{"error": {"code","message","request_id","details"}}`.

- Оркестратор **может вернуть HTTP 200**, даже если inference упал: смотри поле `error` в JSON.
- `schema_valid=false` + `schema_errors` — модель (или mock) вернула JSON, не проходящий JSON‑schema.
- `artifact_rel`:
  - ? S3-?????? ? ???? ??? `S3_PREFIX` (???????? `extracts/YYYY-MM-DD/<id>.json`)
  - ? local-?????? ? ???? ???????????? `ARTIFACTS_DIR` (???????? `extracts\YYYY-MM-DD\<id>.json`)

---

## 7) E2E Smoke tests (curl)

Ниже — примеры для **PowerShell** и **Bash**.  
Ключевой момент: для JSON‑эндпоинтов мы **сохраняем body в файл** и отправляем `--data-binary "@file"` (без лишнего перевода строки).

---

### 7.1 PowerShell (Windows) — используй `curl.exe`

```powershell
# ====== CONFIG ======
$BASE   = "http://127.0.0.1:8080"         # или https://<vps-host>
$APIKEY = "oAJcMSdufcInKdy9PRb-sb-5H-vDquybRQSpym3fZg4"
$AUTH   = "Authorization: Bearer $APIKEY"
$CURL_W = "`nHTTP %{http_code}`n"

New-Item -ItemType Directory -Force -Path ".\tmp" | Out-Null

# ====== 0) HEALTH + AUTH ======
curl.exe -sS -w $CURL_W "$BASE/v1/health"
curl.exe -sS -w $CURL_W -H $AUTH "$BASE/v1/me"

# ====== 1) SYNC: /v1/extract (локальный файл -> base64) ======
$IMG = ".\data\batch_smoke\images\cord_0000.jpg"
$IMG2 = ".\data\batch_smoke\images\cord_0001.jpg"
$IMG3 = ".\data\batch_smoke\images\cord_0002.jpg"

$IMG_B64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($IMG))

$PREFIX = "smoke_mock_"   # например smoke_mock_ / smoke_vllm_
$RID = $PREFIX + [guid]::NewGuid().ToString("N")

$body = @{
  task_id      = "receipt_fields_v1"
  image_base64 = $IMG_B64
  mime_type    = "image/jpeg"
  request_id   = $RID
} | ConvertTo-Json -Depth 6

$bodyFile = ".\tmp\extract_body.json"
Set-Content -Path $bodyFile -Value $body -Encoding utf8 -NoNewline

curl.exe -sS -w $CURL_W -X POST "$BASE/v1/extract" `
  -H $AUTH `
  -H "Content-Type: application/json" `
  -H "X-Request-ID: $RID" `
  --data-binary "@$bodyFile"

# ====== 2) SYNC: /v1/batch_extract_upload (multipart) ======
$RUN_ID = $PREFIX + (Get-Date -Format "yyyyMMddTHHmmss")


curl.exe -sS -w $CURL_W -X POST "$BASE/v1/batch_extract_upload" `
  -H $AUTH `
  -F "task_id=receipt_fields_v1" `
  -F "persist_inputs=true" `
  -F "concurrency=2" `
  -F "run_id=$RUN_ID" `
  -F "files=@$IMG;type=image/jpeg" `
  -F "files=@$IMG2;type=image/jpeg" `
  -F "files=@$IMG3;type=image/jpeg"

# ====== 3) SYNC: /v1/batch_extract_rerun (по сохранённым inputs) ======
$body_rerun = @{
  task_id     = "receipt_fields_v1"
  run_id      = $RUN_ID
  concurrency = 2
  max_days    = 90
  # batch_date = "2026-02-15"             # опционально, если знаешь день
  # new_run_id = "rerun_" + (Get-Date -Format "yyyyMMddTHHmmss")
} | ConvertTo-Json -Depth 6

$bodyFile = ".\tmp\batch_rerun.json"
Set-Content -Path $bodyFile -Value $body_rerun -Encoding utf8 -NoNewline

curl.exe -sS -w $CURL_W -X POST "$BASE/v1/batch_extract_rerun" `
  -H $AUTH `
  -H "Content-Type: application/json" `
  --data-binary "@$bodyFile"

# ====== 4) SYNC: /v1/batch_extract (server-side dir под DATA_ROOT) ======
# images_dir должен быть ПОД DATA_ROOT на сервере/в контейнере.
$body_batch = @{
  task_id     = "receipt_fields_v1"
  images_dir  = "/data/batch_smoke/images"
  glob        = "**/*"
  concurrency = 2
  limit       = 5
  run_id      = "smoke_dir_" + (Get-Date -Format "yyyyMMddTHHmmss")
  # gt_path     = "/data/batch_smoke/gt.json"   # debug only
  # gt_image_key= "image"                       # по умолчанию "image"
} | ConvertTo-Json -Depth 6

$bodyFile = ".\tmp\batch_dir.json"
Set-Content -Path $bodyFile -Value $body_batch -Encoding utf8 -NoNewline

curl.exe -sS -w $CURL_W -X POST "$BASE/v1/batch_extract" `
  -H $AUTH `
  -H "Content-Type: application/json" `
  --data-binary "@$bodyFile"

# ============================
# ASYNC (202 + polling)
# ============================

# ====== 5) ASYNC: /v1/extract_async ======
$RID_ASYNC = "async_" + [guid]::NewGuid().ToString("N")
$body_async = @{
  task_id      = "receipt_fields_v1"
  image_base64 = $IMG_B64
  mime_type    = "image/jpeg"
  request_id   = $RID_ASYNC
} | ConvertTo-Json -Depth 6

$bodyFile = ".\tmp\extract_async.json"
Set-Content -Path $bodyFile -Value $body_async -Encoding utf8 -NoNewline

$job_create = curl.exe -sS -X POST "$BASE/v1/extract_async" `
  -H $AUTH -H "Content-Type: application/json" `
  --data-binary "@$bodyFile"

$job_id = ($job_create | ConvertFrom-Json).job_id
"JOB_ID=$job_id"

# polling:
while ($true) {
  $j = curl.exe -sS -H $AUTH "$BASE/v1/jobs/$job_id" | ConvertFrom-Json
  $progress = if ($null -ne $j.progress) { $j.progress | ConvertTo-Json -Compress } else { "null" }
  "{0} status={1} progress={2}" -f (Get-Date -Format "HH:mm:ss"), $j.status, $progress
  if ($j.status -in @("succeeded","failed","canceled")) { $j | ConvertTo-Json -Depth 30; break }
  Start-Sleep -Seconds 1
}

# ====== 6) ASYNC: /v1/batch_extract_upload_async ======
$job_create = curl.exe -sS -X POST "$BASE/v1/batch_extract_upload_async" `
  -H $AUTH `
  -F "task_id=receipt_fields_v1" `
  -F "concurrency=2" `
  -F "run_id=async_up_$(Get-Date -Format yyyyMMddTHHmmss)" `
  -F "files=@$IMG;type=image/jpeg" `
  -F "files=@$IMG2;type=image/jpeg"

$job_id = ($job_create | ConvertFrom-Json).job_id
"JOB_ID=$job_id"

# ====== 7) ASYNC: /v1/batch_extract_rerun_async ======
$body_rerun_async = @{
  task_id     = "receipt_fields_v1"
  run_id      = $RUN_ID
  concurrency = 2
  max_days    = 90
  new_run_id  = "async_rerun_" + (Get-Date -Format "yyyyMMddTHHmmss")
} | ConvertTo-Json -Depth 6

$bodyFile = ".\tmp\batch_rerun_async.json"
Set-Content -Path $bodyFile -Value $body_rerun_async -Encoding utf8 -NoNewline

$job_create = curl.exe -sS -X POST "$BASE/v1/batch_extract_rerun_async" `
  -H $AUTH -H "Content-Type: application/json" `
  --data-binary "@$bodyFile"

$job_id = ($job_create | ConvertFrom-Json).job_id
"JOB_ID=$job_id"

# cancel (best-effort):
curl.exe -sS -X POST -H $AUTH "$BASE/v1/jobs/$job_id/cancel"

# ====== 8) DEBUG (если включён) ======
curl.exe -sS -w $CURL_W -H $AUTH "$BASE/v1/debug/artifacts?limit=5"
curl.exe -sS -w $CURL_W -H $AUTH "$BASE/v1/debug/artifacts/$RID"
curl.exe -sS -w $CURL_W -H $AUTH "$BASE/v1/debug/artifacts/$RID/raw"
curl.exe -sS -w $CURL_W -H $AUTH -o artifacts_backup.tar.gz "$BASE/v1/debug/artifacts/backup.tar.gz"
```

---

### 7.2 Bash (Linux/macOS)

> Тут тоже сохраняем JSON в файл и шлём `--data-binary @file`.

```bash
set -euo pipefail

BASE="http://127.0.0.1:8080"
APIKEY="XXXXX"
AUTH="Authorization: Bearer ${APIKEY}"

mkdir -p ./tmp

# 0) HEALTH + AUTH
curl -sS "$BASE/v1/health"
curl -sS -H "$AUTH" "$BASE/v1/me"

# 1) extract (base64)
IMG="./data/batch_smoke/images/cord_0000.jpg"
IMG_B64="$(base64 -w0 "$IMG" 2>/dev/null || base64 "$IMG" | tr -d '\n')"
RID="smoke_$(uuidgen | tr -d '-')"

cat > ./tmp/extract.json <<JSON
{"task_id":"receipt_fields_v1","image_base64":"$IMG_B64","mime_type":"image/jpeg","request_id":"$RID"}
JSON

curl -sS -X POST "$BASE/v1/extract" \
  -H "$AUTH" -H "Content-Type: application/json" -H "X-Request-ID: $RID" \
  --data-binary "@./tmp/extract.json"

# 2) batch_extract_upload (multipart)
RUN_ID="smoke_$(date +%Y%m%dT%H%M%S)"
curl -sS -X POST "$BASE/v1/batch_extract_upload" \
  -H "$AUTH" \
  -F "task_id=receipt_fields_v1" \
  -F "persist_inputs=true" \
  -F "concurrency=2" \
  -F "run_id=$RUN_ID" \
  -F "files=@$IMG;type=image/jpeg"

# 3) batch_extract_rerun
cat > ./tmp/batch_rerun.json <<JSON
{"task_id":"receipt_fields_v1","run_id":"$RUN_ID","concurrency":2,"max_days":90}
JSON

curl -sS -X POST "$BASE/v1/batch_extract_rerun" \
  -H "$AUTH" -H "Content-Type: application/json" \
  --data-binary "@./tmp/batch_rerun.json"

# 4) async extract_async
cat > ./tmp/extract_async.json <<JSON
{"task_id":"receipt_fields_v1","image_base64":"$IMG_B64","mime_type":"image/jpeg"}
JSON

create="$(curl -sS -X POST "$BASE/v1/extract_async" \
  -H "$AUTH" -H "Content-Type: application/json" \
  --data-binary "@./tmp/extract_async.json")"
echo "$create"
job_id="$(echo "$create" | python -c 'import sys,json; print(json.load(sys.stdin)["job_id"])')"

# polling
while true; do
  j="$(curl -sS -H "$AUTH" "$BASE/v1/jobs/$job_id")"
  status="$(echo "$j" | python -c 'import sys,json; print(json.load(sys.stdin)["status"])')"
  echo "$(date +%H:%M:%S) status=$status"
  if [[ "$status" == "succeeded" || "$status" == "failed" || "$status" == "canceled" ]]; then
    echo "$j"
    break
  fi
  sleep 1
done
```

---
## 8) Запуск RunPod
- Выбрать GPU (например A5000 24GB).
- Образ: `vllm/vllm-openai:latest`
- Start command (рекомендуется через shell, чтобы работали `${...}`)
```bash
{"entrypoint":["bash","-lc"],"cmd":["vllm serve ${VLLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct} --host 0.0.0.0 --port 8000 --trust-remote-code --max-model-len ${VLLM_MAX_MODEL_LEN:-4096} --download-dir ${HF_HOME:-/workspace/hf} --dtype ${VLLM_DTYPE:-auto} --gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"]}
```
или же просто
```bash
--model Qwen/Qwen3-VL-8B-Instruct --host 0.0.0.0 --port 8000 --trust-remote-code --max-model-len 4096 --limit-mm-per-prompt.video 0
```
- Порт: `8000`
- Env (пример)
VLLM_API_KEY=
HF_TOKEN=


## 9) Troubleshooting (частые ошибки)

- **`vLLM HTTP 404`**: почти всегда неверный `VLLM_BASE_URL` (или забыли `/v1`).  
  Должно быть так, чтобы оркестратор дергал `${VLLM_BASE_URL}/chat/completions`.
- **`schema_validation_failed`**: модель вернула JSON, где нет обязательных полей (см. `required` в schema).  
  Для `INFERENCE_BACKEND=mock` удобно добавлять `examples` в schema (например, все ключи → `null`).

---
