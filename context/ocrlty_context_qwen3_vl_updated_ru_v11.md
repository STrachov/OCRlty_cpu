# OCRlty Qwen3‑VL — проектный контекст (Split: RunPod vLLM GPU + CPU Orchestrator + Cloudflare R2)

> Цель проекта: стабильная и дешёвая inference‑система для извлечения полей из чеков/счетов (receipt/invoice) с воспроизводимой инфраструктурой, артефактами, оценкой качества и безопасным API (ключи/скоупы).

---

## 0) Что изменилось относительно прежней архитектуры

**Было (legacy):** один pod/контейнер на RunPod поднимал и vLLM, и Orchestrator, а артефакты писались на volume.

**Стало (текущая архитектура):**
- **GPU слой:** отдельный RunPod Pod со **стандартным образом `vllm/vllm-openai`**, который поднимает OpenAI‑совместимый API (`/v1/...`) и держит модель.
- **CPU слой:** отдельный сервер (VPS) с **Orchestrator (FastAPI)**: принимает запросы, делает auth, вызывает vLLM, валидирует/парсит результат, сохраняет артефакты.
- **Хранилище артефактов:** Cloudflare **R2 (S3‑compatible)** как source of truth.

Почему так:
- быстрее старт/пересоздание GPU pod (стандартный vLLM образ, меньше кастомного build),
- CPU часть обновляется независимо,
- артефакты больше не зависят от локального диска/volume конкретного pod.

---

## 1) Высокоуровневая архитектура (два сервиса + объектное хранилище)

### 1.1 Компоненты
1) **vLLM OpenAI server (GPU, RunPod)**
- образ: `vllm/vllm-openai` (стандартный)
- порт: **8000**
- модель: `Qwen/Qwen3-VL-8B-Instruct`
- защищён **API key** на уровне vLLM (`VLLM_API_KEY`)

2) **Orchestrator (CPU, VPS, FastAPI)**
- порт: **8080**
- принимает запросы клиентов (`/v1/extract`, `/v1/batch_extract`, `/v1/debug/...`)
- делает auth/scopes (SQLite + HMAC‑hash key)
- вызывает vLLM по `VLLM_BASE_URL`
- сохраняет артефакты в R2/S3
- пишет структурные JSON логи (stdout)

3) **Cloudflare R2 (S3)**
- хранение артефактов: extracts / batches / evals
- bucket один, внутри используется `S3_PREFIX` (например `prod`)

### 1.2 Поток данных (single extract)
Client → **Orchestrator** → (HTTP) **vLLM** → Orchestrator (parse/validate) → **R2** (save artifact) → Client

---

## 2) Сетевые границы и безопасность

### 2.1 Что открыто наружу
- **Открыт только Orchestrator** (VPS).
- **vLLM (RunPod) может быть публично достижим** через RunPod proxy URL, но должен быть:
  - защищён `VLLM_API_KEY`,
  - по возможности ограничен по IP (если RunPod/Firewall позволяет),
  - доступен только Orchestrator’у.

### 2.2 Важное про RunPod proxy URL и POD_ID
RunPod proxy URL имеет вид:
`https://<POD_ID>-8000.proxy.runpod.net/v1`

`POD_ID` может **поменяться** при пересоздании pod или миграции, поэтому:
- Orchestrator должен получать `VLLM_BASE_URL` как конфиг (env/settings),
- (план) добавить менеджер pod’а через RunPod API, который умеет обновлять endpoint при миграции/пересоздании.

---

## 3) Конфигурация (env → settings)

### 3.1 Оркестратор (CPU) — ключевые env
**vLLM**
- `INFERENCE_BACKEND` — `mock` (тестовый, возвращает валидный JSON по схеме) или `vllm` (боевой вызов vLLM).
- `VLLM_BASE_URL` — например `https://<POD_ID>-8000.proxy.runpod.net/v1`
- `VLLM_API_KEY` — ключ для доступа к vLLM
- `VLLM_MODEL` — `Qwen/Qwen3-VL-8B-Instruct`

**vLLM HTTP (keep-alive) и таймауты (Orchestrator → vLLM)**
- `VLLM_HTTP_MAX_CONNECTIONS` — общий лимит соединений httpx (например 50–200)
- `VLLM_HTTP_MAX_KEEPALIVE` — лимит keep-alive соединений (например 20–100)
- `VLLM_HTTP_KEEPALIVE_EXPIRY_S` — TTL keep-alive (например 30–120s)
- `VLLM_CONNECT_TIMEOUT_S` — connect timeout
- `VLLM_READ_TIMEOUT_S` — read timeout (самый важный для долгих ответов модели)
- `VLLM_WRITE_TIMEOUT_S` — write timeout
- `VLLM_POOL_TIMEOUT_S` — pool timeout

**Jobs (async / 202 + polling)**
- `JOBS_BACKEND` — `local` (in-process) или `celery`
- `JOBS_DB_PATH` — путь к SQLite БД jobs (например `/data/db/jobs.db`)
- `JOBS_MAX_CONCURRENCY` — глобальный лимит одновременно выполняющихся job (local runner)
- `CELERY_BROKER_URL` — (если `JOBS_BACKEND=celery`) например `redis://redis:6379/0`
- `CELERY_RESULT_BACKEND` — (опционально) backend результатов Celery (можно оставить пустым и хранить результат в SQLite)

**R2 / S3 артефакты**
- `S3_BUCKET` — имя bucket (один bucket)
- `S3_ENDPOINT_URL` — `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`
- `S3_REGION` — `auto`
- `S3_PREFIX` — например `prod` или `ocrlty` (префикс внутри bucket)
- `S3_ALLOW_OVERWRITE` — по умолчанию 0 (безопасно, не перетирать артефакты)
- `S3_PRESIGN_TTL_S` — TTL для presigned url (если включим позже)
- `S3_FORCE_PATH_STYLE` — обычно 0

**AWS‑совместимые креды для R2 (boto3)**
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

**Логирование**
- `LOG_LEVEL` — `INFO` (или `DEBUG`)
- `LOG_FORMAT` — `json` (по умолчанию)

**Auth**
- `AUTH_ENABLED=1`
- `AUTH_DB_PATH=...` (SQLite)
- `API_KEY_PEPPER=...` (секрет для HMAC хэша)

> В коде настройки читаются через `settings` (Pydantic Settings). Это важно: `.env` подхватывается settings, но не обязан попадать в `os.getenv()`.

### 3.2 vLLM (GPU pod) — env
- `VLLM_MODEL=Qwen/Qwen3-VL-8B-Instruct`
- `VLLM_MAX_MODEL_LEN=4096` (пример)
- `VLLM_API_KEY=...`
- `HF_TOKEN=...` (если нужно для скачивания модели)

---

## 4) Развёртывание GPU (RunPod) — стандартный vLLM + модель + API key

### 4.1 Настройка pod
- Выбрать GPU (например A5000 24GB).
- Образ: `vllm/vllm-openai`
- Start command (рекомендуется через shell, чтобы работали `${...}`)
Если нужно использовать значение переменных среды в качестве параметров запуска то только в виде json объекта такого вида (если просто строка с ${VLLM_...} - то не работает):
```bash
{"entrypoint":["bash","-lc"],"cmd":["vllm serve ${VLLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct} --host 0.0.0.0 --port 8000 --trust-remote-code --max-model-len ${VLLM_MAX_MODEL_LEN:-4096} --download-dir ${HF_HOME:-/workspace/hf} --dtype ${VLLM_DTYPE:-auto} --gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"]}
```
Если же со всеми параметрами определились то так (модель указаывается первой без ключа --model - иначе ошибка):
```bash
Qwen/Qwen3-VL-8B-Instruct --host 0.0.0.0 --port 8000 --trust-remote-code --max-model-len 4096
```
- Порт: `8000`
- Env (пример)
VLLM_API_KEY=...
HF_TOKEN=...

### 4.3 Smoke tests (PowerShell)
```powershell
$env:VLLM_BASE_URL="https://POD_ID-8000.proxy.runpod.net/v1"
$env:VLLM_API_KEY="YOUR_KEY"
curl.exe -sS "$env:VLLM_BASE_URL/models" -H "Authorization: Bearer $env:VLLM_API_KEY"
```

---

## 5) Orchestrator (CPU) — назначение и ответственности

Orchestrator выполняет:
- аутентификацию/авторизацию клиентов (API key + scopes),
- подготовку промпта/схемы по `task_id`,
- вызов vLLM (OpenAI chat completions, multimodal),
- разбор JSON, валидацию по jsonschema,
- сохранение артефактов (успехи и ошибки) в R2/S3,
- отдачу клиенту краткого результата + `request_id` + `artifact_rel` (указатель на артефакт).
- запуск долгих операций через **job-модель** (202 Accepted + polling по `/v1/jobs/{job_id}`) без удержания HTTP-соединения.

---

## 6) Артефакты: хранение в Cloudflare R2 (S3)

### 6.1 Структура ключей в bucket
Внутри bucket используется `S3_PREFIX` (например `prod`), далее:

- `{prefix}/extracts/YYYY-MM-DD/<request_id>.json`
- `{prefix}/batches/YYYY-MM-DD/<run_id>.json`
- `{prefix}/evals/YYYY-MM-DD/<eval_id>.json`

### 6.2 Принципы
### 6.3 `artifact_index` (SQLite) — быстрый поиск артефактов
- индексирует артефакты по `(kind, artifact_id)` и владельцу (owner_key_id).
- хранит `full_ref` (как реально читать: S3 key с `S3_PREFIX` или абсолютный local path) и `rel_ref` (UI‑friendly ссылка: key без `S3_PREFIX` или путь относительно `ARTIFACTS_DIR`).
- в UI/ответах предпочтительно оперировать `rel_ref`/`artifact_rel`, а на сервере резолвить в `full_ref`.

- Артефакты сохраняются **и при ошибках** (inference/parse/validate), чтобы всегда было “куда смотреть”.
- Сейчас `input` в extract‑артефакте не содержит base64 вообще: при наличии base64 в запросе в артефакт пишем только image_base64_len, а для воспроизводимости используем image_ref/input_ref (в batch всегда есть input_ref; в sync image_ref появляется только если input был сохранён как ref).
- По умолчанию включена защита от перезаписи (если `S3_ALLOW_OVERWRITE=0`).

---

## 7) Логирование (минимально необходимое)

### 7.1 Формат
- JSON лог‑строки в stdout (идеально для Docker/journald).
- Корреляция через `request_id`:
  - берётся из `X-Request-ID` или генерируется,
  - возвращается клиенту в header `X-Request-ID`.

### 7.2 События (минимум)
- `request_start`, `request_end`, `request_error`
- `vllm_call_start`, `vllm_call_end`, `vllm_call_error`
- `s3_put`, `s3_get`, `s3_retry`

---

## 8) API Orchestrator (порт 8080)

### 8.1 Health
`GET /health`

### 8.2 Single extract (оставлен скорее для тестов)
`POST /v1/extract`

- вход: `image_base64` (боевой вариант) или `image_url` (только debug)
- ответ: `request_id`, `result`, `schema_ok`, `schema_errors`, `timings_ms`, `artifact_rel`

### 8.3 Batch extract с файлами на сервере (оставлен скорее для тестов)
`POST /v1/batch_extract`
### 8.4 Batch extract с файлами загруженными пользователем (с возможностью указать нужно ли их сохранять на S3)
`POST /v1/batch_extract_upload`
### 8.5 Повторный Batch extract с файлами сохраненными на S3 но с другой задачей или параметрами (в качестве указателя используется run_id)
`POST /v1/batch_extract_rerun`

> Контракты (рекомендуемый каноничный вариант):
> - `/v1/batch_extract` и `/v1/batch_extract_rerun` — **application/json** (JSON body).
> - `/v1/batch_extract_upload` — **multipart/form-data** (файлы), опционально `persist_inputs=true` для последующего rerun.
>
> Auto-eval vs GT (если передан `gt_path`):
> - включается только при `DEBUG_MODE=1`;
> - рекомендуется требовать scope `debug:run` (а `debug:read_raw` оставить для чтения raw/артефактов);
> - `gt_image_key` по умолчанию: **`"file"`** (совпадает с ключом `image` в `items[]` batch-артефакта; сравнение по basename).

### 8.6 Jobs API (async для долгих операций)

Для операций, которые могут выполняться дольше обычного HTTP‑таймаута (batch/rerun/большие uploads), используется **job‑модель**:
- запрос создаёт job и возвращает **`202 Accepted` + `job_id`**
- клиент делает polling: `GET /v1/jobs/{job_id}` до финального статуса
- отмена: `POST /v1/jobs/{job_id}/cancel` (best‑effort)

**Статусы:** `queued`, `running`, `succeeded`, `failed`, `canceled`.

**Хранение:** jobs пишутся в SQLite (`JOBS_DB_PATH`).  
**Исполнение (переключатель):**
- `JOBS_BACKEND=local` — in‑process runner с ограничением `JOBS_MAX_CONCURRENCY`
  - при старте процесса: `running` помечаются как `failed` (stale_after_restart),
  - `queued` requeue’ятся обратно в runner (best‑effort).
- `JOBS_BACKEND=celery` — тот же API, но enqueue идёт в очередь (Redis/Celery).

#### 8.6.1 “Низкоуровневые” jobs эндпоинты (router `jobs`)
Эти эндпоинты создают job напрямую (каноничный путь для любых будущих job‑типов):
- `POST /v1/jobs/batch_extract_dir` → `202` + `job_id` (обработка server-side dir)
- `POST /v1/jobs/batch_extract_rerun` → `202` + `job_id` (повторный прогон из сохранённых inputs)
- `GET /v1/jobs/{job_id}` → статус/прогресс/результат
- `GET /v1/jobs?status=queued|running|succeeded|failed|canceled` → список (admin видит всё, не‑admin — только своё)
- `POST /v1/jobs/{job_id}/cancel` → best‑effort отмена (ставит `cancel_requested=1`; queued может стать `canceled` сразу)

**Прогресс:** `GET /v1/jobs/{job_id}` возвращает поле `progress` (JSON).
- Для batch‑jobs (upload/rerun) прогресс уже “UI‑friendly”: `{stage, total, done, ok, err, current}` с throttling через `JOBS_PROGRESS_EVERY_N` и `JOBS_PROGRESS_MIN_SECONDS`.
- Для прочих job‑типов прогресс может оставаться минимальным.

#### 8.6.2 “Высокоуровневые” async wrappers (router `extract`, те же контракты что sync)
Чтобы клиенту было проще мигрировать, добавлены “параллельные” async‑эндпоинты, которые принимают **те же тела**, что и sync‑варианты, но возвращают job:

- `POST /v1/extract_async` → `202 {job_id, poll_url}`
- `POST /v1/batch_extract_async` → `202 {job_id, poll_url}`
- `POST /v1/batch_extract_rerun_async` → `202 {job_id, poll_url}`
- `POST /v1/batch_extract_upload_async` (multipart) → `202 {job_id, poll_url}`

Ключевой момент для `*_upload_async`:
- файлы **сразу сохраняются** как `artifacts/inputs/...` (или в R2/S3, если включено),
- в JobsStore кладётся только список `{file, input_ref, mime_type}` (без base64/байтов),
- job выполняется по `input_ref` (повторяемо и безопасно для SQLite).

> Важно: чтобы polling работал, `jobs_router` должен быть подключён в `main.py`: `app.include_router(jobs_router)`.

### 8.9 Runs API (для UI, batch artifact = продуктовый объект)

- `GET /v1/runs?limit&cursor` — список batch‑прогонов (используем `artifact_index` как run registry v1).
- `GET /v1/runs/{run_id}` — вернуть **полный batch‑артефакт** (summary + `items[]`), где каждый item содержит минимум для UI: `parsed`, `schema_valid`, `schema_errors`, `error`, `error_history`, `file`, `request_id`, `input_ref`, `artifact_rel`.
- `GET /v1/artifacts?ref=<artifact_rel>` — drill‑down: вернуть полный JSON extract‑артефакта для конкретного item (auth‑gated, без debug).

> Примечание про пагинацию: пока батчи небольшие — UI может получать batch целиком. Когда батчи станут большими, понадобится серверная пагинация/чанкинг (см. TODO P2).

### 8.7 Debug artifacts (S3‑режим)
(доступ только при `DEBUG_MODE=1` + scope `debug:read_raw`)

- `GET /v1/debug/artifacts` — листинг (через S3 list)
- `GET /v1/debug/artifacts/{request_id}` — прочитать JSON артефакта
- `GET /v1/debug/artifacts/{request_id}/raw` — только `raw_model_text`
- `GET /v1/debug/artifacts/backup.tar.gz` — on‑demand backup (с лимитом на кол-во объектов)

### 8.8 GT формат для auto-eval (минимум)
GT файл — JSON (список объектов или dict с records), где в каждой записи есть поле с именем/путём картинки.
**По умолчанию используется ключ `file`** (как в твоём GT). Сопоставление идёт по **basename** (т.е. `Path(value).name`).

Важно: в batch-артефакте имя картинки обычно хранится в `items[].image`, но eval-логика умеет брать имя и из `file|image|image_file`
в каждом `item` (на всякий случай).

Пример (list):
```json
[
  {"file": "receipts/receipt_001.png", "fields": {"total": "12.34", "date": "2025-01-31"}},
  {"file": "receipt_002.jpg", "fields": {"total": "9.99"}}
]
```

Если GT использует другой ключ (например, `image`), просто передай `gt_image_key=image`.

Требования безопасности:
- `gt_path` должен быть под `DATA_ROOT` или `ARTIFACTS_DIR`;
- auto-eval включается только при `DEBUG_MODE=1`.

---

## 9) Аутентификация (кратко, текущий подход)

- API key передаётся как:
  - `Authorization: Bearer <api_key>` (канонично)
  - или `X-API-Key: <api_key>`
- В БД хранится **только HMAC‑SHA256 hash** (pepper в env).
- Scopes:
  - `extract:run`
  - `debug:run`
  - `debug:read_raw`
  - (admin scopes — позже)

---

## 10) Рекомендации по конфиденциальности (inputs)

Текущий подход:
- inputs не храним в облаке только при явном указании пользователя (для повторного прогона, например),
- все остальные держим локально на CPU с коротким TTL (или удаляем сразу после обработки),

---

## 11) Тестирование end‑to‑end (CPU Orchestrator + Jobs)

Ниже — набор команд для проверки **sync** и **async (202 + polling)** эндпоинтов.

### 11.1 PowerShell (Windows) — используй `curl.exe` и сохранение тела json запроса во временном файле

```powershell
# ====== CONFIG ======
$BASE   = "http://127.0.0.1:8080"         # или https://<vps-host>
$APIKEY = "oAJcMSdufcInKdy9PRb-sb-5H-vDquybRQSpym3fZg4"
$AUTH   = "Authorization: Bearer $APIKEY"
$CURL_W = "`nHTTP %{http_code}`n"

New-Item -ItemType Directory -Force -Path ".\tmp" | Out-Null

# ====== 0) HEALTH + AUTH ======
curl.exe -sS -w $CURL_W "$BASE/health"
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

### 11.2 Bash (Linux/macOS) — (опционально) с `jq`

```bash
BASE="http://127.0.0.1:8080"
APIKEY="<PASTE_YOUR_API_KEY_HERE>"
AUTH="Authorization: Bearer ${APIKEY}"

# 0) HEALTH + AUTH
curl -sS "$BASE/health"
curl -sS -H "$AUTH" "$BASE/v1/me"

# 1) ASYNC: extract_async (пример с заранее подготовленным base64)
IMG_B64="$(base64 -w0 receipt.png)"
create="$(curl -sS -X POST "$BASE/v1/extract_async" \
  -H "$AUTH" -H "Content-Type: application/json" \
  --data-binary "{\"task_id\":\"receipt_fields_v1\",\"image_base64\":\"$IMG_B64\",\"mime_type\":\"image/png\"}")"
echo "$create"
job_id="$(echo "$create" | jq -r .job_id)"

# polling (status + progress):
while true; do
  j="$(curl -sS -H "$AUTH" "$BASE/v1/jobs/$job_id")"
  status="$(echo "$j" | jq -r .status)"
  progress="$(echo "$j" | jq -c .progress)"
  echo "$(date +%H:%M:%S) status=$status progress=$progress"
  if [[ "$status" == "succeeded" || "$status" == "failed" || "$status" == "canceled" ]]; then
    echo "$j" | jq .
    break
  fi
  sleep 1
done

# 2) ASYNC: batch_extract_upload_async (multipart)
create="$(curl -sS -X POST "$BASE/v1/batch_extract_upload_async" \
  -H "$AUTH" \
  -F "task_id=receipt_fields_v1" \
  -F "concurrency=2" \
  -F "run_id=async_up_$(date +%Y%m%dT%H%M%S)" \
  -F "files=@./receipt.png;type=image/png")"
echo "$create"
job_id="$(echo "$create" | jq -r .job_id)"
curl -sS -H "$AUTH" "$BASE/v1/jobs/$job_id" | jq .

# 3) Низкоуровневый jobs: batch_extract_dir
create="$(curl -sS -X POST "$BASE/v1/jobs/batch_extract_dir" \
  -H "$AUTH" -H "Content-Type: application/json" \
  --data-binary '{
    "images_dir":"/workspace/src/data",
    "glob":"**/*",
    "exts":["png","jpg","jpeg","webp"],
    "limit":10,
    "task_id":"receipt_fields_v1",
    "concurrency":2,
    "run_id":null
  }')"
job_id="$(echo "$create" | jq -r .job_id)"
echo "job_id=$job_id"
```

**Если `/v1/jobs/...` отдаёт 404:** проверь, что в `main.py` подключён `jobs_router` (`app.include_router(jobs_router)`), и что `JOBS_BACKEND=local`/`celery` выставлен корректно.


---

## 12) TODO 

### P0 — UI-ready API (минимум, чтобы начать интерфейс)
* Под прогоном (run) понимается батч прогон.
1) **Runs list поверх `artifact_index` (без нового run registry)**. 
- добавить `ArtifactIndex.list(...)`:
  - фильтры: `kind="batch"`, `owner_key_id`, `limit`, `cursor`
  - сортировка по дате создания (как по возрастанию так и по убыванию)
- эндпоинт:
  - `GET /v1/runs?limit&cursor&sort=asc` → список прогонов (run_id=`artifact_id`, day, created_at, artifact_ref/rel)

2) **Run details = полный batch artifact**
- эндпоинт:
  - `GET /v1/runs/{batch_artifact_id}` → вернуть полный JSON батч артефакта (на данный момент он уже содержит нужные данные для UI: `parsed`, `schema_valid`, `schema_errors`, `error`, `error_history`, `file`, `request_id`, `input_ref`, `artifact_rel`)
  
3) **Drill-down по одному item (дешёвый вариант, 1 чтение из storage)**
- эндпоинт:
  - `GET /v1/extracts/{request_id}` → вернуть полный JSON extract-артефакта (auth-gated, без debug)

- серверная логика/проверки:
  1) найти артефакт по индексу: `(kind="extract", artifact_id=request_id)` → получить `full_ref`
  2) проверить владельца: `owner_key_id` из `artifact_index` == текущий `principal.key_id`
     - (админ-роль может читать всех, если так задумано)
  3) прочитать JSON из storage по `full_ref` и вернуть его
  4) выдача `raw` — только при соответствующем scope (`debug:read_raw`/аналог), иначе `raw=null` (если `raw` присутствует в файле)

- примечания:
  - этот вариант **не проверяет**, что `request_id` относится к конкретному `run_id`; гарантируется только “пользователь видит свои артефакты”.
  - строгую проверку “item ∈ run” можно добавить позже отдельным эндпоинтом или через таблицу `run_items` (P2/P3).


4) **CORS для браузерного UI + минимальная модель auth для UI**
- allowlist origins (dev/prod)
- на старте можно API key вручную (позже — токены/сессии)

---

### P1 **RunPod pod manager** в Orchestrator:
- start/stop/create pod
- обновление `VLLM_BASE_URL` при миграции/пересоздании
- retries на cold start

### P2 — Контроль стоимости/безопасности (когда начнет использоваться чаще)
5) **Лимиты payload**
- max image bytes (sync base64)
- max files per upload
- max total bytes per batch
- явные `error_code`: `payload_too_large`, `too_many_files`

6) **Rate limits / quotas**
- per key_id: req/min, jobs/hour
- `error_code`: `rate_limited`

7) **Presigned uploads (опционально, не обязательный путь)**
- цель: “тонкий клиент” + разгрузка оркестратора от передачи больших файлов
- `POST /v1/uploads/create` → presigned PUT’ы + будущие `input_ref`
- `POST /v1/uploads/complete` → старт `batch_extract_upload_async`
- TTL + CORS на bucket
- важно: persist inputs включать только когда нужен rerun/аудит (не всегда)

8) **Progress для batch async (уже реализовано; довести мелочи под UI)**
- уже есть: `_emit_progress` + `progress_cb`, counters `total/done/ok/err/current`, throttling через `JOBS_PROGRESS_EVERY_N` и `JOBS_PROGRESS_MIN_SECONDS`, стадии `extracting/finalizing/succeeded/failed`
- (опционально):
  - добавить стадию `loading_inputs` (для upload/rerun до начала обработки)
  - добавить стадию `canceled` при cooperative cancel (когда будет сделано)
  - убедиться, что runner/executor всегда прокидывает `progress_cb` для `batch_extract_upload_async` и `batch_extract_rerun_async`


---

### P3 — Эксплуатация и эволюция формата (когда появятся большие батчи/много пользователей)
9) **Наблюдаемость**
- `/metrics` Prometheus: latency, job_duration, batch_throughput, retries, vLLM statuses
- единая таксономия `error.code`/`error.detail.code` для UI и аналитики

10) **Кооперативная отмена batch**
- check `cancel_requested` перед каждым item
- аккуратно завершать: partial batch artifact + job `canceled`

11) **Пагинация items (реальная)**
- когда batch.json станет слишком большим для UI/сети:
  - перейти на chunking (`items_0001.json`, `items_0002.json`) или JSONL
  - или завести таблицу `run_items`
- после этого добавить:
  - `GET /v1/runs/{run_id}/items?status=...&limit&cursor` (серверная пагинация без чтения всего файла)

---

### P4 — Масштабирование под клиентов
12) **Celery/Redis backend**
- compose-профиль
- `JOBS_BACKEND=celery` без изменения API



---

## 13) Эксплуатационные заметки (R2/S3, .env, логирование, VPS)

### Cloudflare R2: S3 credentials и типовые ошибки

Для записи артефактов в Cloudflare R2 через boto3 нужны **S3-compatible credentials**:

- `AWS_ACCESS_KEY_ID` — **32 символа** (token id, hex)
- `AWS_SECRET_ACCESS_KEY` — **64 hex** (sha256)

Если видишь ошибку вида:
- `Credential access key has length 20, should be 32`

…значит, в .env не правильные значения ключей или библиотека не читает .env.

Рекомендации:
- проверь, какие значения у переменных в  AWS_ACCESS_KEY_ID и AWS_SECRET_ACCESS_KEY в .env
- проверь, загружается ли .env при запуске.

### Почему значения из `.env` могут “не подхватываться” (и что считается правильным)

В проекте часть настроек читается через `pydantic-settings` (оно умеет читать `.env`), а часть — через `os.getenv()` (например, boto3 креды).
Это **разные источники**:

- `.env` прочитан pydantic → значения доступны как `settings.*`
- но это **не означает**, что эти значения появились в `os.environ`
- boto3 (и вообще `os.getenv()`) видит только то, что реально находится в окружении процесса

**Best practice по средам:**

- **Docker / docker-compose (рекомендуется для VPS)**: использовать `env_file: [.env]` в compose — тогда переменные попадут в окружение контейнера, и boto3 их увидит.
- **Локальный запуск uvicorn (без Docker)**: запускать с `--env-file .env` (если поддерживается), либо явно загружать `.env` в `os.environ` через `python-dotenv` на старте приложения.

Нюанс: S3 клиент в коде обычно кэшируется (singleton). Поэтому изменение env без перезапуска процесса/контейнера может не дать эффекта — **перезапуск обязателен** при смене AWS/R2 ключей или endpoint’ов.

### Логирование на небольшом VPS (1 vCPU / 1GB RAM / 20GB SSD)

Для такого VPS нецелесообразно поднимать локально ELK/Loki: это быстро “съест” память и диск.
Практичный подход:

- писать логи приложения в `stdout/stderr`
- смотреть их через Docker (`docker compose logs`) или journald (если выбран journald driver)
- включить **ротацию логов Docker**, чтобы не заполнить диск:
  - `max-size: "10m"`
  - `max-file: "5"`

Если нужно централизованное хранение и поиск:
- предпочтительнее hosted-решения (Grafana Cloud Loki / др.) или периодическая выгрузка логов (например, раз в сутки) в R2.

### VPS деплой: Docker + systemd (без противоречий)

Если “всё должно работать через Docker”, это нормально сочетается с systemd:

- приложение запускается и живёт **в контейнере**
- `systemd` (опционально) используется как “менеджер” на уровне VPS:
  - поднять `docker compose up -d` при старте системы
  - обеспечить автозапуск после ребута
  - контролировать зависимости (`docker.service`, сеть)

Логи при этом обычно читаются через Docker:
- `docker compose logs -f orchestrator`
- или `docker logs -f orchestrator`


---

