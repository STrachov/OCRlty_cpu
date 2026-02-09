# OCRlty (Qwen3-VL) — CPU Orchestrator + RunPod vLLM + Cloudflare R2 (S3)

Сервис для извлечения структурированных данных из чеков/счетов с помощью мультимодальной LLM (например `Qwen/Qwen3-VL-8B-Instruct`), с воспроизводимыми артефактами, аутентификацией, логированием и batch-прогонами.

## TL;DR (как это работает)

**Архититектура (split):**

- **GPU слой (RunPod):** стандартный образ `vllm/vllm-openai` поднимает OpenAI-совместимый API (`/v1/...`) и держит модель.
- **CPU слой (VPS):** FastAPI Orchestrator:
  - принимает запросы клиентов,
  - проверяет API key + scopes,
  - вызывает vLLM,
  - валидирует/парсит JSON,
  - сохраняет артефакты в **Cloudflare R2 (S3)**,
  - пишет **JSON logs** (stdout) с корреляцией `request_id`.

---

## Репозиторий (ожидаемая структура)

```
.
├─ app/
│  ├─ main.py
│  ├─ settings.py
│  ├─ auth.py
│  ├─ vllm_client.py
│  ├─ prompts/
│  └─ schemas/
├─ Dockerfile
├─ entrypoint.sh
├─ requirements.in
├─ requirements.txt   (желательно pinned)
└─ .github/workflows/
   └─ build-orchestrator.yml
```

---

## Предварительные требования

### 1) Cloudflare R2 (S3)
- Создай **один bucket** (например `ocrlty`).
- Создай **R2 API Token** (S3 credentials) с доступом **Read/Write** к этому bucket.
- Важно: **не делай bucket публичным**.

### 2) RunPod (GPU) + vLLM
- Запусти pod с GPU (например A5000).
- Образ: `vllm/vllm-openai`
- Порт: `8000`
- Включи авторизацию на стороне vLLM через API key (`VLLM_API_KEY`).

### 3) VPS (CPU orchestrator)
- Ubuntu 24.04, Docker + Docker Compose.
- На 1 vCPU / 1GB RAM рекомендуется включить swap (2GB), иначе при пиках можно поймать OOM.

---

## Конфигурация (env → settings)

### Оркестратор (CPU) — основные переменные

**vLLM**
- `VLLM_BASE_URL` — например: `https://<POD_ID>-8000.proxy.runpod.net/v1`
- `VLLM_API_KEY` — ключ доступа к vLLM
- `VLLM_MODEL` — например: `Qwen/Qwen3-VL-8B-Instruct`

**R2 / S3**
- `S3_BUCKET` — имя bucket (например `ocrlty`)
- `S3_ENDPOINT_URL` — `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`
- `S3_REGION` — `auto`
- `S3_PREFIX` — префикс внутри bucket (например `prod`)
- `S3_ALLOW_OVERWRITE` — `0|1` (по умолчанию `0`, безопаснее)
- `S3_PRESIGN_TTL_S` — TTL для presigned URL (если включишь позже)
- `S3_FORCE_PATH_STYLE` — обычно `0`

**AWS creds для boto3 (R2)**
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

**Auth**
- `AUTH_ENABLED=1`
- `API_KEY_PEPPER=...` (обязателен при AUTH_ENABLED=1)
- `AUTH_DB_PATH=/data/auth/auth.db` (SQLite, путь в контейнере)

**Логи**
- `LOG_LEVEL=INFO|DEBUG|WARNING|ERROR`
- `LOG_FORMAT=json|text` (рекомендуется `json`)

> ⚠️ Важно: настройки читаются через `settings` (pydantic settings), поэтому `.env` подхватывается корректно.

---

## Логирование (минимум, но достаточно)

- JSON-события в stdout.
- Корреляция через `request_id`:
  - берётся из `X-Request-ID` (если есть),
  - иначе генерируется,
  - всегда возвращается клиенту в ответе как header `X-Request-ID`.

События:
- `request_start`, `request_end`, `request_error`
- `vllm_call_start`, `vllm_call_end`, `vllm_call_error`
- `s3_put`, `s3_get`, `s3_retry`

---

## Артефакты в R2 (S3)

Структура ключей (пример):
- `{S3_PREFIX}/extracts/YYYY-MM-DD/<request_id>.json`
- `{S3_PREFIX}/batches/YYYY-MM-DD/<run_id>.json`
- `{S3_PREFIX}/evals/YYYY-MM-DD/<eval_id>.json`

В ответах API возвращается `artifact_rel` — относительный “указатель” на артефакт.

---

## Запуск GPU (RunPod vLLM)

### Образ: `vllm/vllm-openai`
### Start command (рекомендуется через shell, чтобы работали `${...}`)
```bash
{"entrypoint":["bash","-lc"],"cmd":["vllm serve ${VLLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct} --host 0.0.0.0 --port 8000 --trust-remote-code --max-model-len ${VLLM_MAX_MODEL_LEN:-4096} --download-dir ${HF_HOME:-/workspace/hf} --dtype ${VLLM_DTYPE:-auto} --gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"]}
```
### Порт: `8000`
### Env (пример)
- `VLLM_API_KEY=...`
- `HF_TOKEN=...` (если надо для скачивания)


  
### Smoke test vLLM (PowerShell)
```powershell
$env:VLLM_BASE_URL="https://POD_ID-8000.proxy.runpod.net/v1"
$env:VLLM_API_KEY="YOUR_KEY"
curl.exe -sS "$env:VLLM_BASE_URL/models" -H "Authorization: Bearer $env:VLLM_API_KEY"
```

---

## Локальная разработка (CPU orchestrator)

### Установка зависимостей (рекомендуется pinned)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

Если используешь `requirements.in`, то лучше генерировать pinned:
```bash
pip install pip-tools
pip-compile requirements.in -o requirements.txt
```


> ⚠️ Примечание про pip-tools  
> Для корректной работы `pip-tools==7.5.2` (в частности `pip-compile`) рекомендуется использовать `pip<26` (например `25.3`):
>
> ```bash
> python -m pip install -U "pip<26" "pip-tools==7.5.2"
> ```



### Запуск
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

---

## Деплой на VPS (Docker)

### 1) docker-compose.yml (пример)
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

### 2) .env на VPS (пример)
```env
# vLLM
VLLM_BASE_URL=https://POD_ID-8000.proxy.runpod.net/v1
VLLM_API_KEY=...
VLLM_MODEL=Qwen/Qwen3-VL-8B-Instruct

# Auth
AUTH_ENABLED=1
API_KEY_PEPPER=...
AUTH_DB_PATH=/data/auth/auth.db

# R2/S3
S3_BUCKET=ocrlty
S3_ENDPOINT_URL=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
S3_REGION=auto
S3_PREFIX=prod
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# Logs
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### 3) Запуск
```bash
docker compose up -d
docker compose logs -f
curl -fsS http://127.0.0.1:8080/health
```

---


## Управление пользователями и API ключами (SQLite auth DB)

Оркестратор хранит пользователей/ключи в SQLite (по умолчанию путь внутри контейнера: `AUTH_DB_PATH=/data/auth/auth.db`).  
В docker-compose это должно быть **на volume**, чтобы база не терялась при пересоздании контейнера:

```yaml
volumes:
  - ./data:/data
```

### Инициализация базы
```bash
python -m app.auth_cli init-db
# или указать путь явно:
python -m app.auth_cli --db /data/auth/auth.db init-db
```

### Создать ключ
> Требуется `API_KEY_PEPPER` в окружении (используется для безопасного хеширования ключей; сырой ключ в БД не хранится).

```bash
export API_KEY_PEPPER="your-secret-pepper"
python -m app.auth_cli create-key --key-id client-1 --role client
```

Опционально можно переопределить scopes (CSV или JSON-массив):
```bash
python -m app.auth_cli create-key --key-id dbg-1 --role debugger --scopes "extract:run,debug:run,debug:read_raw"
# или:
python -m app.auth_cli create-key --key-id dbg-1 --role debugger --scopes '["extract:run","debug:run"]'
```

⚠️ Сгенерированный `api_key` показывается **только один раз** в stdout. Сохрани его сразу.

### Посмотреть список ключей (без секретов)
```bash
python -m app.auth_cli list-keys
# или:
python -m app.auth_cli --db /data/auth/auth.db list-keys
```

### Отозвать ключ
```bash
python -m app.auth_cli revoke-key --key-id client-1
```


## API (основное)

### Health
`GET /health`

### Single extract
`POST /v1/extract`

- вход: `image_base64` (боевой вариант)
- (опционально debug): `image_url` (только если включён debug режим + scope)
- выход: результат + `request_id` + `artifact_rel`

Пример (PowerShell, если используешь `image_url` и debug разрешён):
```powershell
$env:ORCH="http://<VPS>:8080"
$env:API_KEY="YOUR_ORCH_KEY"

$body = @"
{
  "task_id": "receipt_fields_v1",
  "image_url": "https://www.learnopencv.com/wp-content/uploads/2018/06/receipt.png"
}
"@

curl.exe -sS "$env:ORCH/v1/extract" `
  -H "Authorization: Bearer $env:API_KEY" `
  -H "Content-Type: application/json" `
  -H "X-Request-ID: test_req_001" `
  -d $body
```

### Batch extract (текущее состояние)
`POST /v1/batch_extract` — сейчас работает с локальной папкой на CPU сервере.

**План (следующий этап):**
- перейти на batch через manifest (список URL/presigned/S3 keys), чтобы не зависеть от локального диска сервера.

---

## CI (GitHub Actions): сборка образа по тегу

Рекомендуемый паттерн:
- сборка/push image запускается **только по тегам** `orch-v*` или вручную через `workflow_dispatch`.

Пример релиза:
```bash
git tag orch-v0.3.0
git push origin orch-v0.3.0
```

---

## Безопасность и конфиденциальность

- **Не сохраняй input-изображения** в облаке без необходимости.
- R2 bucket держи **private**.
- В логах **не должно быть** base64, сырого OCR текста, полных payload’ов.
- Используй `artifact_rel` + артефакты для дебага вместо логирования данных.

---

## Troubleshooting (коротко)

- **vLLM недоступен / таймауты:** проверь `VLLM_BASE_URL`, `VLLM_API_KEY`, состояние pod в RunPod.
- **S3 ошибки:** проверь `S3_ENDPOINT_URL`, `S3_BUCKET`, `AWS_ACCESS_KEY_ID/SECRET`, и что bucket private.
- **На VPS 1GB падает по памяти:** включи swap, снизь concurrency, отключи DEBUG, оставь 1 worker.

---

## Roadmap (ближайшее)
- Batch через manifest (R2/S3, presigned URLs).
- Pod manager через RunPod API (start/stop, обновление endpoint).
- `/readyz` (проверка доступности vLLM и R2).
