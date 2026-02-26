# AGENTS.md

## Проект: OCR Orchestrator + vLLM

Система инференса для извлечения структурированных данных из чеков/счетов.

Архитектура:
- GPU: RunPod pod с `vllm/vllm-openai` (OpenAI-совместимый API)
- CPU: Orchestrator (FastAPI) — auth, валидация, jobs, артефакты
- Storage: Cloudflare R2 (S3-compatible) — source of truth

---

# 1. Главные правила (НЕ НАРУШАТЬ)

1. Публичные API-контракты стабильны.  
   Нельзя менять формат ответов, ошибок или эндпоинты без явного задания.

2. Артефакты пишутся всегда (включая ошибки).

3. В артефактах НЕЛЬЗЯ хранить base64 изображений.  
   Допустимо только:
   - `image_base64_len`
   - `image_ref`
   - `input_ref`

4. Никогда не логировать секреты и не коммитить ключи.

---

# 2. Архитектура

Поток single extract:

Client  
→ Orchestrator  
→ HTTP вызов vLLM  
→ parse + validate  
→ сохранение артефакта  
→ ответ клиенту  

Снаружи доступен только Orchestrator.  
vLLM защищён `VLLM_API_KEY`.

---

# 3. Ключевые ENV (не переименовывать без задачи)

## Backend
- `INFERENCE_BACKEND=mock|vllm`
- `VLLM_BASE_URL`
- `VLLM_API_KEY`
- `VLLM_MODEL`

## HTTP таймауты (httpx)
- `VLLM_CONNECT_TIMEOUT_S`
- `VLLM_READ_TIMEOUT_S`
- `VLLM_WRITE_TIMEOUT_S`
- `VLLM_POOL_TIMEOUT_S`

## Storage (R2 / S3)
- `S3_BUCKET`
- `S3_ENDPOINT_URL`
- `S3_PREFIX`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `S3_ALLOW_OVERWRITE=0` (по умолчанию)

## Jobs
- `JOBS_BACKEND=local|celery`
- `JOBS_DB_PATH`
- `JOBS_MAX_CONCURRENCY`

## Auth
- `AUTH_ENABLED`
- `AUTH_DB_PATH`
- `API_KEY_PEPPER`

---

# 4. Артефакты

Ключи в bucket:

{S3_PREFIX}/extracts/YYYY-MM-DD/<request_id>.json  
{S3_PREFIX}/batches/YYYY-MM-DD/<run_id>.json  
{S3_PREFIX}/evals/YYYY-MM-DD/<eval_id>.json  

Правила:
- Не перезаписывать артефакты (если `S3_ALLOW_OVERWRITE=0`)
- В API-ответах использовать `artifact_rel`, не полный путь

---

# 5. API (канонические эндпоинты)

- `GET /v1/health`
- `POST /v1/extract`
- `POST /v1/batch_extract`
- `POST /v1/batch_extract_upload`
- `POST /v1/batch_extract_rerun`
- `GET /v1/jobs/{job_id}`
- `POST /v1/jobs/{job_id}/cancel`

Формат ошибки:

{
  "error": {
    "code": "...",
    "message": "...",
    "request_id": "...",
    "details": {}
  }
}

---

# 6. Jobs (async правила)

- Async endpoints → `202` с `{job_id, poll_url}`
- Статусы: `queued | running | succeeded | failed | canceled`
- В jobs нельзя хранить raw bytes или base64

---

# 7. Auth, scopes и roles

Ключ передаётся через:
- `Authorization: Bearer <api_key>`
- или `X-API-Key`

В БД хранится только HMAC-SHA256 hash.  
Pepper берётся из `API_KEY_PEPPER`.

Scopes:
- `extract:run`
- `debug:run`
- `debug:read_raw`

Roles.
В системе определены фиксированные пресеты ролей.  
Их нельзя изменять без отдельной задачи.

ROLE_PRESET_SCOPES:

{
  "client": ["extract:run"],
  "debugger": ["extract:run", "debug:run"],
  "admin": ["extract:run", "debug:run", "debug:read_raw"]
}

Правила:

- Не добавлять новые роли без явного задания.
- Не расширять существующие scopes.
- `debug:read_raw` разрешён только для admin.
- UI должен опираться на эти пресеты, а не вычислять scopes динамически.
- Любое изменение этой структуры требует миграции и обновления документации.

---

# 8. Логирование

Формат: JSON в stdout.

Не логировать:
- API ключи
- Authorization header
- AWS секреты

---

# 9. Как агент должен вносить изменения

Перед изменениями:
1. Краткий план (3–7 шагов)
2. Список файлов, которые будут изменены

Во время изменений:
- Делать минимальный diff
- Не рефакторить «по пути»
- Не менять контракты без задания
- Добавлять тесты, если меняется поведение

Запрещено без явного указания:
- Менять формат API
- Удалять поля из артефактов
- Отключать запись артефактов при ошибках
- Хранить inputs по умолчанию

---

# 10. Smoke проверки

Проверить vLLM:

GET $VLLM_BASE_URL/models  
Authorization: Bearer $VLLM_API_KEY  

Проверить Orchestrator:

GET /v1/health  
