# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# (Опционально) нужны для Pillow / curl для healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl \
      libjpeg62-turbo \
      zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Если у тебя есть requirements.txt — это лучший вариант
# (если нет — см. ниже комментарий "requirements.txt")
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Код приложения
COPY app /app/app

# Entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080

# Простая проверка живости
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT:-8080}/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
