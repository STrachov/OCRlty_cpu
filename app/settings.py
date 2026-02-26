from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices, field_validator

class Settings(BaseSettings):
    # --- Primary DB (PostgreSQL) ---
    # Single source of truth for relational data (jobs/auth/artifact_index).
    DATABASE_URL: str = "postgresql+psycopg://postgres:postgres@localhost:5432/ocrlty"

    # --- Auth ---
    AUTH_ENABLED: bool = True
    API_KEY_PEPPER: str

    # --- Debug / local dev ---
    # If False, all debug features/endpoints are disabled, even if a key has debug scopes.
    DEBUG_MODE: bool = False

    # Switch inference backend without changing code.
    # - "vllm": call the vLLM OpenAI-compatible API
    # - "mock": return deterministic fake responses (for local testing without GPU/model)
    INFERENCE_BACKEND: str = "vllm"

    # Hard caps (apply to both vLLM and mock backends)
    MAX_PROMPT_CHARS: int = 20000
    MAX_TOKENS_CAP: int = 256

    # --- App ---
    ARTIFACTS_DIR: str = "/data/artifacts"

    VLLM_MODEL: str = "Qwen/Qwen3-VL-8B-Instruct"
    VLLM_BASE_URL: str = "http://127.0.0.1:8000"
    VLLM_API_KEY: str = ""
    
    VLLM_TIMEOUT_S: int = 120

    VLLM_CONNECT_TIMEOUT_S: float = 10.0
    VLLM_READ_TIMEOUT_S: float = 180.0
    VLLM_WRITE_TIMEOUT_S: float = 30.0
    VLLM_POOL_TIMEOUT_S: float = 10.0

    VLLM_RETRY_MAX_ATTEMPTS: int = 3
    VLLM_RETRY_BASE_DELAY_S: float = 0.25
    VLLM_RETRY_MAX_DELAY_S: float = 2.0
    VLLM_RETRY_JITTER_S: float = 0.25

    VLLM_HTTP_MAX_CONNECTIONS: int = 100
    VLLM_HTTP_MAX_KEEPALIVE: int = 20
    VLLM_HTTP_KEEPALIVE_EXPIRY_S: float = 30.0
    # --- Logging ---
    # Defaults are suitable for Docker (stdout) + log collectors (Loki/ELK).
    LOG_LEVEL: str = "INFO"          # DEBUG | INFO | WARNING | ERROR
    LOG_FORMAT: str = "json"         # json | text

    # --- Jobs ---
    JOBS_BACKEND: str = "local"          # local | celery
    JOBS_MAX_CONCURRENCY: int = 2

    # Progress reporting for long-running jobs (batch_* async).
    # - JOBS_PROGRESS_EVERY_N=1 is handy for debugging (update after every item).
    # - In production set higher values (e.g. 10/25) to reduce DB write frequency.
    JOBS_PROGRESS_EVERY_N: int = 1
    JOBS_PROGRESS_MIN_SECONDS: float = 0.0

    # --- Celery (only if JOBS_BACKEND=celery) ---
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str | None = None



    # --- Object storage (S3 / Cloudflare R2) ---
    # Set S3_BUCKET to enable storing artifacts in S3-compatible storage.
    # For Cloudflare R2, also set:
    #   S3_ENDPOINT_URL=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
    #   S3_REGION=auto
    #
    # Credentials: boto3 will read AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY from env.
    S3_BUCKET: str = ""
    S3_ENDPOINT_URL: str | None = None
    S3_REGION: str = "auto"
    # Support both S3_PREFIX and legacy S3_ARTIFACTS_PREFIX env var names.
    S3_PREFIX: str = Field(default="prod", validation_alias=AliasChoices("S3_PREFIX", "S3_ARTIFACTS_PREFIX"))
    S3_ALLOW_OVERWRITE: bool = False
    S3_PRESIGN_TTL_S: int = 3600
    S3_FORCE_PATH_STYLE: bool = False

    @field_validator("S3_BUCKET", mode="before")
    @classmethod
    def _strip_s3_bucket(cls, v):
        return str(v or "").strip()

    @field_validator("S3_ENDPOINT_URL", mode="before")
    @classmethod
    def _strip_s3_endpoint(cls, v):
        s = str(v or "").strip()
        return s or None

    @field_validator("S3_REGION", mode="before")
    @classmethod
    def _strip_s3_region(cls, v):
        return str(v or "auto").strip()

    @field_validator("S3_PREFIX", mode="before")
    @classmethod
    def _strip_s3_prefix(cls, v):
        return str(v or "ocrlty").strip().strip("/")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
