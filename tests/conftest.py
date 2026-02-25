import os
from pathlib import Path

import pytest

# Ensure Settings() can be created during module imports in tests.
os.environ.setdefault("API_KEY_PEPPER", "test-pepper")
os.environ.setdefault("AUTH_ENABLED", "0")


@pytest.fixture
def tmp_artifacts_dir(tmp_path, monkeypatch):
    d = tmp_path / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ARTIFACTS_DIR", str(d))
    return d


@pytest.fixture
def principal():
    from app.auth_store import ApiPrincipal

    return ApiPrincipal(api_key_id=1, key_id="test-key", role="user", scopes={"extract:run"})


@pytest.fixture
def principal_debug():
    from app.auth_store import ApiPrincipal

    return ApiPrincipal(
        api_key_id=2,
        key_id="debug-key",
        role="debugger",
        scopes={"extract:run", "debug:run", "debug:read_raw"},
    )


@pytest.fixture
def tmp_jobs_db_path(tmp_path) -> str:
    p = tmp_path / "jobs.db"
    return str(p)
