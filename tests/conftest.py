import os

import pytest

# Ensure Settings() can be created during module imports in tests.
os.environ.setdefault("API_KEY_PEPPER", "test-pepper")
os.environ.setdefault("AUTH_ENABLED", "0")


@pytest.fixture
def principal():
    from app.auth_store import ApiPrincipal

    return ApiPrincipal(api_key_id=1, key_id="test-key", role="user", scopes={"extract:run"})
