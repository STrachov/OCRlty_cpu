import os

import pytest

from app.auth_store import ApiPrincipal
from app.services.jobs_store import JobsStore


def _principal() -> ApiPrincipal:
    return ApiPrincipal(api_key_id=1, key_id="k1", role="user", scopes={"extract:run"})


def test_jobs_store_requires_postgres_url():
    with pytest.raises(RuntimeError):
        JobsStore("sqlite:///tmp/jobs.db")


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL is not set")
def test_job_lifecycle_success():
    store = JobsStore(TEST_DATABASE_URL or "")
    job = store.create_job(kind="extract", request={"task_id": "receipt_fields_v1"}, principal=_principal())

    assert job.status == "queued"

    store.mark_running(job.job_id)
    j2 = store.get_job(job.job_id)
    assert j2 is not None
    assert j2.status == "running"

    store.mark_succeeded(job.job_id, result_ref="jobs/2026-02-25/x.json", result_meta={"type": "extract"})
    j3 = store.get_job(job.job_id)
    assert j3 is not None
    assert j3.status == "succeeded"
    assert j3.result_ref == "jobs/2026-02-25/x.json"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL is not set")
def test_request_cancel_and_mark_canceled():
    store = JobsStore(TEST_DATABASE_URL or "")
    job = store.create_job(kind="extract", request={}, principal=_principal())

    store.request_cancel(job.job_id)
    store.mark_canceled(job.job_id)

    j2 = store.get_job(job.job_id)
    assert j2 is not None
    assert j2.status == "canceled"
    assert j2.cancel_requested is True


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL is not set")
def test_mark_stale_running_as_failed():
    store = JobsStore(TEST_DATABASE_URL or "")
    job = store.create_job(kind="extract", request={}, principal=_principal())
    store.mark_running(job.job_id)

    affected = store.mark_stale_running_as_failed()
    assert affected >= 1

    j2 = store.get_job(job.job_id)
    assert j2 is not None
    assert j2.status in {"failed", "canceled"}
