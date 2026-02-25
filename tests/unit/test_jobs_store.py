from app.auth_store import ApiPrincipal
from app.services.jobs_store import JobsStore


def _principal() -> ApiPrincipal:
    return ApiPrincipal(api_key_id=1, key_id="k1", role="user", scopes={"extract:run"})


def test_job_lifecycle_success(tmp_jobs_db_path):
    store = JobsStore(tmp_jobs_db_path)
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


def test_request_cancel_and_mark_canceled(tmp_jobs_db_path):
    store = JobsStore(tmp_jobs_db_path)
    job = store.create_job(kind="extract", request={}, principal=_principal())

    store.request_cancel(job.job_id)
    store.mark_canceled(job.job_id)

    j2 = store.get_job(job.job_id)
    assert j2 is not None
    assert j2.status == "canceled"
    assert j2.cancel_requested is True


def test_mark_stale_running_as_failed(tmp_jobs_db_path):
    store = JobsStore(tmp_jobs_db_path)
    job = store.create_job(kind="extract", request={}, principal=_principal())
    store.mark_running(job.job_id)

    affected = store.mark_stale_running_as_failed()
    assert affected >= 1

    j2 = store.get_job(job.job_id)
    assert j2 is not None
    assert j2.status in {"failed", "canceled"}
