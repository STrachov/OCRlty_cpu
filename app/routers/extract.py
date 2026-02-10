from fastapi import APIRouter
from app.handlers import batch_extract_upload, batch_extract_rerun, extract, batch_extract, ExtractResponse, BatchExtractResponse

router = APIRouter(prefix="/v1", tags=["extract"])
router.post("/extract", response_model=ExtractResponse)(extract)
router.post("/batch_extract", response_model=BatchExtractResponse)(batch_extract)
router.post("/batch_extract_upload", response_model=BatchExtractResponse)(batch_extract_upload)
router.post("/batch_extract_rerun", response_model=BatchExtractResponse)(batch_extract_rerun)

