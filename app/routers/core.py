from fastapi import APIRouter
from app.handlers import me

router = APIRouter(prefix="/v1", tags=["core"])
router.get("/me")(me)
