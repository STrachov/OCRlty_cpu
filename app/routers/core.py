from fastapi import APIRouter
from app.handlers import me, health

router = APIRouter(prefix="/v1", tags=["core"])
router.get("/me")(me)
router.get("/health")(health)
