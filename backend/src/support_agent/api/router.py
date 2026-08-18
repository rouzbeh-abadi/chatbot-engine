from fastapi import APIRouter

from support_agent.api.routes.chat import router as chat_router
from support_agent.api.routes.health import router as health_router

router = APIRouter()

router.include_router(health_router)
router.include_router(chat_router)