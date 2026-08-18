from fastapi import FastAPI

from support_agent.api.router import router

app = FastAPI(
    title="Support Agent API",
    version="0.1.0",
)

app.include_router(router)