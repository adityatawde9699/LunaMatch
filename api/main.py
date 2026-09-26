"""LunaMatch FastAPI application."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.registration import router

app = FastAPI(title="LunaMatch API", version="0.1.0")
origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
origins.extend(origin.strip().rstrip("/") for origin in
               os.environ.get("LUNAMATCH_CORS_ORIGINS", "").split(",") if origin.strip())
app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])
app.include_router(router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Simple process health endpoint."""
    return {"status": "ok"}
