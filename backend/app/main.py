from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router, load_persisted_results


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_persisted_results()
    yield


app = FastAPI(
    title="Classroom Voice Analytics API",
    description="Offline-first pipeline: VAD-chunked parallel transcription, "
                "speaker diarization, and classroom engagement metrics.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {
        "service": "Classroom Voice Analytics API",
        "version": "0.2.0",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
