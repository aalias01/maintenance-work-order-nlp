from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api.schemas import ClassifyRequest, ClassifyResponse, HealthResponse
import api.predictor as predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load_all()
    yield


app = FastAPI(
    title="Maintenance Work Order NLP API",
    description="Classifies maintenance work orders by failure category and retrieves similar past cases. "
                "Uses DistilBERT + LoRA fine-tune with sentence-transformer similarity search.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080", "http://127.0.0.1:8080",
        "https://workorders.alvinalias.com",         # canonical demo (Primary)
        "https://maintenance-work-order-nlp.vercel.app",  # legacy, 308-redirects to subdomain
        # Portfolio landing live-model playground (alvinalias.com).
        "https://alvinalias.com", "https://www.alvinalias.com",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", response_class=JSONResponse)
def root():
    return {
        "project": "Maintenance Work Order NLP",
        "endpoints": {"health": "GET /health", "classify": "POST /classify", "docs": "GET /docs"},
        "github": "https://github.com/aalias01/maintenance-work-order-nlp",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    s = predictor.status()
    return HealthResponse(
        status="ok" if predictor.is_ready() else "degraded",
        **s,
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    """Classify a maintenance work order text and return similar past cases."""
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded. Run notebooks first.")
    try:
        return predictor.classify(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
