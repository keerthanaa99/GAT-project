"""
GAT-XIDS FastAPI Application Entry Point
========================================
Exposes RESTful endpoints for health verification, model introspection,
and flow-similarity graph intrusion detection analysis.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    AnalyzeCSVResponse,
    ErrorResponse
)
from backend.app.services.model_service import model_service

logger = logging.getLogger("gat_xids.backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and caches model and scaler upon startup."""
    logger.info("Initializing GAT-XIDS Backend Service...")
    try:
        model_service.initialize()
        logger.info("GAT-XIDS Backend Service initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize model service on startup: {e}")
    yield
    logger.info("Shutting down GAT-XIDS Backend Service.")


app = FastAPI(
    title="GAT-XIDS API",
    description="Explainable Graph Attention Network for Medical IoT Intrusion Detection API",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for local development frontend
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check():
    """
    Returns application status, model checkpoint readiness, and dependency verification.
    """
    return model_service.get_health_status()


@app.get("/api/model-info", response_model=ModelInfoResponse, tags=["Introspection"])
def get_model_info():
    """
    Returns actual information from the saved checkpoint, feature schema, and label mappings.
    """
    try:
        return model_service.get_model_info()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model info: {str(e)}"
        )


@app.post(
    "/api/analyze-csv",
    response_model=AnalyzeCSVResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file or empty CSV"},
        413: {"model": ErrorResponse, "description": "File too large"},
        422: {"model": ErrorResponse, "description": "Schema mismatch / missing features"}
    },
    tags=["Analysis"]
)
async def analyze_csv(
    file: UploadFile = File(..., description="CICIoMT2024 network flow CSV file"),
    max_records: int = Query(2000, ge=1, le=10000, description="Max flow records to analyze")
):
    """
    Accepts a network flow CSV, cleans features, constructs a flow-similarity graph,
    and runs GAT classification without label contamination.
    """
    return await model_service.analyze_csv(file=file, max_records=max_records)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )
