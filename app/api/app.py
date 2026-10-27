from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config.constants import PROJECT_ROOT, VISUALS_DIR, METRICS_PATH
from app.api.schemas import LoanApplicationRequest, PredictionResponse, HealthResponse
from app.api.services import ModelService, model_service
from utils.helpers import load_json
from utils.logger import get_logger

logger = get_logger(__name__)

# Directory paths
API_DIR = Path(__file__).resolve().parent
APP_DIR = API_DIR.parent
FRONTEND_DIR = APP_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
TEMPLATES_DIR = FRONTEND_DIR / "templates"
INDEX_HTML = TEMPLATES_DIR / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to initialize models and services on startup."""
    logger.info("Starting up CrediLens AI API service...")
    try:
        model_service.load_model()
        if model_service.is_ready():
            logger.info("Champion calibrated model and threshold loaded successfully.")
        else:
            logger.warning("Model artifacts could not be loaded at startup.")
    except Exception as e:
        logger.error(f"Error initializing model service during startup: {e}")
    yield
    logger.info("Shutting down CrediLens AI API service...")


app = FastAPI(
    title="CrediLens AI — Credit Risk Assessment API",
    description="Underwriting credit risk scoring and explainable ML API using calibrated XGBoost.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files (CSS, JS)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    logger.info(f"Mounted static files from: {STATIC_DIR.resolve()}")
else:
    logger.warning(f"Static directory not found at: {STATIC_DIR}")


@app.get("/", tags=["Frontend"])
async def serve_index():
    """Serve the primary Credit Ledger underwriting dashboard."""
    if not INDEX_HTML.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dashboard template (index.html) not found."
        )
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint to verify API and model readiness."""
    is_ready = model_service.is_ready()
    threshold = model_service.pipeline.threshold if is_ready else None

    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        service="CrediLens AI Credit Risk Service",
        model_loaded=is_ready,
        model_path=str(model_service.model_path),
        threshold=threshold,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_loan_risk(application: LoanApplicationRequest):
    """
    Score loan applicant credit risk.

    Returns:
    - default_probability: Calibrated default probability
    - default_prediction: 0 (Low Risk / Approved) or 1 (High Risk / Default Likely)
    - threshold: Operating decision threshold
    - Result: UI-compatible string ("Low Risk" or "High Risk")
    - risk_category: Descriptive category
    - decision: Actionable underwriting decision
    """
    if not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready. Verify that artifacts/models/credit_risk_model.pkl exists."
        )

    try:
        response = model_service.predict(application)
        return response
    except Exception as e:
        logger.error(f"Inference request failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.get("/metrics", tags=["Model Analytics"])
async def get_metrics():
    """Retrieve test benchmark metrics and cross-validation performance."""
    metrics_path = Path(METRICS_PATH)
    if not metrics_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metrics file not found. Train the pipeline first."
        )
    return load_json(metrics_path)


@app.get("/visuals/{filename}", tags=["Visualizations"])
async def get_visual_artifact(filename: str):
    """Retrieve saved evaluation and explainability visual diagrams."""
    file_path = Path(VISUALS_DIR) / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Visual artifact '{filename}' not found."
        )
    return FileResponse(file_path)
