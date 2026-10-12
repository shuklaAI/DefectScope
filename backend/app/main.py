from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import CORS_ORIGINS, RESULTS_DIR, UPLOAD_DIR
from app.database import Base, engine
from app import models  # noqa: F401 - ensures models are registered before create_all
from app.api import routes_inspect, routes_dashboard, routes_process, routes_model
from app.pipeline.classifier import is_trained_model_available

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI-Based Industrial Surface Defect Detection and Classification API",
    description="Digital Image Processing pipeline (OpenCV/NumPy/scikit-image/SciPy) with an "
                "optional trained scikit-learn RandomForest classifier evaluated on the real "
                "NEU-CLS steel surface defect dataset. See /api/model/info for real training "
                "metrics; nothing here is fabricated.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep the legacy results mount for image URLs already stored in inspection
# history. New result pages use /inspection-assets, which does not overlap the
# frontend's /results page route.
app.mount("/inspection-assets", StaticFiles(directory=str(RESULTS_DIR)), name="inspection-assets")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

app.include_router(routes_inspect.router)
app.include_router(routes_dashboard.router)
app.include_router(routes_process.router)
app.include_router(routes_model.router)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "defect-detection-backend",
        "trained_model_available": is_trained_model_available(),
    }
