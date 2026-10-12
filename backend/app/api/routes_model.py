from fastapi import APIRouter

from app import schemas
from app.pipeline.classifier import is_trained_model_available, get_model_metadata, get_evaluation_results

router = APIRouter(prefix="/api", tags=["model"])


@router.get("/model/info", response_model=schemas.ModelInfoResponse)
def model_info():
    if not is_trained_model_available():
        return schemas.ModelInfoResponse(
            model_available=False,
            message="No trained model found. Run `python ml/train_classifier.py` from the "
                    "backend directory against backend/data/raw/NEU-CLS/ to train one.",
        )
    metadata = get_model_metadata()
    evaluation = get_evaluation_results()
    return schemas.ModelInfoResponse(
        model_available=True,
        metadata=metadata,
        evaluation=evaluation,
    )
