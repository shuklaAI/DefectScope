from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import datetime


class StageResult(BaseModel):
    step: int
    key: str
    title: str
    technique: str
    explanation: str
    parameters: Dict[str, Any]
    image_url: Optional[str] = None
    histogram: Optional[Dict[str, List[float]]] = None


class InspectionResponse(BaseModel):
    id: int
    timestamp: datetime.datetime
    original_filename: str
    mode: str                              # "demo" | "real"
    model_mode: str                        # "DEMO / SYNTHETIC DATA" | "TRAINED MODEL"
    result: str                            # "DEFECT_DETECTED" | "NO_DEFECT" | "CLASSIFIED"
    predicted_class: Optional[str]
    num_regions: int
    defect_area_percent: Optional[float]
    candidate_regions_disclaimer: Optional[str]
    classifier_used: str                   # "rule_based_demo" | "random_forest_trained"
    model_probability: Optional[float]     # real predict_proba output (real mode only)
    heuristic_score: Optional[float]       # rule-based heuristic weight (demo mode only)
    per_class_probability: Optional[Dict[str, float]]
    model_version: Optional[str]
    processing_time_ms: float
    original_image_url: str
    highlighted_image_url: Optional[str]
    stages: List[StageResult]
    features: Optional[Dict[str, float]]

    model_config = {"from_attributes": True}


class InspectionSummary(BaseModel):
    id: int
    timestamp: datetime.datetime
    original_filename: str
    mode: str
    model_mode: str
    result: str
    predicted_class: Optional[str]
    model_probability: Optional[float]
    heuristic_score: Optional[float]
    num_regions: int
    defect_area_percent: Optional[float]   # candidate-region area (DIP segmentation), not verified
    model_version: Optional[str]
    processing_time_ms: float
    original_image_url: str

    model_config = {"from_attributes": True}


class DashboardStats(BaseModel):
    """Statistics computed ONLY from stored inspection records (both demo
    and real-mode runs). Never mixed with model evaluation metrics, which
    live in ModelInfo/EvaluationResults below."""
    total_inspections: int
    demo_inspections: int
    real_inspections: int
    # Kept per mode on purpose: Demo labels come from a rule-based heuristic on
    # synthetic/sample images, Real labels from the trained model. Pooling them
    # (or computing a pooled "defect rate") would mix incomparable things.
    demo_defect_detected: int
    demo_no_defect: int
    demo_class_distribution: Dict[str, int]
    real_class_distribution: Dict[str, int]
    avg_processing_time_ms: float
    recent_inspections: List[InspectionSummary]
    has_data: bool


class ModelMetadata(BaseModel):
    model_type: str
    model_version: str
    feature_names: List[str]
    class_names: List[str]
    dataset_name: str
    dataset_class_counts: Dict[str, int]
    total_samples: int
    train_samples: int
    test_samples: int
    random_seed: int
    training_date_utc: str


class EvaluationResults(BaseModel):
    """Real metrics computed on the held-out test split during training.
    Never computed from, or mixed with, inspection history."""
    evaluated_on: str
    test_set_size: int
    train_set_size: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class_report: Dict[str, Any]
    confusion_matrix: List[List[int]]
    class_order: List[str]
    class_distribution: Dict[str, int]
    evaluated_at_utc: str


class ModelInfoResponse(BaseModel):
    model_available: bool
    metadata: Optional[ModelMetadata] = None
    evaluation: Optional[EvaluationResults] = None
    message: Optional[str] = None
