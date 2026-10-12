import datetime

from sqlalchemy import Column, Integer, String, Float, DateTime, Text

from app.database import Base


class Inspection(Base):
    """
    One row per uploaded image run through the pipeline. `mode` and
    `classifier_used` together determine how every other numeric field
    should be interpreted (see app/pipeline/classifier.py docstring) - the
    API and frontend never blend demo-mode heuristic scores with real-mode
    model probabilities.
    """
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), index=True)

    original_filename = Column(String, nullable=False)
    stored_filename = Column(String, nullable=False)

    mode = Column(String, nullable=False, default="demo")           # "demo" | "real"
    model_mode = Column(String, nullable=False, default="DEMO / SYNTHETIC DATA")

    result = Column(String, nullable=False)          # "DEFECT_DETECTED" | "NO_DEFECT" | "CLASSIFIED"
    predicted_class = Column(String, nullable=True)
    num_regions = Column(Integer, default=0)
    defect_area_percent = Column(Float, nullable=True)

    classifier_used = Column(String, default="rule_based_demo")  # or "random_forest_trained"
    model_probability = Column(Float, nullable=True)  # real predict_proba (real mode only)
    heuristic_score = Column(Float, nullable=True)     # rule-based weight (demo mode only)
    model_version = Column(String, nullable=True)

    processing_time_ms = Column(Float, nullable=False)

    original_image_path = Column(String, nullable=False)
    highlighted_image_path = Column(String, nullable=True)

    stages_json = Column(Text, nullable=False)
    features_json = Column(Text, nullable=True)
    per_class_probability_json = Column(Text, nullable=True)
