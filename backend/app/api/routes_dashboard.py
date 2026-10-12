"""
Dashboard statistics are computed ONLY from stored inspection records
(activity counts, processing time, predicted-class distribution across
whatever has actually been inspected). Model accuracy/precision/recall/F1
NEVER come from here - those are training-time evaluation metrics served
separately by GET /api/model/info (see routes_model.py) and rendered in a
clearly separate "Model Evaluation" section on the frontend Dashboard page.
"""
from collections import Counter

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app import models, schemas
from app.database import get_db
from app.api.routes_inspect import record_to_summary

router = APIRouter(prefix="/api", tags=["dashboard"])


@router.get("/dashboard", response_model=schemas.DashboardStats)
def dashboard(db: Session = Depends(get_db)):
    records = db.query(models.Inspection).order_by(models.Inspection.timestamp.desc()).all()

    total = len(records)
    if total == 0:
        return schemas.DashboardStats(
            total_inspections=0, demo_inspections=0, real_inspections=0,
            demo_defect_detected=0, demo_no_defect=0,
            demo_class_distribution={}, real_class_distribution={},
            avg_processing_time_ms=0.0, recent_inspections=[], has_data=False,
        )

    demo_records = [r for r in records if r.mode == "demo"]
    real_records = [r for r in records if r.mode == "real"]

    def distribution(rows):
        return dict(Counter(r.predicted_class for r in rows if r.predicted_class))

    return schemas.DashboardStats(
        total_inspections=total,
        demo_inspections=len(demo_records),
        real_inspections=len(real_records),
        demo_defect_detected=sum(r.result == "DEFECT_DETECTED" for r in demo_records),
        demo_no_defect=sum(r.result == "NO_DEFECT" for r in demo_records),
        demo_class_distribution=distribution(demo_records),
        real_class_distribution=distribution(real_records),
        avg_processing_time_ms=round(sum(r.processing_time_ms for r in records) / total, 2),
        recent_inspections=[record_to_summary(r) for r in records[:10]],
        has_data=True,
    )
