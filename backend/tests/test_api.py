"""
API-level tests using FastAPI's TestClient (no separate server process
needed). Run from the backend/ directory with: pytest tests/ -v

These exercise the actual FastAPI app object, including a genuine
upload -> process -> classify -> store -> retrieve round trip for Demo
Mode, and well-formed responses for the model-info endpoint regardless of
whether a trained model happens to be present on this machine.
"""
import math
from pathlib import Path

import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

from app.pipeline.features import FEATURE_NAMES

VALID_CLASSES = {"crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"}
NEU_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "NEU-CLS"


def _make_test_image_bytes():
    img = np.full((150, 200, 3), 180, dtype=np.uint8)
    rng = np.random.default_rng(1)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    cv2.line(img, (20, 20), (150, 120), (60, 60, 60), 2)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="module")
def client():
    # Storage/DB isolation is configured in tests/conftest.py.
    from app.main import app
    return TestClient(app)


def test_tests_use_isolated_storage():
    """Regression guard: tests must never write to the real backend/storage."""
    from app import config
    assert "defectscope_tests_" in str(config.STORAGE_DIR)
    assert "defectscope_tests_" in config.DATABASE_URL


class TestHealth:
    def test_health_endpoint(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ok"
        assert "trained_model_available" in body


class TestModelInfo:
    def test_model_info_reports_availability(self, client):
        res = client.get("/api/model/info")
        assert res.status_code == 200
        body = res.json()
        assert "model_available" in body
        if not body["model_available"]:
            assert body["message"]


class TestInspectDemoMode:
    def test_oversized_upload_is_rejected_with_bounded_read(self, client, monkeypatch):
        from app.api import upload_utils
        monkeypatch.setattr(upload_utils, "MAX_UPLOAD_BYTES", 8)
        res = client.post(
            "/api/inspect",
            files={"file": ("large.png", b"x" * 9, "image/png")},
            data={"mode": "demo"},
        )
        assert res.status_code == 400
        assert "File too large" in res.json()["detail"]

    def test_processing_endpoint_applies_upload_limit(self, client, monkeypatch):
        from app.api import upload_utils
        monkeypatch.setattr(upload_utils, "MAX_UPLOAD_BYTES", 8)
        res = client.post(
            "/api/process/grayscale",
            files={"file": ("large.png", b"x" * 9, "image/png")},
        )
        assert res.status_code == 400
        assert "File too large" in res.json()["detail"]

    def test_invalid_file_type_rejected(self, client):
        res = client.post(
            "/api/inspect",
            files={"file": ("test.txt", b"not an image", "text/plain")},
            data={"mode": "demo"},
        )
        assert res.status_code == 400

    def test_corrupted_image_rejected(self, client):
        res = client.post(
            "/api/inspect",
            files={"file": ("bad.png", b"\x89PNG\r\n garbage", "image/png")},
            data={"mode": "demo"},
        )
        assert res.status_code == 400

    def test_invalid_mode_rejected(self, client):
        res = client.post(
            "/api/inspect",
            files={"file": ("t.png", _make_test_image_bytes(), "image/png")},
            data={"mode": "not_a_real_mode"},
        )
        assert res.status_code == 400

    def test_full_demo_inspection_round_trip(self, client):
        """Upload -> process -> classify -> store -> retrieve, end to end."""
        res = client.post(
            "/api/inspect",
            files={"file": ("scratch.png", _make_test_image_bytes(), "image/png")},
            data={"mode": "demo"},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["mode"] == "demo"
        assert body["classifier_used"] == "rule_based_demo"
        assert body["result"] in ("DEFECT_DETECTED", "NO_DEFECT")
        assert len(body["stages"]) == 11
        assert body["features"] is not None
        assert body["stages"][0]["image_url"].startswith("/inspection-assets/")
        image_res = client.get(body["stages"][0]["image_url"])
        assert image_res.status_code == 200
        assert image_res.headers["content-type"] == "image/png"
        assert image_res.content.startswith(b"\x89PNG\r\n\x1a\n")
        legacy_image_res = client.get(body["stages"][0]["image_url"].replace(
            "/inspection-assets/", "/results/", 1
        ))
        assert legacy_image_res.status_code == 200
        missing_image_res = client.get("/inspection-assets/missing-image.png")
        assert missing_image_res.status_code == 404
        assert missing_image_res.headers["content-type"].startswith("application/json")
        assert client.get("/api/health").status_code == 200
        inspection_id = body["id"]

        get_res = client.get(f"/api/inspections/{inspection_id}")
        assert get_res.status_code == 200
        assert get_res.json()["id"] == inspection_id

    def test_list_inspections(self, client):
        res = client.get("/api/inspections")
        assert res.status_code == 200
        rows = res.json()
        assert isinstance(rows, list)
        assert rows, "the demo round-trip test above should have stored a record"
        for key in ("timestamp", "original_filename", "predicted_class", "model_mode",
                    "processing_time_ms", "defect_area_percent", "num_regions", "model_version"):
            assert key in rows[0], f"history row missing '{key}'"

    def test_timestamps_are_explicit_utc(self, client):
        """A naive ISO string would be parsed as local time by browsers."""
        row = client.get("/api/inspections").json()[0]
        assert row["timestamp"].endswith(("Z", "+00:00")), row["timestamp"]

    def test_get_nonexistent_inspection_404(self, client):
        res = client.get("/api/inspections/999999")
        assert res.status_code == 404


class TestDashboard:
    def test_dashboard_reflects_stored_inspections(self, client):
        res = client.get("/api/dashboard")
        assert res.status_code == 200
        body = res.json()
        assert body["has_data"] is True
        assert body["total_inspections"] >= 1
        assert body["demo_inspections"] >= 1
        # demo and real statistics are reported separately, never pooled
        assert "demo_class_distribution" in body and "real_class_distribution" in body
        assert "defect_rate_percent" not in body
        assert not any("accuracy" in k or "f1" in k for k in body)


def _first_neu_image(class_name):
    files = sorted((NEU_DIR / class_name).glob("*.jpg")) if NEU_DIR.exists() else []
    return files[0] if files else None


class TestInspectRealMode:
    """End to end on an actual NEU-CLS image: upload -> FastAPI -> DIP pipeline ->
    feature extraction -> scaler -> RandomForest -> API response."""

    def test_real_image_end_to_end(self, client):
        from app.pipeline.classifier import is_trained_model_available
        img_path = _first_neu_image("scratches")
        if img_path is None or not is_trained_model_available():
            pytest.skip("NEU dataset or trained model not present")

        res = client.post(
            "/api/inspect",
            files={"file": (img_path.name, img_path.read_bytes(), "image/jpeg")},
            data={"mode": "real"},
        )
        assert res.status_code == 200, res.text
        body = res.json()

        assert body["mode"] == "real"
        assert body["model_mode"] == "TRAINED MODEL"
        assert body["classifier_used"] == "random_forest_trained"
        assert body["predicted_class"] in VALID_CLASSES
        assert 0.0 <= body["model_probability"] <= 1.0
        assert body["heuristic_score"] is None            # demo-only field never set in real mode
        assert abs(sum(body["per_class_probability"].values()) - 1.0) < 1e-2
        assert body["processing_time_ms"] > 0
        assert body["model_version"]
        assert len(body["stages"]) == 11
        assert all(s["image_url"] for s in body["stages"])  # every stage has a real image
        assert "not be interpreted as verified defect boundaries" in body["candidate_regions_disclaimer"]

        # feature vector: exactly the 23 contract features, all finite
        assert set(body["features"]) == set(FEATURE_NAMES)
        assert len(body["features"]) == 23
        assert all(math.isfinite(v) for v in body["features"].values())

    def test_uniform_image_empty_segmentation_is_handled(self, client):
        from app.pipeline.classifier import is_trained_model_available
        if not is_trained_model_available():
            pytest.skip("trained model not present")
        ok, buf = cv2.imencode(".png", np.full((200, 200, 3), 150, dtype=np.uint8))
        res = client.post("/api/inspect", files={"file": ("flat.png", buf.tobytes(), "image/png")},
                          data={"mode": "real"})
        assert res.status_code == 200
        body = res.json()
        assert body["num_regions"] == 0
        assert all(math.isfinite(v) for v in body["features"].values())


class TestMissingModel:
    def test_real_mode_without_model_returns_503_not_a_fake_result(self, client, tmp_path, monkeypatch):
        from app.pipeline import classifier as clf_mod
        monkeypatch.setattr(clf_mod, "TRAINED_MODEL_PATH", tmp_path / "nope.joblib")
        monkeypatch.setattr(clf_mod, "_model_cache", None)

        res = client.post("/api/inspect",
                          files={"file": ("t.png", _make_test_image_bytes(), "image/png")},
                          data={"mode": "real"})
        assert res.status_code == 503
        assert "train_classifier" in res.json()["detail"]

        info = client.get("/api/model/info").json()
        assert info["model_available"] is False
        assert info["evaluation"] is None

    def test_demo_mode_still_works_without_model(self, client, tmp_path, monkeypatch):
        from app.pipeline import classifier as clf_mod
        monkeypatch.setattr(clf_mod, "TRAINED_MODEL_PATH", tmp_path / "nope.joblib")
        monkeypatch.setattr(clf_mod, "_model_cache", None)
        res = client.post("/api/inspect",
                          files={"file": ("t.png", _make_test_image_bytes(), "image/png")},
                          data={"mode": "demo"})
        assert res.status_code == 200
        assert res.json()["classifier_used"] == "rule_based_demo"
