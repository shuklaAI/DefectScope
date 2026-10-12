"""
Unit tests for the individual DIP pipeline building blocks. Run from the
backend/ directory with:  pytest tests/ -v
"""
import numpy as np
import cv2
import pytest

from app.pipeline import preprocessing, filtering, edge_detection, segmentation, morphology, features


def _synthetic_bgr(w=200, h=200, base=180, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), base, dtype=np.uint8)
    noise = rng.normal(0, 5, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _encode(img):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


class TestPreprocessing:
    def test_load_valid_image(self):
        img = _synthetic_bgr()
        decoded = preprocessing.load_image_from_bytes(_encode(img))
        assert decoded.shape == img.shape

    def test_load_invalid_image_raises(self):
        with pytest.raises(ValueError):
            preprocessing.load_image_from_bytes(b"not an image")

    def test_resize_and_normalize_shape(self):
        img = _synthetic_bgr(100, 80)
        out = preprocessing.resize_and_normalize(img, (256, 256))
        assert out.shape == (256, 256, 3)
        assert out.dtype == np.uint8

    def test_grayscale_conversion(self):
        img = _synthetic_bgr()
        gray = preprocessing.to_grayscale(img)
        assert gray.ndim == 2
        assert gray.shape == img.shape[:2]


class TestFiltering:
    def test_median_denoise_reduces_impulse_noise(self):
        img = np.full((100, 100), 150, dtype=np.uint8)
        img[10, 10] = 255  # impulse noise pixel
        img[20, 20] = 0
        denoised = filtering.denoise(img, method="median")
        assert denoised[10, 10] != 255
        assert denoised[20, 20] != 0

    def test_clahe_changes_the_image(self):
        img = _synthetic_bgr()
        gray = preprocessing.to_grayscale(img)
        enhanced = filtering.enhance_contrast(gray, method="clahe")
        assert enhanced.shape == gray.shape
        assert not np.array_equal(enhanced, gray)

    def test_histogram_sums_to_pixel_count(self):
        gray = np.random.default_rng(0).integers(0, 256, (50, 50), dtype=np.uint8)
        hist = filtering.compute_histogram(gray)
        assert sum(hist["counts"]) == 50 * 50


class TestEdgeDetection:
    def test_canny_on_blank_image_has_no_edges(self):
        blank = np.full((100, 100), 128, dtype=np.uint8)
        edges = edge_detection.canny_edges(blank)
        assert np.count_nonzero(edges) == 0

    def test_canny_detects_a_hard_edge(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 255
        edges = edge_detection.canny_edges(img)
        assert np.count_nonzero(edges) > 0


class TestSegmentation:
    def test_otsu_threshold_returns_binary_mask(self):
        img = np.full((100, 100), 60, dtype=np.uint8)
        img[:, 50:] = 200
        mask, thresh_val = segmentation.otsu_threshold(img)
        assert set(np.unique(mask)).issubset({0, 255})
        assert 0 < thresh_val < 255


class TestMorphology:
    def test_opening_removes_small_noise_speck(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 255  # single-pixel speck
        cleaned = morphology.clean_mask(mask, kernel_size=3)
        assert np.count_nonzero(cleaned) == 0

    def test_connected_components_filters_by_area(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 20, 255, -1)  # big component, area ~1256
        mask[5, 5] = 255  # tiny 1px noise speck
        _, components = morphology.connected_components(mask, min_area=25)
        assert len(components) == 1
        assert components[0]["area"] > 1000

    def test_max_area_fraction_discards_global_blob(self):
        mask = np.full((100, 100), 255, dtype=np.uint8)  # entire frame flagged
        _, components = morphology.connected_components(mask, min_area=25, max_area_fraction=0.4)
        assert len(components) == 0


class TestFeatures:
    def test_feature_vector_length_matches_names(self):
        gray = np.random.default_rng(0).integers(0, 256, (200, 200), dtype=np.uint8).astype(np.uint8)
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 15, 255, -1)
        _, components = morphology.connected_components(mask, min_area=25)
        feats = features.extract_features(gray, mask, components)
        assert set(feats.keys()) == set(features.FEATURE_NAMES)
        vec = features.feature_vector(feats)
        assert vec.shape == (1, len(features.FEATURE_NAMES))

    def test_no_components_returns_zeroed_regional_features(self):
        gray = np.full((100, 100), 128, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        feats = features.extract_features(gray, mask, [])
        assert feats["num_components"] == 0
        assert feats["largest_area"] == 0.0

    def test_uniform_image_produces_no_nan_or_inf(self):
        """A completely uniform (zero-variance) image is an edge case that
        can otherwise drive GLCM correlation or circularity to NaN/Inf."""
        gray = np.full((200, 200), 150, dtype=np.uint8)
        mask = np.zeros((200, 200), dtype=np.uint8)
        feats = features.extract_features(gray, mask, [])
        vec = features.feature_vector(feats)
        assert np.all(np.isfinite(vec))
        assert not np.any(np.isnan(vec))

    def test_degenerate_component_produces_no_nan_or_inf(self):
        """A single-pixel component has zero perimeter in some edge cases -
        must not propagate NaN/Inf through circularity/solidity/eccentricity."""
        gray = np.random.default_rng(0).integers(0, 256, (100, 100)).astype(np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 255
        from app.pipeline import morphology
        _, components = morphology.connected_components(mask, min_area=0)
        feats = features.extract_features(gray, mask, components)
        vec = features.feature_vector(feats)
        assert np.all(np.isfinite(vec))

    def test_feature_names_match_trained_model_contract(self):
        """Guards against silent train/inference skew: if a trained model
        exists, its saved feature contract must match the current
        feature-extraction code exactly (name and order)."""
        from app.pipeline.classifier import is_trained_model_available, get_model_metadata
        if not is_trained_model_available():
            pytest.skip("No trained model present in this environment")
        metadata = get_model_metadata()
        assert metadata["feature_names"] == features.FEATURE_NAMES


class TestClassifierFeatureContract:
    def test_mismatched_feature_contract_raises(self, tmp_path, monkeypatch):
        """If a saved model's feature_names no longer match the current
        FEATURE_NAMES, loading it must raise a clear error rather than
        silently predicting on a misaligned vector."""
        import json
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from app.pipeline import classifier as clf_mod

        model_path = tmp_path / "defect_classifier.joblib"
        scaler_path = tmp_path / "scaler.joblib"
        metadata_path = tmp_path / "metadata.json"

        joblib.dump(RandomForestClassifier(), model_path)
        joblib.dump(StandardScaler(), scaler_path)
        metadata_path.write_text(json.dumps({
            "feature_names": ["totally_different_feature"],
            "label_map": {"0": "crazing"},
            "model_version": "0.0.0",
        }))

        monkeypatch.setattr(clf_mod, "TRAINED_MODEL_PATH", model_path)
        monkeypatch.setattr(clf_mod, "SCALER_PATH", scaler_path)
        monkeypatch.setattr(clf_mod, "METADATA_PATH", metadata_path)
        monkeypatch.setattr(clf_mod, "_model_cache", None)

        with pytest.raises(clf_mod.FeatureContractError):
            clf_mod.load_model_bundle()
