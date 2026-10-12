"""
Standalone single-operation endpoints. These are not required for the main
inspection flow (which runs the full pipeline in one call via /api/inspect)
but are exposed so individual DIP operations can be exercised/tested in
isolation, e.g. from the API docs page (/docs) during a viva.
"""
import cv2
from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from fastapi.responses import Response

from app.pipeline import preprocessing, filtering, segmentation, morphology
from app.api.upload_utils import read_upload_limited

router = APIRouter(prefix="/api/process", tags=["processing"])


def _decode(contents: bytes):
    try:
        return preprocessing.load_image_from_bytes(contents)
    except ValueError as e:
        raise HTTPException(400, str(e))


def _encode_png(img) -> Response:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(500, "Failed to encode result image")
    return Response(content=buf.tobytes(), media_type="image/png")


@router.post("/grayscale")
async def process_grayscale(file: UploadFile = File(...)):
    img = _decode(await read_upload_limited(file))
    gray = preprocessing.to_grayscale(preprocessing.resize_and_normalize(img))
    return _encode_png(gray)


@router.post("/filter")
async def process_filter(file: UploadFile = File(...), method: str = Query("median", enum=["median", "gaussian", "wiener"])):
    img = _decode(await read_upload_limited(file))
    gray = preprocessing.to_grayscale(preprocessing.resize_and_normalize(img))
    denoised = filtering.denoise(gray, method=method)
    return _encode_png(denoised)


@router.post("/segment")
async def process_segment(file: UploadFile = File(...), method: str = Query("otsu", enum=["otsu", "adaptive"])):
    img = _decode(await read_upload_limited(file))
    gray = preprocessing.to_grayscale(preprocessing.resize_and_normalize(img))
    enhanced = filtering.enhance_contrast(filtering.denoise(gray))
    if method == "otsu":
        mask, _ = segmentation.otsu_threshold(enhanced)
    else:
        mask = segmentation.adaptive_threshold(enhanced)
    return _encode_png(mask)


@router.post("/morphology")
async def process_morphology(file: UploadFile = File(...), operation: str = Query("open_close", enum=["open_close", "skeleton"])):
    img = _decode(await read_upload_limited(file))
    gray = preprocessing.to_grayscale(preprocessing.resize_and_normalize(img))
    enhanced = filtering.enhance_contrast(filtering.denoise(gray))
    mask, _ = segmentation.otsu_threshold(enhanced)
    if operation == "open_close":
        result = morphology.clean_mask(mask)
    else:
        result = morphology.skeletonize(morphology.clean_mask(mask))
    return _encode_png(result)
