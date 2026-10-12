from fastapi import HTTPException, UploadFile

from app.config import MAX_UPLOAD_MB

MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)


async def read_upload_limited(file: UploadFile) -> bytes:
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"File too large. Limit is {MAX_UPLOAD_MB} MB.")
    return contents
