"""Part 7 (given): the FastAPI surface the Topic 3 project asks you to deploy.

    uvicorn seesense_lab.api:app --reload
    curl -F "file=@some_image.jpg" http://127.0.0.1:8000/predict

The image arrives as multipart/form-data. Only `to_db_record(result)` would ever be
persisted (the project writes it to Supabase); this lab just logs it.
"""

import logging
import os
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile

from .predict import Predictor, to_db_record

log = logging.getLogger("seesense")
app = FastAPI(title="See-Sense (lab)")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    path = os.environ.get("SEESENSE_CHECKPOINT", "checkpoints/model.pt")
    if not os.path.exists(path):
        raise HTTPException(503, f"No checkpoint at {path}. Run `python -m seesense_lab.train` first.")
    return Predictor.from_checkpoint(path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image larger than 5 MB")
    try:
        result = get_predictor().predict(data)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    log.info("would store: %s", to_db_record(result))
    return result
