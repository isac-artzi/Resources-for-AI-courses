"""FastAPI model service -- Cloud #2 (deployed on Render.com).

Responsibilities:
  * expose the char Transformer behind HTTP so the Streamlit UI can call it over
    HTTPS,
  * read dataset parameters from Supabase before training,
  * write run rows (pretrain + finetune), model artifacts, and sequence metadata
    (hash only) back.

There is NO UI code here and NO business logic in the UI -- separation of
concerns across the three clouds. This is "Fine-It": transfer learning with a
causal character Transformer -- pretrain a next-char LM, fine-tune a classifier
(vs a from-scratch baseline), and sample text from the LM.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api import db
from api.training import classify, finetune, generate, pretrain_lm
from shared.data import (
    CLASS_NAMES,
    N_CLASSES,
    VOCAB_SIZE,
    decode,
    encode,
    generate_one,
)
from shared.schemas import (
    ClassesResponse,
    Dataset,
    DatasetCreate,
    FinetuneRequest,
    FinetuneResponse,
    GenerateRequest,
    GenerateResponse,
    Health,
    PredictRequest,
    PredictResponse,
    PredictSampleRequest,
    PretrainRequest,
    PretrainResponse,
    Run,
    Version,
)

app = FastAPI(
    title="Fine-It API",
    description="Pretrain/finetune char Transformer for the three-cloud stack.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


def _git_sha() -> str:
    if os.environ.get("RENDER_GIT_COMMIT"):
        return os.environ["RENDER_GIT_COMMIT"][:7]
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _project_ref() -> str | None:
    url = os.environ.get("SUPABASE_URL", "")
    if url.startswith("https://"):
        return url.split("//", 1)[1].split(".", 1)[0]
    return None


def _artifact_or_404(run_id: int) -> str:
    model_b64 = db.get_run_artifact(run_id)
    if model_b64 is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    return model_b64


def _seq_sha256(sequence: list[int]) -> str:
    return hashlib.sha256(json.dumps(sequence).encode()).hexdigest()


def _dataset_or_404(dataset_id: int) -> dict:
    dataset = db.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset_id not found")
    return dataset


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
@app.post("/datasets", response_model=Dataset, tags=["datasets"])
def create_dataset(req: DatasetCreate) -> Dataset:
    """Register a synthetic pretraining corpus (parameters only; text from seed)."""
    row = db.insert_dataset(
        name=req.name,
        n_rows=req.n_rows,
        seq_len=req.seq_len,
        vocab_size=VOCAB_SIZE,
        n_classes=N_CLASSES,
        seed=req.seed,
    )
    return Dataset(**{k: row[k] for k in Dataset.model_fields})


# ---------------------------------------------------------------------------
# pretraining  (self-supervised LM)
# ---------------------------------------------------------------------------
@app.post("/pretrain", response_model=PretrainResponse, tags=["training"])
def pretrain(req: PretrainRequest) -> PretrainResponse:
    """Pretrain the next-char LM on the corpus; persist a pretrain run + artifact."""
    dataset = _dataset_or_404(req.dataset_id)
    val_loss, model_b64, _hist = pretrain_lm(
        n_rows=dataset["n_rows"],
        seq_len=dataset["seq_len"],
        seed=dataset["seed"],
        lr=req.lr,
        batch_size=req.batch_size,
        epochs=req.epochs,
    )
    run = db.insert_pretrain_run(
        dataset_id=req.dataset_id,
        lr=req.lr,
        batch_size=req.batch_size,
        epochs=req.epochs,
        val_loss=val_loss,
        model_b64=model_b64,
    )
    return PretrainResponse(run_id=run["id"], val_loss=val_loss)


# ---------------------------------------------------------------------------
# fine-tuning  (supervised; pretrained vs scratch)
# ---------------------------------------------------------------------------
@app.post("/finetune", response_model=FinetuneResponse, tags=["training"])
def finetune_endpoint(req: FinetuneRequest) -> FinetuneResponse:
    """Fine-tune a classifier (optionally warm-started) + a from-scratch baseline."""
    dataset = _dataset_or_404(req.dataset_id)
    pretrain_b64 = None
    if req.pretrain_run_id is not None:
        pretrain_b64 = _artifact_or_404(req.pretrain_run_id)

    pretrained, scratch, model_b64 = finetune(
        n_labeled=req.n_labeled,
        seq_len=dataset["seq_len"],
        seed=dataset["seed"],
        lr=req.lr,
        batch_size=req.batch_size,
        epochs=req.epochs,
        test_size=req.test_size,
        pretrain_b64=pretrain_b64,
    )
    run = db.insert_finetune_run(
        dataset_id=req.dataset_id,
        pretrain_run_id=req.pretrain_run_id,
        lr=req.lr,
        batch_size=req.batch_size,
        epochs=req.epochs,
        pretrained=pretrained,
        scratch=scratch,
        model_b64=model_b64,
    )
    return FinetuneResponse(
        run_id=run["id"],
        pretrained=pretrained,
        scratch=scratch,
    )


@app.get("/runs/{run_id}", response_model=Run, tags=["training"])
def get_run(run_id: int) -> Run:
    row = db.get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    return Run(**{k: row.get(k) for k in Run.model_fields})


@app.get("/runs", response_model=list[Run], tags=["training"])
def list_runs() -> list[Run]:
    rows = db.latest_runs(limit=50)
    return [Run(**{k: r.get(k) for k in Run.model_fields}) for r in rows]


# ---------------------------------------------------------------------------
# generation  (temperature sampling from the LM)
# ---------------------------------------------------------------------------
@app.post("/generate", response_model=GenerateResponse, tags=["generation"])
def generate_endpoint(req: GenerateRequest) -> GenerateResponse:
    """Sample characters from a pretrain run's LM head with temperature."""
    model_b64 = _artifact_or_404(req.run_id)
    prompt_ids = [t for t in encode(req.prompt) if t != 0]
    if not prompt_ids:
        raise HTTPException(status_code=422, detail="prompt has no known characters")
    new_ids = generate(
        model_b64, prompt_ids, req.length, req.temperature, req.seed
    )
    return GenerateResponse(
        prompt=req.prompt,
        generated=decode(new_ids),
        text=decode(prompt_ids + new_ids),
    )


# ---------------------------------------------------------------------------
# prediction  (classify a string's dialect)
# ---------------------------------------------------------------------------
def _predict_sequence(
    run_id: int, sequence: list[int], true_class: str | None
) -> PredictResponse:
    model_b64 = _artifact_or_404(run_id)
    result = classify(model_b64, sequence)
    sha256 = _seq_sha256(sequence)
    db.insert_sequence_metadata(
        run_id=run_id,
        sha256=sha256,
        length=len(sequence),
        label=result["label"],
        class_name=result["class_name"],
        confidence=result["confidence"],
    )
    return PredictResponse(
        run_id=run_id,
        sequence=sequence,
        sequence_sha256=sha256,
        true_class=true_class,
        **result,
    )


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(req: PredictRequest) -> PredictResponse:
    """Classify a character sequence's dialect. Only the hash + length are logged."""
    if any(t < 0 or t >= VOCAB_SIZE for t in req.sequence):
        raise HTTPException(
            status_code=422, detail=f"character ids must be in [0, {VOCAB_SIZE})"
        )
    return _predict_sequence(req.run_id, req.sequence, true_class=None)


@app.post("/predict_sample", response_model=PredictResponse, tags=["prediction"])
def predict_sample(req: PredictSampleRequest) -> PredictResponse:
    """Generate a synthetic string, classify its dialect, and log the hash."""
    kind = (
        req.true_class
        if (req.true_class is None or req.true_class < N_CLASSES)
        else None
    )
    seq, label = generate_one(seed=req.seed, kind=kind)
    return _predict_sequence(
        req.run_id, [int(t) for t in seq], true_class=CLASS_NAMES[label]
    )


# ---------------------------------------------------------------------------
# meta / ops
# ---------------------------------------------------------------------------
@app.get("/classes", response_model=ClassesResponse, tags=["meta"])
def classes() -> ClassesResponse:
    return ClassesResponse(classes=CLASS_NAMES, n_classes=N_CLASSES)


@app.get("/healthz", response_model=Health, tags=["ops"])
def healthz() -> Health:
    supabase_ok = db.ping()
    model_ok = torch.tensor([1.0]).sum().item() == 1.0
    status = "ok" if (supabase_ok and model_ok) else "degraded"
    return Health(status=status, model_loader=model_ok, supabase=supabase_ok)


@app.get("/version", response_model=Version, tags=["ops"])
def version() -> Version:
    return Version(
        git_sha=_git_sha(),
        torch_version=torch.__version__,
        supabase_project_ref=_project_ref(),
    )
