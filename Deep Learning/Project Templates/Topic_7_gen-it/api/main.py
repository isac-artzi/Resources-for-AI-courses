"""FastAPI model service -- Cloud #2 (deployed on Render.com).

Responsibilities:
  * expose the VAE behind HTTP so the Streamlit UI can call it over HTTPS,
  * read dataset parameters from Supabase before training,
  * write run rows, model artifacts, and image metadata (hash only) back.

There is NO UI code here and NO business logic in the UI -- separation of
concerns across the three clouds. This is "Gen-It": a variational autoencoder
that learns a 2-D latent space over synthetic images and then generates,
reconstructs, and interpolates in that space.
"""
from __future__ import annotations

import hashlib
import os
import subprocess

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api import db
from api.training import (
    generate as vae_generate,
    interpolate as vae_interpolate,
    latent_coords,
    reconstruct as vae_reconstruct,
    train_vae,
)
from shared.data import CLASS_NAMES, IMG_SIZE, N_CLASSES, generate_images, generate_one
from shared.schemas import (
    ClassesResponse,
    Dataset,
    DatasetCreate,
    GenerateRequest,
    Health,
    ImageResponse,
    InterpolateRequest,
    InterpolateResponse,
    LatentPoint,
    LatentScatterRequest,
    LatentScatterResponse,
    ReconstructRequest,
    ReconstructResponse,
    Run,
    TrainRequest,
    TrainResponse,
    VAEMetrics,
    Version,
)

app = FastAPI(
    title="Gen-It API",
    description="Variational autoencoder (generate / reconstruct / interpolate) for the three-cloud stack.",
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


def _sample(seed: int, true_class: int | None):
    """Draw one synthetic image; return (gray_uint8, label)."""
    kind = true_class if (true_class is None or true_class < N_CLASSES) else None
    return generate_one(img_size=IMG_SIZE, noise=0.1, seed=seed, kind=kind)


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
@app.post("/datasets", response_model=Dataset, tags=["datasets"])
def create_dataset(req: DatasetCreate) -> Dataset:
    """Register a synthetic image dataset (parameters only; pixels come from seed)."""
    row = db.insert_dataset(
        name=req.name,
        n_rows=req.n_rows,
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        noise=req.noise,
        seed=req.seed,
    )
    return Dataset(**{k: row[k] for k in Dataset.model_fields})


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
@app.post("/train", response_model=TrainResponse, tags=["training"])
def train(req: TrainRequest) -> TrainResponse:
    """Read dataset params from Supabase, train the VAE, persist run + artifact."""
    dataset = db.get_dataset(req.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset_id not found")

    metrics, model_b64, _loss = train_vae(
        n_rows=dataset["n_rows"],
        img_size=dataset["img_size"],
        noise=dataset["noise"],
        seed=dataset["seed"],
        lr=req.lr,
        batch_size=req.batch_size,
        epochs=req.epochs,
    )
    run = db.insert_run(
        dataset_id=req.dataset_id,
        lr=req.lr,
        batch_size=req.batch_size,
        epochs=req.epochs,
        metrics=metrics,
        model_b64=model_b64,
    )
    return TrainResponse(run_id=run["id"], metrics=VAEMetrics(**metrics))


@app.get("/runs/{run_id}", response_model=Run, tags=["training"])
def get_run(run_id: int) -> Run:
    row = db.get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    return Run(**{k: row[k] for k in Run.model_fields})


@app.get("/runs", response_model=list[Run], tags=["training"])
def list_runs() -> list[Run]:
    rows = db.latest_runs(limit=50)
    return [Run(**{k: r[k] for k in Run.model_fields}) for r in rows]


# ---------------------------------------------------------------------------
# generate  (decode a latent point; the interactive slider endpoint)
# ---------------------------------------------------------------------------
@app.post("/generate", response_model=ImageResponse, tags=["generate"])
def generate(req: GenerateRequest) -> ImageResponse:
    """Decode a chosen latent point (or a prior sample) into a new image."""
    model_b64 = _artifact_or_404(req.run_id)
    image_png, z = vae_generate(model_b64, req.z, req.seed)
    return ImageResponse(run_id=req.run_id, image_png=image_png, z=z)


# ---------------------------------------------------------------------------
# reconstruct  (encode a synthetic sample, decode it, score the error)
# ---------------------------------------------------------------------------
@app.post("/reconstruct", response_model=ReconstructResponse, tags=["reconstruct"])
def reconstruct(req: ReconstructRequest) -> ReconstructResponse:
    """Generate a synthetic image, encode it to its posterior mean, decode it.

    Only the input image's sha256 hash + shape + reconstruction error are
    logged -- never pixels.
    """
    model_b64 = _artifact_or_404(req.run_id)
    gray, label = _sample(req.seed, req.true_class)
    result = vae_reconstruct(model_b64, gray)
    sha256 = hashlib.sha256(gray.tobytes()).hexdigest()
    db.insert_image_metadata(
        run_id=req.run_id,
        sha256=sha256,
        width=int(gray.shape[1]),
        height=int(gray.shape[0]),
        recon_mse=result["recon_mse"],
    )
    return ReconstructResponse(
        run_id=req.run_id,
        image_sha256=sha256,
        true_class=CLASS_NAMES[label],
        **result,
    )


# ---------------------------------------------------------------------------
# interpolate  (walk a straight line between two latents, decode each step)
# ---------------------------------------------------------------------------
@app.post("/interpolate", response_model=InterpolateResponse, tags=["interpolate"])
def interpolate(req: InterpolateRequest) -> InterpolateResponse:
    """Encode two synthetic images and decode a straight-line latent walk between them."""
    model_b64 = _artifact_or_404(req.run_id)
    gray_a, _ = _sample(req.seed_a, req.class_a)
    gray_b, _ = _sample(req.seed_b, req.class_b)
    result = vae_interpolate(model_b64, gray_a, gray_b, req.steps)
    return InterpolateResponse(run_id=req.run_id, **result)


# ---------------------------------------------------------------------------
# latent scatter  (encode a batch to visualise the latent space)
# ---------------------------------------------------------------------------
@app.post("/latent_scatter", response_model=LatentScatterResponse, tags=["latent"])
def latent_scatter(req: LatentScatterRequest) -> LatentScatterResponse:
    """Encode a batch of labelled samples to their latent coordinates (for a scatter)."""
    model_b64 = _artifact_or_404(req.run_id)
    images, labels = generate_images(req.n_points, img_size=IMG_SIZE, noise=0.1, seed=req.seed)
    coords = latent_coords(model_b64, images)
    points = [
        LatentPoint(z=z, label=int(lbl), class_name=CLASS_NAMES[int(lbl)])
        for z, lbl in zip(coords, labels)
    ]
    latent_dim = len(coords[0]) if coords else 0
    return LatentScatterResponse(run_id=req.run_id, points=points, latent_dim=latent_dim)


# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------
@app.get("/classes", response_model=ClassesResponse, tags=["meta"])
def classes() -> ClassesResponse:
    """Return the class labels used to colour the latent scatter."""
    return ClassesResponse(classes=CLASS_NAMES, n_classes=N_CLASSES)


# ---------------------------------------------------------------------------
# ops
# ---------------------------------------------------------------------------
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
