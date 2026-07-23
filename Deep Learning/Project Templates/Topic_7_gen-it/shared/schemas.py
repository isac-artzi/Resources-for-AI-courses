"""Pydantic request/response models shared by the API and (optionally) the UI.

The contract between the three clouds. "Gen-It" is a **variational autoencoder**
(VAE): it learns a 2-D latent space over synthetic images, then generates,
reconstructs, and interpolates in that space. The *shape* of the contract follows
the base template; the fields swap classification metrics for VAE metrics
(reconstruction loss / KL / ELBO) and add the generative endpoints.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class DatasetCreate(BaseModel):
    """Request body for POST /datasets."""

    name: str = Field(..., min_length=1, max_length=100)
    n_rows: int = Field(1500, ge=100, le=20_000)
    noise: float = Field(0.1, ge=0.0, le=1.0, description="Pixel noise std.")
    seed: int = Field(42, description="Seed for the synthetic image generator.")


class Dataset(BaseModel):
    """A dataset row as stored in Supabase."""

    id: int
    name: str
    n_rows: int
    img_size: int
    n_classes: int
    created_at: datetime


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
class TrainRequest(BaseModel):
    """Request body for POST /train."""

    dataset_id: int
    lr: float = Field(0.001, gt=0.0, description="Adam learning rate.")
    batch_size: int = Field(64, ge=1)
    epochs: int = Field(20, ge=1, le=500)


class VAEMetrics(BaseModel):
    recon_loss: float = Field(..., description="Per-image reconstruction BCE (held-out).")
    kl: float = Field(..., description="Per-image KL(q(z|x) || N(0, I)).")
    elbo: float = Field(..., description="Evidence lower bound = -(recon_loss + kl).")


class Run(BaseModel):
    """A training-run row as stored in Supabase (metrics only)."""

    id: int
    dataset_id: int
    lr: float
    batch_size: int
    epochs: int
    recon_loss: float
    kl: float
    elbo: float
    created_at: datetime


class TrainResponse(BaseModel):
    run_id: int
    metrics: VAEMetrics


# ---------------------------------------------------------------------------
# Generation  (sample the prior, or decode a chosen latent point)
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    run_id: int
    z: Optional[List[float]] = Field(
        None, description="Latent coordinates to decode; None samples z ~ N(0, I)."
    )
    seed: int = Field(0, description="Sampling seed when z is not given.")


class ImageResponse(BaseModel):
    run_id: int
    image_png: str = Field(..., description="base64 PNG of the decoded image.")
    z: List[float] = Field(..., description="The latent point that was decoded.")


# ---------------------------------------------------------------------------
# Reconstruction  (encode a sample, then decode it)
# ---------------------------------------------------------------------------
class ReconstructRequest(BaseModel):
    run_id: int
    seed: int = Field(0, description="Seed for the generated input sample.")
    true_class: Optional[int] = Field(
        None, ge=0, description="Force the sample's class; None picks one at random."
    )


class ReconstructResponse(BaseModel):
    run_id: int
    input_png: str = Field(..., description="base64 PNG of the (synthetic) input.")
    recon_png: str = Field(..., description="base64 PNG of the reconstruction.")
    z: List[float] = Field(..., description="Posterior mean latent of the input.")
    recon_mse: float = Field(..., description="Mean squared reconstruction error.")
    image_sha256: str = Field(..., description="Hash of the input image (no pixels stored).")
    true_class: Optional[str] = None


# ---------------------------------------------------------------------------
# Interpolation  (walk a straight line between two latents)
# ---------------------------------------------------------------------------
class InterpolateRequest(BaseModel):
    run_id: int
    seed_a: int = Field(0, description="Seed for the first endpoint sample.")
    seed_b: int = Field(1, description="Seed for the second endpoint sample.")
    class_a: Optional[int] = Field(None, ge=0)
    class_b: Optional[int] = Field(None, ge=0)
    steps: int = Field(8, ge=2, le=32, description="Number of frames along the line.")


class InterpolateResponse(BaseModel):
    run_id: int
    frames: List[str] = Field(..., description="base64 PNGs from endpoint a to b.")
    z_a: List[float]
    z_b: List[float]


# ---------------------------------------------------------------------------
# Latent scatter  (encode a batch to visualise the latent space)
# ---------------------------------------------------------------------------
class LatentScatterRequest(BaseModel):
    run_id: int
    n_points: int = Field(200, ge=10, le=2000)
    seed: int = Field(123)


class LatentPoint(BaseModel):
    z: List[float]
    label: int
    class_name: str


class LatentScatterResponse(BaseModel):
    run_id: int
    points: List[LatentPoint]
    latent_dim: int


# ---------------------------------------------------------------------------
# Meta / Ops
# ---------------------------------------------------------------------------
class ClassesResponse(BaseModel):
    classes: List[str]
    n_classes: int


class Health(BaseModel):
    status: str
    model_loader: bool
    supabase: bool


class Version(BaseModel):
    git_sha: str
    torch_version: str
    supabase_project_ref: Optional[str]
