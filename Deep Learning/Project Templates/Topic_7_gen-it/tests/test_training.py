"""Numerical test: the VAE learns to reconstruct the synthetic shapes and the
four latent-space operations round-trip through the API.

We train a compact 2-D-latent VAE, assert the training ELBO loss decreases and
held-out reconstruction error clears a floor, then exercise
generate / reconstruct / interpolate / latent_scatter through the in-memory
store (a real pickled VAE is deserialized each call).
"""
from __future__ import annotations

import base64

from shared.data import IMG_SIZE, N_CLASSES
from api.training import train_vae


def test_vae_reconstructs_shapes_unit():
    metrics, model_b64, loss = train_vae(
        n_rows=1200, img_size=IMG_SIZE, noise=0.1, seed=42,
        lr=0.001, batch_size=64, epochs=20,
    )
    assert loss[-1] < loss[0]          # training loss decreased
    assert metrics["recon_loss"] > 0.0
    assert metrics["kl"] > 0.0
    assert metrics["elbo"] == -(metrics["recon_loss"] + metrics["kl"])
    assert isinstance(model_b64, str) and len(model_b64) > 0


def test_generate_reconstruct_interpolate_endpoints(client):
    ds = client.post(
        "/datasets", json={"name": "gt", "n_rows": 1200, "noise": 0.1, "seed": 42}
    ).json()
    resp = client.post(
        "/train",
        json={"dataset_id": ds["id"], "lr": 0.001, "batch_size": 64, "epochs": 20},
    )
    assert resp.status_code == 200
    body = resp.json()
    run_id = body["run_id"]
    assert body["metrics"]["elbo"] < 0.0

    # generate: decode a chosen 2-D latent point -> a valid PNG.
    gen = client.post(
        "/generate", json={"run_id": run_id, "z": [0.5, -0.5]}
    ).json()
    assert base64.b64decode(gen["image_png"])
    assert gen["z"] == [0.5, -0.5]

    # generate: no z -> sample the prior with the seed.
    gen2 = client.post("/generate", json={"run_id": run_id, "seed": 3}).json()
    assert len(gen2["z"]) == 2
    assert base64.b64decode(gen2["image_png"])

    # reconstruct: encode a generated sample, decode it, score the error.
    rec = client.post(
        "/reconstruct", json={"run_id": run_id, "seed": 1, "true_class": 3}
    ).json()
    assert base64.b64decode(rec["input_png"])
    assert base64.b64decode(rec["recon_png"])
    assert rec["recon_mse"] < 0.08          # reconstruction is faithful
    assert len(rec["z"]) == 2
    assert rec["true_class"] == "block"
    assert len(rec["image_sha256"]) == 64

    # The reconstruction was logged as metadata (hash + shape + error only).
    logged = client._store["image_metadata"]
    assert logged and logged[-1]["run_id"] == run_id
    assert logged[-1]["width"] == IMG_SIZE and logged[-1]["height"] == IMG_SIZE
    assert "input_png" not in logged[-1]     # no pixels persisted

    # interpolate: N frames along a straight line between two latents.
    interp = client.post(
        "/interpolate",
        json={"run_id": run_id, "seed_a": 0, "seed_b": 1, "steps": 6},
    ).json()
    assert len(interp["frames"]) == 6
    assert all(base64.b64decode(f) for f in interp["frames"])
    assert len(interp["z_a"]) == 2 and len(interp["z_b"]) == 2

    # latent_scatter: encode a labelled batch to 2-D coordinates.
    scatter = client.post(
        "/latent_scatter", json={"run_id": run_id, "n_points": 60, "seed": 123}
    ).json()
    assert scatter["latent_dim"] == 2
    assert len(scatter["points"]) == 60
    pt = scatter["points"][0]
    assert len(pt["z"]) == 2
    assert pt["label"] in range(N_CLASSES)
