"""Numerical test: pretraining transfers.

The dialect classes share an identical unigram distribution and differ ONLY in
their Markov transition structure, so a bag-of-characters view is useless. A
network that learned the language during next-char pretraining therefore beats an
identically-trained from-scratch baseline on the small labelled set -- the whole
point of the template. We assert the warm-started model both clears a high
accuracy floor AND out-scores the scratch baseline, and that the LM head samples
strings of the requested length.
"""
from __future__ import annotations

from api.training import finetune, generate, pretrain_lm
from shared.data import N_CLASSES, SEQ_LEN, encode


def test_pretraining_beats_scratch_unit():
    val_loss, pretrain_b64, hist = pretrain_lm(
        n_rows=4000, seq_len=SEQ_LEN, seed=42, lr=0.005, batch_size=64, epochs=8
    )
    assert hist[-1] < hist[0]                     # LM training loss decreased
    assert isinstance(pretrain_b64, str) and pretrain_b64

    pretrained, scratch, model_b64 = finetune(
        n_labeled=240, seq_len=SEQ_LEN, seed=42,
        lr=0.004, batch_size=32, epochs=6, test_size=0.25,
        pretrain_b64=pretrain_b64,
    )
    assert pretrained is not None
    assert pretrained["accuracy"] > 0.85         # warm-start learns the dialects
    assert pretrained["accuracy"] > scratch["accuracy"]  # transfer helps
    assert isinstance(model_b64, str) and model_b64

    # The LM head samples exactly `length` new token ids.
    new_ids = generate(pretrain_b64, encode("ab"), length=24, temperature=0.8, seed=0)
    assert len(new_ids) == 24
    assert all(1 <= t < N_CLASSES + 2 for t in new_ids)  # real chars, never PAD


def test_pretrain_finetune_generate_endpoints(client):
    ds = client.post(
        "/datasets", json={"name": "gt", "n_rows": 4000, "seq_len": SEQ_LEN, "seed": 42}
    ).json()

    pre = client.post(
        "/pretrain",
        json={"dataset_id": ds["id"], "lr": 0.005, "batch_size": 64, "epochs": 8},
    )
    assert pre.status_code == 200
    pretrain_run_id = pre.json()["run_id"]
    assert pre.json()["val_loss"] > 0.0

    ft = client.post(
        "/finetune",
        json={
            "dataset_id": ds["id"],
            "pretrain_run_id": pretrain_run_id,
            "n_labeled": 240,
            "lr": 0.004,
            "batch_size": 32,
            "epochs": 6,
            "test_size": 0.25,
        },
    )
    assert ft.status_code == 200
    body = ft.json()
    assert body["pretrained"]["accuracy"] > 0.85
    assert body["pretrained"]["accuracy"] > body["scratch"]["accuracy"]
    finetune_run_id = body["run_id"]

    # Sample text from the pretrained LM.
    gen = client.post(
        "/generate",
        json={"run_id": pretrain_run_id, "prompt": "ab", "length": 16, "temperature": 0.8},
    ).json()
    assert len(gen["generated"]) == 16
    assert gen["text"].startswith("ab")

    # Classify a generated sample; only the hash + length are logged.
    pred = client.post(
        "/predict_sample", json={"run_id": finetune_run_id, "seed": 3, "true_class": 2}
    ).json()
    assert pred["label"] in range(N_CLASSES)
    assert 0.0 <= pred["confidence"] <= 1.0
    assert len(pred["probs"]) == N_CLASSES
    assert pred["true_class"] == "dialect-3"
    assert len(pred["sequence_sha256"]) == 64

    logged = client._store["sequence_metadata"]
    assert logged and logged[-1]["run_id"] == finetune_run_id
    assert logged[-1]["length"] == len(pred["sequence"])
    assert "sequence" not in logged[-1]           # no raw characters persisted
