"""
train/embed.py — turn text into fixed-length vectors ONCE, offline, and cache them.

TRAINING TIER. The encoder never leaves this directory.

    python -m train.embed --offline          # deterministic hashing encoder
    python -m train.embed                    # a frozen sentence-transformer

THE ARCHITECTURE NOTE, RESTATED AS CODE
---------------------------------------
This is where the transformer lives, and it is the only place. It runs on your
laptop or a Colab runtime, it is FROZEN — never fine-tuned, so no gradient ever
flows into it — and its output is a fixed-length float vector that gets written
to a cache file. `api/` never imports this module. Nothing downstream of the
cache knows or cares that a transformer produced it.

Three consequences worth stating plainly, because they are the reasons for the
design rather than side effects of it:

  * **Cost.** The encoder runs once per unique string, not once per request.
    A reward model trained on 2,400 comparisons touches 4,800 responses; a
    served model would touch one per request forever.
  * **Determinism.** A frozen encoder makes the embedding a property of the
    text. Re-running training does not move the features underneath the head,
    so a change in held-out accuracy is a change you caused.
  * **Deployability.** The head that consumes these vectors is 60 KB. The thing
    that produced them is 90 MB of weights plus PyTorch. Only one of those two
    fits the serving budget, which is why `train/reward_model.py` trains a
    second head on TF-IDF features and why THAT is the one the service loads.

WHAT TO REPORT
--------------
Build step 3 asks for two numbers: how long the embedding pass took and how
large the cached vectors are. `main()` prints both. They are not trivia — they
are the two numbers that decide whether the embedding head could have been
served at all, and a student who cannot state them has not understood what the
split is buying.

THE OFFLINE FALLBACK
--------------------
`HashingEncoder` is a deterministic, dependency-free stand-in for a sentence
encoder, for environments with no access to the model hub. It is NOT a sentence
encoder and its vectors carry no semantics beyond token identity. It is here so
that the rest of the pipeline runs; results computed on it are labelled
`encoder='hashing-offline'` wherever they are persisted.

It does, however, reproduce ONE property of a real mean-pooled encoder on
purpose, and you should know which: **mean pooling leaks length**. Averaging n
roughly-orthogonal token vectors gives a result whose norm falls like
1/sqrt(n), so ||e(y)|| is a smooth, monotone, linearly-decodable function of
response length in both the real encoder and this one. That is the mechanism
behind the length-bias result in `train/reward_model.py`, and it is a genuine
property of mean-pooled transformer embeddings, not an artefact of the
fallback. If you want to see it, `python -m train.embed --offline --diagnose`
prints the norm-versus-length correlation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time

import numpy as np

from shared.preprocess import tokenise

CACHE_DIR = pathlib.Path("cache")
HASHING_DIM = 256          # the fallback's dimension; a real MiniLM is 384
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ===========================================================================
# The offline encoder
# ===========================================================================


class HashingEncoder:
    """Deterministic token-hashing encoder with mean pooling.

    Each token maps to a fixed pseudo-random unit vector, seeded from a BLAKE2b
    digest of the token itself. The sentence vector is the mean of its tokens'
    vectors. That is exactly the shape of the real pipeline — per-token vectors,
    mean-pooled — with a hash standing in for a transformer.

    Why BLAKE2b and not `hash()`: Python's built-in `hash()` for strings is
    SALTED PER PROCESS (PYTHONHASHSEED), so a cache built in one run would be
    garbage in the next, with no error and no way to notice except a reward
    model that stopped working. A cryptographic digest is stable across
    processes, machines and Python versions.

    Vectors are cached in a dict as they are seen. The vocabulary here is a few
    hundred types, so this is bounded; on real text you would want an LRU or a
    fixed-size hashing trick instead of an unbounded dict.
    """

    name = "hashing-offline"

    def __init__(self, dim: int = HASHING_DIM) -> None:
        self.dim = dim
        self._cache: dict[str, np.ndarray] = {}

    def _token_vector(self, token: str) -> np.ndarray:
        v = self._cache.get(token)
        if v is None:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            rng = np.random.default_rng(int.from_bytes(digest, "little"))
            v = rng.normal(size=self.dim)
            v /= np.linalg.norm(v)          # unit vectors, so mean pooling is an average
            self._cache[token] = v
        return v

    def encode(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        del batch_size  # signature-compatible with the real encoder; unused here
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = tokenise(text)
            if not toks:
                continue        # an all-zero row, which is the honest encoding of nothing
            out[i] = np.mean([self._token_vector(t) for t in toks], axis=0)
        return out


class SentenceEncoder:
    """A FROZEN sentence-transformer. The real path.

    Imported inside `__init__` rather than at module scope so that this file
    stays importable — and `HashingEncoder` usable — in an environment with
    neither torch nor transformers installed.

    `torch.no_grad()` and `.eval()` are not optimisations here, they are the
    statement that this encoder is frozen. Leaving the model in training mode
    would apply dropout, so the same string would embed differently on two
    passes and the cache would be a lie.

    Mean pooling is written out rather than taken from the model's pooler
    because the pooler of a BERT-family checkpoint is the [CLS] head, which was
    trained for next-sentence prediction and is a poor sentence representation.
    Mean pooling over the attention mask is what the sentence-transformers
    recipe actually does, and doing it by hand is one loop that shows you where
    the padding goes.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.name = model_name
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()                          # frozen: no dropout, no updates
        for p in self.model.parameters():
            p.requires_grad_(False)                # frozen: and no gradients either
        self.dim = int(self.model.config.hidden_size)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        torch = self._torch
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    # 256 tokens, not the model's 512. UltraFeedback responses
                    # have a long tail and the quadratic attention cost is paid
                    # on the longest item in the batch, not the mean. State the
                    # truncation length in your README: it is a modelling
                    # decision, and a reward model that never sees the last
                    # paragraph cannot be blamed for ignoring it.
                    max_length=256,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**enc).last_hidden_state       # (B, T, H)
                mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
                # Masked mean. Dividing by the mask sum rather than by T is the
                # whole trick: averaging over padding pulls every short sequence
                # toward zero by an amount that depends on the OTHER items in
                # the batch, which makes the embedding batch-order dependent.
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                out[i : i + len(batch)] = pooled.cpu().numpy()
        return out


def make_encoder(offline: bool, model_name: str = DEFAULT_MODEL):
    """One switch, as in train/data.py. Explicit, never a silent fallback."""
    return HashingEncoder() if offline else SentenceEncoder(model_name)


# ===========================================================================
# The cache
# ===========================================================================


def _cache_path(encoder_name: str, tag: str, cache_dir: pathlib.Path = CACHE_DIR
                ) -> pathlib.Path:
    # The ENCODER NAME is in the filename. Two different encoders producing two
    # caches with the same name is how you train a head on MiniLM vectors and
    # evaluate it on hashing vectors, get 50% accuracy, and spend an evening
    # debugging the training loop.
    safe = encoder_name.replace("/", "_")
    return cache_dir / f"embeddings_{safe}_{tag}.npz"


def embed_texts(
    texts: list[str],
    encoder,
    tag: str,
    cache_dir: pathlib.Path = CACHE_DIR,
    force: bool = False,
) -> tuple[np.ndarray, dict]:
    """Embed `texts`, using the cache if it is valid. Returns (matrix, stats).

    The cache is keyed on a digest of the text list, not just on the filename.
    A cache that is reused because the file exists — while the dataset under it
    changed — is worse than no cache: the reward head trains on vectors for
    responses it is not being shown, and nothing anywhere reports an error.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(encoder.name, tag, cache_dir)

    key = hashlib.sha256("\x00".join(texts).encode("utf-8")).hexdigest()
    if path.exists() and not force:
        z = np.load(path, allow_pickle=False)
        if str(z["key"]) == key:
            return np.asarray(z["X"], dtype=np.float32), {
                "encoder": encoder.name,
                "cached": True,
                "n_texts": len(texts),
                "dim": int(z["X"].shape[1]),
                "seconds": 0.0,
                "cache_bytes": path.stat().st_size,
                "cache_path": str(path),
            }

    t0 = time.perf_counter()
    X = encoder.encode(texts)
    elapsed = time.perf_counter() - t0

    # float32, for the same reason train/export.py writes float32: it halves the
    # cache for no measurable loss, and cache size is a number you are
    # accountable for in the README.
    np.savez_compressed(path, X=X.astype(np.float32), key=np.asarray(key))
    return X, {
        "encoder": encoder.name,
        "cached": False,
        "n_texts": len(texts),
        "dim": int(X.shape[1]),
        "seconds": round(elapsed, 3),
        "texts_per_second": round(len(texts) / elapsed, 1) if elapsed > 0 else None,
        "cache_bytes": path.stat().st_size,
        "cache_path": str(path),
    }


def embed_dataset(
    dataset,
    encoder,
    cache_dir: pathlib.Path = CACHE_DIR,
    force: bool = False,
    include_prompt: bool = False,
):
    """Embed every response of a `PreferenceDataset`.

    Returns `(vectors, stats)` where `vectors` maps split -> (chosen, rejected)
    matrices, each (n_pairs, dim).

    `include_prompt=False` IS A DELIBERATE AND COSTLY CHOICE. A reward model
    should score a response IN THE CONTEXT OF A PROMPT — r(x, y), not r(y) —
    and dropping the prompt means "42" is a uniformly mediocre answer rather
    than an excellent answer to one question. Both heads here are r(y).

    Two reasons, and you should disagree with them in your report if you can
    argue it:

      1. `POST /score` takes text and nothing else. That is the contract the
         product brief specifies, and a deployed head trained on r(x, y) but
         served r(y) would be evaluated on a distribution it never saw — the
         text-domain version of a preprocessing step applied in training and
         forgotten at serving time.
      2. The two heads are compared on held-out accuracy, and that comparison
         is only about the FEATURES if everything else is held fixed. Giving
         the embedding head the prompt and the TF-IDF head not would make the
         comparison about the input unit instead.

    Set `include_prompt=True` and re-run to see what conditioning is worth; on
    real UltraFeedback it is worth several accuracy points, and the gap is a
    good thing to quote in the limitations section.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    stats: dict[str, dict] = {}
    tag_suffix = "withprompt" if include_prompt else "responseonly"

    def render(row: dict, field_name: str) -> str:
        return f"{row['prompt']} [SEP] {row[field_name]}" if include_prompt else row[field_name]

    for split, rows in (("train", dataset.train), ("test", dataset.test)):
        if not rows:
            continue
        chosen = [render(r, "chosen") for r in rows]
        rejected = [render(r, "rejected") for r in rows]
        # One encode call over both, so the cache holds one array and the
        # "how long did it take" number covers the whole pass rather than half.
        X, st = embed_texts(chosen + rejected, encoder, f"{split}_{tag_suffix}", cache_dir, force)
        n = len(rows)
        out[split] = (X[:n], X[n:])
        stats[split] = st
    return out, stats


def norm_versus_length(texts: list[str], encoder) -> float:
    """Pearson r between ||e(y)|| and token count. The length-leak diagnostic.

    Prints negative and large in magnitude for any mean-pooled encoder, because
    the mean of n unit vectors has expected norm ~1/sqrt(n). Run it on the real
    encoder too — the effect is weaker there but it is present, and seeing it
    yourself is more convincing than being told.
    """
    X = encoder.encode(texts)
    norms = np.linalg.norm(X, axis=1)
    lens = np.asarray([len(tokenise(t)) for t in texts], dtype=np.float64)
    ok = lens > 0
    return float(np.corrcoef(norms[ok], lens[ok])[0, 1])


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Embed the preference dataset, once, offline.")
    ap.add_argument("--offline", action="store_true",
                    help="use the deterministic hashing encoder instead of a model hub")
    ap.add_argument("--pairs", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--force", action="store_true", help="ignore an existing cache")
    ap.add_argument("--diagnose", action="store_true",
                    help="print the norm-versus-length correlation and exit")
    args = ap.parse_args(argv)

    from train.data import load_preferences

    ds = load_preferences(args.offline, n_pairs=args.pairs, seed=args.seed)
    enc = make_encoder(args.offline, args.model)

    if args.diagnose:
        sample = [r["chosen"] for r in ds.train[:500]] + [r["rejected"] for r in ds.train[:500]]
        r = norm_versus_length(sample, enc)
        print(json.dumps({"encoder": enc.name, "norm_vs_length_pearson_r": round(r, 4)}, indent=2))
        return {"norm_vs_length_pearson_r": r}

    _, stats = embed_dataset(ds, enc, force=args.force)
    total_bytes = sum(s["cache_bytes"] for s in stats.values())
    total_seconds = sum(s["seconds"] for s in stats.values())
    summary = {
        "encoder": enc.name,
        "per_split": stats,
        # THE TWO NUMBERS BUILD STEP 3 ASKS FOR.
        "total_seconds": round(total_seconds, 3),
        "cache_megabytes": round(total_bytes / 1e6, 3),
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":  # pragma: no cover - a CLI entry point
    main()
