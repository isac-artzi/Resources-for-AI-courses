# Why the deployed application must not import PyTorch

## The number

Measured on a standard Linux wheel, in a clean environment:

| | resident memory |
|---|---|
| `import torch` (default wheel) | ~490 MB |
| `import torch` (CPU-only wheel) | ~340 MB |
| `import torch` + FastAPI | ~531 MB |
| This entire deployed stack (Streamlit + NumPy + Supabase client + FastAPI + uvicorn) | **82 MB** |

Streamlit Community Cloud **guarantees 690 MB** and may allocate up to 2.7 GB
depending on load. A PyTorch-importing service therefore spends most of its
guaranteed allocation before loading a single weight, on bundled CUDA shared
libraries that a CPU application will never call. Under load — which is when
you care — it will not start.

## The rule

1. **Train in PyTorch, outside the deployed app.** `train/` runs on your
   laptop or a Colab GPU runtime. It streams per-episode telemetry to Supabase
   and writes a policy artifact at the end.
2. **Serve a NumPy artifact.** `train/export.py` writes the weights to `.npz`;
   `api/policy.py` evaluates the forward pass in NumPy. Tabular policies are an
   array lookup. A two-layer network is `h = np.maximum(0, W0 @ x + b0)` then
   `softmax(W1 @ h + b1)`.
3. **Enforce it in CI.** `tests/test_no_torch.py` asserts `"torch" not in
   sys.modules` after importing the app. `requirements-serve.txt` does not list
   torch, and CI installs only that file.

## Diagnosing a failure

The guard catches *transitive* imports, which is the failure that actually
happens: you add a helper to `shared/`, the helper imports something from
`train/`, and `train/` imports torch. Grepping `api/` would not find it.

```bash
python -X importtime -c "import api.main" 2>&1 | grep -i torch
```

The first `torch` line names the module that pulled it in. Move that import
inside the function that needs it, or move the function into `train/`.

## What the rule is actually about — two decisions from this topic

The rule is not "few dependencies." It is **the artifact you deploy is not the
training graph.** This topic makes two calls in opposite directions, and the
reasoning is the point:

**Gymnasium IS in `requirements-serve.txt`.** `POST /rollout` runs bounded
evaluation episodes server-side — a standing endpoint, not an extra — so the
deployed process has to construct the environment. Measured on this stack:

| | disk | resident, on top of `numpy` + `fastapi` | import time |
|---|---|---|---|
| `gymnasium` | 2.1 MB | **+4.9 MB** | +66 ms |
| `torch` | ~800 MB | +490 MB | ~1.5 s |

A 2 MB pure-Python environment API over NumPy is not a training graph. A tensor
library that maps a bundled CUDA runtime at import time is.

**SciPy is NOT.** `train/compare.py` needs a Student *t* distribution for the
confidence intervals and the equivalence test. SciPy is ~90 MB, and the
serving tier has no use for it — so it stays in `requirements-train.txt`, and
`compare.py` falls back to a self-contained implementation (a continued-fraction
incomplete beta and a bisected inverse CDF) when it is absent. That fallback is
the path CI takes, since CI installs the serving requirements only, and
`tests/test_topic2.py` checks the two implementations against each other to 15
significant figures. A hand-rolled statistical function nobody validated is
worse than no statistics at all.

CI runs the guard as its own step, in a fresh process, for a reason worth
knowing: by the time `pytest` has imported `train/`, SciPy is already in
`sys.modules` and an in-process assertion would be meaningless.

```bash
python -c "import sys, api.main; \
  assert 'torch' not in sys.modules; \
  assert 'scipy' not in sys.modules"
```

## Why this is not an artificial constraint

The lesson generalises well past this course: **the artifact you deploy is not
the training graph.** A team that cannot state how large its deployed
application is does not yet have a production system. Writing the forward pass
by hand, once, is the clearest possible demonstration that inference is matrix
arithmetic and nothing more — and it is exactly what an ONNX or TensorRT export
does for you later, when you have stopped being surprised by it.
