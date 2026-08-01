# Why the deployed application must not import PyTorch

## The number

Measured on a standard Linux wheel, in a clean environment:

| | resident memory |
|---|---|
| `import torch` (default wheel) | ~490 MB |
| `import torch` (CPU-only wheel) | ~340 MB |
| `import torch` + FastAPI | ~531 MB |
| `import torch` + `transformers` + a 124M-parameter causal LM | ~1.1 GB |
| This entire deployed stack (Streamlit + NumPy + Supabase client + FastAPI + uvicorn) | **82 MB** |

Streamlit Community Cloud **guarantees 690 MB** and may allocate up to 2.7 GB
depending on load. A PyTorch-importing service therefore spends most of its
guaranteed allocation before loading a single weight, on bundled CUDA shared
libraries that a CPU application will never call. Under load — which is when
you care — it will not start.

## The rule

1. **Train in PyTorch, outside the deployed app.** `train/` runs on your
   laptop or a Colab GPU runtime. It streams telemetry to Supabase and writes a
   policy artifact at the end.
2. **Serve a NumPy artifact.** `train/export.py` writes the weights to `.npz`;
   the serving tier evaluates the forward pass in NumPy. A tabular policy is an
   array lookup. A two-layer network is `h = np.maximum(0, W0 @ x + b0)` then
   an output layer.
3. **Enforce it in CI.** `tests/test_no_torch.py` asserts `"torch" not in
   sys.modules` after importing the app. `requirements-serve.txt` does not list
   torch, and CI installs only that file.

## What Topic 6 adds: the rule applied to language

This topic is where the rule stops being about a policy and starts being about
an entire modality, and it forces two decisions that are worth stating as
architecture rather than discovering as constraints.

### 1. The encoder runs offline, and only offline

The transformer is used in the **training tier** to turn text into fixed-length
vectors, once per unique string, and the vectors are cached. It is frozen — no
gradient ever flows into it — so the embedding is a property of the text rather
than of the training run. `api/` never imports `train/embed.py`.

The reward model is a small head on top of those vectors and exports to `.npz`
exactly like every policy in this course.

### 2. Generation is offline; the service **scores**

There is no `/generate`. A 124M-parameter causal LM plus PyTorch is over a
gigabyte resident, and generation is the expensive direction anyway. So the
training tier generates completions once, persists them to `completions`, and
`GET /completions` reads rows. The deployed model is the reward head, which is
four NumPy operations.

This is the shape of a real inference budget: the expensive model runs where
you can afford it, the cheap one runs where the traffic is.

### 3. A third rule this topic needs: **scikit-learn is not a serving dependency either**

The deployed head's input is a TF-IDF vector. Fitting the vectoriser is
training-tier work; *applying* it is fifteen lines of NumPy, and
`api/reward.py` contains those fifteen lines. The vocabulary and the IDF vector
travel inside the same `.npz` as the weights, so the artifact carries its own
featuriser and there is exactly one thing to deploy and one checksum to record.

The temptation, the first time `tests/test_equivalence.py` fails, is to import
scikit-learn in `api/reward.py` and make the disagreement go away.
`requirements-serve.txt` does not list it and CI asserts `"sklearn" not in
sys.modules`, so that change does not build.

## Diagnosing a failure

The guard catches *transitive* imports, which is the failure that actually
happens: you add a helper to `shared/`, the helper imports something from
`train/`, and `train/` imports torch. Grepping `api/` would not find it.

```bash
python -X importtime -c "import api.main" 2>&1 | grep -iE "torch|sklearn|transformers"
```

The first matching line names the module that pulled it in. Move that import
inside the function that needs it, or move the function into `train/`.

Note the pattern this repository uses everywhere for the legitimate cases:
`train/export.py` imports torch *inside* `export_torch_mlp`, and
`train/data.py` imports `datasets` *inside* `load_real`. Both files stay
importable in an environment that has neither.

## The same rule inside the test suite

`tests/` runs in the serving environment, and three of this topic's tests need
torch. They run it in a **subprocess** (`run_torch_script` in
`tests/conftest.py`) rather than importing it, because `sys.modules` is
per-process: one `import torch` anywhere in the session would leave it there
for `tests/test_no_torch.py`, which would then fail for a reason that has
nothing to do with the deployment — and the obvious "fix" would be to weaken
the guard.

## Why this is not an artificial constraint

The lesson generalises well past this course: **the artifact you deploy is not
the training graph.** A team that cannot state how large its deployed
application is does not yet have a production system. Writing the forward pass
by hand, once, is the clearest possible demonstration that inference is matrix
arithmetic and nothing more — and it is exactly what an ONNX or TensorRT export
does for you later, when you have stopped being surprised by it.

In this topic there is a second lesson layered on top. Writing the TF-IDF
transform by hand shows you that **the feature pipeline is part of the model**.
A team that ships weights and re-derives the featuriser at serving time has
shipped half a model, and the half they kept is the half that cannot tell them
it is wrong.
