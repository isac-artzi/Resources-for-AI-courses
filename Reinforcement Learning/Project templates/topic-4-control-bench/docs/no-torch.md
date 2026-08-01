# Why the deployed application must not import PyTorch

## The number

Measured on a standard Linux wheel, in a clean environment:

| | resident memory |
|---|---|
| `import torch` (default wheel) | ~490 MB |
| `import torch` (CPU-only wheel) | ~340 MB |
| `import torch` + FastAPI | ~531 MB |
| This entire deployed stack (Streamlit + NumPy + Gymnasium + Supabase client + FastAPI + uvicorn) | **82 MB** |

Streamlit Community Cloud **guarantees 690 MB** and may allocate up to 2.7 GB
depending on load. A PyTorch-importing service therefore spends most of its
guaranteed allocation before loading a single weight, on bundled CUDA shared
libraries that a CPU application will never call. Under load — which is when you
care — it will not start.

## The rule

1. **Train in PyTorch, outside the deployed app.** `train/` runs on your laptop
   or a Colab GPU runtime. It streams per-episode telemetry to Supabase and
   writes policy artifacts at the end.
2. **Serve NumPy artifacts.** `train/export.py` writes the weights to `.npz`;
   `api/forward.py` evaluates the forward pass in NumPy.
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

## Topic 4: three artifacts, two heads, and one new failure mode

Topic 3 made the rule cost something: the deployed policy became a network, and
a transposed weight matrix is still a matrix. Topic 4 adds a second head and a
third artifact, and with them a failure the discrete case cannot produce.

**The continuous head.** The SAC actor emits `2 × action_dim` numbers — a mean
and a log standard deviation — which are then squashed by `tanh` and rescaled
into the environment's action range. Reproducing that in NumPy is three lines
(`api/forward.gaussian_head` and `api/forward.squashed_action`), and each of the
three has a way of being silently wrong:

1. **Splitting the output.** Nothing in the archive shape distinguishes a
   two-output Gaussian actor from a two-action discrete policy. That is why the
   archive carries an explicit `head` key: without it the loader would run
   `argmax` over a mean and a log standard deviation, which is a perfectly
   well-typed way to be completely wrong and raises nothing.
2. **The log σ clamp.** SAC clamps `log σ` to `[-20, 2]` during training. The
   clamp is part of the function the network computes, not a safety net around
   it — apply it on one side only and the two implementations diverge exactly
   where the raw head is extreme, which is the tail of the input space where you
   will not look. The bounds travel inside the archive.
3. **The action scale.** `tanh` gives `[-1, 1]`; Pendulum's torque is `[-2, 2]`.
   A policy deployed without the scale applies half the torque it trained with,
   and the symptom — an agent that swings most of the way up and stalls — reads
   as undertraining rather than as a units bug.

`tests/test_equivalence.py` compares the pre-squash mean, the clamped log σ AND
the final action separately for exactly this reason: which of the three
disagrees tells you which of the three mistakes you made.

**The test suite still needs torch, and this process still must not have it.**
`sys.modules` is per-process, so a single `import torch` at the top of any test
module leaves it there for every test that runs afterwards — including
`tests/test_no_torch.py`. The guard would start failing for a reason unrelated
to the deployment, collection order would decide whether the build was green,
and the obvious "fix" would be to weaken the guard. The answer is a
**subprocess**: see `run_torch_script` in `tests/conftest.py`. Note also that
`train/entropy_sweep.py` imports `train.sac` inside its functions rather than at
module scope, so that its pure aggregation helper can be unit-tested in the
torch-free test process.

**Measured agreement for this template's committed artifacts**, across 256 wide
random observations, against a stated tolerance of 1e-5:

| artifact | quantity compared | max abs difference |
|---|---|---|
| `a2c_cartpole` | action probabilities | ~8.9e-08 |
| `ppo_acrobot` | action probabilities | ~7.3e-08 |
| `sac_pendulum` | pre-squash mean | ~2.4e-07 |
| `sac_pendulum` | clamped log σ | ~2.8e-07 |
| `sac_pendulum` | deterministic action | ~4.6e-07 |

The residual is float32 weights evaluated in float64; anything at 1e-3 or above
is a bug, not rounding.

## Why this is not an artificial constraint

The lesson generalises well past this course: **the artifact you deploy is not
the training graph.** A team that cannot state how large its deployed
application is does not yet have a production system. Writing the forward pass
by hand, once, is the clearest possible demonstration that inference is matrix
arithmetic and nothing more — and it is exactly what an ONNX or TensorRT export
does for you later, when you have stopped being surprised by it.
