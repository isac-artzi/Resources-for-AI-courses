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

## The one thing that is in the serving requirements and looks like it should not be

`gymnasium`. It is there because `POST /rollout` runs evaluation episodes
server-side, which means the service tier needs the environment itself.

That is not a hole in the rule, because the rule is about a number. Gymnasium
is a few hundred kilobytes of Python on top of NumPy; FrozenLake is pure Python
with no renderer instantiated (see `envs.make_env`, which leaves `render_mode`
as `None` on purpose). Torch is 490 MB of bundled CUDA libraries a CPU app will
never call. **Ask what a dependency costs and what it is for, not what field it
belongs to.**

The line to hold: if a future topic needs a Box2D or MuJoCo environment — real
binary wheels, real memory — then `/rollout` moves to the training tier and the
service returns pre-computed trajectories, rather than this file growing.

## Diagnosing a failure

The guard catches *transitive* imports, which is the failure that actually
happens: you add a helper to `shared/`, the helper imports something from
`train/`, and `train/` imports torch. Grepping `api/` would not find it.

```bash
python -X importtime -c "import api.main" 2>&1 | grep -i torch
```

The first `torch` line names the module that pulled it in. Move that import
inside the function that needs it, or move the function into `train/`.

## Why this is not an artificial constraint

The lesson generalises well past this course: **the artifact you deploy is not
the training graph.** A team that cannot state how large its deployed
application is does not yet have a production system. Writing the forward pass
by hand, once, is the clearest possible demonstration that inference is matrix
arithmetic and nothing more — and it is exactly what an ONNX or TensorRT export
does for you later, when you have stopped being surprised by it.
