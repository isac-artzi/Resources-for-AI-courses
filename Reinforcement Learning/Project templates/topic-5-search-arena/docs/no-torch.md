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

## Why this is not an artificial constraint

The lesson generalises well past this course: **the artifact you deploy is not
the training graph.** A team that cannot state how large its deployed
application is does not yet have a production system. Writing the forward pass
by hand, once, is the clearest possible demonstration that inference is matrix
arithmetic and nothing more — and it is exactly what an ONNX or TensorRT export
does for you later, when you have stopped being surprised by it.

---

## Topic 5 addendum: the guard that catches the framework you ARE allowed to install

The rule above is about PyTorch's 490 MB. This topic surfaces a second, quieter
version of the same problem, and it is worth reading because the first one is
easy and this one is not.

`gymnasium` **is** in `requirements-serve.txt`. It has to be: `POST /rollout`
runs evaluation episodes server-side, so the deployed process must be able to
construct an environment. It is a few megabytes of pure Python over NumPy, and
the no-torch rule was never about the word "RL".

But this topic's search agents do not need it. `Position`, the whole of
`search/`, and the learned evaluator in `search/net.py` work on plain lists and
NumPy arrays; nothing on the `POST /act` path ever constructs an environment.
`requirements-serve.txt` says so in its header. **An unchecked claim in a
comment is a wish**, and `tests/test_no_torch.py` cannot check this one, because
gymnasium being present in `sys.modules` is not by itself a failure.

So there are two extra guards:

* `tests/test_import_graph.py` imports `api.main` in a **clean subprocess** and
  asserts `gymnasium` is absent from `sys.modules`. A subprocess, because by the
  time that test runs some earlier test has already called `make_env()`, and
  asserting on the test process would measure the suite's import history rather
  than the application's import graph.
* CI **uninstalls gymnasium entirely** and then plays a move
  (`.github/workflows/ci.yml`). If the search path has grown a dependency, the
  import fails outright.

### What it caught

The first version of `envs/connect_four.py` had this at module scope:

```python
try:
    import gymnasium as gym
    _GYM_BASE = gym.Env
except Exception:
    _GYM_BASE = object          # "optional" — so the serving image doesn't need it
```

Every service process imported gymnasium anyway. **A defensive import still
imports**; the `try` only changes what happens when the import fails, not
whether it is attempted. The fix was structural rather than cosmetic: split the
module into `envs/connect_four.py` (the rules, pure) and `envs/gym_env.py` (the
wrapper), and import the second one *inside* `make_env()`.

### The generalisable lesson

The size of your deployed application is a property of its **import graph**, not
of its requirements file. `requirements-serve.txt` says what may be installed;
`python -X importtime -c "import api.main"` says what is actually paid for. Those
are different questions, and only the second one shows up in the memory graph at
three in the morning.

```bash
python -X importtime -c "import api.main" 2>&1 | sort -k2 -n -r | head -20
```
