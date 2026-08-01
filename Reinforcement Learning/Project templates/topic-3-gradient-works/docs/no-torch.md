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

## Topic 3: the guard stops being free

Until now the deployed policy was a table and the rule cost nothing — nobody was
tempted to import torch to index an array. From this topic the artifact is a
network, and two new pressures appear:

1. **The test suite needs torch.** The required NumPy/PyTorch equivalence test
   cannot be written without it. But `sys.modules` is per-process, so a single
   `import torch` at the top of any test module leaves it there for every test
   that runs afterwards — including `tests/test_no_torch.py`. The guard would
   start failing for a reason unrelated to the deployment, collection order
   would decide whether the build was green, and the obvious "fix" would be to
   weaken the guard. The right answer is a **subprocess**: see
   `run_torch_script` in `tests/conftest.py`. It also happens to mirror
   reality, since training and serving really are separate processes.

2. **The forward pass can be silently wrong.** A transposed weight matrix is
   still a matrix, and NumPy will multiply it without complaint. An agent that
   scored 500 in training then behaves randomly in the deployed app, with no
   error anywhere. That is why the equivalence check is a pytest test rather
   than a line in a checklist, and why `api/forward.check_layer_shapes` fails
   at load time rather than at first request.

The measured agreement for this template's committed artifact is
**max |NumPy − PyTorch| ≈ 9e-8** across 256 wide random observations, against a
stated tolerance of 1e-5. The residual is float32 weights evaluated in float64;
anything at 1e-3 or above is a bug, not rounding.

## Why this is not an artificial constraint

The lesson generalises well past this course: **the artifact you deploy is not
the training graph.** A team that cannot state how large its deployed
application is does not yet have a production system. Writing the forward pass
by hand, once, is the clearest possible demonstration that inference is matrix
arithmetic and nothing more — and it is exactly what an ONNX or TensorRT export
does for you later, when you have stopped being surprised by it.
