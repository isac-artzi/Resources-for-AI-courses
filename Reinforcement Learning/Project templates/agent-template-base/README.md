<!-- =========================================================================
     TEMPLATE README

     The first fifteen lines are graded. Replace the bracketed placeholders
     before your first commit — not the night before submission.
     ========================================================================= -->

# [Product Name] — [one-line description a stakeholder would recognise]

[**One paragraph, before any code or commands.** What does this agent do, who
would use it, and why is it useful? Write it for the client in the product
brief, not for the grader. If the paragraph only makes sense to someone who
has read the assignment, rewrite it.]

| | |
|---|---|
| **Live app** | https://[your-app].streamlit.app |
| **Supabase project** | `[your-project-ref]` — schema in [`db/migrations/`](db/migrations/) |
| **Service tier (local)** | `uvicorn api.main:app --port 8000` → http://127.0.0.1:8000/docs |
| **Service capture** | [link to your screen recording of the service serving `POST /act` and `GET /docs`] |
| **Author / section** | [name] |

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.11–3.13. Not 3.14.
pip install -r requirements-train.txt                 # laptop / Colab
cp .env.example .env                                  # then fill in your keys

# 1. apply the schema — paste db/migrations/001_init.sql into the Supabase SQL editor
python -m db.seed

# 2. train (writes telemetry to Supabase as it goes, then exports an artifact)
python -m train.train

# 3. serve, and demonstrate it serving
uvicorn api.main:app --reload --port 8000

# 4. the user interface
streamlit run ui/app.py

# 5. the gate
pytest -q && ruff check .
```

## What is where

```
api/        the service tier. Owns the policy. Imports NumPy, never a framework.
  main.py     the six standing endpoints + /docs
  policy.py   the entire serving-side forward pass
ui/         the presentation tier. No policy code, no training code, no writes.
  app.py      the Streamlit application
  service.py  the one switch between in-process and HTTP service calls
train/      the training tier. Runs on your laptop or in Colab. Never deployed.
  export.py   PyTorch -> .npz. The seam between the tiers.
shared/     the contracts. Pydantic models, settings, the data-tier interface,
            and preprocess.py — every transformation, in one importable place.
envs/       the Gymnasium environment, exposed as make_env().
db/         migrations and a seed script. The schema is checked in and tested.
tests/      the standing four, all driven through an HTTP test client.
policies/   exported artifacts. Committed, because they are what you deployed.
```

## Architecture — two clouds, three tiers

**Presentation** (Streamlit Community Cloud, deployed) → **Service** (FastAPI,
in this repository, run under uvicorn locally and imported in-process in
production) → **Data** (Supabase Postgres, deployed).

The service tier is a real application with real Pydantic contracts, exercised
over HTTP by the test suite. In the deployed app the Streamlit tier imports the
same handlers instead of crossing the network; `SERVICE_MODE` is the only thing
that changes. Separation of concerns is a property of the code, not of the
hosting bill — a repository with clean tiers can move one onto its own host in
an afternoon.

**Why a database tier at all.** A supervised product can be judged from one
number on a held-out set. A reinforcement learning product cannot: the
deliverable is a comparison of learning curves, and every honest claim has the
form *configuration A reached return R in N episodes and configuration B did
not*. That is a query, not a plot. With every episode of every run as a row,
you cannot accidentally report the one seed that worked, and the comparison
tables the rubric asks for fall out of a `GROUP BY`.

## The no-PyTorch-in-serving rule

`import torch` alone occupies roughly **490 MB** of resident memory against
Streamlit Community Cloud's **690 MB** guarantee — most of the budget spent
before a single weight is loaded, on bundled CUDA libraries a CPU application
never calls. The entire deployed stack here measures **82 MB**.

So: train in PyTorch outside the deployed app, export the weights to a NumPy
`.npz`, and evaluate the forward pass in NumPy. For a tabular policy that is an
array lookup; for a two-layer network it is two matrix multiplies, a ReLU and a
softmax. `tests/test_no_torch.py` asserts `"torch" not in sys.modules` after
importing the app, and `requirements-serve.txt` does not list it. If the guard
fails, the build fails. See [`docs/no-torch.md`](docs/no-torch.md).

## Free-Tier Notes

*Required section. State how this product handles each limit.*

| Limit | Value | How this product handles it |
|---|---|---|
| Streamlit Cloud memory | 690 MB guaranteed | [82 MB measured; no framework in the serving path] |
| Streamlit Cloud sleep | after 12 h idle | [wakes on first request; note the cold start in your demo] |
| Supabase storage | 500 MB | [per-episode rows, not per-step; N runs ≈ M MB] |
| Supabase projects | 2 active per person | [one project reused across every topic] |
| Supabase pause | after 1 week idle | [UI degrades visibly; see the health banner] |
| Python version | 3.11–3.13 | [pinned in CI; 3.14 has no Box2D wheels] |

## Theoretical Brief

*350–600 words, mirrored in the Streamlit "Concepts" tab. Where your topic's
objectives call for a derivation, the derivation belongs here.*

## Quantitative Analysis

*A results table comparing at least two configurations, plus a written
interpretation. Every number must be reproducible from `experiments` and
`episodes`, and the seeds must be named. A learning curve with no seed and no
row count is not evidence.*

| Configuration | Seeds | Mean return (± SE) | Episodes to threshold | Query |
|---|---|---|---|---|
| | | | | |

## AI-Assistance Disclosure

*Required. What did you generate, with which tool, and how did you verify it?
Generated code must be read, understood and tested by you; blind paste-through
is not acceptable.*

## Limitations & Responsible Use

*At least four concrete limitations, each with how you would test whether it
binds in a deployment scenario. Then foreseeable misuse, reward-specification
risk, and the worldview reflection your topic calls for.*
