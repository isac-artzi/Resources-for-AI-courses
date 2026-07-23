# Attend-It — Three-Cloud Template (LSTM + Attention Sequence Classifier)

> The same three-cloud architecture as the base template
> (**Streamlit UI ⇄ FastAPI Model API ⇄ Supabase Data**), with the middle box
> swapped for a **PyTorch LSTM + additive (Bahdanau) attention** doing sequence
> classification with per-token explanations. The deploy steps are identical, so
> follow the main [three-cloud TUTORIAL](../Topic_1_three-cloud/TUTORIAL.md).

## Live deployment URLs (fill these in)

| Tier | Platform | URL |
|------|----------|-----|
| **UI** | Streamlit Community Cloud | `https://<your-app>.streamlit.app` |
| **API** | Render.com | `https://<your-api>.onrender.com` |
| **Data** | Supabase | `https://<your-project-ref>.supabase.co` |

> Replace the placeholders above with your real URLs once deployed.

---

## What it does

Attend-It classifies short integer-token sequences into one of four classes
(**synthetic**: `alpha`, `beta`, `gamma`, `delta`) with an LSTM + attention, and
— crucially — returns the **per-timestep attention weights** that show *which
token* the network relied on. Each sequence's class is decided by a single class
marker placed right after a randomly-positioned trigger token, so a model that
solves the task must learn to *attend* to that position — making the attention
bar a genuine explanation. You pick the learning rate, batch size, and epochs;
the API trains with Adam + cross-entropy, reports held-out **accuracy /
macro-F1**, and persists every run. The UI lets you train, type or generate a
sequence to classify, view the attention bar chart, review run history, and read
the model card. Only the sequence's **sha256 hash** is logged — never the tokens.

## What changed from the base template (the reusable pattern in action)

The three-cloud split and the file layout are identical to *Regress-It*. Only the
middle box changed:

| Aspect | Regress-It | Attend-It |
|--------|----------------------|---------------------|
| Model | `nn.Linear(1,1)` + SGD | LSTM + additive attention + Adam |
| Task | regression | multi-class sequence classification |
| Metrics | MSE / MAE / R² | accuracy / macro-F1 |
| Input | one scalar `x` | a sequence of token ids |
| Explainability | — | **per-timestep attention weights** |
| Extra endpoints | — | `/predict_sample`, `/classes` |
| Privacy invariant | — | store sequence **hash**, never tokens |
| Tables | datasets · runs · predictions | datasets · runs · **run_artifacts** · sequence_metadata |

Everything else — UI as a thin client (no torch), API as the only writer,
Supabase as the single source of truth, service-role vs anon keys, RLS on
`runs`, the four test categories — is unchanged.

## Architecture

```
┌──────────────────────┐   HTTPS/JSON    ┌──────────────────────────┐   service-role   ┌──────────────────┐
│  Streamlit Cloud     │ ──────────────► │  FastAPI on Render        │ ───────────────► │  Supabase        │
│  (ui/app.py)         │   token seq     │  (api/main.py)            │   full access    │  Postgres        │
│  thin client, no ML  │                 │  LSTM + attention (torch) │                  │  datasets/runs/  │
│  plots attention bar │                 │                          │                  │  run_artifacts/  │
│                      │ ◄────anon key,  │                          │                  │  sequence_       │
│                      │   read-only ────┼──────────────────────────┼──────────────────►│  metadata        │
└──────────────────────┘   SELECT runs   └──────────────────────────┘                  │  (RLS: anon can  │
                                                                                        │   only SELECT    │
                                                                                        │   runs)          │
                                                                                        └──────────────────┘
```

## Project structure

```
attend-it/
├── README.md                 # This file
├── MODEL_CARD.md             # Model details, attention, privacy, limitations
├── shared/
│   ├── schemas.py            # Pydantic API contract
│   └── data.py               # Synthetic trigger-token sequence generator + class contract
├── api/                      # FastAPI tier (deploys to Render)
│   ├── main.py               # Endpoints
│   ├── training.py           # LSTM + attention, artifact (de)serialization
│   ├── db.py                 # Supabase (service-role) data access
│   ├── configs/default.yaml
│   └── requirements.txt
├── ui/
│   ├── app.py                # 5-tab thin client (type / generate + attention bar)
│   ├── requirements.txt      # No torch
│   └── .streamlit/secrets.toml.example
├── db/
│   ├── migrations/001_init.sql
│   └── seed.py
├── tests/                    # pytest suite
├── render.yaml               # Render blueprint
├── requirements-dev.txt
└── .env.example
```

## Quickstart (local)

```bash
cd attend-it

python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

pytest -q            # tests pass; the live-Supabase test skips without creds

cp .env.example .env                                   # API: SUPABASE_URL + SERVICE key
cp ui/.streamlit/secrets.toml.example ui/.streamlit/secrets.toml

uvicorn api.main:app --reload --port 8000              # terminal 1
streamlit run ui/app.py                                # terminal 2
```

To deploy to the three clouds, follow **Part E** of the main
[TUTORIAL](../Topic_1_three-cloud/TUTORIAL.md): apply
`db/migrations/001_init.sql` in the Supabase SQL Editor → deploy the API from
`render.yaml` on Render → deploy the UI on Streamlit Community Cloud.

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/datasets` | Register a synthetic sequence dataset (params only) |
| `POST` | `/train` | Train the model, persist run + model artifact, return metrics |
| `GET`  | `/runs/{run_id}` | Fetch one run |
| `GET`  | `/runs` | List recent runs |
| `POST` | `/predict` | Classify a token sequence; return attention; log the hash |
| `POST` | `/predict_sample` | Generate + classify a sample; return attention |
| `GET`  | `/classes` | The class labels the model predicts over |
| `GET`  | `/healthz` | Liveness / DB ping |
| `GET`  | `/version` | Build SHA + torch version |

## Checklist

- [ ] Three live URLs listed at the top of this README
- [ ] `datasets`, `runs`, `run_artifacts`, `sequence_metadata` tables with RLS
- [ ] 6+ API endpoints
- [ ] 5 Streamlit tabs (Concepts, Train, Predict, Run History, Model Card)
- [ ] Attention training with held-out accuracy / macro-F1 + per-token explanations
- [ ] pytest suite passing
- [ ] `MODEL_CARD.md` completed
