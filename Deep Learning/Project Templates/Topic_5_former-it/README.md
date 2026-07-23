# Former-It — Three-Cloud Template (From-Scratch Transformer Encoder)

> The same three-cloud architecture as the base template
> (**Streamlit UI ⇄ FastAPI Model API ⇄ Supabase Data**), with the middle box
> swapped for a **from-scratch tiny Transformer encoder** (positional encoding +
> multi-head self-attention) doing sequence classification with per-head
> attention heatmaps. The deploy steps are identical, so follow the main
> [three-cloud TUTORIAL](../Topic_1_three-cloud/TUTORIAL.md).

## Live deployment URLs (fill these in)

| Tier | Platform | URL |
|------|----------|-----|
| **UI** | Streamlit Community Cloud | `https://<your-app>.streamlit.app` |
| **API** | Render.com | `https://<your-api>.onrender.com` |
| **Data** | Supabase | `https://<your-project-ref>.supabase.co` |

> Replace the placeholders above with your real URLs once deployed.

---

## What it does

Former-It decides whether a short symbol sequence is a **palindrome** using a
Transformer encoder built from scratch, and — crucially — returns the
**per-layer, per-head self-attention matrices** so you can see *which positions
attend to which*. Because the palindrome rule compares position `i` with position
`n-1-i`, a self-attention head can learn the **anti-diagonal**, which the heatmaps
make visible. You pick the learning rate, batch size, and epochs; the API trains
with Adam + cross-entropy, reports held-out **accuracy / macro-F1**, and persists
every run. The UI lets you train, type or generate a sequence to classify, view
the attention heatmaps, review run history, and read the model card. Only the
sequence's **sha256 hash** is logged — never the symbols.

## What changed from the base template (the reusable pattern in action)

The three-cloud split and the file layout are identical to *Regress-It*. Only the
middle box changed:

| Aspect | Regress-It | Former-It |
|--------|----------------------|---------------------|
| Model | `nn.Linear(1,1)` + SGD | from-scratch Transformer encoder + Adam |
| Task | regression | palindrome vs random (binary) |
| Metrics | MSE / MAE / R² | accuracy / macro-F1 |
| Input | one scalar `x` | a sequence of symbol ids |
| Explainability | — | **per-head self-attention heatmaps** |
| Extra endpoints | — | `/predict_sample`, `/classes` |
| Privacy invariant | — | store sequence **hash**, never symbols |
| Tables | datasets · runs · predictions | datasets · runs · **run_artifacts** · sequence_metadata |

Everything else — UI as a thin client (no torch), API as the only writer,
Supabase as the single source of truth, service-role vs anon keys, RLS on
`runs`, the four test categories — is unchanged.

## Architecture

```
┌──────────────────────┐   HTTPS/JSON    ┌──────────────────────────┐   service-role   ┌──────────────────┐
│  Streamlit Cloud     │ ──────────────► │  FastAPI on Render        │ ───────────────► │  Supabase        │
│  (ui/app.py)         │   symbol seq    │  (api/main.py)            │   full access    │  Postgres        │
│  thin client, no ML  │                 │  Transformer encoder      │                  │  datasets/runs/  │
│  plots attn heatmaps │                 │  (from scratch, torch)    │                  │  run_artifacts/  │
│                      │ ◄────anon key,  │                          │                  │  sequence_       │
│                      │   read-only ────┼──────────────────────────┼──────────────────►│  metadata        │
└──────────────────────┘   SELECT runs   └──────────────────────────┘                  │  (RLS: anon can  │
                                                                                        │   only SELECT    │
                                                                                        │   runs)          │
                                                                                        └──────────────────┘
```

## Project structure

```
former-it/
├── README.md                 # This file
├── MODEL_CARD.md             # Model details, attention, privacy, limitations
├── shared/
│   ├── schemas.py            # Pydantic API contract
│   └── data.py               # Synthetic palindrome sequence generator + class contract
├── api/                      # FastAPI tier (deploys to Render)
│   ├── main.py               # Endpoints
│   ├── training.py           # From-scratch Transformer encoder + artifact (de)serialization
│   ├── db.py                 # Supabase (service-role) data access
│   ├── configs/default.yaml
│   └── requirements.txt
├── ui/
│   ├── app.py                # 5-tab thin client (type / generate + attention heatmaps)
│   ├── requirements.txt      # No torch (matplotlib only, for heatmaps)
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
cd former-it

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
| `POST` | `/train` | Train the Transformer, persist run + model artifact, return metrics |
| `GET`  | `/runs/{run_id}` | Fetch one run |
| `GET`  | `/runs` | List recent runs |
| `POST` | `/predict` | Classify a symbol sequence; return per-head attention; log the hash |
| `POST` | `/predict_sample` | Generate + classify a sample; return attention |
| `GET`  | `/classes` | The class labels the model predicts over |
| `GET`  | `/healthz` | Liveness / DB ping |
| `GET`  | `/version` | Build SHA + torch version |

## Checklist

- [ ] Three live URLs listed at the top of this README
- [ ] `datasets`, `runs`, `run_artifacts`, `sequence_metadata` tables with RLS
- [ ] 6+ API endpoints
- [ ] 5 Streamlit tabs (Concepts, Train, Predict, Run History, Model Card)
- [ ] Transformer training with held-out accuracy / macro-F1 + per-head attention
- [ ] pytest suite passing
- [ ] `MODEL_CARD.md` completed
