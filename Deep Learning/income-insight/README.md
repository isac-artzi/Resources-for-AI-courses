# Income-Insight — Three-Cloud Template (Tabular Classifier)

> The same three-cloud architecture as the base template
> (**Streamlit UI ⇄ FastAPI Model API ⇄ Supabase Data**), with the middle box
> swapped for a **PyTorch MLP + sklearn preprocessing pipeline** doing tabular
> binary classification. Start from *Regress-It*; the deploy steps are identical,
> so follow the main
> [three-cloud TUTORIAL](../three-cloud/TUTORIAL.md).

## Live deployment URLs (fill these in)

| Tier | Platform | URL |
|------|----------|-----|
| **UI** | Streamlit Community Cloud | `https://<your-app>.streamlit.app` |
| **API** | Render.com | `https://<your-api>.onrender.com` |
| **Data** | Supabase | `https://<your-project-ref>.supabase.co` |

> Replace the placeholders with your real URLs once deployed.

---

## What it does

Income-Insight predicts whether a person's income exceeds \$50K from a handful of
tabular features (Adult-Income shaped, but **synthetic**). You pick the hidden
width, learning rate, batch size, and epochs; the API standardizes the numeric
features, one-hot-encodes the categoricals, trains an MLP with Adam + BCE loss,
reports held-out **accuracy / precision / recall / F1 / ROC-AUC**, and persists
every run. The UI lets you train, score individual records, review run history,
and the API exposes a fairness `/audit` view over logged predictions.

## What changed from the base template (the reusable pattern in action)

The three-cloud split and the file layout are identical to *Regress-It*. Only the
middle box changed:

| Aspect | Regress-It | Income-Insight |
|--------|------------|----------------|
| Model | `nn.Linear(1,1)` + SGD | MLP (`Linear→ReLU→Linear`) + Adam |
| Preprocessing | none | sklearn `ColumnTransformer` (scale + one-hot) |
| Task | regression | binary classification |
| Metrics | MSE / MAE / R² | accuracy / precision / recall / F1 / ROC-AUC |
| Feature | one scalar `x` | a record of 7 named features |
| Extra endpoints | — | `/predict_batch`, `/schema`, `/audit` |
| Tables | datasets · runs · predictions | datasets · runs · **run_artifacts** · predictions |

Everything else — UI as a thin client, API as the only writer, Supabase as the
single source of truth, service-role vs anon keys, RLS on `runs`, the four test
categories — is unchanged.

## Architecture

```
┌──────────────────────┐   HTTPS/JSON    ┌──────────────────────────┐   service-role   ┌──────────────────┐
│  Streamlit Cloud     │ ──────────────► │  FastAPI on Render        │ ───────────────► │  Supabase        │
│  (ui/app.py)         │                 │  (api/main.py)            │   full access    │  Postgres        │
│  thin client, no ML  │                 │  MLP + sklearn pipeline   │                  │  datasets/runs/  │
│                      │ ◄────anon key,  │                          │                  │  run_artifacts/  │
│                      │   read-only ────┼──────────────────────────┼──────────────────►│  predictions     │
└──────────────────────┘   SELECT runs   └──────────────────────────┘                  │  (RLS: anon can  │
                                                                                        │   only SELECT    │
                                                                                        │   runs)          │
                                                                                        └──────────────────┘
```

## Project structure

```
income-insight/
├── README.md                 # This file
├── MODEL_CARD.md             # Model details, fairness, limitations
├── shared/
│   ├── schemas.py            # Pydantic API contract
│   └── data.py               # Synthetic Adult-Income generator + feature contract
├── api/                      # FastAPI tier (deploys to Render)
│   ├── main.py               # Endpoints
│   ├── training.py           # MLP + sklearn ColumnTransformer, artifact (de)serialization
│   ├── db.py                 # Supabase (service-role) data access
│   ├── configs/default.yaml
│   └── requirements.txt
├── ui/
│   ├── app.py                # 5-tab thin client (form built from /schema)
│   ├── requirements.txt      # No torch / no sklearn
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
cd income-insight

python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

pytest -q            # 6 pass; the live-Supabase test skips without creds

cp .env.example .env                                   # API: SUPABASE_URL + SERVICE key
cp ui/.streamlit/secrets.toml.example ui/.streamlit/secrets.toml

uvicorn api.main:app --reload --port 8000              # terminal 1
streamlit run ui/app.py                                # terminal 2
```

To deploy to the three clouds, follow **Part E** of the main
[TUTORIAL](../three-cloud/TUTORIAL.md): apply
`db/migrations/001_init.sql` in the Supabase SQL Editor → deploy the API from
`render.yaml` on Render → deploy the UI on Streamlit Community Cloud.

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/datasets` | Generate a synthetic tabular dataset |
| `POST` | `/train` | Train the MLP, persist run + model artifact, return metrics |
| `GET`  | `/runs/{run_id}` | Fetch one run |
| `GET`  | `/runs` | List recent runs |
| `POST` | `/predict` | Classify one record; log it |
| `POST` | `/predict_batch` | Classify many records; log each |
| `GET`  | `/schema` | Feature contract (drives the UI form) |
| `GET`  | `/audit` | Positive-prediction rate grouped by a categorical feature |
| `GET`  | `/healthz` | Liveness / DB ping |
| `GET`  | `/version` | Build SHA + torch/sklearn versions |

## Checklist

- [ ] Three live URLs listed at the top of this README
- [ ] `datasets`, `runs`, `run_artifacts`, `predictions` tables with RLS
- [ ] 6+ API endpoints
- [ ] 5 Streamlit tabs (Concepts, Train, Predict, Run History, Model Card)
- [ ] MLP training with held-out accuracy/precision/recall/F1/ROC-AUC
- [ ] pytest suite passing
- [ ] `MODEL_CARD.md` completed
