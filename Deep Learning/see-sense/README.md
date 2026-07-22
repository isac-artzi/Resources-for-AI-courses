# See-Sense — Three-Cloud Template (CNN Image Classifier + Grad-CAM)

> The same three-cloud architecture as the base template
> (**Streamlit UI ⇄ FastAPI Model API ⇄ Supabase Data**), with the middle box
> swapped for a **PyTorch CNN + Grad-CAM** doing image classification with
> visual explanations. The deploy steps are identical, so follow the main
> [three-cloud TUTORIAL](../three-cloud/TUTORIAL.md).

## Live deployment URLs (fill these in)

| Tier | Platform | URL |
|------|----------|-----|
| **UI** | Streamlit Community Cloud | `https://<your-app>.streamlit.app` |
| **API** | Render.com | `https://<your-api>.onrender.com` |
| **Data** | Supabase | `https://<your-project-ref>.supabase.co` |

> Replace the placeholders above with your real URLs once deployed.

---

## What it does

See-Sense classifies small grayscale images into one of four geometric shape
classes (**synthetic**: horizontal bar, vertical bar, diagonal line, filled
block) with a compact CNN, and — crucially — returns a **Grad-CAM** heatmap that
shows *where the network looked* to make each call. You pick the learning rate,
batch size, and epochs; the API trains with Adam + cross-entropy, reports
held-out **accuracy / macro-F1**, and persists every run. The UI lets you train,
upload an image or generate a sample to classify, view the Grad-CAM overlay,
review run history, and read the model card. Only the image's **sha256 hash** is
logged — never the pixels.

## What changed from the base template (the reusable pattern in action)

The three-cloud split and the file layout are identical to *Regress-It*. Only the
middle box changed:

| Aspect | Regress-It | See-Sense |
|--------|----------------------|---------------------|
| Model | `nn.Linear(1,1)` + SGD | small CNN (2 conv blocks) + Adam |
| Task | regression | multi-class image classification |
| Metrics | MSE / MAE / R² | accuracy / macro-F1 |
| Input | one scalar `x` | a 28×28 grayscale image |
| Explainability | — | **Grad-CAM** heatmap overlay |
| Extra endpoints | — | `/predict_sample`, `/classes` |
| Privacy invariant | — | store image **hash**, never pixels |
| Tables | datasets · runs · predictions | datasets · runs · **run_artifacts** · image_metadata |

Everything else — UI as a thin client (no torch, no image libs), API as the only
writer, Supabase as the single source of truth, service-role vs anon keys, RLS on
`runs`, the four test categories — is unchanged.

## Architecture

```
┌──────────────────────┐   HTTPS/JSON    ┌──────────────────────────┐   service-role   ┌──────────────────┐
│  Streamlit Cloud     │ ──────────────► │  FastAPI on Render        │ ───────────────► │  Supabase        │
│  (ui/app.py)         │  + multipart    │  (api/main.py)            │   full access    │  Postgres        │
│  thin client, no ML  │    upload       │  CNN + Grad-CAM (PIL)     │                  │  datasets/runs/  │
│  base64-decodes PNGs │                 │                          │                  │  run_artifacts/  │
│                      │ ◄────anon key,  │                          │                  │  image_metadata  │
│                      │   read-only ────┼──────────────────────────┼──────────────────►│  (RLS: anon can  │
└──────────────────────┘   SELECT runs   └──────────────────────────┘                  │   only SELECT    │
                                                                                        │   runs)          │
                                                                                        └──────────────────┘
```

## Project structure

```
see-sense/
├── README.md                 # This file
├── MODEL_CARD.md             # Model details, Grad-CAM, privacy, limitations
├── shared/
│   ├── schemas.py            # Pydantic API contract
│   └── data.py               # Synthetic shape-image generator + class contract
├── api/                      # FastAPI tier (deploys to Render)
│   ├── main.py               # Endpoints
│   ├── training.py           # CNN + Grad-CAM, artifact (de)serialization, PNG overlay
│   ├── db.py                 # Supabase (service-role) data access
│   ├── configs/default.yaml
│   └── requirements.txt
├── ui/
│   ├── app.py                # 5-tab thin client (upload / generate + Grad-CAM view)
│   ├── requirements.txt      # No torch / no image libs
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
cd see-sense

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
| `POST` | `/datasets` | Register a synthetic image dataset (params only) |
| `POST` | `/train` | Train the CNN, persist run + model artifact, return metrics |
| `GET`  | `/runs/{run_id}` | Fetch one run |
| `GET`  | `/runs` | List recent runs |
| `POST` | `/predict` | Classify an uploaded image; return Grad-CAM; log the hash |
| `POST` | `/predict_sample` | Generate + classify a sample; return Grad-CAM |
| `GET`  | `/classes` | The class labels the model predicts over |
| `GET`  | `/healthz` | Liveness / DB ping |
| `GET`  | `/version` | Build SHA + torch version |

## Checklist

- [ ] Three live URLs listed at the top of this README
- [ ] `datasets`, `runs`, `run_artifacts`, `image_metadata` tables with RLS
- [ ] 6+ API endpoints
- [ ] 5 Streamlit tabs (Concepts, Train, Predict, Run History, Model Card)
- [ ] CNN training with held-out accuracy / macro-F1 + Grad-CAM explanations
- [ ] pytest suite passing
- [ ] `MODEL_CARD.md` completed
