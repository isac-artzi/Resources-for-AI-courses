# Fine-It — Three-Cloud Template (Pretrain + Fine-Tune Char Transformer)

> The same three-cloud architecture as the base template
> (**Streamlit UI ⇄ FastAPI Model API ⇄ Supabase Data**), with the middle box
> swapped for a **causal character Transformer** trained in two phases:
> self-supervised **next-character pretraining**, then **fine-tuning** a
> classifier head — compared against a from-scratch baseline to show the transfer
> gap. The deploy steps are identical, so follow the main
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

Fine-It demonstrates **transfer learning**. First it **pretrains** a causal
character Transformer to predict the next character across an unlabelled corpus.
Then it **fine-tunes** a classification head on a *small* labelled set — once
warm-started from the pretrained trunk, once from scratch — and reports both, so
the **transfer gap** is a bar chart you can see. The classification task is to
name which **dialect** produced a string; every dialect has the *same* character
frequencies and differs only in its character-to-character transitions, so the
model must actually learn the language (pretraining) to win. You can also **sample
text** from the LM with temperature. You pick the hyperparameters; the API trains
with Adam + cross-entropy, reports held-out **loss / accuracy / macro-F1**, and
persists every run. Only a string's **sha256 hash** is logged — never the
characters.

## What changed from the base template (the reusable pattern in action)

The three-cloud split and the file layout are identical to *Regress-It*. Only the
middle box changed:

| Aspect | Regress-It | Fine-It |
|--------|----------------------|---------------------|
| Model | `nn.Linear(1,1)` + SGD | causal char Transformer + Adam |
| Task | regression | **pretrain LM** → fine-tune dialect classifier |
| Metrics | MSE / MAE / R² | LM `val_loss` + accuracy / macro-F1 |
| Input | one scalar `x` | a sequence of character ids |
| Interactivity | — | **pretrained-vs-scratch bar** + **text generation** |
| Extra endpoints | — | `/pretrain`, `/finetune`, `/generate`, `/predict_sample`, `/classes` |
| Privacy invariant | — | store sequence **hash**, never characters |
| Tables | datasets · runs · predictions | datasets · runs (`run_type`) · **run_artifacts** · sequence_metadata |

Everything else — UI as a thin client (no torch), API as the only writer,
Supabase as the single source of truth, service-role vs anon keys, RLS on
`runs`, the four test categories — is unchanged.

## Architecture

```
┌──────────────────────┐   HTTPS/JSON    ┌──────────────────────────┐   service-role   ┌──────────────────┐
│  Streamlit Cloud     │ ──────────────► │  FastAPI on Render        │ ───────────────► │  Supabase        │
│  (ui/app.py)         │  pretrain /     │  (api/main.py)            │   full access    │  Postgres        │
│  thin client, no ML  │  finetune /     │  causal char Transformer  │                  │  datasets/runs/  │
│  bar chart + gen text│  generate       │  (from scratch, torch)    │                  │  run_artifacts/  │
│                      │ ◄────anon key,  │                          │                  │  sequence_       │
│                      │   read-only ────┼──────────────────────────┼──────────────────►│  metadata        │
└──────────────────────┘   SELECT runs   └──────────────────────────┘                  │  (RLS: anon can  │
                                                                                        │   only SELECT    │
                                                                                        │   runs)          │
                                                                                        └──────────────────┘
```

## Project structure

```
fine-it/
├── README.md                 # This file
├── MODEL_CARD.md             # Model details, transfer learning, privacy, limitations
├── shared/
│   ├── schemas.py            # Pydantic API contract
│   └── data.py               # Synthetic dialect (Markov) generator + class contract
├── api/                      # FastAPI tier (deploys to Render)
│   ├── main.py               # Endpoints
│   ├── training.py           # Char Transformer (LM + class heads) + artifact (de)serialization
│   ├── db.py                 # Supabase (service-role) data access
│   ├── configs/default.yaml
│   └── requirements.txt
├── ui/
│   ├── app.py                # 7-tab thin client (pretrain → finetune bar → generate → predict)
│   ├── requirements.txt      # No torch (pandas only)
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
cd fine-it

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
| `POST` | `/datasets` | Register a synthetic corpus (params only) |
| `POST` | `/pretrain` | Pretrain the next-char LM, persist a pretrain run + artifact |
| `POST` | `/finetune` | Fine-tune (warm-start + scratch), persist a finetune run, return both |
| `GET`  | `/runs/{run_id}` | Fetch one run |
| `GET`  | `/runs` | List recent runs |
| `POST` | `/generate` | Temperature-sample characters from a pretrain run's LM |
| `POST` | `/predict` | Classify a character sequence's dialect; log the hash |
| `POST` | `/predict_sample` | Generate + classify a sample |
| `GET`  | `/classes` | The dialect labels the model predicts over |
| `GET`  | `/healthz` | Liveness / DB ping |
| `GET`  | `/version` | Build SHA + torch version |

## Checklist

- [ ] Three live URLs listed at the top of this README
- [ ] `datasets`, `runs` (with `run_type`), `run_artifacts`, `sequence_metadata` tables with RLS
- [ ] 6+ API endpoints
- [ ] 7 Streamlit tabs (Concepts, Pretrain, Fine-tune, Generate, Predict, Run History, Model Card)
- [ ] Pretrain LM (held-out loss) → fine-tune with pretrained-vs-scratch accuracy gap
- [ ] pytest suite passing
- [ ] `MODEL_CARD.md` completed
