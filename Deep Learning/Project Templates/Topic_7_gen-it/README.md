# Gen-It — Three-Cloud Template (Variational Autoencoder)

> The same three-cloud architecture as the base template
> (**Streamlit UI ⇄ FastAPI Model API ⇄ Supabase Data**), with the middle box
> swapped for a **variational autoencoder (VAE)** that learns a **2-D latent
> space** over synthetic images and then generates, reconstructs, and
> interpolates in it. The deploy steps are identical, so follow the main
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

Gen-It demonstrates **latent-variable generative modelling**. A VAE compresses
each 28×28 synthetic shape into a **2-D latent** vector and reconstructs it,
trained **unsupervised** by maximising the ELBO (reconstruction + KL). Because
the latent space is 2-D, the UI exposes it four ways: **Generate** decodes a
point you pick with two sliders (or a prior sample); **Reconstruct** encodes an
image to its posterior mean and decodes it, scoring the pixel error;
**Interpolate** walks a straight line between two images' latents; and **Latent
Scatter** plots where each (known) class lands. You pick the hyperparameters; the
API trains with Adam, reports held-out **reconstruction loss / KL / ELBO**, and
persists every run. Only an image's **sha256 hash** + shape + reconstruction MSE
is logged — never the pixels.

## What changed from the base template (the reusable pattern in action)

The three-cloud split and the file layout are identical to *Regress-It*. Only the
middle box changed:

| Aspect | Regress-It | Gen-It |
|--------|----------------------|---------------------|
| Model | `nn.Linear(1,1)` + SGD | MLP **VAE** (2-D latent) + Adam |
| Task | regression | **unsupervised** reconstruction + generation |
| Metrics | MSE / MAE / R² | reconstruction loss / KL / **ELBO** |
| Input | one scalar `x` | a 28×28 image |
| Interactivity | — | **latent sliders** + **interpolation** + **2-D latent scatter** |
| Extra endpoints | — | `/generate`, `/reconstruct`, `/interpolate`, `/latent_scatter`, `/classes` |
| Privacy invariant | — | store image **hash + recon error**, never pixels |
| Tables | datasets · runs · predictions | datasets · runs · **run_artifacts** · image_metadata |

Everything else — UI as a thin client (no torch), API as the only writer,
Supabase as the single source of truth, service-role vs anon keys, RLS on
`runs`, the four test categories — is unchanged.

## Architecture

```
┌──────────────────────┐   HTTPS/JSON    ┌──────────────────────────┐   service-role   ┌──────────────────┐
│  Streamlit Cloud     │ ──────────────► │  FastAPI on Render        │ ───────────────► │  Supabase        │
│  (ui/app.py)         │  generate /     │  (api/main.py)            │   full access    │  Postgres        │
│  thin client, no ML  │  reconstruct /  │  variational autoencoder  │                  │  datasets/runs/  │
│  sliders + scatter   │  interpolate    │  (from scratch, torch)    │                  │  run_artifacts/  │
│                      │ ◄────anon key,  │                          │                  │  image_metadata  │
│                      │   read-only ────┼──────────────────────────┼──────────────────►│  (RLS: anon can  │
└──────────────────────┘   SELECT runs   └──────────────────────────┘                  │   only SELECT    │
                                                                                        │   runs)          │
                                                                                        └──────────────────┘
```

## Project structure

```
gen-it/
├── README.md                 # This file
├── MODEL_CARD.md             # Model details, VAE, privacy, limitations
├── shared/
│   ├── schemas.py            # Pydantic API contract
│   └── data.py               # Synthetic shape-image generator + class contract
├── api/                      # FastAPI tier (deploys to Render)
│   ├── main.py               # Endpoints
│   ├── training.py           # VAE (encoder/decoder) + latent operations + artifact (de)serialization
│   ├── db.py                 # Supabase (service-role) data access
│   ├── configs/default.yaml
│   └── requirements.txt
├── ui/
│   ├── app.py                # 8-tab thin client (train → generate → reconstruct → interpolate → scatter)
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
cd gen-it

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
| `POST` | `/datasets` | Register a synthetic image dataset (params only) |
| `POST` | `/train` | Train the VAE, persist a run + artifact, return metrics |
| `GET`  | `/runs/{run_id}` | Fetch one run |
| `GET`  | `/runs` | List recent runs |
| `POST` | `/generate` | Decode a chosen latent point (or a prior sample) |
| `POST` | `/reconstruct` | Encode a synthetic sample, decode it, log the hash |
| `POST` | `/interpolate` | Decode a straight-line latent walk between two images |
| `POST` | `/latent_scatter` | Encode a labelled batch to 2-D latent coordinates |
| `GET`  | `/classes` | The class labels used to colour the latent scatter |
| `GET`  | `/healthz` | Liveness / DB ping |
| `GET`  | `/version` | Build SHA + torch version |

## Checklist

- [ ] Three live URLs listed at the top of this README
- [ ] `datasets`, `runs`, `run_artifacts`, `image_metadata` tables with RLS
- [ ] 6+ API endpoints
- [ ] 8 Streamlit tabs (Concepts, Train, Generate, Reconstruct, Interpolate, Latent Scatter, Run History, Model Card)
- [ ] VAE trains unsupervised (held-out recon loss / KL / ELBO)
- [ ] pytest suite passing
- [ ] `MODEL_CARD.md` completed
