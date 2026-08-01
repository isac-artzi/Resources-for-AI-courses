# EntityFinder — Topic 5 starter skeleton

> **A named-entity-recognition service on the three-cloud stack.**
> Streamlit Community Cloud (UI) ⇄ FastAPI on Render.com (API) ⇄ Supabase Postgres (data).
>
> This is a **skeleton, not a solution.** The infrastructure is finished and
> deployable as-is. The NLP layer — every function in `api/nlp.py` — is yours.

Same layout as the Topic 1 template, with the middle box swapped: instead of one
model producing tokens, two models produce **spans**, and a human gets to
disagree with them. The write path back from that human is the part of this
build that is easy to skip and impossible to fake — a review queue you cannot
write to is not a review queue.

---

## Live deployment URLs (fill these in)

| Tier | Platform | URL |
|------|----------|-----|
| **UI** | Streamlit Community Cloud | `https://<your-app>.streamlit.app` |
| **API** | Render.com | `https://<your-api>.onrender.com` |
| **Data** | Supabase | `https://<your-project-ref>.supabase.co` |

Replace these with your real URLs before you submit. Graders open them.

---

## What is given, and what is yours

| | Given, working | Yours to write |
|---|---|---|
| **API** | `api/main.py` routes, CORS, `/healthz`, `/version`, `/review_queue`, the `/review` write path | every function in `api/nlp.py` |
| **Data** | `api/db.py`, the four-table schema in `db/migrations/001_init.sql`, RLS policies | nothing — but read the SQL comments |
| **UI** | `ui/app.py`: all five tabs, inline highlighting, the review form, API client | the **Concepts** tab content |
| **Ops** | `render.yaml`, `.env.example`, secrets example, `.gitignore` | your actual secrets (never committed) |
| **Tests** | infrastructure, schema, and review-write tests that pass on a fresh fork | make the `contract` tests pass |
| **Docs** | this file, `MODEL_CARD.md` scaffold | fill in the model card |

If you catch yourself building a database helper, a health endpoint, or an HTML
highlighter, stop — they already exist, and the time belongs to the NER work.

---

## Get it running (about 20 minutes)

### 1. Fork and install

```bash
git clone <your-fork-url> && cd Topic_5_entityfinder
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

### 2. Prove the skeleton works before you change anything

```bash
pytest -m "not contract"
```

Everything should pass, including the review round-trip against the in-memory
database. If it doesn't, fix your environment now — debugging a broken install
while also debugging your first tag decoder is twice the work.

### 3. See what you have to build

```bash
pytest -m contract
```

Every failure is a function in `api/nlp.py` with a docstring explaining what it
must return. That list is your assignment.

### 4. Run the two tiers locally

```bash
uvicorn api.main:app --reload            # terminal 1 → http://127.0.0.1:8000/docs
streamlit run ui/app.py                  # terminal 2 → http://localhost:8501
```

The UI works immediately. The Extract tab will say "not implemented yet" until
you write the code — the API returns `501` for a `NotImplementedError`, and the
UI renders that as a message pointing at the function you still owe.

### 5. Wire up Supabase

1. Create a project at supabase.com (free tier is fine).
2. SQL Editor → New query → paste `db/migrations/001_init.sql` → Run.
3. `cp .env.example .env` and fill in the URL and the **service-role** key.
4. `python db/seed.py` to put two training runs and one deliberately
   low-confidence prediction in the database, so the Review Queue tab has
   something to show you before your model does.
5. `cp ui/.streamlit/secrets.toml.example ui/.streamlit/secrets.toml` and fill in
   the URL and the **anon** key.

> Two different keys, on purpose. The service-role key can write and lives only
> on the API side. The anon key can only `SELECT` from `runs`, because that is
> the only policy the migration creates, and it is the one the browser sees.
> That is also why the Review Queue posts to the API instead of writing to
> Postgres directly: an anon INSERT policy would let anyone with your public key
> invent reviewer decisions in your audit trail.

### 6. Deploy

**API → Render.** New → Blueprint → point at your fork. `render.yaml` does the
rest. Afterwards, set `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` in the service's
environment settings.

**UI → Streamlit Community Cloud.** New app → your fork → main file
`ui/app.py`. Paste the three keys into the Secrets box.

Confirm both `GET /healthz` and `GET /version` answer from the live Render URL,
then record all three URLs in the table at the top of this file.

---

## The order to implement in

`api/nlp.py` is written to be worked top to bottom, but the two functions that
need no model at all are the two that decide whether your results mean anything.
Do them first, on a laptop, offline.

1. `decode_bio_spans` — tags in, character spans out. Pure function, fifteen
   minutes, and a bug here is invisible during training and fatal in production.
2. `entity_level_scores` — exact match on (start, end, type), plus the per-type
   breakdown. This is the number the whole report rests on.
3. `profile_dataset` — splits, entity types, label distribution. Count the `O`
   tokens; you will use that number twice.
4. `word_features` / `sentence_features` / `train_crf` — the transparent
   baseline. Prefixes, suffixes, POS tags, and a window over the neighbours.
5. `load_model` / `train_transformer` — align labels to subwords with
   `word_ids()`, fine-tune, evaluate on the same split as the CRF.
6. `predict_bio_tags` / `extract_entities` — the serving path. Offset mappings,
   not counting.

---

## Things that will cost you an evening if nobody tells you

- **Entity-level F1 is not token-level F1.** Most tokens in any NER corpus are
  `O`. Tag everything `O` and token accuracy sails past 90% while the service
  finds nothing. If you report the token number — and the libraries will happily
  hand it to you — you are reporting a different and much easier task than the
  one your product does.
- **A right type with a wrong boundary is a full miss.** Gold says "New York
  Stock Exchange", you say "York Stock Exchange": that is one false positive and
  one false negative, which scores *worse* than predicting nothing. There is no
  partial credit at entity level, and there shouldn't be — a downstream lookup
  gets nothing from a span that is close.
- **Subword tokenization breaks character offsets.** The tokenizer may split,
  lowercase, or strip accents before your model ever sees a word. Get offsets
  from `return_offsets_mapping=True` and map back with them. Do not count
  characters, and do not use `str.find()` — it returns the *first* "Smith" in
  the document every time, so the second one gets the first one's span.
- **The corpus is newswire from decades ago.** It knows the politicians,
  companies, and sports teams that were in the news then. Your users have
  clinical notes, or support tickets, or contracts. The transfer will not be
  quiet or small; measure it on your own text and put the number in the model
  card instead of hoping nobody checks.
- **A softmax score is not a probability of being correct.** It is a normalised
  score over labels, and it is highest exactly where the model is most sure —
  including when the model is sure and wrong. Use it to rank the review queue,
  not to tell a user how likely an answer is to be right.
- **Never overwrite a prediction with its correction.** Once the row says what
  the human said, nobody can tell what the model said, and the pair of the two
  is the single most valuable thing in your database — it is the labelled
  example of a mistake this model makes. Corrections go in `reviews`;
  `entities` is append-only. `api/db.insert_review` explains it at length.
- **Render free tier sleeps.** The first request after idle takes 30–60 seconds,
  and a transformer cold start is on top of that. Your UI is not broken; it is
  waking up. Say so in your demo.
- **Load models once.** Module-level cache. A per-request load will make the
  Extract tab look like it has hung.
- **Do not put the service-role key in `secrets.toml`.** It ends up in the
  browser. Anyone can then write to your database — including to `reviews`.
- **Commit `.env.example`, never `.env`.** Both are in this repo's `.gitignore`
  for the second one.

---

## What you submit

The assignment's deliverables list is in the course assignment policy. This
repository is where the README, the model card, and the two live URLs live —
make sure they are current before you submit, because they are the first thing
opened. Your report needs entity-level precision, recall, and F1 for both
models, and three error categories with a worked example of each; the Review
Queue is where you will find those examples already sorted for you.
