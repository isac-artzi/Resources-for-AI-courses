# TokenForge — Topic 1 starter skeleton

> **A text-preprocessing and tokenization service on the three-cloud stack.**
> Streamlit Community Cloud (UI) ⇄ FastAPI on Render.com (API) ⇄ Supabase Postgres (data).
>
> This is a **skeleton, not a solution.** The infrastructure is finished and
> deployable as-is. The NLP layer — every function in `api/nlp.py` — is yours.

This is also the **base template** for the rest of the course. Topics 2–7 reuse
this exact layout with the middle box swapped, so the hour you spend
understanding it now is an hour you do not spend again in Topic 5.

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
| **API** | `api/main.py` routes, CORS, `/healthz`, `/version`, error handling | every function in `api/nlp.py` |
| **Data** | `api/db.py`, the schema in `db/migrations/001_init.sql`, RLS policies | nothing — but read the SQL comments |
| **UI** | `ui/app.py`: all five tabs, API client, error states, history query | the **Concepts** tab content |
| **Ops** | `render.yaml`, `.env.example`, secrets example, `.gitignore` | your actual secrets (never committed) |
| **Tests** | infrastructure + schema tests that pass on a fresh fork | make the `contract` tests pass |
| **Docs** | this file, `MODEL_CARD.md` scaffold | fill in the model card |

If you catch yourself building a database helper or a health endpoint, stop —
it already exists, and the time belongs to the NLP work instead.

---

## Get it running (about 20 minutes)

### 1. Fork and install

```bash
git clone <your-fork-url> && cd Topic_1_tokenforge
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

### 2. Prove the skeleton works before you change anything

```bash
pytest -m "not contract"
```

Everything should pass. If it doesn't, fix your environment now — debugging a
broken install while also debugging your first tokenizer is twice the work.

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

The UI works immediately. The product tabs will say "not implemented yet" until
you write the code — the API returns `501` for a `NotImplementedError`, and the
UI renders that as a message pointing at the function you still owe.

### 5. Wire up Supabase

1. Create a project at supabase.com (free tier is fine).
2. SQL Editor → New query → paste `db/migrations/001_init.sql` → Run.
3. `cp .env.example .env` and fill in the URL and the **service-role** key.
4. `python db/seed.py` to put two demo rows in the History tab.
5. `cp ui/.streamlit/secrets.toml.example ui/.streamlit/secrets.toml` and fill in
   the URL and the **anon** key.

> Two different keys, on purpose. The service-role key can write and lives only
> on the API side. The anon key can only `SELECT` from `runs`, because that is
> the only policy the migration creates, and it is the one the browser sees.

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

`api/nlp.py` is written to be worked top to bottom. Each function is small.

1. `word_tokenize` — the hardest easy function in the file. Contractions,
   hashtags, emoji, and URLs all break the obvious answer.
2. `strip_punctuation`, `remove_stopwords`, `stem`, `lemmatize` — one rule each.
3. `preprocess` — composition plus the `steps_applied` audit trail.
4. `load_tokenizer`, `subword_tokenize` — the Hugging Face side. Cache the
   tokenizer; a per-request load will make Render feel broken.
5. `vocabulary_overlap` — Jaccard over the piece sets.

---

## Things that will cost you an evening if nobody tells you

- **Render free tier sleeps.** The first request after idle takes 30–60 seconds.
  Your UI is not broken; it is waking up. Say so in your demo.
- **Load tokenizers once.** Module-level cache. Otherwise every keystroke in the
  Compare tab re-downloads a vocabulary.
- **Do not put the service-role key in `secrets.toml`.** It ends up in the
  browser. Anyone can then write to your database.
- **Do not disable RLS to make the History tab work.** If the tab is empty, the
  policy is right and your insert is failing — check the API logs.
- **A near-zero OOV rate is correct.** Subword tokenizers rarely emit `[UNK]`.
  Look at the *piece count* instead: that is where rare words show up.
- **Commit `.env.example`, never `.env`.** Both are in this repo's `.gitignore`
  for the second one.

---

## What you submit

The assignment's deliverables list is in the course assignment policy. This
repository is where the README, the model card, and the two live URLs live —
make sure they are current before you submit, because they are the first thing
opened.
