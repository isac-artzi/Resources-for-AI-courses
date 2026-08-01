# GenText — Topic 3 starter skeleton

> **A controllable text-generation service on the three-cloud stack.**
> Streamlit Community Cloud (UI) ⇄ FastAPI on Render.com (API) ⇄ Supabase Postgres (data).
>
> This is a **skeleton, not a solution.** The infrastructure is finished and
> deployable as-is. The generation layer — every function in `api/nlp.py` — is
> yours.

Same layout as Topic 1, with the middle box swapped: a pretrained decoder
transformer instead of a tokenizer. If you understood the TokenForge template,
you already know where everything in this one lives.

---

## Live deployment URLs (fill these in)

| Tier | Platform | URL |
|------|----------|-----|
| **UI** | Streamlit Community Cloud | `https://<your-app>.streamlit.app` |
| **API** | Render.com | `https://<your-api>.onrender.com` |
| **Data** | Supabase | `https://<your-project-ref>.supabase.co` |
| **Model** | Hugging Face Hub | `https://huggingface.co/<you>/<your-checkpoint>` |

Replace these with your real URLs before you submit. Graders open them, and they
check `GET /healthz` and `GET /version` against the API one.

---

## What is given, and what is yours

| | Given, working | Yours to write |
|---|---|---|
| **API** | `api/main.py`: `/generate`, `/rate`, `/history`, `/healthz`, `/version`, CORS, error handling | every function in `api/nlp.py` |
| **Data** | `api/db.py`, the two-table schema in `db/migrations/001_init.sql`, RLS policies, `db/seed.py` | nothing — but read the SQL comments |
| **UI** | `ui/app.py`: all four tabs, decoding controls, rating form, error states | the **Concepts** tab content |
| **Ops** | `render.yaml`, `.env.example`, secrets example, `.gitignore` | your actual secrets (never committed) |
| **Tests** | infrastructure + schema tests that pass on a fresh fork | make the `contract` tests pass |
| **Docs** | this file, `MODEL_CARD.md` scaffold | fill in the model card |

If you catch yourself building a database helper, a health endpoint, or a
Streamlit slider, stop — it already exists, and the time belongs to the decoding
work instead.

---

## Get it running (about 20 minutes)

### 1. Fork and install

```bash
git clone <your-fork-url> && cd Topic_3_gentext
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

That pulls in torch, which is a large download. Start it and read the next
section while it runs.

### 2. Prove the skeleton works before you change anything

```bash
pytest -m "not contract and not cloud"
```

Everything should pass. If it doesn't, fix your environment now — debugging a
broken install while also debugging your first sampling loop is twice the work.

### 3. See what you have to build

```bash
pytest -m "contract and not network and not train"
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
2. SQL Editor → New query → paste `db/migrations/001_init.sql` → Run. It creates
   **both** tables; do not stop halfway.
3. `cp .env.example .env` and fill in the URL and the **service-role** key.
4. `python db/seed.py` to put one training run and three generations in the
   History tab, one of them already rated by two people who disagreed.
5. `cp ui/.streamlit/secrets.toml.example ui/.streamlit/secrets.toml` and fill in
   the URL and the **anon** key.

> Two different keys, on purpose. The service-role key can write and lives only
> on the API side. The anon key can only `SELECT`, because that is the only
> policy the migration creates, and it is the one the browser sees. That is also
> why the rating form posts to `/rate` instead of writing to Postgres directly.

### 6. Train offline, then point the service at the result

Fine-tuning does **not** happen on Render. Run `api/nlp.fine_tune` on your own
machine or a free GPU notebook, push the checkpoint to the Hugging Face Hub, and
write the training run to Supabase:

```python
from api import db
from api.nlp import fine_tune, load_corpus
from shared.schemas import TrainingConfig

corpus = load_corpus("data/corpus.txt", source="...", domain="...")
run = fine_tune(corpus, TrainingConfig(base_model="gpt2",
                                       model_version="gentext-reviews-v2"))
db.insert_training_run(base_model=run.base_model,
                       model_version=run.model_version,
                       hyperparameters=run.hyperparameters,
                       corpus_source=corpus.source,
                       corpus_sha256=corpus.sha256,
                       corpus_sentence_count=corpus.sentence_count,
                       held_out_perplexity=run.held_out_perplexity)
```

Then set `MODEL_NAME` to your checkpoint id and `MODEL_VERSION` to the same
string you recorded, in `.env` locally and in the Render dashboard.

### 7. Deploy

**API → Render.** New → Blueprint → point at your fork. `render.yaml` does the
rest. Afterwards, set `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` in the service's
environment settings.

**UI → Streamlit Community Cloud.** New app → your fork → main file
`ui/app.py`. Paste the three keys into the Secrets box.

Confirm both `GET /healthz` and `GET /version` answer from the live Render URL,
then record all the URLs in the table at the top of this file.

---

## The order to implement in

`api/nlp.py` is written to be worked top to bottom. Each function is small.

1. `distinct_n` — ten lines, no model needed, and it gives you a number to look
   at from the very first generation. Start here.
2. `build_generation_kwargs` — the five-branch switch. This is the function the
   whole topic is about; get it right before you generate anything.
3. `load_model` — cached, eval mode, pad token set.
4. `generate` — tokenize, seed, call, decode, strip the prompt, measure.
   At this point the Generate and Compare Decoding tabs come alive.
5. `record_rating` and `rater_agreement` — you need these before your rating
   session, not after it.
6. `load_corpus` — filter, then count, then hash.
7. `perplexity` — on the held-out split, and only then on generated text.
8. `fine_tune` — last, offline, once the rest of the product already works with
   the base model. A working demo on `gpt2` beats a half-finished fine-tune.

---

## Things that will cost you an evening if nobody tells you

- **The free Render instance cannot fine-tune a decoder.** Not slowly — at all.
  It runs out of memory, or the request times out, and the failure looks like a
  network error. Train offline, publish the checkpoint, and let the service load
  it. If your deploy log mentions a training step, you have built something that
  will not deploy.
- **Render free tier sleeps.** The first request after idle takes 30–60 seconds,
  plus however long the weights take to load. Your UI is not broken; it is waking
  up. Say so in your demo instead of clicking again.
- **Temperature 0 is not greedy decoding here.** In the maths it is the limit of
  greedy; in the library it is a division by zero that produces NaNs or an
  exception depending on the version. Hosted APIs that accept `temperature=0`
  quietly branch to greedy internally. Do the same, explicitly.
- **Beam search collapses diversity.** It optimizes for the highest-scoring
  sequence, which for open-ended text is the blandest one, and widening the beam
  makes it blander. If your beam output reads like a form letter, that is beam
  search working correctly — say so rather than trying to fix it with more
  training.
- **A model that repeats itself is a decoding problem, not a training problem.**
  "It is the best because it is the best because it is the best" comes out of
  greedy and narrow-beam decoding, and another epoch will not touch it. Switch to
  sampling, raise `repetition_penalty` slightly, or set `no_repeat_ngram_size` —
  then say which, and what it cost, because `no_repeat_ngram_size` also forbids
  repetition you wanted.
- **Storing the raw prompt after promising to hash it.** It is one line, it makes
  the History tab nicer to read, and it breaks the product's only privacy claim.
  There is a test that fails if a persisted schema grows a `prompt` field. Leave
  it there.
- **Load the model once.** Module-level cache. A per-request load turns a
  two-second generation into a ninety-second one, and you will blame Render.
- **Strip the prompt off the output.** Hugging Face returns prompt +
  continuation. Leave the prompt in and every diversity number is inflated by
  text the model did not write — worst exactly when the output is short and
  repetitive, which is the case you needed to catch.
- **Do not disable RLS to make the History tab work.** If the tab is empty, the
  policy is right and your insert is failing — check the API logs.
- **Perplexity across different tokenizers is not comparable.** Your fine-tune
  against its own base model, yes. Against a model with a different vocabulary,
  no — and that comparison is a common way to claim an improvement that is not
  there.
- **Rate blind.** If the rater can see that an output came from beam search, the
  rating measures their opinion of beam search. Export, shuffle, hide the
  strategy column, then score.
- **Commit `.env.example`, never `.env`.** Both are in this repo's `.gitignore`
  for the second one. The same goes for `artifacts/` and `data/`.

---

## What you submit

The assignment's deliverables list is in the course assignment policy. This
repository is where the README, the model card, and the live URLs live — make
sure they are current before you submit, because they are the first thing opened.

Two things the report needs that are easy to leave until too late: the twenty
rated outputs with two independent scores each, and the disagreement analysis
that comes out of them. Start rating as soon as `generate` works.
