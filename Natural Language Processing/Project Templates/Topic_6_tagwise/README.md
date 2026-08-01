# TagWise — Topic 6 starter skeleton

> **A part-of-speech tagging service on the three-cloud stack.**
> Streamlit Community Cloud (UI) ⇄ FastAPI on Render.com (API) ⇄ Supabase Postgres (data).
>
> This is a **skeleton, not a solution.** The infrastructure is finished and
> deployable as-is. The NLP layer — every function in `api/nlp.py` — is yours.

The layout is the base template from Topic 1 with the middle box swapped. If you
worked through that one, the API, the database layer, and the deploy config will
look familiar and you can go straight to `api/nlp.py`.

---

## Live deployment URLs (fill these in)

| Tier | Platform | URL |
|------|----------|-----|
| **UI** | Streamlit Community Cloud | `https://<your-app>.streamlit.app` |
| **API** | Render.com | `https://<your-api>.onrender.com` |
| **Data** | Supabase | `https://<your-project-ref>.supabase.co` |

Replace these with your real URLs before you submit. Graders open them.

---

## What you are building, and why the baseline is a lookup table

Two taggers, evaluated on the same split with the same tag set:

* **The baseline** counts how often each word takes each tag in the training
  split, remembers the winner, and handles words it has never seen with a short
  list of readable rules (does it contain digits? does it start with a capital?
  does it end in `-ly`?).
* **The transformer** is a token classifier fine-tuned on the same data.

The baseline is a lookup table on purpose, and the choice is the point of the
topic. It needs no training — you build it with a counter and it is finished in
a second. It is completely inspectable — you can print it, and when it tags a
word wrong you can find the exact entry that did it. And it is already good: on
ordinary English a most-frequent-tag lookup gets roughly nine words in ten
right. Measure your own number and report it.

A strong, transparent opponent is worth far more here than a weak one. Because
the baseline is right most of the time, the interesting question is not "does the
transformer win" but *which words does the lookup miss, and why can it never get
them*. The answer is always the same shape:

> "book" is a NOUN in *read the book* and a VERB in *book me a flight*. The
> lookup table maps a word to a tag. It has one entry for "book". It must
> therefore be wrong about one of those two sentences, and no amount of extra
> training data can fix that, because the fix requires looking at the words on
> either side.

That is the argument the whole topic exists to make: context is what a
context-free tagger cannot buy, and a contextual model is what buys it. Your
report shows the gap with a confusion matrix and at least three worked examples,
then measures how much of it your transformer actually closes.

---

## What is given, and what is yours

| | Given, working | Yours to write |
|---|---|---|
| **API** | `api/main.py` routes, CORS, `/healthz`, `/version`, `/tag`, `/runs`, error handling | every function in `api/nlp.py` |
| **Data** | `api/db.py`, the two-table schema in `db/migrations/001_init.sql`, RLS policies | nothing — but read the SQL comments |
| **UI** | `ui/app.py`: all five tabs, tag colouring, confusion-matrix rendering, history query | the **Concepts** tab content |
| **Ops** | `render.yaml`, `.env.example`, secrets example, `.gitignore` | your actual secrets (never committed) |
| **Tests** | infrastructure + schema tests that pass on a fresh fork | make the `contract` tests pass |
| **Docs** | this file, `MODEL_CARD.md` scaffold | fill in the model card |

If you catch yourself building a database helper or a health endpoint, stop —
it already exists, and the time belongs to the NLP work instead.

---

## Get it running (about 20 minutes)

### 1. Fork and install

```bash
git clone <your-fork-url> && cd Topic_6_tagwise
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

### 2. Prove the skeleton works before you change anything

```bash
pytest -m "not contract"
```

Everything should pass. If it doesn't, fix your environment now — debugging a
broken install while also debugging your first tagger is twice the work.

### 3. See what you have to build

```bash
pytest -m contract
```

Every failure is a function in `api/nlp.py` with a docstring explaining what it
must return. That list is your assignment. Add `-m "contract and not network"`
to skip the two that need a fine-tuned model on disk.

### 4. Run the two tiers locally

```bash
uvicorn api.main:app --reload            # terminal 1 → http://127.0.0.1:8000/docs
streamlit run ui/app.py                  # terminal 2 → http://localhost:8501
```

The UI works immediately. The product tabs will say "not implemented yet" until
you write the code — the API returns `501` for a `NotImplementedError`, and the
UI renders that as a message pointing at the function you still owe.

### 5. Get the corpus

Download the train, dev, and test `.conllu` files for an English treebank from
<https://universaldependencies.org/#english-treebanks> into `data/` (gitignored),
and put the treebank's name and revision in `api/configs/default.yaml`. Run
`describe_corpus` on all three splits and paste the numbers into your README and
model card before you build anything.

### 6. Wire up Supabase

1. Create a project at supabase.com (free tier is fine).
2. SQL Editor → New query → paste `db/migrations/001_init.sql` → Run.
3. `cp .env.example .env` and fill in the URL and the **service-role** key.
4. `python db/seed.py` to put demo rows in the History and comparison tabs.
5. `cp ui/.streamlit/secrets.toml.example ui/.streamlit/secrets.toml` and fill in
   the URL and the **anon** key.

> Two different keys, on purpose. The service-role key can write and lives only
> on the API side. The anon key can only `SELECT`, because those are the only
> policies the migration creates, and it is the one the browser sees.

### 7. Deploy

**API → Render.** New → Blueprint → point at your fork. `render.yaml` does the
rest. Afterwards, set `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` in the service's
environment settings.

**UI → Streamlit Community Cloud.** New app → your fork → main file
`ui/app.py`. Paste the three keys into the Secrets box.

Confirm both `GET /healthz` and `GET /version` answer from the live Render URL,
then record all three URLs in the table at the top of this file.

---

## Two tables, and why

`runs` holds one row per tagger **build**: the lookup table you built, the
transformer you fine-tuned, with hyperparameters, accuracy, macro-F1, and the
confusion matrix. A handful of rows for the whole project.

`taggings` holds one row per **served request**: the hashed sentence, the
predicted tag sequence, which model answered, the timestamp. Hundreds of rows,
growing every time someone uses the service.

The History tab reads `taggings`. It has to: a build log is not a request log. If
you log only runs, the tab has no data source and shows four rows that say
nothing about what the service has actually done — you would not be able to
answer "how many sentences did we tag last week", "how often did the baseline hit
a word it had never seen in real traffic", or "did anything change when v2 went
out". Those are questions about requests, and only the request log has them.

Neither table holds a sentence. Both hold a sha256 of one.

---

## The order to implement in

`api/nlp.py` is written to be worked top to bottom.

1. `load_treebank`, `describe_corpus` — the corpus, its tag set, its splits, and
   its tag distribution. Do this first; the distribution tells you the floor.
2. `build_lookup_table`, `fallback_tag`, `tag_with_lookup` — the whole baseline.
   It is short, and by the end of it you have a working tagger.
3. `align_tags_to_subwords` — read the docstring twice. This one fails silently.
4. `fine_tune_transformer`, `load_tagger_model`, `tag_with_transformer` — run the
   fine-tune from your own script, offline. A web request must never train.
5. `split_words`, `tag_sentence` — serving. Now POST /tag answers.
6. `token_accuracy`, `per_tag_f1`, `macro_f1`, `confusion_matrix`,
   `evaluate_tagger` — the numbers, and the figure your report is built around.

---

## Things that will cost you an evening if nobody tells you

- **Subword pieces are not words.** The tokenizer turns "unhappiness" into three
  pieces; your gold data has one tag for it. Align the tag to the **first piece
  of each word and mask the rest with `-100`**, and at prediction time read the
  output at the first piece only. Get this wrong and nothing crashes: the model
  trains, the loss falls, and accuracy comes out mediocre in a way that looks
  like "needs more epochs" rather than "the labels are shifted by one word".
  Before training, print two sentences as a table of piece, word id, and label
  and check it by eye.
- **A badly fine-tuned transformer loses to the lookup table.** This is not a
  hypothetical. A ninety-per-cent baseline is a high bar, and three epochs at the
  wrong learning rate will not clear it. If it happens to you, report the number
  and your diagnosis. That is a finding. Re-running until the graph looks right
  and reporting only that run is the thing this course is trying to train out of
  you.
- **Nearly all of the baseline's remaining error is unknown words.** The table
  handles seen words almost perfectly by construction, so the error concentrates
  on words it has never seen — new proper nouns, technical terms, typos,
  anything from a different domain. Measure your accuracy on unknown tokens
  separately from known ones. The two numbers are far apart, and the gap is your
  best argument for everything else in the topic.
- **Macro-F1 is dominated by your rare tags.** It averages per-tag F1 without
  weighting, so a tag with twenty instances counts as much as one with twenty
  thousand. A model can be right on nine tokens in ten and still score under 0.5
  if it never predicts one rare tag. Report accuracy and macro-F1 together, look
  at the per-tag table before you explain a gap, and expect macro-F1 to bounce
  between runs for reasons that are about the rare tags, not the model.
- **Use the 17-tag universal set.** Seventeen tags make a 17x17 confusion matrix:
  289 cells, readable at a glance, and the pairs you care about are visible in
  it. A 45-tag set gives you 2,025 cells that nobody reads, splits your training
  data more thinly, and buries NOUN/VERB inside distinctions no part of this
  product uses.
- **Build the lookup table on the training split only.** Building it on all the
  data gives you an excellent, meaningless accuracy number, and nothing crashes
  to tell you.
- **Match the treebank's tokenization.** The treebank splits "don't" into "do" +
  "n't" and separates punctuation. If your `split_words` disagrees, your live
  demo sees words your models never trained on, and the errors will not appear in
  any of your test numbers.
- **Render free tier sleeps.** The first request after idle takes 30–60 seconds,
  and loading a transformer on top of that is slower still. Your UI is not
  broken; it is waking up. Say so in your demo.
- **Load the model once.** Module-level cache. A per-request load will run the
  free instance out of memory, and the baseline should keep answering when the
  transformer cannot load.
- **Do not commit weights or treebank files.** `data/` and `models/` are
  gitignored. Push the model to the Hugging Face Hub and point the API at it.
- **Do not put the service-role key in `secrets.toml`.** It ends up in the
  browser. Anyone can then write to your database.
- **Do not disable RLS to make the History tab work.** If the tab is empty, the
  policy is right and your insert is failing — check the API logs.
- **Commit `.env.example`, never `.env`.** Both are in this repo's `.gitignore`
  for the second one.

---

## What you submit

The assignment's deliverables list is in the course assignment policy. This
repository is where the README, the model card, the confusion matrices, the
worked examples of ambiguity, and the two live URLs live — make sure they are
current before you submit, because they are the first thing opened.
