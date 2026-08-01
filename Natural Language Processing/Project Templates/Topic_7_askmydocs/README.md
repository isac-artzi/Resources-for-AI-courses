# AskMyDocs — Topic 7 starter skeleton

> **A retrieval-augmented question-answering service on the three-cloud stack.**
> Streamlit Community Cloud (UI) ⇄ FastAPI on Render.com (API) ⇄ Supabase
> Postgres with pgvector (data).
>
> This is a **skeleton, not a solution.** The infrastructure is finished and
> deployable as-is. The NLP layer — every function in `api/nlp.py` — is yours.

The layout is the same one you have used since Topic 1: same tiers, same
`shared/schemas.py` contract, same test markers, same deploy story. What is new
is the middle box. Instead of a model that classifies or tags a string, you are
building a pipeline: chunk a document collection, embed it, store the vectors,
retrieve the closest passages for a question, and condition a language model on
them. This is also the bridge into the conversational and agentic topics later
in the program, where retrieval and tool use get developed properly.

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
| **API** | `api/main.py` routes, CORS, `/healthz`, `/version`, the corpus-separation guards, error handling | every function in `api/nlp.py` |
| **Data** | `api/db.py`, the five-table schema in `db/migrations/001_init.sql`, the `match_chunks` SQL function, the vector index, RLS policies | nothing — but read the SQL comments |
| **UI** | `ui/app.py`: all four tabs, API client, the with/without comparison view, the audit query | the **Concepts** tab content |
| **Ops** | `render.yaml`, `.env.example`, secrets example, `.gitignore` | your actual secrets (never committed) |
| **Tests** | infrastructure + schema tests that pass on a fresh fork | make the `contract` tests pass |
| **Docs** | this file, `MODEL_CARD.md` scaffold | fill in the model card |

If you catch yourself writing a database helper, a health endpoint, or a
similarity query in SQL, stop — it already exists, and the time belongs to the
NLP work instead.

---

## The one rule that is not negotiable

**The corpus you fine-tune the generator on and the collection you retrieve from
must be disjoint.**

Here is why, and it is worth reading before you choose your documents, because
choosing badly is expensive to undo. The product's central claim is a
comparison: the same question, answered by the same model, once with retrieved
passages in the prompt and once without. If the generator was fine-tuned on the
very documents the retriever later hands it, then the "without retrieval" answer
is already correct — the model memorised it during training. Retrieval then
appears to add nothing at all. Your comparison table shows two identical columns,
and you cannot tell whether that is because retrieval does not help or because
you contaminated the experiment. You will have measured your own bookkeeping.

This template enforces the rule in three independent places, and you should not
weaken any of them to make a demo work:

1. **`api/main.py`** refuses a `POST /embed` whose documents already exist in the
   other corpus, with a 409 and a message naming the document. It refuses before
   a single vector is written, so there is never a half-contaminated store.
2. **`nlp.check_corpus_disjointness`** catches the near-duplicate: the same
   document reformatted, a PDF and its HTML version, a note with the header
   stripped. None of those share a hash and every one of them contaminates the
   experiment exactly as badly as a literal copy.
3. **The database** has a unique index on `documents.content_sha256`, so the same
   text cannot exist in both corpora even if the API check is bypassed.

Perplexity follows the same rule: report it on a **held-out split of the
fine-tuning corpus**, split by document rather than inside one. Perplexity on
data the model was trained on measures memorisation and will look excellent.

---

## Get it running (about 30 minutes, plus the install)

### 1. Fork and install

```bash
git clone <your-fork-url> && cd Topic_7_askmydocs
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

This one is a big install — `torch` is most of it. Expect several minutes and a
couple of gigabytes.

### 2. Prove the skeleton works before you change anything

```bash
pytest -m "not contract"
```

Everything should pass. If it doesn't, fix your environment now — debugging a
broken install while also debugging your first retriever is twice the work.

### 3. See what you have to build

```bash
pytest -m contract
```

Every failure is a function in `api/nlp.py` with a docstring explaining what it
must return. That list is your assignment. On a train with no signal, use
`pytest -m "contract and not network"` for the subset that needs no model.

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
   **The first line enables pgvector.** If you skip it or run the file in pieces,
   every `vector(384)` column fails with "type vector does not exist" and you
   end up with a `documents` table and no `chunks` table.
3. `cp .env.example .env` and fill in the URL and the **service-role** key.
4. `python db/seed.py` to put demo rows in the Retrieval Audit tab.
5. `cp ui/.streamlit/secrets.toml.example ui/.streamlit/secrets.toml` and fill in
   the URL and the **anon** key.

> Two different keys, on purpose. The service-role key can write and lives only
> on the API side. The anon key can only `SELECT`, because those are the only
> policies the migration creates, and it is the one the browser sees.

### 6. Ingest your collection

Send documents to `POST /embed` with `"corpus": "retrieval"`. Send your
fine-tuning corpus with `"corpus": "finetune"` — it is registered but never
embedded, because nothing should be able to retrieve it. Then check `GET
/sources`, or the sidebar in the UI, which flags any title that appears on both
sides.

### 7. Deploy

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

1. `load_documents`, `count_tokens` — get the collection in and measure it. Do
   this first: the "at least 50 documents, and here is the total token count"
   claim in your report should be a number you printed.
2. `chunk_spans` — pure arithmetic, no models, fully testable offline. Get the
   overlap right here and `chunk_document` becomes easy.
3. `chunk_document` — window over token ids, decode back to text. Read ten of
   your own chunks before you embed five thousand.
4. `embedding_dimension`, `embed_texts` — and normalise the vectors.
5. `cosine_similarity` — twelve lines, and writing it once stops the formula
   being a black box.
6. `vector_search` — embed the query with the same model, call
   `api.db.match_chunks`, return ranked results.
7. `build_prompt`, `generate` — the language-model half.
8. `answer_question` — compose it all, and fill in every field of the response.
9. `perplexity` — on the held-out fine-tuning split.
10. `normalise_for_comparison`, `shingles`, `check_corpus_disjointness` — the
    evidence that steps 1 and 9 were about two different sets of documents.

---

## Things that will cost you an evening if nobody tells you

- **Embedding dimensions must match, and a mismatch fails at query time, not at
  write time.** The `chunks.embedding` column is `vector(384)` because
  `all-MiniLM-L6-v2` produces 384 numbers. Swap to a 768-dimensional model and
  change it in only one place, and the write may succeed while the query later
  raises "different vector dimensions 768 and 384" from inside a user request —
  days after the change that caused it. Change it in three places at once: the
  migration's column, the `match_chunks` signature, and `nlp.EMBEDDING_DIM`. Then
  re-embed the whole collection; old vectors cannot be converted.

- **Cosine similarity over un-normalised vectors is not cosine similarity.**
  `cos(a, b) = (a·b) / (|a||b|)`. Drop the denominator and you have a dot product
  that scores long passages higher purely for being long. The scores still look
  reasonable — 0.7, 0.8, nothing alarming — and your top-5 is quietly ranked by
  passage length. Normalise once in `embed_texts` and the problem cannot recur
  downstream. pgvector's `<=>` handles it for you; your own reranker, your local
  ChromaDB run, and your notebook do not.

- **A chunk boundary that splits a sentence will retrieve a passage that does not
  contain the answer.** One chunk ends with "the contract was terminated in" and
  the next starts with "March 2021, after the third". Neither contains the fact.
  Both embed fine, both get retrieved with respectable scores, and the generator
  invents the missing half — with a citation, which makes the invention look
  sourced. The 10–20 percent overlap exists for exactly this. Snapping boundaries
  to sentence ends helps further, at the cost of uneven chunk lengths. Whatever
  you do, print ten chunks and read them.

- **A retrieved passage the model then ignores looks identical to a retrieval
  failure.** Same wrong answer, completely different fix: one is a chunking or
  embedding problem, the other is a prompting problem. The only way to tell them
  apart is the `retrievals` table sitting next to the `answers` table, which is
  why the schema has both and why `cited_chunk_ids` is a required field rather
  than a nicety. Check the gap between retrieved and cited in the Ask tab.

- **Free-tier memory will not hold a large generator.** A 7B model does not load
  in Render's free plan, and the failure mode is a silent restart with no useful
  log line — the deploy just says the service failed. Either use a small causal
  model you have watched start (the GPT-2 family, or a small instruction-tuned
  model in the 100M–500M range) or call a hosted inference API and keep only the
  embedding model in your process. Say which you chose in the MODEL_CARD, and say
  what memory you actually observed.

- **pgvector needs the extension enabled before the migration will run.**
  `create extension if not exists vector;` is the first line of
  `001_init.sql` for that reason. Supabase ships the extension but does not turn
  it on in a new project. If you paste the file in pieces, or start from the
  `documents` table, the `vector(384)` column fails and everything downstream of
  it silently does not exist.

- **Render free tier sleeps.** The first request after idle takes 30–60 seconds,
  and this service also has to load models. Your UI is not broken; it is waking
  up. Say so in your demo.

- **Load models once.** Module-level cache, and import them inside functions so
  `/healthz` does not pay for `import torch` on every cold start.

- **Do not put the service-role key in `secrets.toml`.** It ends up in the
  browser. Anyone can then write to your database.

- **Do not disable RLS to make the Retrieval Audit tab work.** If the tab is
  empty, the policy is right and your insert is failing — check the API logs.

- **The `chunks` table is anon-readable, so your passages are public.** That is
  the correct trade for an auditable demo built on public documents and the wrong
  one for a real knowledge base. Choose your collection accordingly.

- **Commit `.env.example`, never `.env`.** Both are in this repo's `.gitignore`
  for the second one.

---

## What you submit

The assignment's deliverables list is in the course assignment policy. This
repository is where the README, the model card, and the two live URLs live —
make sure they are current before you submit, because they are the first thing
opened.

Two things the rubric will look for that are easy to leave until last: the
comparison table of at least ten questions answered with and without retrieval,
with three cases where the retrieved passage prevented a hallucination and the
passage quoted; and the evidence that your two corpora were disjoint. Both are
much easier to produce while you are building than to reconstruct afterwards.

---

## Reading

- Jurafsky, D., & Martin, J. H. *Speech and Language Processing* (3rd ed. draft),
  chapters 3 and 9, and chapter 10 for review.
  https://web.stanford.edu/~jurafsky/slp3/
- Vaswani, A., et al. (2017). Attention Is All You Need. https://arxiv.org/abs/1706.03762
- Alammar, J. (2018). The Illustrated Transformer.
  https://jalammar.github.io/illustrated-transformer/
- Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive
  NLP Tasks. https://arxiv.org/abs/2005.11401
