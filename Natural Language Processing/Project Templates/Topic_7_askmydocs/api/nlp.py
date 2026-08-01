"""THE FILE YOU ARE HERE TO WRITE.

Everything else in this template is finished: the FastAPI app boots, the
database layer works, the Streamlit UI renders, the deploy config is correct.
This module is the hole in the middle. Every function below raises
``NotImplementedError`` and every one of them has a docstring telling you what
it must return and why.

HOW TO WORK THROUGH IT
----------------------
1. Run ``pytest -m contract`` — the contract tests fail, listing what's missing.
2. Implement one function.
3. Re-run. Go green. Move to the next.

The order below is the order you should implement in. Each function is small;
if yours is growing past ~40 lines, you are probably solving the next one too.

WHAT "DONE" LOOKS LIKE
----------------------
``pytest`` passes with no skips other than the Supabase round-trip (which needs
real credentials), the Ask tab returns an answer with citations, and the
Retrieval Audit tab shows real similarity scores pulled from Postgres.

THE ONE DESIGN RULE THAT IS NOT NEGOTIABLE
------------------------------------------
The corpus you fine-tune the generator on and the collection you retrieve from
must be DISJOINT. Not "mostly". Disjoint.

Here is why, and it is worth reading twice because the whole assignment rests on
it. The point of this product is a comparison: the same question, answered by
the same model, once with retrieved passages in the prompt and once without. If
the generator was fine-tuned on the very documents the retriever later hands it,
then the "without retrieval" answer is already correct — the model memorised it
during training. Retrieval then appears to add nothing, your comparison shows a
flat line, and you have no way to tell whether that is because retrieval does
not help or because you contaminated the experiment. You will have measured your
own bookkeeping.

``check_corpus_disjointness`` below is the check. ``api/main.py`` calls it on
ingest, ``db/migrations/001_init.sql`` has a unique index that enforces it a
second time at the database, and ``CorpusOverlapError`` is what a violation
raises. Do not weaken any of the three to make a demo work.

A NOTE ON LIBRARIES AND MEMORY
------------------------------
Use ``sentence-transformers`` for the embedding model and Hugging Face
``transformers`` for the generator. Import them INSIDE the functions, not at the
top of this module: a module-level ``import torch`` costs seconds of cold start
on every single request path, including ``/healthz``, and the health check is
the one endpoint that must answer fast.

Render's free plan gives you a small memory allowance. A 7-billion-parameter
generator will not load in it, and the process will be killed with no useful log
line — the deploy just says the service failed. Pick a small causal model you
have actually watched start (the GPT-2 family and the small instruction-tuned
models in the 100M–500M range are the realistic options), or call a hosted
inference API and keep only the embedding model in your own process. Whichever
you choose, name it in MODEL_CARD.md and say which memory it needed.
"""
from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence, Tuple

from shared.schemas import (
    AskResponse,
    ChunkingParams,
    ChunkRecord,
    DisjointnessReport,
    DocumentIn,
    PerplexityReport,
    RetrievedChunk,
)

# ---------------------------------------------------------------------------
# Defaults. Change them if you have a reason, and record the reason in the
# MODEL_CARD — every one of these three strings ends up in a logged row, which
# is what lets you say later "the answers got better at v3" and prove it.
#
# EMBEDDING_MODEL's output dimension must match the vector(...) column in
# db/migrations/001_init.sql. all-MiniLM-L6-v2 produces 384 numbers, which is
# why the migration says vector(384). Swap the model and you must edit the
# migration too — see the warning on embedding_dimension() below.
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
GENERATOR_MODEL = "distilgpt2"
DEFAULT_K = 5

# The instruction that separates a RAG system from a chatbot with extra text in
# the prompt. "If the passages do not contain the answer, say so" is the line
# that turns a hallucination into an honest refusal — and refusals are cheap to
# grade, which is exactly why the rubric likes them.
REFUSAL_INSTRUCTION = (
    "Answer using only the numbered passages provided. "
    "Cite the passage numbers you used. "
    "If the passages do not contain the answer, say that you do not know."
)


class CorpusOverlapError(RuntimeError):
    """Raised when the fine-tuning corpus and the retrieval collection intersect.

    IMPLEMENTED FOR YOU (it is four lines and the whole assignment depends on it
    being raised in the right place). ``api/main.py`` turns this into an HTTP 409
    with the message you put in it, so write a message a tired human can act on:
    which document, which corpus, and what the overlap was.
    """


def sha256_text(text: str) -> str:
    """Hash the input so queries are reproducible without storing anyone's text.

    IMPLEMENTED FOR YOU — this is the privacy rule the whole product depends on,
    so it is not left to chance. The ``queries`` table stores this hash and never
    the question. A knowledge-management team's questions are often more
    sensitive than their documents ("is the Cairo office being closed?"), so the
    audit trail records that a question was asked, what came back for it, and
    nothing about what it said.

    Use it everywhere you would be tempted to log a raw string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 1. Getting documents in, and counting what is in them
# ---------------------------------------------------------------------------
def load_documents(paths: Sequence[str]) -> List[DocumentIn]:
    """Read a document collection off disk into ``DocumentIn`` records.

    Parameters
    ----------
    paths
        File paths, directory paths, or a mix. A directory means "every readable
        text file underneath it".

    Returns
    -------
    One ``DocumentIn`` per document, with ``title`` and ``source`` filled in.
    ``source`` is not optional and it is not decoration: it is what a citation
    in the Ask tab points at, and the product brief asks you to document where
    the collection came from. A source of "doc1.txt" is a fail; a path, a URL,
    or a dataset identifier is a pass.

    The assignment needs at least 50 documents, so this will be run on a folder,
    not a file. Two things will bite you:

    * **Encoding.** Real collections contain files that are not UTF-8. Decide
      what you do — skip, or decode with ``errors="replace"`` — and say which in
      your report. Silently crashing on document 34 of 50 is the default.
    * **PDF and HTML are not text.** If your collection is PDFs, extract the text
      in a separate step and keep the extraction quality in mind: a two-column
      PDF extracted naively interleaves the columns line by line, and every chunk
      you build from it is nonsense that will still embed and still be retrieved.

    Skip anything you cannot read rather than raising; return what you got, and
    log the count so the "50 documents" claim in your report is a number you
    measured rather than a number you assumed.
    """
    raise NotImplementedError("Implement load_documents() — see the docstring above.")


def count_tokens(text: str, tokenizer_name: str = EMBEDDING_MODEL) -> int:
    """Count tokens in ``text`` with a named tokenizer.

    Returns 0 for empty input.

    "300–500 tokens" is meaningless until you say whose tokens. Words are not
    tokens, characters are not tokens, and two different models disagree with
    each other by 20–30 percent on the same paragraph. Count with the tokenizer
    that belongs to the model you are about to use, and record the name — that is
    why ``ChunkingParams`` carries ``tokenizer_name``.

    Do NOT count special tokens (``[CLS]``, ``</s>``) here. They are added by the
    model at encode time and counting them makes your chunk arithmetic drift by
    two tokens per chunk, which is invisible until you are debugging why a
    passage got truncated.

    Cache the tokenizer in a module-level dict keyed by name. Loading it per call
    turns a 50-document ingest into a 50-download ingest.
    """
    raise NotImplementedError("Implement count_tokens() with a module-level tokenizer cache.")


# ---------------------------------------------------------------------------
# 2. Chunking — the step that quietly decides whether retrieval can work at all
# ---------------------------------------------------------------------------
def chunk_spans(n_tokens: int, params: ChunkingParams) -> List[Tuple[int, int]]:
    """Compute the (start, end) token spans of every chunk, as pure arithmetic.

    This function touches no model and no text. It exists on its own because the
    off-by-one errors in windowing are much easier to see — and to test — when
    they are not tangled up with a tokenizer.

    Returns
    -------
    A list of ``(start, end)`` pairs, ``end`` exclusive, in document order. The
    rules the tests check:

    * ``n_tokens == 0`` returns ``[]``.
    * A document shorter than one chunk returns exactly one span, ``(0, n_tokens)``.
      Do not pad it and do not drop it.
    * Consecutive spans advance by ``params.stride_tokens``.
    * Every span except possibly the last is exactly ``chunk_size_tokens`` long.
    * The last span ends at ``n_tokens`` — no token is left out of every chunk.
    * Neighbouring spans overlap by ``params.overlap_tokens``.
    * No zero-length spans, and no duplicate spans at the tail. The classic bug
      is a final window that starts past the end of the document and produces an
      empty chunk, which then embeds to a vector of near-zeros that is
      surprisingly close to everything.

    ``ChunkingParams`` already computes ``overlap_tokens`` and ``stride_tokens``
    for you; use them rather than recomputing, so the recorded parameters and the
    actual behaviour cannot disagree.
    """
    raise NotImplementedError("Implement chunk_spans() — pure arithmetic, no models.")


def chunk_document(document: DocumentIn, params: ChunkingParams) -> List[ChunkRecord]:
    """Cut one document into overlapping passages and record how you did it.

    Returns
    -------
    ``ChunkRecord``s in document order with ``ordinal`` counting from 0, the
    decoded ``text`` of each passage, its ``token_count``, and the
    ``start_token`` / ``end_token`` offsets from ``chunk_spans``.

    Method: tokenize once, window over the token ids with ``chunk_spans``, decode
    each window back to text. Tokenizing once and slicing is both faster and more
    honest than re-tokenizing every window, because it guarantees the offsets you
    store actually describe the text you store.

    THE TRAP, and it is the most expensive one in this assignment. A window over
    raw tokens will cut sentences in half. You will store a chunk that ends with
    "the contract was terminated in" and a chunk that begins with "March 2021,
    after the third". Neither passage contains the fact. Both embed fine. The
    retriever cheerfully returns one of them with a similarity of 0.71 and the
    generator, given half a sentence, invents the other half — and the citation
    makes the invention look sourced. The overlap is your first defence, which is
    exactly what it is for. A second is to snap boundaries to the nearest
    sentence end within a small window. If you do that, your chunks are no longer
    all the same length, which is fine; record what you did and why, because the
    "record the chunking parameters you chose and why" line in the brief is
    asking for precisely this paragraph.

    Whatever you choose: read ten of your own chunks before you embed 5,000 of
    them. Print them. Look at where they start and stop. Nobody who did this
    regretted the twenty minutes.
    """
    raise NotImplementedError("Implement chunk_document() using chunk_spans().")


# ---------------------------------------------------------------------------
# 3. Embeddings
# ---------------------------------------------------------------------------
def embedding_dimension(model_name: str = EMBEDDING_MODEL) -> int:
    """Return the number of dimensions ``model_name`` produces.

    Load the model and ask it; do not hard-code a table of numbers you copied
    from a blog post.

    WHY THIS IS ITS OWN FUNCTION. The pgvector column in the migration is
    declared ``vector(384)`` because that is what the default model emits. If you
    switch to a model that emits 768, three things happen, in this order: the
    insert fails with a dimension error (good — you find out immediately), or
    worse, you switch the model used at QUERY time only and the insert never
    happens at all, so nothing complains until a user asks a question and the
    similarity operator raises "different vector dimensions 768 and 384" from
    inside a request. That failure is at query time, not at write time, and by
    then the mismatch is a week old.

    Call this at ingest and compare it with ``EMBEDDING_DIM``. Refuse loudly on a
    mismatch. Then edit the migration and re-embed everything: you cannot mix two
    dimensions in one column, and there is no partial migration — the old vectors
    are unusable and must be rebuilt.
    """
    raise NotImplementedError("Implement embedding_dimension() by querying the model.")


def embed_texts(texts: Sequence[str], model_name: str = EMBEDDING_MODEL) -> List[List[float]]:
    """Embed a batch of strings into unit-length vectors.

    Returns
    -------
    One list of floats per input, in the same order, each of length
    ``embedding_dimension(model_name)``. Empty input returns ``[]``.

    NORMALISE THEM. Every vector must have an L2 norm of 1.0 (within floating
    point tolerance) before you return it, and the tests check that. Here is the
    reason, and it is the second-most-common silent bug in RAG projects after
    dimension mismatch:

        cosine(a, b) = (a · b) / (|a| · |b|)

    pgvector's ``<=>`` operator gives you cosine distance and handles this for
    you. But the moment you compute similarity yourself — in a local ChromaDB
    run, in a notebook, in a reranker — a bare dot product over un-normalised
    vectors is NOT cosine similarity. It is cosine similarity multiplied by the
    two magnitudes, which means long passages score higher than short ones purely
    for being long. Your top-5 fills up with your longest chunks, the scores look
    plausible (0.8-ish, nothing alarming), and the retrieval is quietly ranked by
    document length. Normalising once, here, makes the dot product and cosine
    similarity the same thing everywhere downstream.

    Batch the encode call rather than looping one string at a time, cache the
    model at module level, and remember that this runs on Render's free plan: a
    50-document ingest is thousands of chunks and you want it to finish.
    """
    raise NotImplementedError("Implement embed_texts() — and normalise the vectors.")


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors, in [-1, 1].

    Small enough to write by hand, and worth writing by hand once so the formula
    stops being a black box. Divide by both magnitudes; do not assume the inputs
    are normalised, even though ``embed_texts`` normalises them — this function
    is also what you will use to sanity-check vectors that came back from the
    database, and those may have been written by an older version of your code.

    Raise ``ValueError`` on a length mismatch, with both lengths in the message.
    That error is the dimension trap catching itself, and a message saying
    "384 vs 768" saves an afternoon.

    Return 0.0 when either vector is all zeros rather than dividing by zero.
    """
    raise NotImplementedError("Implement cosine_similarity().")


# ---------------------------------------------------------------------------
# 4. Retrieval
# ---------------------------------------------------------------------------
def vector_search(
    question: str,
    k: int = DEFAULT_K,
    corpus: str = "retrieval",
) -> List[RetrievedChunk]:
    """Embed the question and return the k most similar chunks, closest first.

    Returns
    -------
    Up to ``k`` ``RetrievedChunk``s with ``rank`` running 1, 2, 3, ... and
    ``similarity`` descending. Fewer than ``k`` is fine when the store is small.
    An empty store returns ``[]`` — it must not raise.

    How to get the neighbours:

    * **In production**, call ``api.db.match_chunks(embedding, k, corpus)``. That
      helper is written for you and calls the ``match_chunks`` SQL function the
      migration creates, which does the search inside Postgres with pgvector's
      ``<=>`` cosine-distance operator and the ivfflat index. Similarity is
      ``1 - distance``.
    * **In local development**, ChromaDB is the store the brief suggests, and it
      is genuinely easier to iterate against. Keep the two behind this one
      function so the rest of your code never knows which is live, and make sure
      you use the SAME embedding model for both. A collection embedded with one
      model and queried with another returns confident nonsense — the numbers are
      in range, the ordering is meaningless.

    Embed the query with the same model you embedded the chunks with. Say it out
    loud before you write the line, because it is the single most common cause of
    "retrieval returns unrelated passages" and it produces no error at all.

    ``corpus`` defaults to "retrieval" and should stay there. There is no
    legitimate reason for the retriever to reach into the fine-tuning corpus, and
    if it can, the with/without comparison is dead.
    """
    raise NotImplementedError("Implement vector_search() — see the docstring above.")


# ---------------------------------------------------------------------------
# 5. Generation
# ---------------------------------------------------------------------------
def build_prompt(question: str, passages: Sequence[RetrievedChunk]) -> str:
    """Assemble the prompt that conditions generation on the retrieved passages.

    Returns
    -------
    One string containing, at minimum: an instruction telling the model to answer
    only from the passages and to say so when it cannot (``REFUSAL_INSTRUCTION``
    is a starting point), the passages themselves NUMBERED so the model has
    something to cite, and the question.

    With an empty ``passages`` list, return a no-context prompt: the question and
    an instruction, nothing else. That is the control condition and it must go
    through this same function, because if the two conditions use differently
    worded prompts you are comparing prompts, not comparing retrieval.

    Things that decide whether this works:

    * **Number the passages** ("[1] ...", "[2] ..."). A model cannot cite what
      has no name, and ``cited_chunk_ids`` in the response is you mapping those
      numbers back to real chunk ids.
    * **Order matters.** Many models attend most reliably to the beginning and
      the end of a long prompt. Putting your best passage in the middle of five
      is a real way to lose it. Try rank-1-first and try rank-1-last, and say in
      your report which you shipped.
    * **Watch the length.** k=5 passages of 400 tokens is 2,000 tokens of context
      before you have written the question. A small generator may have a 1,024-
      token window, in which case the tokenizer silently drops the end of your
      prompt — usually the question. Count the tokens (``count_tokens``), put the
      number in ``AskResponse.prompt_token_count``, and truncate deliberately
      rather than letting it happen to you.
    """
    raise NotImplementedError("Implement build_prompt().")


def generate(prompt: str, max_new_tokens: int = 256, model_name: str = GENERATOR_MODEL) -> str:
    """Run the causal language model on ``prompt`` and return only the new text.

    Returns the continuation with the prompt stripped off. Returning the prompt
    concatenated with the answer is the classic first-attempt bug: it makes the
    Ask tab display the whole context back at the user, and it makes the
    with/without comparison unreadable.

    Practical notes:

    * Load the model once at module level. Loading it per request on a free-plan
      instance means every question takes a minute and some of them time out.
    * Set the decoding parameters explicitly and identically for the with- and
      without-retrieval runs. If retrieval is on for one and temperature is
      different for the other, you have two variables and no experiment.
    * Greedy decoding (``do_sample=False``) makes the comparison reproducible,
      which matters when you are quoting specific answers in a report someone
      will try to re-run.
    * Set the pad token if the tokenizer has none (GPT-2 family does not), or
      batching will raise an error whose text does not mention padding.
    * If you use a hosted inference API instead of loading weights, that is
      allowed and often the right call on a free plan — put the endpoint and the
      model id in the MODEL_CARD, keep the API key in an environment variable,
      and handle the request timing out rather than letting a 504 reach the user.
    """
    raise NotImplementedError("Implement generate().")


def answer_question(
    question: str,
    k: int = DEFAULT_K,
    use_retrieval: bool = True,
) -> AskResponse:
    """The product, in one function: retrieve, condition, generate, report.

    This is what ``POST /ask`` calls. Compose the functions above:

    1. If ``use_retrieval``: ``vector_search(question, k)``. Otherwise no search
       at all — not a search you throw away, no search, because the control
       condition has to be honest about what it cost.
    2. ``build_prompt(question, passages)``.
    3. ``generate(prompt)``.
    4. Work out which passages the answer actually cited and put their chunk ids
       in ``cited_chunk_ids``.

    Fill EVERY field of ``AskResponse``:

    * ``retrieved`` — the passages, with scores and ranks, even though it makes
      the JSON large. The Retrieval Audit tab is built on it.
    * ``cited_chunk_ids`` — the subset the answer used. Parsing "[2]" out of the
      generated text is a reasonable way to do this for a small model; say in
      your report how you did it and how often it was wrong, because a model that
      cites [7] when you gave it five passages is telling you something.
    * ``retrieval_used`` — the flag the comparison is built on.
    * ``prompt_token_count``, ``generator_model``, ``embedding_model`` — the
      reproducibility fields. ``embedding_model`` is None when retrieval was off.

    Leave ``query_id`` as None. ``api/main.py`` fills it in after logging, so
    that this function stays pure and testable with no database.

    A degenerate case worth handling before a grader finds it: retrieval returns
    nothing (empty store, or every similarity below a floor you set). Do not
    quietly fall through to an unconditioned generation that looks like a RAG
    answer. Either return the honest "I do not have a document that covers this"
    or set ``retrieval_used=False`` so the log says what actually happened.
    """
    raise NotImplementedError("Implement answer_question() — compose the functions above.")


# ---------------------------------------------------------------------------
# 6. Evaluation: perplexity on the held-out FINE-TUNING split
# ---------------------------------------------------------------------------
def perplexity(
    texts: Sequence[str],
    model_name: str = GENERATOR_MODEL,
    split: str = "held_out",
) -> PerplexityReport:
    """Perplexity of ``model_name`` over ``texts``. Report it on held-out data.

    Returns a ``PerplexityReport`` with the model id, the split name, how many
    documents and tokens were scored, and the number itself.

    Perplexity is exp(mean negative log-likelihood per token): "on average, how
    many equally likely options was the model choosing between at each token?" A
    perplexity of 20 means the model was about as uncertain as if it were picking
    uniformly from 20 words. Lower is better. It is not comparable across
    tokenizers — a model that splits text into more, smaller pieces gets an
    easier job per piece — so never put two models' perplexities in one table
    without saying they share a tokenizer.

    How to compute it without getting it wrong:

    * Score each text in windows the model can actually hold, and accumulate the
      SUM of the negative log-likelihoods and the SUM of the token counts, then
      exponentiate the ratio at the end. Averaging per-text perplexities instead
      weights a 30-token document the same as a 3,000-token one and gives a
      different, wrong answer.
    * The labels are the input shifted by one. Hugging Face causal models do that
      shift internally when you pass ``labels=input_ids``, which is convenient
      right up to the moment you also do it yourself and score the model on
      predicting the token before each token.
    * Do not count padding tokens. Mask them to -100 or your perplexity improves
      every time you pad, which is a wonderful result and completely fake.
    * ``torch.no_grad()``, model in ``eval()`` mode.

    WHICH TEXTS. The held-out split of the FINE-TUNING corpus. Not the training
    split — perplexity on data the model was fine-tuned on measures memorisation,
    and it will look excellent. Not the retrieval collection — that corpus is
    supposed to be text this model has never seen, and scoring it tells you
    nothing about the fine-tuning and quietly invites you to start training on
    it. Split the fine-tuning corpus by DOCUMENT, not by sentence: two halves of
    the same document share vocabulary, names, and phrasing, and splitting inside
    one leaks the test set into the training set in a way that is very hard to
    see afterwards.
    """
    raise NotImplementedError("Implement perplexity() — see the docstring above.")


# ---------------------------------------------------------------------------
# 7. The disjointness check. Read the module docstring again if you skipped it.
# ---------------------------------------------------------------------------
def check_corpus_disjointness(
    finetune_texts: Sequence[str],
    retrieval_texts: Sequence[str],
    shingle_size: int = 8,
    max_jaccard: float = 0.10,
    raise_on_overlap: bool = True,
) -> DisjointnessReport:
    """Prove the fine-tuning corpus and the retrieval collection do not overlap.

    Parameters
    ----------
    shingle_size
        Length in words of the overlapping n-grams ("shingles") used for
        near-duplicate detection. 8 is a reasonable default: long enough that
        ordinary English does not collide by chance, short enough to survive
        reformatting and small edits.
    max_jaccard
        The threshold. A pair of documents whose shingle sets overlap by more
        than this is treated as the same document wearing a different hat.
    raise_on_overlap
        True (the default, and what ``api/main.py`` uses) raises
        ``CorpusOverlapError``. False returns the report anyway, which is what
        you want when you are writing the evidence table for your report and
        would rather see all of the problems than the first one.

    Returns
    -------
    A ``DisjointnessReport``: how many documents on each side, how many exact
    duplicates, the worst Jaccard score seen, the offending pairs, and a
    ``disjoint`` boolean.

    Two levels of check, and you need both:

    1. **Exact.** sha256 of each normalised text (``sha256_text``). Cheap, catches
       the copy-paste case, and this is the check the database also enforces with
       a unique index on ``documents.content_sha256``.
    2. **Near-duplicate.** Jaccard over word shingles. This is the one that earns
       its keep. The same press release published on two URLs, a PDF and its HTML
       version, a document with a header stripped — none of these share a hash,
       and every one of them contaminates the experiment exactly as badly as a
       literal copy would.

    Normalise before you shingle: lowercase, collapse whitespace, drop
    punctuation. Otherwise "The Board met on 3 March." and "the board met on 3
    march" are different documents to your checker and identical documents to the
    model.

    The error message on ``CorpusOverlapError`` is part of the deliverable. It
    should name which document on each side collided and what the score was, so
    the person reading the 409 in the Streamlit tab can go and delete the right
    file. "Corpora overlap" is not a message; "finetune doc 12 and retrieval doc
    3 share 0.94 of their 8-word shingles" is.

    Comparing every document against every other document is quadratic. At 50
    documents that is 2,500 comparisons and completely fine. If your collection
    grows past a few thousand, the standard fix is to hash the shingles and
    compare only documents that share a hash bucket — worth a sentence in your
    report if you do it, and worth not bothering with if you do not.
    """
    raise NotImplementedError("Implement check_corpus_disjointness() — see the docstring above.")


def normalise_for_comparison(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Used by both checks.

    Small helper, but put it here rather than inlining it twice: the exact-hash
    check and the shingle check MUST normalise identically, or a document can
    pass one and fail the other and you will not be able to reproduce either.
    """
    raise NotImplementedError("Implement normalise_for_comparison().")


def shingles(text: str, size: int = 8) -> set:
    """The set of overlapping ``size``-word n-grams in ``text``.

    "the cat sat on the mat" with size=3 gives {"the cat sat", "cat sat on",
    "sat on the", "on the mat"} — four shingles, one per starting position that
    has enough words after it.

    Return an empty set when the text has fewer than ``size`` words. Do not pad
    it out: a two-word document is not similar to everything, and padding makes
    short documents collide with each other and trip the overlap check for no
    reason.

    Normalise with ``normalise_for_comparison`` first.
    """
    raise NotImplementedError("Implement shingles().")


# ---------------------------------------------------------------------------
# Optional, but this is where the report's headline table comes from.
# ---------------------------------------------------------------------------
def compare_with_and_without(
    questions: Sequence[str],
    k: int = DEFAULT_K,
    grounding_notes: Optional[dict] = None,
):
    """Answer each question twice — with retrieval and without — and tabulate it.

    The assignment asks for at least ten questions and at least three cases where
    the retrieved passage stopped the generator inventing something, each with
    the passage quoted. This function is how you produce that table without doing
    it by hand at 2am the night before.

    Return a ``ComparisonReport`` (see ``shared/schemas.py``). Hold everything
    constant except retrieval: same model, same decoding settings, same prompt
    template, same k. Then read the pairs yourself — this is a judgement call and
    no metric substitutes for it. The interesting rows are not the ones where
    retrieval helped; they are the ones where it did not, and working out whether
    that was a chunking failure, an embedding failure, or the generator ignoring
    a passage it was handed is the best paragraph in the report.

    Choose questions whose answers are ONLY in your collection. A question a
    pretrained model can already answer from its own weights tells you nothing
    about your retriever, and half the class will fill their table with them.
    """
    raise NotImplementedError("Implement compare_with_and_without() for your report table.")
