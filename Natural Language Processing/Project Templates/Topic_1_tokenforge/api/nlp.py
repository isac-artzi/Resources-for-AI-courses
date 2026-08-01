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
if yours is growing past ~30 lines, you are probably solving the next one too.

WHAT "DONE" LOOKS LIKE
----------------------
``pytest`` passes with no skips other than the Supabase round-trip (which needs
real credentials), and the three UI tabs render real numbers.

A NOTE ON LIBRARIES
-------------------
Use the Hugging Face ``transformers`` / ``tokenizers`` stack for the subword
work. For classical preprocessing, write the logic yourself — the stop list, the
punctuation strip, and a simple suffix stemmer are twenty lines each and you
learn more from writing them than from importing them. Lemmatization is the one
place a library is reasonable; if you use one, say so in your MODEL_CARD.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List

from shared.schemas import (
    PreprocessOptions,
    PreprocessResponse,
    TokenizerResult,
)

# ---------------------------------------------------------------------------
# A starting stop list. It is deliberately short and deliberately English-only.
# One of your discussion points is what it breaks: "not", "no", and "never" are
# stop words by this list, which means "the service was not good" and "the
# service was good" preprocess to the same tokens. Decide what to do about that
# and defend the decision in your report.
# ---------------------------------------------------------------------------
DEFAULT_STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if then than that this these those of in on at by for
    with about against between into through during to from up down out off over
    under again further is are was were be been being have has had do does did
    i me my we our you your he him his she her it its they them their what which
    who whom as so no not nor only own same too very s t can will just don should
    now
    """.split()
)


def sha256_text(text: str) -> str:
    """Hash the input so runs are reproducible without storing anyone's text.

    IMPLEMENTED FOR YOU — this is the privacy rule the whole product depends on,
    so it is not left to chance. Use it everywhere you would be tempted to log
    the raw string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 1. Word tokenization
# ---------------------------------------------------------------------------
def word_tokenize(text: str) -> List[str]:
    """Split raw text into word-level tokens — the 'before' side of the UI.

    Returns
    -------
    A list of tokens in document order. Empty input returns an empty list.

    What counts as one token is your call, but the choice has to survive the
    challenging text in the discussion question: contractions ("don't"), hashtags
    ("#NLP"), emoji, URLs, and code. A bare ``text.split()`` will glue punctuation
    onto words; a naive ``re.findall(r"\\w+")`` will shred "don't" into "don" and
    "t" and drop every emoji on the floor. Pick a rule, write it down in the
    MODEL_CARD, and make sure the Compare tab shows what it cost you.

    Hint: ``regex``-style character classes over Unicode categories get you
    further than ASCII-only patterns.
    """
    raise NotImplementedError("Implement word_tokenize() — see the docstring above.")


# ---------------------------------------------------------------------------
# 2. The classical preprocessing steps
# ---------------------------------------------------------------------------
def strip_punctuation(tokens: List[str]) -> List[str]:
    """Remove punctuation-only tokens and strip edge punctuation from the rest.

    "hello," -> "hello";  "!!!" -> dropped;  "don't" -> "don't" (interior
    punctuation stays);  "co-op" -> "co-op".

    Returns a new list; do not mutate the input.
    """
    raise NotImplementedError("Implement strip_punctuation().")


def remove_stopwords(tokens: List[str], stopwords: frozenset[str] = DEFAULT_STOPWORDS) -> List[str]:
    """Drop tokens that appear in the stop list.

    Compare case-insensitively but return the tokens as they came in — if the
    caller did not ask for lowercasing, you should not lowercase behind their
    back. This kind of hidden side effect is exactly what the 'steps_applied'
    audit trail is meant to catch.
    """
    raise NotImplementedError("Implement remove_stopwords().")


def stem(tokens: List[str]) -> List[str]:
    """Reduce tokens to a crude stem by suffix stripping.

    A Porter-style stemmer is overkill here. Handle the common English suffixes
    (-ing, -ed, -ly, -es, -s) with a few ordered rules and accept that you will
    produce non-words: "running" -> "run" but "flies" -> "fli". That ugliness is
    the point — your report should contrast it with what lemmatize() returns.

    Do not stem tokens shorter than 4 characters; the rules do more harm than
    good there.
    """
    raise NotImplementedError("Implement stem().")


def lemmatize(tokens: List[str]) -> List[str]:
    """Reduce tokens to dictionary form ("was" -> "be", "mice" -> "mouse").

    Unlike stem(), every output should be a real word. A small irregular-forms
    table plus a few regular rules is enough for this assignment; a library is
    also acceptable, but if you import one, name it in the MODEL_CARD and pin
    the version in api/requirements.txt.
    """
    raise NotImplementedError("Implement lemmatize().")


def preprocess(text: str, options: PreprocessOptions) -> PreprocessResponse:
    """Run the classical pipeline and report what happened at every step.

    The order is fixed (see PreprocessOptions): lowercase → strip punctuation →
    remove stop words → remove digits → stem OR lemmatize. If both ``stem`` and
    ``lemmatize`` are true, prefer lemmatize and record that you did.

    You MUST populate ``steps_applied`` with one entry per step that actually
    ran, in order. The tab that renders it is what turns this from a black box
    into something an analyst can trust — and it is graded.

    ``run_id`` is filled in by the caller in api/main.py, not here. Leave it None.
    """
    raise NotImplementedError("Implement preprocess() — compose the functions above.")


# ---------------------------------------------------------------------------
# 3. Subword tokenization
# ---------------------------------------------------------------------------
def load_tokenizer(name: str):
    """Load and cache a Hugging Face tokenizer by id.

    Render's free tier gives you limited memory and a cold start on every deploy,
    so loading a tokenizer per request will make the service feel broken. Cache
    them in a module-level dict keyed by name.

    Raise a ValueError with a readable message if the id is unknown — it will be
    surfaced to the user as a 400, and "tokenizer 'bert-base-uncase' not found"
    is a much better error than a stack trace.
    """
    raise NotImplementedError("Implement load_tokenizer() with a module-level cache.")


def subword_tokenize(text: str, tokenizer_name: str) -> TokenizerResult:
    """Tokenize with one subword tokenizer and report the full result.

    Every field of TokenizerResult must be real:

    * ``tokens`` — keep the tokenizer's own markers. "##ing" and "Ġthe" look
      like noise but they encode where the word boundaries are, and stripping
      them destroys the thing the Compare tab is meant to show.
    * ``algorithm`` — WordPiece / BPE / Unigram. Determine it from the tokenizer
      rather than hard-coding a lookup table, if you can.
    * ``unknown_count`` — how many pieces are the tokenizer's unk token.
    * ``oov_rate`` — unknown_count / token_count, guarding against divide-by-zero
      on empty input.

    A well-trained subword tokenizer will give you an OOV rate at or near zero on
    ordinary English, which looks like a bug the first time you see it. It isn't.
    Feed it something it has never seen — a chemical name, a URL, a language in
    another script — and watch the piece count explode instead. That contrast is
    your worked example.
    """
    raise NotImplementedError("Implement subword_tokenize().")


def vocabulary_overlap(results: List[TokenizerResult]) -> Dict[str, float]:
    """Pairwise Jaccard overlap between the piece sets each tokenizer produced.

    Key format is "name_a|name_b" with the names in the order given. Return an
    empty dict when there are fewer than two results.

    Jaccard = |A ∩ B| / |A ∪ B| over the SETS of pieces (not the sequences).
    Expect a low number — two tokenizers with different algorithms and different
    training corpora agree on far less than students predict, and saying why is
    a good paragraph in your report.
    """
    raise NotImplementedError("Implement vocabulary_overlap().")
