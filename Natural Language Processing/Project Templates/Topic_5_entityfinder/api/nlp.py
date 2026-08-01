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

The order below is the order you should implement in. Notice that the two
functions with no model in them — ``decode_bio_spans`` and
``entity_level_scores`` — are also the two that decide whether your numbers mean
anything. Write them early, test them hard, and do it before you have a GPU bill.

WHAT "DONE" LOOKS LIKE
----------------------
``pytest`` passes with no skips other than the Supabase round-trip (which needs
real credentials), the Extract tab highlights real spans in the original text,
and the Review Queue has rows in it because your low-confidence predictions were
logged.

THE THREE MISTAKES THIS FILE IS ARRANGED TO PREVENT
---------------------------------------------------
1. **Token-level metrics reported as if they were entity-level.** Most tokens in
   any NER corpus are ``O``. A model that predicts ``O`` for everything scores
   well above 90% token accuracy and finds zero entities. Entity-level F1 is the
   only number that answers the question the product asks.
2. **Character offsets reconstructed by counting.** Subword tokenization splits
   "Lovelace" into pieces and may normalise, lowercase, or strip accents on the
   way. The only reliable route back to the original string is the tokenizer's
   offset mapping. Everything in this module is typed in character offsets to
   keep you honest about that.
3. **Confidence treated as a probability of being right.** A softmax gives you a
   number in [0, 1] that sums to one across labels. That is not the same as "I
   am 93% likely to be correct", and transformers are famously happy to be
   confidently wrong on entities they never saw in training. The threshold that
   fills your review queue is a triage tool, not a calibrated probability.

A NOTE ON LIBRARIES
-------------------
Use ``sklearn-crfsuite`` for the CRF and the Hugging Face ``transformers`` stack
for the token classifier — both are already pinned in api/requirements.txt.
Write the feature functions yourself; that is where the learning is. Whatever
you use for POS tags, pin it and name it in the MODEL_CARD, because your CRF's
behaviour depends on a tagger you did not train.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence

from shared.schemas import (
    DatasetProfile,
    Entity,
    EntityScores,
    Span,
    TokenPrediction,
    TrainingRun,
)

# ---------------------------------------------------------------------------
# The label set. Four types, from the corpus in the course materials:
#   PER  people          ORG  organisations
#   LOC  locations       MISC everything else the annotators called an entity
#
# MISC is the interesting one. It is a residue category — nationalities, event
# names, product names — and it is where inter-annotator agreement is weakest,
# which is why nearly every published system scores worst on it. If you swap in
# a domain of your own (drug names, case citations, ticket numbers), change this
# tuple, and then say in your report what you lost and gained by doing so.
# ---------------------------------------------------------------------------
ENTITY_TYPES: tuple[str, ...] = ("PER", "ORG", "LOC", "MISC")

#: The BIO tag inventory that ENTITY_TYPES implies. "B-" opens an entity, "I-"
#: continues the one before it, "O" is outside any entity. Two adjacent entities
#: of the same type are exactly why "B-" exists: without it, "Paris London" is
#: one two-word location.
BIO_TAGS: tuple[str, ...] = ("O",) + tuple(
    f"{prefix}-{t}" for t in ENTITY_TYPES for prefix in ("B", "I")
)

#: Default cutoff for the review queue. Anything the model scores below this
#: goes in front of a human. It is a starting point and a placeholder for a
#: decision your team owes an argument for: set it high and you drown reviewers
#: in work; set it low and the errors you most need to catch never surface.
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.85


def sha256_text(text: str) -> str:
    """Hash the input so extractions are reproducible without storing the document.

    IMPLEMENTED FOR YOU — this is the privacy rule the whole product depends on,
    so it is not left to chance. The documents an extraction team uploads are
    exactly the kind of text that should not sit in a course project's database.
    Use this everywhere you would be tempted to log the raw string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_spans(entities: Sequence[Entity]) -> List[Span]:
    """Reduce entities to the ``(start_char, end_char, entity_type)`` triples used
    for scoring.

    IMPLEMENTED FOR YOU, because it is three lines and because it makes the
    scoring contract impossible to misread: the surface string and the
    confidence play no part in whether a prediction is correct.
    """
    return [(e.start_char, e.end_char, e.entity_type) for e in entities]


# ---------------------------------------------------------------------------
# 1. The data
# ---------------------------------------------------------------------------
def profile_dataset(name: str = "conll2003", split: Optional[str] = None) -> DatasetProfile:
    """Load the labelled NER corpus and report what is actually in it.

    Returns
    -------
    A ``DatasetProfile`` with every field populated from counting, not from a
    number you read somewhere: sentence counts per split, the entity types
    present, and the token count for every BIO tag including ``O``.

    Why this is a graded function and not a footnote:

    * **The O share is the headline.** Count it. Then compute what token-level
      accuracy a model gets by predicting ``O`` for every token in the test
      split. Put that number in your report next to your real F1 — it is the
      single most convincing argument for entity-level evaluation you can make,
      and it costs you four lines of code.
    * **Check the tagging scheme rather than assuming it.** IOB1 only uses ``B-``
      to separate two adjacent entities of the same type; IOB2 starts every
      entity with ``B-``. Decoding IOB1 data with an IOB2 decoder silently merges
      neighbouring entities and quietly lowers your recall. Look at the tags in
      your copy and set ``tagging_scheme`` from what you see.
    * **The corpus is newswire from decades ago.** It knows the organisations,
      the politicians, and the sports teams that were in the news then. If your
      users care about clinical notes, support tickets, or contracts, this data
      will not carry you there, and the drop will not be small. Say so in the
      MODEL_CARD under out-of-scope, with an example you actually ran.

    Cite the corpus in your report. The citation is in the course materials.
    """
    raise NotImplementedError("Implement profile_dataset() — see the docstring above.")


# ---------------------------------------------------------------------------
# 2. The CRF baseline
# ---------------------------------------------------------------------------
def word_features(tokens: Sequence[str], i: int, pos_tags: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Build the feature dictionary for one token, for the CRF.

    Returns
    -------
    A flat ``{feature_name: value}`` dict for ``tokens[i]``. ``sklearn-crfsuite``
    takes strings, booleans, and numbers; keep the keys stable, because the
    feature names ARE your model's vocabulary and you have to list them in the
    MODEL_CARD.

    Required by the assignment, at minimum:

    * **prefixes and suffixes** — the first and last 2–3 characters. This is how
      a CRF learns morphology it was never told about: "-ton" and "-burg" end
      place names, "-ing" does not end anything.
    * **POS tag** of the token. Proper nouns are where entities live.
    * shape features: is it capitalised, all-caps, title case, does it contain a
      digit, a hyphen, a period.

    And the feature that matters most and gets forgotten most: **the window**.
    Include the same features for ``tokens[i-1]`` and ``tokens[i+1]`` under
    prefixed keys ("-1:", "+1:"), plus explicit ``BOS`` / ``EOS`` flags at the
    edges. A CRF sees only what you hand it. "Washington" is a person, a place,
    or an organisation depending entirely on the words around it, and if those
    words are not in this dict, the model cannot use them.

    That constraint is the whole pedagogical point of the baseline. Write the
    features, then look at what the transformer gets for free.
    """
    raise NotImplementedError("Implement word_features() — prefixes, suffixes, POS, window.")


def sentence_features(tokens: Sequence[str], pos_tags: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Apply ``word_features`` across a whole sentence.

    Returns one dict per token, in order. This is the ``X`` that
    ``sklearn_crfsuite.CRF.fit`` expects (a list of these, one per sentence).
    Keep it thin — all the thinking belongs in ``word_features``.
    """
    raise NotImplementedError("Implement sentence_features().")


def train_crf(profile: DatasetProfile, **hyperparameters: Any) -> TrainingRun:
    """Fit the CRF baseline and return everything needed to reproduce it.

    Returns
    -------
    A ``TrainingRun`` with ``model_type="crf"``, the hyperparameters you used
    (``c1``, ``c2``, ``max_iterations``, the algorithm), the list of feature
    names from ``word_features``, and entity-level ``metrics`` on the held-out
    split — computed with ``entity_level_scores``, not with the token-level
    accuracy the library hands you.

    Persist the model to disk and record where. The API loads it at start-up;
    re-fitting a CRF inside a request handler will make the service look broken.

    Two things to notice while it converges, both of which belong
    in the report:

    * It trains in minutes on a laptop with no GPU, and you can print the
      highest-weighted feature for each label and read it like a sentence. That
      transparency is a real engineering property, not a consolation prize.
    * Its errors are systematic and legible. Where it fails, you can usually name
      the missing feature. Hold on to that, because when the transformer fails
      you will not be able to say the same.
    """
    raise NotImplementedError("Implement train_crf() — fit, evaluate, persist, describe.")


# ---------------------------------------------------------------------------
# 3. The transformer token classifier
# ---------------------------------------------------------------------------
def load_model(name: str):
    """Load and cache a token-classification model and its fast tokenizer.

    Render's free tier gives you limited memory and a cold start on every deploy,
    so loading a model per request will make the service feel broken. Cache in a
    module-level dict keyed by ``name`` and return whatever pair you find easiest
    to use downstream (model + tokenizer, or a pipeline).

    Load the **fast** tokenizer. The slow one cannot return offset mappings, and
    without offset mappings you cannot produce character spans — see
    ``predict_bio_tags``.

    Raise ``ValueError`` with a readable message when the id is unknown; it is
    surfaced as a 400, and "model 'bert-base-cassed' not found" beats a stack
    trace in someone else's log.
    """
    raise NotImplementedError("Implement load_model() with a module-level cache.")


def train_transformer(profile: DatasetProfile, **hyperparameters: Any) -> TrainingRun:
    """Fine-tune a transformer for token classification on the same splits.

    Returns a ``TrainingRun`` with ``model_type="transformer"``, the checkpoint
    name, learning rate, batch size, epochs and seed in ``hyperparameters``, and
    entity-level ``metrics`` on the same held-out split the CRF was scored on.
    Same splits or the comparison is not a comparison.

    The step everyone gets wrong is **label alignment**. Your labels are one per
    word; the tokenizer gives you one row per subword. "Lovelace" may become
    three pieces, and each piece needs a label before the model can be trained.
    Use ``word_ids()`` from the fast tokenizer to map pieces back to words, label
    the first piece of a word with that word's tag, and set the rest to -100 so
    the loss ignores them. If you skip this, the model trains on labels that are
    off by one from the third multi-piece word onwards and you will spend a day
    convinced the architecture is broken.

    Train this somewhere with a GPU — a notebook service is fine — and commit the
    resulting artefacts or push them to a model hub. Render's free plan is for
    serving, not for fine-tuning.
    """
    raise NotImplementedError("Implement train_transformer() — align labels, fine-tune, evaluate.")


# ---------------------------------------------------------------------------
# 4. Serving: text in, character-anchored entities out
# ---------------------------------------------------------------------------
def predict_bio_tags(text: str, model: str = "transformer") -> List[TokenPrediction]:
    """Run one model over the text and return per-token tags with real offsets.

    Returns
    -------
    A ``TokenPrediction`` per token, in order, each carrying its BIO tag, its
    ``start_char`` / ``end_char`` in the ORIGINAL text, and the model's score for
    that tag.

    ``model`` is ``"transformer"`` or ``"crf"``. Both paths return the same type;
    that is what lets the comparison tab and the review queue treat them
    identically, and what lets you swap one for the other later without touching
    the UI.

    For the transformer, get offsets from the tokenizer:
    ``tokenizer(text, return_offsets_mapping=True)``. Do not compute them by
    walking the string and adding token lengths. The tokenizer may lowercase,
    strip accents, or split on characters you did not expect, and ``str.find()``
    returns the FIRST occurrence, which means the second "Smith" in a document
    gets the first "Smith"'s offsets. Drop the special tokens ``[CLS]`` /
    ``[SEP]``, which come back with the offset span ``(0, 0)``.

    For the CRF, you control tokenization, so record each token's start and end
    as you split — it is easy, but only if you do it at split time rather than
    trying to recover it afterwards.

    The confidence is the model's score for the tag it chose on that token. See
    the module docstring: it is a triage signal, not a probability of correctness.
    """
    raise NotImplementedError("Implement predict_bio_tags() — use offset mappings.")


def decode_bio_spans(predictions: Sequence[TokenPrediction], text: str) -> List[Entity]:
    """Turn a BIO tag sequence into entity spans. Pure function, no model needed.

    Returns
    -------
    ``Entity`` objects in document order. For every one of them,
    ``entity.text == text[entity.start_char:entity.end_char]`` — slice the
    original text, never join the tokens back together with spaces. Joining
    turns "New   York" into "New York" and "don't" into "do n't", and the
    reviewer sees a string that does not exist in their document.

    The rules:

    * ``B-X`` opens a new entity of type X, always, even when the previous token
      was also ``B-X`` or ``I-X``. That is how two adjacent entities of the same
      type stay separate.
    * ``I-X`` continues an open entity of type X.
    * ``O`` closes any open entity.
    * The span runs from the first token's ``start_char`` to the last token's
      ``end_char``, which correctly includes the whitespace between tokens.

    Then there are the sequences that should not happen and do:

    * ``I-X`` with nothing open. Models emit this constantly. Pick a policy —
      start a new entity, or drop the token — write it down in the MODEL_CARD,
      and be consistent. It changes your recall.
    * ``I-Y`` continuing an open X. Same deal: close and reopen, or absorb.

    Assign each entity a confidence from its member tokens. The minimum is the
    defensible default (a span is only as trustworthy as its weakest token) and
    the mean is the flattering one. Whichever you choose, say which, because the
    review queue's contents depend on it.

    Write this function first and test it with hand-written tag sequences. It has
    no dependencies, it takes fifteen minutes, and a bug here is invisible in
    training and fatal in production.
    """
    raise NotImplementedError("Implement decode_bio_spans() — see the rules above.")


def extract_entities(text: str, model: str = "transformer") -> List[Entity]:
    """The serving path: raw text in, character-anchored entities out.

    Compose ``predict_bio_tags`` and ``decode_bio_spans``. This is the function
    ``POST /extract`` calls, and it should stay short — if it is growing, the
    logic belongs in one of the two functions it calls.

    Handle long documents. A transformer has a fixed maximum length, and text
    past it is silently truncated: no error, no warning, just an entity-free
    second half. Split into sentences or windows, run each, and **add the window
    offset back** to every entity's ``start_char`` and ``end_char``. Forgetting
    that shift is the single most common way this template's highlighting ends
    up rainbow-coloured nonsense two-thirds of the way down a page.

    Empty or whitespace-only input returns an empty list, not an exception.
    """
    raise NotImplementedError("Implement extract_entities() — predict, decode, offset-correct.")


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------
def entity_level_scores(gold: Sequence[Span], predicted: Sequence[Span]) -> EntityScores:
    """Entity-level precision, recall, and F1 by exact match. The graded number.

    A predicted span is a true positive only when some gold span has the same
    ``start_char``, the same ``end_char``, AND the same type. Anything else is a
    false positive; every unmatched gold span is a false negative. Match each
    gold entity at most once.

    That definition has a consequence worth sitting with: a boundary error is
    punished twice. Predict "York Stock Exchange" where the gold is "New York
    Stock Exchange" and you score one false positive and one false negative —
    strictly worse than predicting nothing at all. Getting the type right does
    not earn partial credit. This is deliberate; a downstream system that looks
    up your span in a database gets nothing useful from a span that is close.

    Fill ``per_type`` as well as the aggregate. The overall F1 is dominated by
    whichever type is most frequent, and the type you are worst at is the one
    your report should be about.

    Compute it yourself rather than reaching for a library. It is a couple of
    counters, and writing it is how you stop confusing entity-level with
    token-level for good. Once yours works, checking it against an established
    implementation is a reasonable thing to do — but do that as a check, after
    you have your own answer, and reconcile any disagreement rather than
    shrugging at it.

    Edge cases the tests check: define 0/0 as 0 rather than crashing, so empty
    gold with empty predictions returns zeros with ``support=0``; predictions
    against empty gold give precision 0.
    """
    raise NotImplementedError("Implement entity_level_scores() — exact match, per type.")
