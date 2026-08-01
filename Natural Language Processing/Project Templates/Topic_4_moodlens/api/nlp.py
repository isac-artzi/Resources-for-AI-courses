"""THE FILE YOU ARE HERE TO WRITE.

Everything else in this template is finished: the FastAPI app boots, the
database layer works, the Streamlit UI renders all six tabs, the deploy config
is correct. This module is the hole in the middle. Every function below raises
``NotImplementedError`` and every one of them has a docstring telling you what
it must return and why.

HOW TO WORK THROUGH IT
----------------------
1. Run ``pytest -m contract`` — the contract tests fail, listing what's missing.
2. Implement one function.
3. Re-run. Go green. Move to the next.

The order below is the order you should implement in, and it is not arbitrary:
the baseline comes before the transformer so that when the transformer finally
runs you already know what number it has to beat. A fine-tuned model with no
baseline next to it is a number with nothing to compare against, and "0.91" on
its own is not a result.

WHAT "DONE" LOOKS LIKE
----------------------
``pytest`` passes with no skips other than the Supabase round-trip and the
training test (which need real credentials and real compute), the Score Text and
Aspect Breakdown tabs return real answers, and the Model Performance and Bias
Audit tabs are reading metrics your own training run wrote to Supabase rather
than the seed row.

TWO RULES ABOUT IMPORTS
-----------------------
1. Import ``torch``, ``transformers``, ``sklearn`` and ``datasets`` INSIDE the
   functions that need them, not at the top of this file. The test suite, the
   schema tests and the FastAPI app all import this module; a top-level torch
   import makes a 200 ms test run take fifteen seconds and makes the API's cold
   start worse than it already is.
2. Nothing in this module may import ``api.db`` or ``streamlit``. NLP logic does
   not know where it is deployed.

A NOTE ON WHAT TO WRITE YOURSELF
--------------------------------
Use Hugging Face for the transformer and scikit-learn for the TF-IDF baseline —
reimplementing either teaches you nothing this topic is about. Write the metric
functions yourself, or at least check them against something you wrote yourself.
``classification_report`` is easy to call and easy to misread, and "macro-F1"
being an average over classes rather than over examples is exactly the kind of
detail that turns into a wrong number in a report.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from shared.schemas import (
    AspectMetrics,
    AspectSentiment,
    DocumentMetrics,
    SentimentPrediction,
    SliceMetrics,
)

# ---------------------------------------------------------------------------
# THE ASPECTS.
#
# These three are a MODELLING DECISION, not a property of the data. Nobody
# labelled a movie review with "acting"; you decided that "acting" is a thing
# reviews talk about, and every number in your Aspect Breakdown tab inherits
# that decision. Two consequences you have to live with and document:
#
#   * Coverage. A review that praises the pacing and the score has said nothing
#     about any of these three unless you decide that "pacing" belongs under
#     "plot" and "score" under "production". Write that mapping down. If you do
#     not, you will make it up differently on Tuesday than you did on Monday.
#   * Overlap. "The lead was wooden and the script gave her nothing" is one
#     sentence about two aspects, and whichever one you assign it to, you are
#     wrong about the other.
#
# If you switch to a product-review corpus, the natural three are price,
# quality and service — same argument, different words. Change the tuple, change
# the ASPECT_CUES below, and change the MODEL_CARD. Do not leave film aspects
# pointing at a corpus of headphone reviews and hope nobody scrolls.
# ---------------------------------------------------------------------------
ASPECTS: Tuple[str, ...] = ("acting", "plot", "production")

# A starting cue list per aspect. It is deliberately short and deliberately
# crude, because its job is to be the thing you outgrow: keyword matching cannot
# tell "the plot was not bad" from "the plot was bad", and it cannot tell that
# "she carried the film" is about acting. Use it to get the plumbing working end
# to end, then replace it with something that reads the sentence — and keep the
# keyword version around so your report can quantify what replacing it bought.
ASPECT_CUES: Dict[str, Tuple[str, ...]] = {
    "acting": ("acting", "actor", "actress", "cast", "performance", "played", "role"),
    "plot": ("plot", "story", "script", "writing", "ending", "pacing", "twist"),
    "production": (
        "direction",
        "directed",
        "cinematography",
        "visuals",
        "effects",
        "soundtrack",
        "score",
        "editing",
    ),
}

# ---------------------------------------------------------------------------
# THE SLICES for the bias audit.
#
# Two slices, and they are different in kind on purpose.
#
#   review_length  — OBSERVED. You measure it. Nothing was inferred, so a gap
#                    between buckets is a fact about your model.
#   genre          — OBSERVED ONLY IF YOUR CORPUS SHIPS IT. IMDb's public
#                    sentiment split does not. If you assign genre with a
#                    classifier or a keyword rule, the slice is INFERRED, and a
#                    gap you find is a gap in your classifier, your genre
#                    guesser, or both, with no way to tell which. Set
#                    ``observed=False`` on those SliceMetrics and say so in the
#                    tab. Auditing a model with another model's output is
#                    circular, and the circularity is the finding.
#
# The buckets are given so that everyone's audit is comparable. The thresholds
# are in whitespace tokens, not characters.
# ---------------------------------------------------------------------------
SLICE_SPECS: Dict[str, Dict[str, Any]] = {
    "review_length": {
        "observed": True,
        "buckets": ("short", "medium", "long"),
        "description": "short < 60 tokens, medium 60-199, long >= 200",
    },
    "genre": {
        "observed": False,
        "buckets": ("drama", "comedy", "horror", "other"),
        "description": (
            "Only observed if your corpus carries a genre field. If you inferred "
            "it, mark the slice inferred and treat the numbers as a hypothesis."
        ),
    },
}


def sha256_text(text: str) -> str:
    """Hash the input so predictions are auditable without storing anyone's text.

    IMPLEMENTED FOR YOU — this is the privacy rule the whole product depends on,
    so it is not left to chance. Use it everywhere you would be tempted to log
    the raw review.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ===========================================================================
# 1. Data
# ===========================================================================
def load_reviews(split: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load labelled reviews for one split.

    Parameters
    ----------
    split
        "train", "validation" or "test". If your corpus ships only train and
        test — IMDb does — carve the validation set out of train yourself, with
        a fixed seed, and never let a validation review appear in test. Say in
        your report how you split it.
    limit
        Cap the number of rows. Useful while you are debugging; report the
        number you actually trained and evaluated on.

    Returns
    -------
    A list of dicts, one per review, each with at least::

        {"text": str, "label": "positive" | "negative"}

    and optionally ``"aspects"`` (a dict aspect -> label, for the subset you
    have aspect labels on) and any observed slice attributes such as ``"genre"``.

    Notes you will need for the report
    ----------------------------------
    * IMDb is balanced 50/50 and it is long-form: the median review is several
      hundred words, written by someone who chose to sit down and write a
      review. Your production input is a 12-word comment typed on a phone.
      Accuracy on IMDb will therefore be flattering. That is not a reason to
      avoid the corpus; it is a reason to say so out loud next to the number.
    * IMDb carries NO aspect labels. If you want per-aspect precision and recall
      you must either annotate a subset yourself — a few hundred reviews, with a
      written annotation guideline, is enough and is honest work — or use an
      aspect-annotated corpus instead. Whichever you pick, the choice and its
      cost go in the model card.
    * Truncation is a data decision, not an implementation detail. A 512-token
      transformer sees roughly the first two thirds of a long IMDb review, and
      reviews often turn in the last paragraph ("...but I still enjoyed it").
      Decide head, tail, or chunk-and-pool, and measure it.
    """
    raise NotImplementedError("Implement load_reviews() — see the docstring above.")


# ===========================================================================
# 2. The TF-IDF baseline
#
# Build this FIRST. It trains in seconds on a laptop, it gives you the number
# the transformer has to beat, and on long balanced movie reviews it will be
# closer to the transformer than you expect. Some of the report's most useful
# sentences come from the cases where a linear model over word counts is enough.
# ===========================================================================
def fit_baseline(train: Sequence[Dict[str, Any]]) -> Any:
    """Fit a TF-IDF + linear classifier baseline and return the fitted object.

    Return whatever your ``baseline_predict`` needs — an sklearn ``Pipeline`` of
    ``TfidfVectorizer`` and ``LogisticRegression`` is the obvious answer and is
    fine. Fit the vectorizer on the TRAINING SPLIT ONLY. Fitting it on all the
    text first, then splitting, leaks test vocabulary into training and inflates
    every number you go on to report; it is the most common quiet mistake in
    this whole assignment.

    Choices worth a paragraph each in the report, because each one moves the
    score: n-gram range (unigrams alone cannot see "not good"), minimum document
    frequency, whether you strip stop words at all when "not" and "no" are stop
    words in most default lists, and sublinear term frequency on long documents.

    Keep the fitted object picklable and save it, or you will retrain it every
    time you want a number.
    """
    raise NotImplementedError("Implement fit_baseline().")


def baseline_predict(model: Any, texts: Sequence[str]) -> List[SentimentPrediction]:
    """Score texts with the fitted baseline.

    Returns one ``SentimentPrediction`` per input, in input order, with
    ``model_name="tfidf-baseline"``.

    Logistic regression gives you ``predict_proba``, so fill
    ``raw_probability_positive`` honestly. If you have not fitted a calibrator
    for the baseline, set ``calibrated=False`` and copy the raw value into
    ``probability_positive``. Do not quietly pretend a raw softmax or a raw
    logistic output is calibrated — the whole point of the calibration plot is
    to catch that, and it will.
    """
    raise NotImplementedError("Implement baseline_predict().")


# ===========================================================================
# 3. The fine-tuned transformer
# ===========================================================================
def fine_tune_transformer(
    train: Sequence[Dict[str, Any]],
    validation: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
) -> str:
    """Fine-tune a pretrained transformer for sentiment classification.

    Parameters
    ----------
    config
        The ``fine_tune`` block of ``api/configs/default.yaml``: base model id,
        learning rate, epochs, batch size, max sequence length, seed, output
        directory.

    Returns
    -------
    The model version string you will serve under, e.g. ``"moodlens-v2"``. Write
    the weights to ``config["output_dir"]`` and return the version — the caller
    logs that version with the training run, and ``/version`` reports it, and
    that chain is what lets you attribute a bad prediction to a specific build
    three weeks later.

    Requirements
    ------------
    * Validate ``config["base_model"]`` and raise ``ValueError`` for an id you
      cannot load, BEFORE you start downloading anything. Sitting through a
      four-minute download to be told the checkpoint name had a typo is a bad
      afternoon.
    * Set every seed you can reach and record it. "We got 0.91" is not
      reproducible; "we got 0.91 with seed 13, three epochs, lr 2e-5" is.
    * Evaluate on the VALIDATION split during training and pick the checkpoint
      from that. The test split is touched once, at the end, when you are no
      longer making decisions. Every time you look at test and then change
      something, test becomes a little more like train.
    * Log the run: hyperparameters, split sizes, wall-clock time, and the
      metrics you get out. That row in ``runs`` is what the Model Performance
      tab renders.

    Practical
    ---------
    A small checkpoint (distilbert-sized) fine-tunes on CPU in a long coffee
    break and on a free hosted GPU notebook in minutes. Do the training OFF
    Render — the free tier has neither the memory nor the time — commit or
    upload the resulting weights, and let the API do inference only.
    """
    raise NotImplementedError("Implement fine_tune_transformer().")


def load_classifier() -> Any:
    """Load and cache the fine-tuned classifier for serving.

    Cache it in a module-level global named ``_CLASSIFIER`` — /healthz reports
    ``model_loaded`` by looking for exactly that name, without forcing a load.
    A per-request load is not slow, it is broken: on Render's free tier it means every prediction re-reads hundreds of
    megabytes from disk, and the Score Text tab will time out while you stare at
    code that looks correct.

    Read the model location from the environment (see ``.env.example``) so the
    same code serves your local checkpoint and the deployed one. Raise a
    ``ValueError`` with a readable message when the weights are missing — "no
    fine-tuned model at ./models/moodlens-v2; run training first" is a much
    better error than a stack trace out of ``from_pretrained``.

    Return whatever ``predict_sentiment`` needs: the model, its tokenizer, and
    the label mapping. Get the label mapping from the model's own config rather
    than assuming index 1 means positive. That assumption is right about half
    the time, and when it is wrong every metric in your report is exactly
    inverted while looking completely plausible.
    """
    raise NotImplementedError("Implement load_classifier() with a module-level cache.")


def fit_calibrator(
    scores: Sequence[float], labels: Sequence[str]
) -> Callable[[float], float]:
    """Fit a probability calibrator on VALIDATION scores and return it.

    Returns a callable mapping a raw P(positive) in [0, 1] to a calibrated
    P(positive) in [0, 1]. It must be monotonically non-decreasing: calibration
    is allowed to change how confident the model is, never which side of the
    line it lands on.

    Why this exists: a fine-tuned transformer trained with cross-entropy is
    usually overconfident. It says 0.99 for a set of reviews of which 92 per
    cent are actually positive. Nothing in accuracy or F1 notices, because those
    only look at which side of 0.5 the score fell. The product brief asks for
    calibrated probabilities because a downstream team is going to threshold on
    that number ("escalate anything below 0.7 confidence"), and an uncalibrated
    0.7 does not mean what they think it means.

    Platt scaling (a one-feature logistic regression on the scores) or isotonic
    regression both work. Fit on VALIDATION, never on training — the model is
    overconfident on training data by construction, so a calibrator fitted there
    learns to leave it alone. Evaluate on test.
    """
    raise NotImplementedError("Implement fit_calibrator().")


def predict_sentiment(text: str) -> SentimentPrediction:
    """Classify one document and return a calibrated probability.

    Every field of ``SentimentPrediction`` must be real:

    * ``raw_probability_positive`` — softmax output, before calibration.
    * ``probability_positive`` — after calibration. If no calibrator is loaded,
      copy the raw value and set ``calibrated=False``. Say what is true.
    * ``label`` — from the calibrated probability at your documented threshold.
      0.5 is a default, not a law; if you move it, move it for a stated reason
      and record the value in the model card.
    * ``confidence`` — ``max(p, 1 - p)``. Never below 0.5 for a binary model,
      which is worth remembering before you show a "confidence" bar to a user
      and let them read 0.51 as strong.

    Handle the boring cases, because the demo will find them: empty-ish input,
    text longer than the model's maximum length (truncate the way you documented
    in ``load_reviews``, not whichever way is default), and text in a language
    the model never saw.
    """
    raise NotImplementedError("Implement predict_sentiment().")


def predict_sentiment_batch(texts: Sequence[str]) -> List[SentimentPrediction]:
    """Classify several documents at once, returning results in input order.

    Not a loop over ``predict_sentiment`` if you can help it. Tokenize the batch
    together and run one forward pass: on CPU the difference between batched and
    per-item inference for a few dozen short reviews is the difference between
    the batch tab feeling instant and feeling stuck.

    Pad to the longest item in the batch, not to the model maximum, or you will
    spend most of your compute on padding. Empty input list returns an empty
    list — the endpoint should not have to special-case that.
    """
    raise NotImplementedError("Implement predict_sentiment_batch().")


# ===========================================================================
# 4. Aspect-based sentiment
# ===========================================================================
def extract_aspects(text: str) -> List[AspectSentiment]:
    """Return sentiment for each aspect in ``ASPECTS``, with evidence.

    Returns
    -------
    Exactly ``len(ASPECTS)`` entries, one per aspect, in ``ASPECTS`` order. An
    aspect the review never discusses gets ``label="not_mentioned"`` and empty
    evidence. Do not drop it from the list: the Aspect Breakdown tab shows the
    absence, and an analyst learning that four hundred reviews never mention
    service is learning something.

    ``evidence`` must contain snippets that are literal substrings of ``text``.
    Do not paraphrase, do not lowercase, do not clean them up. The whole value
    of the field is that a human can hit Ctrl-F and find it; a "quote" that is
    not in the document is worse than no quote, because it reads as proof.

    How to actually do it — pick one, defend it in the report:

    1. Cue matching plus the document model. Find sentences containing a cue for
       aspect A (``ASPECT_CUES``), run the sentiment model on just those
       sentences. Cheap, transparent, and it fails on "the acting was fine, the
       rest was a disaster" in a way you can demonstrate.
    2. A sentence classifier: split into sentences, assign each to an aspect
       (or none), pool per aspect. Better on mixed reviews, more moving parts.
    3. Fine-tune a model that takes the aspect as input alongside the text
       (the aspect-as-a-question framing). Strongest, and it needs
       aspect-annotated data — which is the cost you priced up in
       ``load_reviews``.

    Whichever you choose, the failure to look for first is the one where the
    document is negative overall and every aspect comes back negative because
    the aspect step is really just reading the document label. If your aspect
    breakdown never disagrees with the overall sentiment, it is decoration. Test
    it deliberately on a mixed review and put the result in the report.
    """
    raise NotImplementedError("Implement extract_aspects().")


# ===========================================================================
# 5. Evaluation
#
# These are pure functions over predictions and gold labels: no model, no
# network, no database. That is why the contract tests for them run in
# milliseconds and why you should write them before you have anything to
# evaluate. A metric function you cannot test on eight hand-written examples is
# a metric function you do not trust.
# ===========================================================================
def evaluate_documents(
    predictions: Sequence[SentimentPrediction],
    gold: Sequence[str],
    model_name: str,
    n_calibration_bins: int = 10,
) -> DocumentMetrics:
    """Held-out document-level metrics for one model.

    Call it twice — once with the transformer's predictions and once with the
    baseline's — and store both. The assignment asks for accuracy, precision,
    recall and macro-F1 for BOTH models; a report with only the transformer's
    numbers is missing half its argument.

    Fill every field:

    * ``per_class`` — precision, recall, F1 and support for each label.
    * ``macro_f1`` — the unweighted mean of the per-class F1s. Not the F1 of the
      pooled counts (that is micro, and on a balanced binary task it equals
      accuracy, which is how people report accuracy twice by accident).
    * ``confusion_matrix`` — ``[gold][predicted]``, indexed by ``labels``. Get
      the orientation right and state it in the tab; a transposed matrix turns
      a false-positive problem into a false-negative one.
    * ``roc_points`` — sweep the threshold over the predicted probabilities and
      record (threshold, fpr, tpr). The curve is a property of the SCORES, not
      of the 0.5 label, which is exactly why it can look excellent for a model
      whose default threshold is in the wrong place.
    * ``calibration_bins`` — bin by predicted probability, then for each bin
      report the mean predicted probability AND the observed fraction of
      positives. Those are two different quantities. If your plot is a clean
      diagonal on the first try, check that you did not plot the mean predicted
      probability against itself; a genuinely well-calibrated uncalibrated
      transformer would be a surprise worth investigating rather than accepting.

    Raise ``ValueError`` if ``predictions`` and ``gold`` differ in length. That
    mistake produces plausible-looking metrics on misaligned data, which is the
    worst kind of bug: silent, and wrong in your favour about half the time.
    """
    raise NotImplementedError("Implement evaluate_documents().")


def evaluate_aspects(
    predicted: Sequence[Sequence[AspectSentiment]],
    gold: Sequence[Dict[str, str]],
) -> List[AspectMetrics]:
    """Per-aspect precision, recall and F1 over the aspect-labelled split.

    Parameters
    ----------
    predicted
        One list of ``AspectSentiment`` per document (what ``extract_aspects``
        returned).
    gold
        One dict per document, mapping aspect -> gold label. An aspect missing
        from the dict has no gold annotation for that document; skip it rather
        than scoring it, and count what you skipped in ``n_evaluated``.

    Returns one ``AspectMetrics`` per aspect in ``ASPECTS``, always — including
    aspects with no gold data at all, reported with ``support=0`` and
    ``n_evaluated=0`` rather than silently omitted. A missing row looks like an
    aspect that scored zero; an explicit zero-support row looks like what it is,
    which is a gap in your annotation.

    Decide and document how ``not_mentioned`` participates. Treating it as a
    fourth class scores the model on noticing absence, which is a real skill.
    Excluding those documents scores it only where a human found something to
    say. The two give very different F1s on the same predictions, so state which
    one you used next to the number.
    """
    raise NotImplementedError("Implement evaluate_aspects().")


def slice_of(record: Dict[str, Any], slice_name: str) -> str:
    """Assign one held-out record to a bucket of one slice.

    Returns one of the buckets declared for ``slice_name`` in ``SLICE_SPECS``.
    Raise ``ValueError`` for a slice name you do not know about.

    ``review_length`` you can measure: count whitespace tokens in
    ``record["text"]`` and apply the documented thresholds. ``genre`` you can
    only read, from a field the corpus gave you. If neither is available and you
    reach for a heuristic, that is allowed — but the resulting ``SliceMetrics``
    must carry ``observed=False``, because from then on you are auditing your
    guess as much as your classifier.
    """
    raise NotImplementedError("Implement slice_of().")


def evaluate_slices(
    records: Sequence[Dict[str, Any]],
    predictions: Sequence[SentimentPrediction],
    slice_name: str,
) -> List[SliceMetrics]:
    """Model performance broken down by one slice — the Bias Audit tab's input.

    ``records`` are held-out rows as ``load_reviews`` returns them — each with
    its gold ``label`` — and ``predictions[i]`` is the model's answer for
    ``records[i]``. Group the records by ``slice_of(record, slice_name)``,
    compute accuracy and macro-F1 within each bucket against those gold labels,
    and return one ``SliceMetrics`` per bucket that actually has data. Set ``observed`` from
    ``SLICE_SPECS[slice_name]["observed"]`` unless you inferred the attribute
    yourself, in which case it is False regardless of what the spec says.

    The assignment needs at least two slices; call this once per slice and store
    the concatenated list in ``runs.metrics["slices"]``.

    Read the ``n`` column before you read the gap. A bucket of 23 reviews will
    happily show a twelve-point accuracy difference that is nothing but sample
    size, and "we found bias" is a serious claim to make on 23 examples. Where a
    bucket is small, say so in the tab rather than letting the bar chart imply a
    precision it does not have. The honest version of this section sometimes
    reads "the gap is 4 points on 800 and 90 reviews respectively, which we
    cannot distinguish from noise", and that sentence scores better than a
    confident wrong one.
    """
    raise NotImplementedError("Implement evaluate_slices().")
