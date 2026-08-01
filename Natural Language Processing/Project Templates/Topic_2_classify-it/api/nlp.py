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

The order below is the order you should implement in. Data first, then the
baseline, then the transformer, then serving. Do not start with the fine-tuning:
if your loader is wrong you will spend an hour training on garbage before you
find out.

WHERE THE TRAINING ACTUALLY RUNS
--------------------------------
Not here. ``fit_baseline`` and ``fine_tune_transformer`` are training entry
points you call from a notebook, a Colab session, or a local script — never from
a request handler. The free Render plan has a few hundred megabytes of RAM; a
DistilBERT fine-tune wants gigabytes and will be killed mid-epoch. Train
somewhere with memory, commit or upload the artifact, and let the web service do
nothing but load it and answer.

WHAT "DONE" LOOKS LIKE
----------------------
``pytest`` passes with no skips other than the Supabase round-trip (which needs
real credentials), the Baseline vs. Transformer tab shows four real metrics for
each model, and the Recent Predictions tab fills up as you use the Classify tab.

A NOTE ON LIBRARIES
-------------------
Use scikit-learn for the baseline (``TfidfVectorizer`` + ``LogisticRegression``,
and ``CalibratedClassifierCV`` if you want it) and the Hugging Face
``transformers`` Trainer API for the fine-tune. Both are already pinned in
``api/requirements.txt``. Write your own metric arithmetic in ``compute_metrics``
at least once before you reach for ``sklearn.metrics`` — you cannot argue about
precision versus recall in your report if you have never counted the four cells
of a confusion matrix by hand.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Tuple

from shared.schemas import (
    LabeledExample,
    LabelSchema,
    Metrics,
    ModelKind,
    PredictionResult,
)

# ---------------------------------------------------------------------------
# Where trained artifacts live. The web service reads from here at startup; it
# never writes here. Keep the directory out of git if the artifacts are large
# (.gitignore already excludes artifacts/) and fetch them at build time instead.
# ---------------------------------------------------------------------------
ARTIFACT_DIR = "artifacts"

# Module-level caches. Loading a model per request is the single easiest way to
# make a correct service feel broken: a DistilBERT checkpoint takes seconds to
# deserialize, and Render's free plan will look like it has hung. Populate these
# once, in load_model().
_MODEL_CACHE: Dict[str, Any] = {}


def sha256_text(text: str) -> str:
    """Hash the input so predictions are auditable without storing anyone's text.

    IMPLEMENTED FOR YOU — this is the privacy rule the whole product depends on,
    so it is not left to chance. The corpus for this product is support messages
    or survey free text: it is other people's writing, and the audit trail does
    not need it. Two identical inputs give the same hash, which is enough to spot
    a repeated question or a replayed test.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 1. The corpus
# ---------------------------------------------------------------------------
def load_labeled_dataset(source: str) -> List[LabeledExample]:
    """Load the labeled corpus into a list of LabeledExample, in file order.

    Parameters
    ----------
    source
        A path to a CSV or JSONL file, or a Supabase table name. Support the CSV
        case at minimum: a file with a ``text`` column and a ``label`` column,
        header row present. The contract test writes exactly that file.

    Returns
    -------
    A list of LabeledExample. Rows with empty text or an empty label are DROPPED,
    not repaired — a blank support message with a label attached is noise the
    annotator left behind, and imputing a value for it invents supervision.

    Do not shuffle here and do not split here. Loading is one job; ordering and
    splitting are another, and mixing them is how a corpus that happens to be
    sorted by label ends up with a training set containing one class.

    THE TRAP: the label column often arrives as ``0``/``1`` integers, or as
    ``"0"``/``"1"`` strings, or as ``True``/``False``. Normalize to the human
    label strings your LabelSchema declares ("spam"/"ham", "urgent"/"routine")
    and do it here, once. If half your pipeline says ``1`` and half says
    ``"spam"``, every join you write afterwards is silently empty.
    """
    raise NotImplementedError(
        "Implement load_labeled_dataset() — see the docstring above."
    )


def validate_dataset(examples: List[LabeledExample]) -> LabelSchema:
    """Check the corpus against the product's rules and describe it.

    Returns
    -------
    A LabelSchema with ``labels``, ``positive_label``, ``class_counts`` and
    ``n_rows`` populated from the data itself. Fill in ``dataset_name``,
    ``dataset_source`` and ``label_definitions`` from your config — the Build
    Steps require the source and the label definitions to be documented, and the
    /schema endpoint is where a reader finds them.

    Raises
    ------
    ValueError
        - if the corpus is empty;
        - if it contains anything other than EXACTLY TWO distinct label values.

    That second rule is the product definition, not a limitation you are working
    around. Single-label with two values: one label per example, two possible
    values. A three-value corpus needs a different loss reduction, a different
    precision/recall definition (macro? micro? per class?), and a different UI,
    and a multilabel corpus needs a different loss function entirely. Both are
    covered in later courses. Fail loudly here rather than producing a model
    whose reported F1 nobody can interpret.

    While you are in here, LOOK at the class counts you just computed. If the
    split is 95/5, accuracy is about to lie to you for the rest of the project —
    ``LabelSchema.minority_share()`` is implemented for you, and the number it
    returns belongs in your report and your model card.
    """
    raise NotImplementedError("Implement validate_dataset().")


def stratified_split(
    examples: List[LabeledExample], test_size: float = 0.2, seed: int = 42
) -> Tuple[List[LabeledExample], List[LabeledExample]]:
    """Split into (train, held_out), preserving each label's proportion.

    Returns
    -------
    Two disjoint lists whose union is the input. Every metric you report comes
    from the second one, and the second one must never be seen during training,
    hyperparameter tuning, or threshold picking.

    Stratify: draw the test fraction from EACH class separately, then combine.
    On a 90/10 corpus a plain random 20% split can hand you a held-out set with
    almost no positives, at which point recall is computed over a handful of rows
    and swings wildly between runs for no reason.

    ``seed`` must make this deterministic. The same seed and the same corpus give
    the same split, every time, on every machine — otherwise you cannot compare
    the baseline's numbers to the transformer's, because they were graded on
    different exams.

    THE TRAP: near-duplicate rows. Many support-ticket corpora contain the same
    boilerplate message dozens of times. If duplicates straddle the split, your
    held-out score is partly a memorization score. De-duplicate on
    ``sha256_text(example.text)`` before splitting and say in your report how
    many rows that removed.
    """
    raise NotImplementedError("Implement stratified_split().")


# ---------------------------------------------------------------------------
# 2. The classical baseline — TF-IDF + logistic regression
# ---------------------------------------------------------------------------
def fit_baseline(
    train: List[LabeledExample], config: Dict[str, Any]
) -> Dict[str, Any]:
    """Fit TF-IDF + logistic regression and persist the artifact.

    Call this from a script or a notebook, not from a request handler.

    Parameters
    ----------
    train
        Training rows only. Passing the whole corpus here is the classic
        leak — your held-out score becomes a training score and everything
        downstream is wrong by an amount you cannot estimate.
    config
        The ``baseline`` block of api/configs/default.yaml: n-gram range, min_df,
        max_features, C, class_weight, max_iter, solver.

    Returns
    -------
    A metadata dict that goes straight into the ``runs`` table:
    ``{"model_kind": "baseline", "model_version": ..., "hyperparameters": {...},
    "artifact_path": ..., "n_train": ...}``. ``hyperparameters`` must contain
    every value that would change the result — including the vectorizer settings,
    which students routinely forget are hyperparameters too.

    Persist the FITTED VECTORIZER AND THE CLASSIFIER TOGETHER (an sklearn
    Pipeline pickled with joblib is the clean way). This is the number-one
    baseline bug: refitting the vectorizer at prediction time produces a
    different vocabulary and a different feature ordering, so the coefficients
    are indexed against features that no longer mean what they meant. The model
    does not crash. It just quietly gets worse.

    Raises ValueError if ``train`` does not contain exactly two distinct labels —
    validate before you spend compute, and do it here rather than letting
    scikit-learn produce a shape error six frames down.

    This baseline is not a formality. On short, keyword-driven text with a large
    corpus, TF-IDF + logistic regression is genuinely hard to beat, it trains in
    seconds, it costs nothing to serve, and it is directly inspectable. If your
    transformer does not clear it, that is a finding, and your report should say
    so instead of hiding it.
    """
    raise NotImplementedError("Implement fit_baseline().")


# ---------------------------------------------------------------------------
# 3. The transformer — fine-tuning DistilBERT with the Trainer API
# ---------------------------------------------------------------------------
def fine_tune_transformer(
    train: List[LabeledExample],
    held_out: List[LabeledExample],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Fine-tune a pretrained encoder for binary classification and persist it.

    Run this in Colab, on a lab machine, or anywhere with real memory. NOT on
    the Render web service, and not inside a request.

    Parameters
    ----------
    config
        The ``transformer`` block of api/configs/default.yaml: checkpoint
        (``distilbert-base-uncased``), max_length, learning_rate, epochs,
        per-device batch size, weight_decay, warmup, seed.

    Returns
    -------
    The same metadata shape as fit_baseline(), so both rows land in ``runs``
    and the Baseline vs. Transformer tab can render them side by side.

    THE SHAPE OF THE WORK
    ---------------------
    * ``AutoTokenizer.from_pretrained(checkpoint)``, then tokenize with
      truncation to ``max_length`` and dynamic padding via
      ``DataCollatorWithPadding``. Padding every example to 512 when your median
      message is 30 tokens makes training several times slower and changes
      nothing about the result.
    * ``AutoModelForSequenceClassification.from_pretrained(checkpoint,
      num_labels=2)``. Two, because the product is binary. Pass ``id2label`` and
      ``label2id`` so the saved artifact carries your human label names — a
      checkpoint that only knows LABEL_0 and LABEL_1 is a decoding bug that will
      surface the next time someone loads it.
    * ``Trainer`` + ``TrainingArguments``, evaluating each epoch on ``held_out``.
    * ``save_pretrained`` the model AND the tokenizer into the artifact
      directory. They are a matched pair; a checkpoint loaded with a different
      tokenizer produces confident nonsense.

    TRAPS, IN THE ORDER STUDENTS HIT THEM
    -------------------------------------
    1. Fine-tuning learning rates are small: 2e-5 to 5e-5. The Adam default of
       1e-3 will destroy the pretrained weights in the first few hundred steps
       and you will get a model that outputs one class for every input. If your
       loss goes flat and accuracy equals the majority-class share, look here
       first.
    2. Two or three epochs is usually right. A pretrained encoder memorizes a
       small corpus fast; by epoch five your training loss is beautiful and your
       held-out F1 has been falling for three epochs.
    3. Class imbalance survives fine-tuning. If your corpus is 95/5, the cheapest
       way to minimize cross-entropy is to always answer with the majority class,
       and the model will find it. Weight the loss, or resample, or at minimum
       select on F1 rather than accuracy — and report which you did.
    4. Set the seed and record it. "It got 0.91 last night" is not a result if
       you cannot get 0.91 again.

    Raises ValueError if the training data does not have exactly two distinct
    labels. Check this on line one, before the checkpoint download.
    """
    raise NotImplementedError("Implement fine_tune_transformer().")


# ---------------------------------------------------------------------------
# 4. Serving
# ---------------------------------------------------------------------------
def load_model(model_kind: ModelKind) -> Any:
    """Load a trained artifact from ARTIFACT_DIR and cache it in _MODEL_CACHE.

    Returns whatever your predict path needs: the joblib Pipeline for
    ``"baseline"``, a ``(tokenizer, model)`` pair for ``"transformer"``. Return
    the SAME object on every subsequent call — check the cache first.

    Raise ValueError with a readable message when the artifact is missing
    ("no baseline artifact in artifacts/; run fit_baseline() first"). The API
    turns ValueError into a 400 and the UI prints it, which is a far better
    experience than a stack trace about a missing pickle.

    Raise ValueError for an unknown model_kind too. The schema already restricts
    the field to two values, but this function is also called from your training
    scripts where nothing validates the string for you.

    Put the transformer in eval mode and disable gradients for inference. Leaving
    autograd on for a forward pass you never back-propagate wastes memory you do
    not have on the free plan, and dropout left active makes the same input
    return different answers on different requests — which looks exactly like a
    bug in your logging.
    """
    raise NotImplementedError("Implement load_model() with a module-level cache.")


def score_texts(texts: List[str], model_kind: ModelKind) -> List[Dict[str, float]]:
    """Raw, UNCALIBRATED per-label scores for a list of texts.

    Returns
    -------
    One dict per input text, in the same order, mapping every label in the
    schema to a score in [0, 1] that sums to 1 across the two labels. For the
    baseline that is ``predict_proba``; for the transformer it is a softmax over
    the two logits.

    Do the whole list in ONE forward pass (one ``vectorizer.transform`` call, one
    batched tokenizer call and one model call). Looping and calling the model
    once per text is roughly an order of magnitude slower for no benefit, and
    /predict_batch is where that shows up.

    Call the returned numbers "scores", not "probabilities", everywhere in your
    own code. A softmax output is a normalized exponential of two logits; nothing
    in training forced it to match an observed frequency, and a fine-tuned
    encoder is usually overconfident — it will say 0.99 on inputs it gets wrong
    about one time in ten. That gap is what calibrate() exists to close.
    """
    raise NotImplementedError("Implement score_texts().")


def calibrate(
    raw_scores: List[Dict[str, float]], model_kind: ModelKind
) -> List[Dict[str, float]]:
    """Map raw model scores onto probabilities that mean what they say.

    Returns the same list-of-dicts shape, same order, values still in [0, 1] and
    still summing to 1 per text, but now calibrated: of all the inputs this
    function assigns 0.80 to, close to 80% should actually carry that label.

    WHY THIS IS A SEPARATE FUNCTION AND NOT ONE LINE OF predict()
    ------------------------------------------------------------
    The product brief promises "a calibrated probability". The softmax output is
    not one. Neither is ``LogisticRegression.predict_proba`` once you have
    changed ``class_weight`` or resampled the data — reweighting shifts the base
    rate the model implicitly assumes, so the numbers no longer match the world
    the model will be used in. Calibration is a second, tiny model fitted on a
    held-out slice that maps scores to observed frequencies: Platt scaling (fit a
    one-feature logistic regression on the scores) or isotonic regression if you
    have a few thousand held-out rows. ``sklearn.calibration`` has both.

    Fit the calibrator on data the classifier did NOT train on, persist it beside
    the model, and load it here. Calibrating on the training set produces a
    calibrator that certifies the model's memorized confidence, which is worse
    than doing nothing because now it looks principled.

    Calibration must preserve ORDER: if text A scored higher than text B before,
    it scores at least as high after. Both Platt and isotonic are monotonic, so
    if your output reorders anything, you have a bug — probably an argsort or a
    label-column mismatch, not a modelling subtlety.

    A defensible fallback while you are still building: return the input
    unchanged and record ``"calibration": "none"`` in the run's hyperparameters,
    so the model card does not claim something the code does not do. Shipping
    uncalibrated numbers labelled "calibrated" is the actual failure here.

    Show the evidence in your report: bucket held-out predictions by predicted
    probability, plot predicted against observed frequency, and show the curve
    before and after. That plot is the whole argument.
    """
    raise NotImplementedError("Implement calibrate().")


def predict(text: str, model_kind: ModelKind) -> PredictionResult:
    """Classify one text and return the full, loggable result.

    Every field of PredictionResult must be real:

    * ``label`` — the argmax label. One of the two values in your LabelSchema,
      spelled the way the schema spells it. Not ``LABEL_1``, not ``1``.
    * ``probability`` — the CALIBRATED probability OF THE PREDICTED LABEL, so it
      is always at least 0.5 in a two-class problem. Returning the probability of
      the positive class regardless of what was predicted is a real bug that
      makes the UI show "routine, 12% confident", which is nonsense.
    * ``model_kind`` — which model answered.
    * ``model_version`` — which ARTIFACT answered. Read it from the artifact's
      own metadata, not from an environment variable, so that a row logged today
      still identifies the right model after three redeploys.
    * ``text_sha256`` — ``sha256_text(text)``. Never the text.

    Leave ``latency_ms`` and ``prediction_id`` as None; api/main.py fills those.

    Implement this by calling score_texts([text], ...) then calibrate(...) —
    one code path for one and for many means the batch endpoint and the single
    endpoint cannot disagree, which they otherwise will, on rounding, at the
    worst possible moment during your demo.
    """
    raise NotImplementedError("Implement predict().")


def predict_batch(texts: List[str], model_kind: ModelKind) -> List[PredictionResult]:
    """Classify many texts in one pass. Same contract as predict(), per element.

    Returns exactly ``len(texts)`` results IN THE SAME ORDER as the input. The
    caller lines these up against their own rows by position; a sort or a set
    operation anywhere in here mislabels their whole file and nothing errors.

    This must be genuinely batched — one vectorizer/tokenizer call, one model
    call — not a list comprehension over predict(). Batching is most of the
    reason this endpoint exists.

    Empty input returns an empty list rather than raising. An empty batch is a
    boring edge case, not an error.
    """
    raise NotImplementedError("Implement predict_batch().")


def label_schema() -> LabelSchema:
    """Describe what this deployed service predicts. Backs GET /schema.

    Return a fully populated LabelSchema: both labels, which one is positive,
    a one-sentence definition of each, the training class counts, the dataset
    name and its public source, and the row count.

    Read it from the artifact metadata or from api/configs/default.yaml — do not
    hard-code a second copy of the label list in this function. Two copies drift,
    and the one that drifts is always the one the UI reads.

    This endpoint is how a reviewer answers "what does 'urgent' mean here?"
    without your notebook. Treat it as documentation that is forced to stay true
    because it is served from the same place the model was built.
    """
    raise NotImplementedError("Implement label_schema().")


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: List[str], y_pred: List[str], positive_label: str
) -> Metrics:
    """Accuracy, precision, recall and F1 for one model on the held-out split.

    Returns a Metrics object. Count the confusion matrix yourself at least once:

        TP = predicted positive AND actually positive
        FP = predicted positive AND actually negative
        FN = predicted negative AND actually positive
        TN = predicted negative AND actually negative

        accuracy  = (TP + TN) / n
        precision = TP / (TP + FP)      "when it says yes, how often is it right"
        recall    = TP / (TP + FN)      "of the real yeses, how many did it find"
        f1        = 2PR / (P + R)

    Guard every denominator. A model that never predicts the positive class has
    TP = FP = 0, and precision is 0/0. Return 0.0 for that case — but notice
    what it is telling you before you move on, because a model that never
    predicts the positive class is exactly what class imbalance produces.

    ``support_positive`` is how many held-out rows actually carry
    ``positive_label``. Report it. Recall over 8 positives moves in steps of
    12.5 percentage points, and quoting it as 0.875 implies a precision the data
    does not support.

    THE TRAP THIS WHOLE FUNCTION EXISTS TO EXPOSE: on a 90/10 corpus, predicting
    the majority label for every single input scores 0.90 accuracy — with recall
    0.00 and F1 0.00. There is a contract test that asserts exactly this. If your
    transformer beats your baseline on accuracy but not on F1, you have not built
    a better classifier; you have built one that is better at agreeing with the
    majority class, and your report needs to say so.

    Raise ValueError if the two lists are different lengths. Zipping mismatched
    predictions and labels produces a number, and the number is meaningless.
    """
    raise NotImplementedError("Implement compute_metrics().")
