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
real credentials), both taggers answer POST /tag, and the comparison tab shows
two accuracy numbers and a confusion matrix you can read.

THE TWO TAGGERS, AND WHY THE BASELINE IS THE INTERESTING ONE
------------------------------------------------------------
The baseline here is a most-frequent-tag lookup: count how often each word takes
each tag in the training split, remember the winner, and tag unseen words with a
handful of readable rules. It has three properties that make it the right thing
to compare against.

* It needs no training. There is no loss curve, no learning rate, no seed. You
  build it with a dictionary and a counter, and it is finished in a second.
* It is completely inspectable. You can print the table. When it tags a word
  wrong you can look up the exact entry that did it and read the rule.
* It is already good — on ordinary English text a most-frequent-tag lookup gets
  roughly nine words in ten right. Measure it yourself and report your number;
  the point is that it is high enough to be a serious opponent.

That last property is what makes the topic work. A tagger that is right ninety
per cent of the time and wrong in a way you can point at gives you a precise
question: *which* words does it miss, and what would it take to get them? The
answer is almost always context. "book" is a NOUN in the table, so "book me a
flight" gets it wrong, and the lookup cannot ever be right about both, because
it has no way to look at the words either side. That is the gap a contextual
model closes, and your job is to show the gap and then measure how much of it
the transformer actually closes on your data.

A NOTE ON LIBRARIES
-------------------
Use the Hugging Face ``transformers`` stack for the token classifier. For the
lookup table, the fallback rules, and the metrics, write the logic yourself —
each is a few lines, and the fallback rules in particular are graded on being
readable. A metrics library is acceptable if you prefer it, but you should be
able to say what macro-F1 is doing to your rare tags without looking it up, and
writing it once is how that happens.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from shared.schemas import (
    ConfusionMatrix,
    EvalReport,
    TaggedToken,
    TagResponse,
)

# ---------------------------------------------------------------------------
# The 17 universal part-of-speech tags. Use this tag set unless you have a
# reason not to, and if you change it, say so everywhere a number is reported.
#
# The size is a design decision, not a detail. Seventeen tags give you a 17x17
# confusion matrix — 289 cells, which a reader scans in a few seconds and can
# point at. The Penn-style tag set has 45 tags and 2,025 cells; nobody reads
# that, and the pairs you care about (NOUN/VERB, ADJ/ADV, DET/PRON) get buried
# in distinctions like singular-versus-plural noun that no downstream feature in
# this product uses. Finer tags also split your training data more thinly, so
# the rare ones get rarer.
# ---------------------------------------------------------------------------
UNIVERSAL_POS_TAGS: Tuple[str, ...] = (
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
)

# A gloss for each tag, used by the UI legend and by your Concepts tab. Read it
# once: several of the confusions you will find are between neighbours here.
TAG_GLOSS: Dict[str, str] = {
    "ADJ": "adjective — old, green, second",
    "ADP": "adposition — in, to, during",
    "ADV": "adverb — very, well, exactly",
    "AUX": "auxiliary — is, has (as a helper), will, should",
    "CCONJ": "coordinating conjunction — and, or, but",
    "DET": "determiner — the, a, this, which",
    "INTJ": "interjection — psst, ouch, hello",
    "NOUN": "common noun — girl, tree, air",
    "NUM": "numeral — one, 2024, IV",
    "PART": "particle — 's, not, the 'to' of an infinitive",
    "PRON": "pronoun — I, you, myself, which",
    "PROPN": "proper noun — Mary, London, NASA",
    "PUNCT": "punctuation — . , ( )",
    "SCONJ": "subordinating conjunction — that, if, while",
    "SYM": "symbol — $, %, +, emoji",
    "VERB": "verb — run, ate, thinking",
    "X": "other — a word that fits nowhere, e.g. a foreign fragment",
}

# Where your fine-tuned model is saved. Keep the trained weights OUT of git (see
# .gitignore) and publish them to the Hugging Face Hub instead, then point this
# at the Hub id. A repository with a few hundred megabytes of checkpoints in it
# is a repository nobody can clone.
TRANSFORMER_MODEL_PATH = "models/tagwise-transformer"


def sha256_text(text: str) -> str:
    """Hash the input so requests are reproducible without storing anyone's text.

    IMPLEMENTED FOR YOU — this is the privacy rule the whole product depends on,
    so it is not left to chance. Use it everywhere you would be tempted to log
    the raw sentence.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 1. The corpus
# ---------------------------------------------------------------------------
def load_treebank(path: str) -> List[List[Tuple[str, str]]]:
    """Read one CoNLL-U file into sentences of (word, tag) pairs.

    Parameters
    ----------
    path
        A ``.conllu`` file from a part-of-speech-annotated treebank — the English
        treebanks at https://universaldependencies.org/#english-treebanks are the
        intended source. Download the train/dev/test files once and keep them out
        of git; record in your MODEL_CARD which treebank and which revision, so
        your accuracy number means something to someone else.

    Returns
    -------
    A list of sentences. Each sentence is a list of ``(form, upos)`` tuples in
    document order. An empty file returns an empty list.

    The format, in the three lines you actually need:

    * Blank lines separate sentences.
    * Lines starting with ``#`` are comments — skip them.
    * Every other line is tab-separated. Column 1 is the ID, column 2 is the
      word form, column 4 is the universal tag. Count from one.

    Two traps in the format itself. Some ID fields look like ``8-9`` (a contracted
    token such as "don't" spanning two real words) and some look like ``5.1`` (an
    empty node used by one of the annotation layers). Neither carries a universal
    tag you want; skip any row whose ID is not a plain integer, or you will
    silently double-count words and every number downstream will be slightly off.

    Keep the sentence boundaries. Flattening the corpus into one long list of
    words costs you nothing for the lookup baseline and makes the transformer
    impossible — context is the entire point, and context stops at the sentence.
    """
    raise NotImplementedError("Implement load_treebank() — see the docstring above.")


def describe_corpus(sentences: List[List[Tuple[str, str]]]) -> Dict[str, Any]:
    """Summarise one split: how big it is and how the tags are distributed.

    Returns a dict with at least these keys:

    * ``n_sentences`` — int
    * ``n_tokens`` — int, counting every word in every sentence
    * ``tagset`` — the sorted list of distinct tags that occur
    * ``tag_counts`` — {tag: int}
    * ``tag_distribution`` — {tag: proportion}, summing to 1.0 (empty input gives
      an empty dict, not a division by zero)

    Run this on train, dev, and test and put all three in your README. It is not
    decoration. Three things fall out of it immediately:

    * The largest tag's share is the accuracy of a tagger that guesses that tag
      for every word. That is the floor. If your transformer is near it, your
      transformer is not working.
    * A tag with a hundred instances in train and eight in test cannot be
      measured reliably, and it is exactly the tag that will swing your macro-F1
      from one run to the next.
    * If the tag distributions of your splits differ noticeably, the splits are
      not comparable and neither are the numbers you compute on them.
    """
    raise NotImplementedError("Implement describe_corpus().")


# ---------------------------------------------------------------------------
# 2. The lookup baseline
# ---------------------------------------------------------------------------
def build_lookup_table(sentences: List[List[Tuple[str, str]]]) -> Dict[str, str]:
    """Build the most-frequent-tag table from the TRAINING split.

    For every word form, count the tags it takes, and keep the most frequent one.
    Ties go to whichever tag is more frequent in the corpus overall; whatever you
    choose, write the rule down in the MODEL_CARD and put it in the run's
    hyperparameters, because a tie-break is a modelling decision even though it
    does not look like one.

    Returns ``{word: tag}``. Nothing else. That is the entire model: no weights,
    no epochs, no seed, and you can print it and read it.

    Build it from the training split only. Building it from all the data is the
    fastest way to produce an accuracy number that is both excellent and
    meaningless, and it is a mistake that hides well because nothing crashes.

    Casing is your call and it is a real trade-off. Lowercasing the keys shrinks
    the table and cuts the unknown-word rate, but it merges "Apple" with "apple"
    and throws away the strongest signal you have for PROPN. Keeping case does
    the reverse: better proper nouns, more unknown words, and the first word of
    every sentence becomes its own entry. Pick one, record it, and if you have
    time, measure both — the difference is a good paragraph in your report.
    """
    raise NotImplementedError("Implement build_lookup_table().")


def fallback_tag(word: str) -> str:
    """Guess a tag for a word the lookup table has never seen. WRITE IT READABLY.

    This function is where nearly all of the baseline's remaining error lives.
    The table handles the words it has seen almost perfectly by construction; the
    unknown words are the ones left, and they are systematically the hard ones —
    new proper nouns, technical terms, typos, numbers, and anything from a domain
    the treebank did not cover.

    That makes this function the pedagogical centre of the baseline, so it is
    graded on being READABLE. A stack of ordered ``if`` statements that a
    non-programmer could follow is the correct answer here. A clever regex that
    encodes eight rules at once is the wrong answer even if it scores higher,
    because the whole reason to keep a baseline is to be able to point at the
    line that produced a wrong tag.

    A reasonable rule set, in an order you should think about rather than copy:

    * digits anywhere, or a decimal/comma-formatted number -> NUM
    * every character is punctuation -> PUNCT
    * a currency or maths symbol -> SYM
    * starts with a capital letter -> PROPN
    * ends in ``-ly`` -> ADV
    * ends in ``-ing``, ``-ed``, ``-ise``, ``-ize`` -> VERB
    * ends in ``-ous``, ``-ful``, ``-able``, ``-ive``, ``-al`` -> ADJ
    * ends in ``-ness``, ``-tion``, ``-ment``, ``-ity`` -> NOUN
    * anything else -> NOUN

    NOUN is the right default because unknown words are overwhelmingly nouns:
    the open classes grow, the closed classes ("the", "of", "and") do not, so a
    word you have never seen is far more likely to be a noun than a determiner.

    Order is doing real work above. Capitalisation is checked before the suffix
    rules, which means a sentence-initial "Running" comes back PROPN instead of
    VERB. That is a genuine error, it will show up in your confusion matrix, and
    it is worth one sentence in your report rather than a hack that hides it.

    Returns one tag from UNIVERSAL_POS_TAGS. Never returns None.
    """
    raise NotImplementedError("Implement fallback_tag() — readably, see the docstring.")


def tag_with_lookup(
    tokens: List[str], lookup: Dict[str, str]
) -> List[TaggedToken]:
    """Tag a tokenised sentence with the table, falling back where it must.

    For each token: if it is in the table, use that tag and set
    ``used_fallback=False``. Otherwise call ``fallback_tag`` and set
    ``used_fallback=True``. Leave ``confidence`` as None — the lookup has no
    probability of its own, and writing 1.0 there would tell the UI something
    untrue, which the UI would then show to a user.

    Apply the same casing rule you used in ``build_lookup_table``. Building the
    table on lowercased forms and then looking up the original casing produces a
    tagger where every capitalised word is unknown, which is a bug that looks
    like a modelling result.

    Notice what this function cannot do: it sees one token at a time. Whatever
    "book" was in the training data, it is that here, in every sentence, forever.
    Hold on to that when you read the confusion matrix — the errors are not
    random, they are the ambiguous words, and they are the same ones every time.
    """
    raise NotImplementedError("Implement tag_with_lookup().")


# ---------------------------------------------------------------------------
# 3. The transformer token classifier
# ---------------------------------------------------------------------------
def align_tags_to_subwords(
    word_ids: Sequence[Optional[int]],
    tags: List[str],
    tag_to_id: Dict[str, int],
) -> List[int]:
    """Spread word-level tags onto subword pieces. THE SILENT-FAILURE FUNCTION.

    Parameters
    ----------
    word_ids
        What the fast tokenizer's ``encoding.word_ids()`` gives you: one entry per
        subword piece, holding the index of the word that piece came from, or
        None for special tokens such as [CLS] and [SEP].
    tags
        One tag per WORD — the gold sequence, same length as the sentence.
    tag_to_id
        Your label map, {tag: integer id}.

    Returns
    -------
    One integer per piece, same length as ``word_ids``:

    * special tokens (word id None) -> ``-100``
    * the FIRST piece of each word -> ``tag_to_id[tag of that word]``
    * every later piece of the same word -> ``-100``

    ``-100`` is not arbitrary. It is the ignore index PyTorch's cross-entropy
    loss skips by default, so those positions contribute nothing to the loss and
    nothing to your training signal.

    WHY THIS EXISTS AT ALL. The transformer does not see words. It sees subword
    pieces, so "unhappiness" might arrive as three pieces and "Zurich" as two,
    while your gold data has exactly one tag per word. The two sequences have
    different lengths, and you have to decide what a tag means for a piece in the
    middle of a word.

    WHY IT IS THE MOST DANGEROUS FUNCTION IN THE FILE. Get it wrong and nothing
    crashes. Zip the tag list against the pieces and every tag after the first
    multi-piece word is attached to the wrong token — the model trains happily,
    the loss goes down, and accuracy comes out mediocre in a way that looks like
    "the model needs more epochs" rather than "the labels are shifted". Students
    lose days here. Before you train anything, print the first two sentences as
    a table of piece, word id, and label id, and check by eye that the first piece
    of each word carries the tag and the rest are -100.

    At prediction time you undo this: read the model's output at the first piece
    of each word and ignore the rest, so you return one tag per word. Whatever
    the model thinks about "##ness" on its own is not a fact about a word, and
    your API contract promises one tag per token.
    """
    raise NotImplementedError("Implement align_tags_to_subwords() — read the whole docstring.")


def fine_tune_transformer(
    train_sentences: List[List[Tuple[str, str]]],
    dev_sentences: List[List[Tuple[str, str]]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Fine-tune a token classifier on the same splits the baseline used.

    Run this OFFLINE, from your own script, not from a request handler. Then call
    ``api.db.insert_run`` with what it returns, so the build is on the record.

    Parameters
    ----------
    config
        At least: ``base_model`` (a checkpoint id), ``learning_rate``,
        ``epochs``, ``batch_size``, ``max_length``, ``seed``. Read the defaults
        from ``api/configs/default.yaml`` so a run can be reproduced from the
        repository alone.

    Returns
    -------
    A dict with the hyperparameters actually used, the path the model was saved
    to, and the dev-set metrics. Everything in it goes into the ``runs`` row.

    The shape of the work:

    1. Build the label map from the training tags, both directions. Store it with
       the model — a saved classifier whose label order you cannot recover is a
       saved classifier you cannot use.
    2. Tokenize with ``is_split_into_words=True``. You already have words; letting
       the tokenizer re-split your sentence is how the alignment breaks before you
       have written a line of alignment code.
    3. Align labels with ``align_tags_to_subwords`` above.
    4. Fine-tune ``AutoModelForTokenClassification`` — a small checkpoint is
       plenty for this task, and it is the difference between minutes and hours
       on the free hardware you have.
    5. Evaluate on dev with ``evaluate_tagger``, save the model to
       ``TRANSFORMER_MODEL_PATH``, return the numbers.

    Set the seed and record it. "It scored better the second time" is not a
    result if you cannot say what changed between the two runs.

    Expect this to be harder to get right than the baseline was, and be honest in
    your report when it is. A badly fine-tuned transformer — too few epochs, a
    learning rate off by an order of magnitude, misaligned labels — can and does
    lose to a lookup table. If that happens to you, that is a finding worth
    reporting, with the number and your diagnosis of why. Quietly deleting the
    run and trying again until the graph points the right way is the thing this
    course is trying to train out of you.
    """
    raise NotImplementedError("Implement fine_tune_transformer() — run it offline.")


def load_tagger_model(path: str = TRANSFORMER_MODEL_PATH):
    """Load and cache the fine-tuned model and its tokenizer.

    Cache both in a module-level dict. Render's free plan gives you a cold start
    on every deploy and limited memory; loading a model per request will make the
    service feel broken, and on a small instance it will run you out of memory.

    Raise a ValueError with a readable message when there is no model at ``path``
    — the route turns that into a 400, and "no fine-tuned model found at
    models/tagwise-transformer; run scripts/train.py first" is a much better
    error than a stack trace. Until you have trained anything, that is the
    correct answer for the transformer, and the baseline still works.
    """
    raise NotImplementedError("Implement load_tagger_model() with a module-level cache.")


def tag_with_transformer(tokens: List[str]) -> List[TaggedToken]:
    """Tag a tokenised sentence with the fine-tuned classifier.

    One TaggedToken per input token — the same length, in the same order. Set
    ``confidence`` to the softmax probability of the chosen tag, and leave
    ``used_fallback`` False: a transformer has no unknown words, because every
    word it has never seen still decomposes into pieces it has.

    The alignment from ``align_tags_to_subwords`` runs in reverse here. Feed the
    words in with ``is_split_into_words=True``, take the argmax at the FIRST piece
    of each word, and drop the rest. If your output is longer than your input,
    you are returning subword predictions and the UI will render "un ##happi
    ##ness" as three words.

    The confidence number is worth showing in the UI even though nobody asked for
    it. The tokens where this model is unsure are, with striking regularity, the
    same ambiguous words the lookup baseline gets wrong — which is a much more
    interesting thing to say in your report than a single accuracy figure.
    """
    raise NotImplementedError("Implement tag_with_transformer().")


# ---------------------------------------------------------------------------
# 4. Serving one request
# ---------------------------------------------------------------------------
def split_words(sentence: str) -> List[str]:
    """Split a raw sentence into the word tokens the taggers will label.

    Returns tokens in order; empty input returns an empty list.

    This is small but it decides everything downstream. A bare ``.split()`` gives
    you "dog." with the full stop attached, which is not a word your lookup table
    has, so it goes to the fallback, which sees a lowercase word with no
    interesting suffix and says NOUN — and now punctuation is being tagged as a
    noun in your live demo. Separate trailing and leading punctuation into their
    own tokens so PUNCT can be predicted as PUNCT.

    Match the treebank's own tokenization as closely as you can, because that is
    what both models were built on. The treebank splits "don't" into "do" + "n't"
    and separates the possessive "'s". Every place your splitter disagrees with
    the treebank is a word your models were never trained to see, and the gap
    shows up as unexplained errors in the demo that were never in your test
    numbers. Write down what your splitter does in the MODEL_CARD.
    """
    raise NotImplementedError("Implement split_words().")


def tag_sentence(sentence: str, model: str = "baseline") -> TagResponse:
    """Tag one sentence with the requested model. The route calls this.

    Steps: split the sentence into words, dispatch on ``model`` ("baseline" ->
    the lookup table plus fallback, "transformer" -> the fine-tuned classifier),
    and assemble the response.

    Every field must be real:

    * ``tokens`` — one TaggedToken per word, in order
    * ``tag_sequence`` — the tags alone, same order, same length as ``tokens``
    * ``unknown_count`` — how many tokens used the fallback (always 0 for the
      transformer)
    * ``model`` — echo back what was asked for

    Raise ValueError for an unknown model name. Leave ``tagging_id`` as None and
    put anything you like in ``model_version``; ``api/main.py`` overwrites the
    version with the deployed one and fills the id in after logging, the same way
    it does for every other route.

    Load the lookup table once, at module import or on first use, from the file
    your build script wrote. Rebuilding it from the treebank on every request
    turns a millisecond into several seconds, and the free plan will time out.
    """
    raise NotImplementedError("Implement tag_sentence() — dispatch to the two taggers.")


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------
def token_accuracy(gold: List[str], predicted: List[str]) -> float:
    """Fraction of tokens whose predicted tag equals the gold tag, in [0, 1].

    Both lists are flat sequences of tags over the same tokens, in the same
    order. Raise ValueError if the lengths differ — that mismatch means an
    alignment bug upstream, and silently zipping to the shorter list would turn a
    loud bug into a slightly disappointing number.

    Empty input returns 0.0.

    Read this number against the floor you computed in ``describe_corpus``: if
    the most common tag covers a sixth of the corpus, then 0.17 is what guessing
    gets you and 0.90 is a lookup table. Accuracy is easy to feel good about
    here, which is why macro-F1 sits next to it.
    """
    raise NotImplementedError("Implement token_accuracy().")


def per_tag_f1(gold: List[str], predicted: List[str]) -> Dict[str, float]:
    """F1 for every tag that appears in either list.

    For each tag: precision = tp / (tp + fp), recall = tp / (tp + fn), and
    F1 = 2PR / (P + R). Return 0.0 for a tag whose precision and recall are both
    zero rather than dividing by zero — a tag your model never predicted and
    never got right scores zero, which is the honest answer.

    Print this table before you print macro-F1. The single number is a summary of
    this table, and the summary is never the interesting part.
    """
    raise NotImplementedError("Implement per_tag_f1().")


def macro_f1(gold: List[str], predicted: List[str]) -> float:
    """Unweighted mean of the per-tag F1 scores, in [0, 1].

    Unweighted is the entire point, and it is also the trap. Macro-F1 gives INTJ,
    with a handful of instances in your test split, exactly the same weight as
    NOUN with thousands. So a model that handles every common tag well and misses
    the two rarest ones loses a large chunk of macro-F1 while its accuracy barely
    moves — the score is dominated by the rare tags, and it will swing between
    runs for reasons that have nothing to do with the model.

    That is not a reason to drop the metric. It is a reason to report accuracy
    and macro-F1 together and to say which tags moved. When your two models'
    macro-F1 scores differ, look at ``per_tag_f1`` before you write a sentence
    about it: the gap is usually two or three low-count tags, not a general
    difference in quality, and claiming otherwise is the kind of overreach a
    reader can check in ten seconds.
    """
    raise NotImplementedError("Implement macro_f1().")


def confusion_matrix(
    gold: List[str], predicted: List[str], labels: Optional[List[str]] = None
) -> ConfusionMatrix:
    """Counts of gold tag (rows) against predicted tag (columns).

    ``labels`` fixes the order of both axes; default to the sorted set of tags
    appearing in either list, or to UNIVERSAL_POS_TAGS if you prefer a fixed
    order across runs (which makes two matrices comparable side by side — worth
    it). ``matrix[i][j]`` is the number of tokens whose gold tag is ``labels[i]``
    and whose predicted tag is ``labels[j]``. Correct predictions land on the
    diagonal. Every cell is an int; the whole matrix sums to the token count.

    This is the figure your report is built around, so make it readable. Store it
    in the run's ``metrics`` as {"confusion": {"labels": [...], "matrix": [[...]]}}
    and the comparison tab will render it.

    What to look for once you have it: the off-diagonal cells that are large. On
    English they are reliably NOUN/VERB ("book", "run", "walk"), ADJ/ADV
    ("fast", "hard", "early"), and DET/PRON ("that", "which", "this"), plus
    PROPN/NOUN wherever capitalisation is doing the work. Those are your three
    ambiguous word classes, and each one needs a worked example: the sentence,
    the gold tag, what each model said, and the reason the lookup could not have
    got it right. "The lookup was wrong" is not the finding. "The lookup is a
    function of the word alone, so it gives 'book' the same tag in 'read the
    book' and 'book me a flight', and it must be wrong about one of them" is.
    """
    raise NotImplementedError("Implement confusion_matrix().")


def evaluate_tagger(
    gold_sentences: List[List[Tuple[str, str]]],
    tag_fn,
    model_name: str,
) -> EvalReport:
    """Run one tagger over a held-out split and produce every number at once.

    Parameters
    ----------
    gold_sentences
        Sentences of (word, gold tag) pairs — the test split, untouched until now.
    tag_fn
        Something that takes a list of words and returns a list of TaggedToken.
        Pass ``lambda ws: tag_with_lookup(ws, table)`` or ``tag_with_transformer``.
        Taking the tagger as an argument is what stops you writing this function
        twice and lets you prove both models were scored by identical code.
    model_name
        Goes into the report so the run row says which tagger this was.

    Flatten the gold tags and the predicted tags into two aligned lists, then
    call the four functions above. Populate every field of EvalReport, including
    the confusion matrix.

    Evaluate both models on the SAME split with the SAME tag set. It sounds too
    obvious to state, and it is still the most common way a comparison in this
    assignment ends up meaningless.
    """
    raise NotImplementedError("Implement evaluate_tagger() — compose the functions above.")
