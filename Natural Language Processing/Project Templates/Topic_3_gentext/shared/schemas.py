"""Pydantic request/response models — the contract between the two clouds.

WHY THIS FILE EXISTS
--------------------
The UI (Streamlit Cloud) and the API (Render.com) are separate deployments that
only ever talk over HTTPS + JSON. This module is the single written-down
description of that JSON. Both tiers import it, so a change here is a change on
both sides at once and you cannot drift.

WHAT YOU SHOULD DO WITH IT
--------------------------
Read it first, before you write any generation code. Every stub in
``api/nlp.py`` is typed in terms of these models, so the shapes below tell you
exactly what your functions have to produce.

You MAY add fields. If your evaluation wants to report, say, a repetition rate
per output, add it to ``GenerateResponse`` and render it in the UI. Do NOT
rename or delete the existing fields — the tests and the grading rubric
reference them by name.

THE ONE FIELD THAT IS NOT NEGOTIABLE
------------------------------------
``prompt_sha256``. There is no ``prompt`` field anywhere in a stored record, and
there should never be one. The product brief promises that prompts are hashed,
and a promise you break in the schema is a promise you break in production.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# The five decoding strategies this service exposes. They are separate values
# rather than a pile of booleans because the whole point of the topic is that
# they are DIFFERENT ALGORITHMS, not different settings of one algorithm:
#
#   greedy      — always take argmax. Deterministic, and repetitive.
#   beam        — keep num_beams partial sequences, return the highest-scoring.
#   temperature — sample from the FULL distribution after scaling the logits.
#   top_k       — sample after truncating to the k most likely tokens.
#   top_p       — sample after truncating to the smallest set whose probability
#                 mass reaches p (nucleus sampling).
Strategy = Literal["greedy", "beam", "temperature", "top_k", "top_p"]


# ---------------------------------------------------------------------------
# Infrastructure responses (already implemented — you should not need to touch)
# ---------------------------------------------------------------------------
class Health(BaseModel):
    """GET /healthz — is the service up, and can it reach the database?"""

    status: str = Field(description='"ok" when the process is serving requests')
    database: str = Field(description='"ok", "unreachable", or "not_configured"')
    model_loaded: bool = Field(
        default=False,
        description="True once a decoder is resident in memory. On the free tier "
        "the first request pays for the load; this flag is how you tell a cold "
        "start apart from a crash.",
    )


class Version(BaseModel):
    """GET /version — which build is live right now?

    Graders use this to confirm the URL in your README is the code you claim.
    ``model_version`` also appears on every generation row, which is what lets
    you say "the outputs got better in v3" and prove it from the table.
    """

    service: str
    git_sha: str = Field(description="Short commit SHA of the running build")
    model_version: str = Field(
        description="Free-text version of the generation configuration, e.g. "
        "'gentext-gpt2-reviews-v2'"
    )
    base_model: str = Field(
        description="Hugging Face id of the decoder underneath, e.g. 'gpt2'"
    )


# ---------------------------------------------------------------------------
# POST /generate
# ---------------------------------------------------------------------------
class DecodingParams(BaseModel):
    """Every knob the Generate tab exposes, in one object.

    This object is stored verbatim on the generation row. That is the audit
    trail: "this text came out of this prompt under these settings". A generation
    you cannot reproduce is a generation you cannot learn anything from.

    Which fields matter depends on ``strategy``, and that is deliberate — part of
    the Concepts tab is explaining that ``top_k`` does nothing under greedy
    decoding, and that ``num_beams`` does nothing under sampling.
    """

    strategy: Strategy = "top_p"

    max_new_tokens: int = Field(
        default=80,
        ge=1,
        le=512,
        description="Tokens to generate, NOT counting the prompt. Cap it: an "
        "unbounded generation on a free instance is how you get a 502.",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Logit scaling before the softmax. <1 sharpens the "
        "distribution, >1 flattens it. Applies to the sampling strategies only.",
    )
    top_k: int = Field(
        default=50,
        ge=0,
        description="Keep the k most likely tokens, renormalize, sample. 0 means "
        "no truncation.",
    )
    top_p: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description="Keep the smallest set of tokens whose cumulative probability "
        "reaches p. 1.0 means no truncation.",
    )
    num_beams: int = Field(
        default=4,
        ge=1,
        le=12,
        description="Beam width. Used by the beam strategy only. Wider is slower "
        "and, past about 5, usually duller.",
    )
    repetition_penalty: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Divides the logits of tokens already produced. 1.0 is off.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for the sampling strategies. Set it when you want "
        "a demo you can repeat; leave it None when you want to show variety.",
    )


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, description="The user's prompt, in the clear "
                        "over HTTPS and hashed the moment it is stored")
    params: DecodingParams = DecodingParams()


class GenerateResponse(BaseModel):
    """Everything the Generate and Compare Decoding tabs need to draw a result."""

    prompt_sha256: str = Field(
        description="sha256 of the prompt. The prompt itself is never persisted."
    )
    generated_text: str = Field(
        description="The continuation ONLY — with the prompt stripped off the "
        "front. Leaving the prompt in inflates every diversity metric you compute "
        "afterwards, because the prompt is shared across all your samples."
    )
    strategy: Strategy
    params: DecodingParams
    prompt_token_count: int
    generated_token_count: int
    distinct_1: Optional[float] = Field(
        default=None, description="Unique unigrams / total unigrams, in [0, 1]"
    )
    distinct_2: Optional[float] = Field(
        default=None,
        description="Unique bigrams / total bigrams. The single most useful number "
        "for spotting a model stuck in a loop.",
    )
    perplexity: Optional[float] = Field(
        default=None,
        description="Perplexity of the generated text under the model, when you "
        "compute it per generation. Low perplexity is not the same as good text — "
        "greedy output has very low perplexity and reads like a broken record.",
    )
    latency_ms: Optional[float] = None
    model_version: Optional[str] = Field(
        default=None, description="Filled in by api/main.py, not by nlp.generate()"
    )
    generation_id: Optional[int] = Field(
        default=None,
        description="Row id in the Supabase generations table, if logging "
        "succeeded. The UI needs it to submit a rating.",
    )


# ---------------------------------------------------------------------------
# POST /rate
# ---------------------------------------------------------------------------
class RateRequest(BaseModel):
    """One rater's independent score for one generation.

    The assignment asks for at least 20 outputs rated by two people working
    independently, and for a report on where they disagreed. That only works if
    each rating is stored separately with the rater's identity attached — a
    single averaged column throws away the disagreement you are being asked to
    analyse.

    ``rater_id`` is a label your team picks ("rater-a", "rater-b"). Do not put a
    real name in it; this table is part of a public repository.
    """

    generation_id: int = Field(gt=0)
    rater_id: str = Field(min_length=1, max_length=32)
    rating: int = Field(ge=1, le=5, description="Overall quality, 1 (unusable) to 5")

    coherence: Optional[int] = Field(default=None, ge=1, le=5)
    fluency: Optional[int] = Field(default=None, ge=1, le=5)
    relevance: Optional[int] = Field(
        default=None, ge=1, le=5, description="Does it answer or continue the prompt?"
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Why you gave that score. This is where the failure modes in "
        "your report come from.",
    )


class RateResponse(BaseModel):
    generation_id: int
    human_rating: Optional[int] = Field(
        description="The aggregate now stored on the row (mean of all raters, "
        "rounded)"
    )
    rating_count: int = Field(description="How many independent ratings exist now")
    stored: bool = Field(
        description="False when Supabase is not configured — the API stays up "
        "either way, but nothing was written."
    )


# ---------------------------------------------------------------------------
# GET /history
# ---------------------------------------------------------------------------
class Generation(BaseModel):
    """One logged generation — this is what the History tab renders.

    Note what is here and what is not: the SETTINGS, the OUTPUT, the HASH of the
    prompt, and the rating. Never the prompt.
    """

    id: int
    prompt_sha256: str
    strategy: str
    decoding_params: Dict[str, Any] = Field(
        default_factory=dict, description="The DecodingParams object, as JSON"
    )
    generated_text: str
    prompt_token_count: Optional[int] = None
    generated_token_count: Optional[int] = None
    distinct_2: Optional[float] = None
    perplexity: Optional[float] = None
    model_version: str
    human_rating: Optional[int] = None
    ratings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Every independent rating, each with its rater_id. Two entries "
        "with different scores is the raw material for your disagreement analysis.",
    )
    created_at: Optional[str] = None


class HistoryResponse(BaseModel):
    generations: List[Generation]


# ---------------------------------------------------------------------------
# Training runs — the SECOND thing that gets persisted.
#
# There is no endpoint for these on purpose: fine-tuning does not happen on the
# web service (see the README). Your offline training script writes a row here
# through api/db.insert_training_run so that every generation's model_version
# can be traced back to the run that produced the weights.
# ---------------------------------------------------------------------------
class TrainingConfig(BaseModel):
    """The fine-tuning configuration you have to report and reproduce."""

    base_model: str = Field(description='Hugging Face id, e.g. "gpt2" or "distilgpt2"')
    model_version: str = Field(
        description="The label you will stamp on every generation made with these "
        "weights, e.g. 'gentext-gpt2-reviews-v2'"
    )
    epochs: float = 1.0
    learning_rate: float = 5e-5
    batch_size: int = 8
    block_size: int = Field(
        default=256, description="Tokens per training example after chunking"
    )
    method: str = Field(
        default="full",
        description='"full" fine-tune, "lora", "prefix", "frozen" (no adaptation). '
        '"frozen" is a legitimate answer if you defend it — but then say so, do not '
        "claim a fine-tune you did not run.",
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict, description="Anything else you varied, as JSON"
    )


class CorpusStats(BaseModel):
    """What ``load_corpus`` has to be able to tell you about your data."""

    source: str = Field(description="Where it came from, specifically enough to find")
    domain: str = Field(description='e.g. "product reviews", "recipe instructions"')
    sentence_count: int = Field(
        description="The assignment sets a floor of 20,000. Count them, do not "
        "estimate them."
    )
    token_count: int
    filters_applied: List[str] = Field(
        default_factory=list,
        description="Every filter, in order: deduplication, length cuts, language "
        "detection, anything you dropped. This is the part reviewers ask about.",
    )
    sha256: str = Field(description="Hash of the filtered corpus, so a run is traceable")


class TrainingRun(BaseModel):
    """A row in the training_runs table."""

    id: int
    base_model: str
    model_version: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    corpus_source: Optional[str] = None
    corpus_sha256: Optional[str] = None
    corpus_sentence_count: Optional[int] = None
    held_out_perplexity: Optional[float] = Field(
        default=None,
        description="Perplexity on the held-out split. The one number that says "
        "whether the fine-tune helped.",
    )
    notes: Optional[str] = None
    created_at: Optional[str] = None
