"""YOUR TO-DO LIST, WRITTEN AS TESTS.

Every test here fails on a fresh fork with NotImplementedError. Each one
describes a behaviour that ``api/nlp.py`` must have. Work top to bottom:

    pytest -m contract -x                          # stop at the first thing left
    pytest -m "contract and not network and not train"   # while you are offline

These are deliberately loose about *how* — they check the contract, not your
algorithm. Passing them is necessary, not sufficient: the report, the human
ratings, and the Compare Decoding tab are where you show you understood what you
built.
"""
from __future__ import annotations

import pytest

from api import nlp
from shared.schemas import DecodingParams

pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# 1. The corpus
# ---------------------------------------------------------------------------
def test_load_corpus_counts_what_is_actually_there(sample_corpus):
    stats = nlp.load_corpus(sample_corpus, source="test fixture", domain="harbour notes")
    assert stats.sentence_count > 0
    assert stats.token_count > stats.sentence_count
    assert stats.source == "test fixture"
    assert stats.domain == "harbour notes"


def test_load_corpus_filters_before_it_counts(sample_corpus):
    """The fixture has 4 non-empty lines, one of them a duplicate of another.

    Whatever filters you apply, the count you report has to describe the text you
    would actually train on. Reporting the pre-filter number and training on the
    post-filter text is how a corpus "shrinks" without anyone noticing.
    """
    stats = nlp.load_corpus(sample_corpus)
    assert stats.sentence_count <= 4
    assert stats.filters_applied, "record every filter you ran, in order"
    assert len(stats.sha256) == 64


# ---------------------------------------------------------------------------
# 2. Decoding strategies — the switch that makes the topic explicit
# ---------------------------------------------------------------------------
def test_greedy_does_not_sample_and_does_not_beam():
    kwargs = nlp.build_generation_kwargs(DecodingParams(strategy="greedy"))
    assert kwargs["do_sample"] is False
    assert kwargs.get("num_beams", 1) == 1
    assert "max_new_tokens" in kwargs


def test_beam_search_uses_more_than_one_beam_and_still_does_not_sample():
    kwargs = nlp.build_generation_kwargs(
        DecodingParams(strategy="beam", num_beams=5)
    )
    assert kwargs["do_sample"] is False
    assert kwargs["num_beams"] == 5


def test_temperature_strategy_samples_from_the_whole_vocabulary():
    """Temperature sampling with no truncation: top_k off, top_p off.

    If you leave the default top_k=50 in place here, this "strategy" is really
    top-k sampling wearing a different label, and your comparison compares
    nothing.
    """
    kwargs = nlp.build_generation_kwargs(
        DecodingParams(strategy="temperature", temperature=1.2)
    )
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == pytest.approx(1.2)
    assert kwargs.get("top_k", 0) == 0
    assert kwargs.get("top_p", 1.0) == pytest.approx(1.0)


def test_top_k_truncates_by_count():
    kwargs = nlp.build_generation_kwargs(DecodingParams(strategy="top_k", top_k=40))
    assert kwargs["do_sample"] is True
    assert kwargs["top_k"] == 40
    assert kwargs.get("top_p", 1.0) == pytest.approx(1.0)


def test_top_p_truncates_by_probability_mass():
    kwargs = nlp.build_generation_kwargs(DecodingParams(strategy="top_p", top_p=0.9))
    assert kwargs["do_sample"] is True
    assert kwargs["top_p"] == pytest.approx(0.9)
    assert kwargs.get("top_k", 0) == 0


def test_temperature_zero_is_routed_to_greedy_not_passed_down():
    """Temperature 0 is greedy decoding in the maths and a crash in the library.

    Dividing the logits by zero gives you NaNs or an exception depending on the
    version. Route it to the greedy branch instead of hoping.
    """
    kwargs = nlp.build_generation_kwargs(
        DecodingParams(strategy="temperature", temperature=0.0)
    )
    assert kwargs["do_sample"] is False
    assert kwargs.get("temperature", 1.0) != 0.0


# ---------------------------------------------------------------------------
# 3. Generation itself. These download a model the first time they run.
#    Offline: pytest -m "contract and not network and not train"
# ---------------------------------------------------------------------------
@pytest.mark.network
def test_generate_fills_every_field_and_hashes_the_prompt():
    prompt = "The morning after the storm, the harbour"
    r = nlp.generate(prompt, DecodingParams(strategy="top_p", max_new_tokens=24))
    assert r.prompt_sha256 == nlp.sha256_text(prompt)
    assert r.generated_text.strip()
    assert r.strategy == "top_p"
    assert r.generated_token_count > 0
    assert r.prompt_token_count > 0
    assert r.distinct_2 is None or 0.0 <= r.distinct_2 <= 1.0
    # model_version and generation_id belong to api/main.py, not to you.
    assert r.model_version is None
    assert r.generation_id is None


@pytest.mark.network
def test_generate_strips_the_prompt_from_the_output():
    """The continuation only. Hugging Face returns prompt + continuation by
    default, and leaving the prompt in inflates every diversity number you
    compute afterwards."""
    prompt = "Zanzibar telescope committee minutes:"
    r = nlp.generate(prompt, DecodingParams(strategy="greedy", max_new_tokens=16))
    assert prompt not in r.generated_text


@pytest.mark.network
def test_a_fixed_seed_makes_sampling_repeatable():
    prompt = "The harbour"
    params = DecodingParams(strategy="top_p", max_new_tokens=20, seed=1234)
    first = nlp.generate(prompt, params)
    second = nlp.generate(prompt, params)
    assert first.generated_text == second.generated_text


@pytest.mark.network
def test_greedy_is_deterministic_without_any_seed():
    prompt = "The harbour"
    params = DecodingParams(strategy="greedy", max_new_tokens=20)
    assert nlp.generate(prompt, params).generated_text == (
        nlp.generate(prompt, params).generated_text
    )


@pytest.mark.network
def test_unknown_model_name_raises_value_error():
    with pytest.raises(ValueError):
        nlp.load_model("definitely-not-a-real-model-id-42")


@pytest.mark.network
def test_load_model_caches():
    """The second call must not re-load. If this is slow, you are not caching,
    and every request on Render will pay the same cost."""
    nlp.load_model(nlp.DEFAULT_MODEL)
    assert nlp.model_is_loaded()
    before = len(nlp._MODEL_CACHE)
    nlp.load_model(nlp.DEFAULT_MODEL)
    assert len(nlp._MODEL_CACHE) == before


# ---------------------------------------------------------------------------
# 4. Automatic evaluation
# ---------------------------------------------------------------------------
def test_distinct_1_on_a_worked_example():
    # "the cat sat on the mat" — 6 tokens, 5 unique.
    assert nlp.distinct_n("the cat sat on the mat", n=1) == pytest.approx(5 / 6)


def test_distinct_2_catches_a_loop():
    # bigrams: (a,b) (b,a) (a,b) (b,a) (a,b) -> 2 unique out of 5.
    assert nlp.distinct_n("a b a b a b", n=2) == pytest.approx(0.4)


def test_distinct_n_is_bounded_and_safe_on_short_input():
    assert nlp.distinct_n("", n=2) == 0.0
    assert nlp.distinct_n("word", n=2) == 0.0
    assert 0.0 <= nlp.distinct_n("one two three four", n=2) <= 1.0


@pytest.mark.network
def test_perplexity_is_a_positive_finite_number():
    ppl = nlp.perplexity(["The harbour was quiet after the storm."])
    assert ppl > 1.0
    assert ppl < float("inf")


@pytest.mark.network
def test_perplexity_is_lower_on_fluent_text_than_on_scrambled_text():
    """A sanity check on the direction of the metric, not on its exact value.

    If your scrambled string scores lower, you have almost certainly averaged
    per-text instead of per-token, or forgotten the label shift.
    """
    fluent = nlp.perplexity(["The boats went out again by noon."])
    scrambled = nlp.perplexity(["noon out by boats went again The."])
    assert fluent < scrambled


# ---------------------------------------------------------------------------
# 5. Human evaluation
# ---------------------------------------------------------------------------
def test_record_rating_returns_the_payload_that_gets_stored():
    payload = nlp.record_rating(
        generation_id=1,
        rater_id="rater-a",
        rating=4,
        dimensions={"fluency": 5, "coherence": 3},
        notes="reads well, drifts off topic",
    )
    for key in ("rater_id", "rating", "dimensions", "recorded_at"):
        assert key in payload
    assert payload["rater_id"] == "rater-a"
    assert payload["rating"] == 4


def test_record_rating_rejects_scores_outside_the_rubric():
    with pytest.raises(ValueError):
        nlp.record_rating(generation_id=1, rater_id="rater-a", rating=7)
    with pytest.raises(ValueError):
        nlp.record_rating(generation_id=1, rater_id="", rating=3)
    with pytest.raises(ValueError):
        nlp.record_rating(
            generation_id=1, rater_id="rater-a", rating=3, dimensions={"fluency": 0}
        )


def test_rater_agreement_measures_the_gap_not_the_average():
    ratings = [
        {"generation_id": 1, "rater_id": "rater-a", "rating": 2},
        {"generation_id": 1, "rater_id": "rater-b", "rating": 4},
        {"generation_id": 2, "rater_id": "rater-a", "rating": 5},
        {"generation_id": 2, "rater_id": "rater-b", "rating": 5},
    ]
    out = nlp.rater_agreement(ratings)
    assert out["pairs_compared"] == 2
    assert out["exact_agreement"] == pytest.approx(0.5)
    assert out["mean_absolute_difference"] == pytest.approx(1.0)
    assert out["disagreements_over_1"] == pytest.approx(0.5)


def test_rater_agreement_ignores_singly_rated_generations():
    """One rater is not agreement. Treating a missing second score as a match
    is how a project reports 100% agreement having compared nothing."""
    out = nlp.rater_agreement(
        [{"generation_id": 9, "rater_id": "rater-a", "rating": 3}]
    )
    assert out["pairs_compared"] == 0
    assert out["exact_agreement"] == 0.0


def test_rater_agreement_on_no_data_returns_zeros_not_an_error():
    out = nlp.rater_agreement([])
    assert out["pairs_compared"] == 0


# ---------------------------------------------------------------------------
# 6. Fine-tuning. Slow, and it needs a GPU or a lot of patience.
#    Deselect with: pytest -m "contract and not train"
#    Run it once, offline, on a small slice of your corpus.
# ---------------------------------------------------------------------------
@pytest.mark.train
def test_fine_tune_returns_a_reproducible_run_record(sample_corpus, tmp_path):
    from shared.schemas import TrainingConfig

    corpus = nlp.load_corpus(sample_corpus, source="test fixture", domain="test")
    cfg = TrainingConfig(
        base_model="distilgpt2",
        model_version="gentext-test-v0",
        epochs=1,
        batch_size=2,
        block_size=64,
    )
    run = nlp.fine_tune(corpus, cfg, output_dir=str(tmp_path / "model"))

    assert run.base_model == "distilgpt2"
    assert run.model_version == "gentext-test-v0"
    assert run.hyperparameters, "the hyperparameters ARE the record — record them"
    assert run.corpus_sha256 == corpus.sha256
    assert run.held_out_perplexity is None or run.held_out_perplexity > 1.0
