"""YOUR TO-DO LIST, WRITTEN AS TESTS.

Every test here fails on a fresh fork with NotImplementedError. Each one
describes a behaviour that ``api/nlp.py`` must have. Work top to bottom:

    pytest -m contract -x                    # stop at the first thing left to do
    pytest -m "contract and not network"     # the offline subset, on a train

These are deliberately loose about *how* — they check the contract, not your
algorithm. Passing them is necessary, not sufficient: the report, the
with/without comparison, and the Retrieval Audit tab are where you show you
understood what you built. In particular, nothing here can check that your
retrieval is any GOOD. A vector_search that returns five random passages with
plausible scores passes every test in this file and fails the assignment.
"""
from __future__ import annotations

import math

import pytest

from api import nlp
from shared.schemas import ChunkingParams, DocumentIn, RetrievedChunk

pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# 1. Loading documents and counting tokens
# ---------------------------------------------------------------------------
def test_load_documents_reads_a_directory_and_records_the_source(tmp_path):
    (tmp_path / "a.txt").write_text("The support rota runs Monday to Thursday.", "utf-8")
    (tmp_path / "b.txt").write_text("Escalation requires two named approvers.", "utf-8")

    docs = nlp.load_documents([str(tmp_path)])
    assert len(docs) == 2
    # Every document must know where it came from — citations point at this.
    assert all(d.source for d in docs)
    assert len({d.source for d in docs}) == 2


def test_load_documents_survives_a_file_it_cannot_decode(tmp_path):
    (tmp_path / "good.txt").write_text("readable text", "utf-8")
    (tmp_path / "bad.bin").write_bytes(b"\xff\xfe\x00\x81\x81")

    docs = nlp.load_documents([str(tmp_path)])
    # Skip it or decode it with replacement — either is fine. Crashing on
    # document 34 of 50 is not.
    assert any(d.text.strip() == "readable text" for d in docs)


@pytest.mark.network
def test_count_tokens_is_zero_for_empty_text():
    assert nlp.count_tokens("", nlp.EMBEDDING_MODEL) == 0


@pytest.mark.network
def test_count_tokens_grows_with_the_text():
    short = nlp.count_tokens("the cat sat", nlp.EMBEDDING_MODEL)
    long = nlp.count_tokens("the cat sat on the mat in the rain", nlp.EMBEDDING_MODEL)
    assert 0 < short < long


# ---------------------------------------------------------------------------
# 2. Chunking. Pure arithmetic first, then the real thing.
# ---------------------------------------------------------------------------
def test_chunk_spans_of_an_empty_document_is_empty():
    assert nlp.chunk_spans(0, ChunkingParams()) == []


def test_chunk_spans_of_a_short_document_is_one_span():
    # 120 tokens is shorter than one 400-token chunk. One span, not padded,
    # not dropped.
    assert nlp.chunk_spans(120, ChunkingParams(chunk_size_tokens=400)) == [(0, 120)]


def test_chunk_spans_overlap_by_the_recorded_amount():
    params = ChunkingParams(chunk_size_tokens=400, overlap_ratio=0.15)  # stride 340
    spans = nlp.chunk_spans(1000, params)
    assert len(spans) >= 2
    assert spans[0] == (0, 400)
    assert spans[1][0] == 340, "second chunk must start one stride in, not one chunk in"
    for (s0, e0), (s1, _e1) in zip(spans, spans[1:]):
        assert e0 - s1 == params.overlap_tokens


def test_chunk_spans_cover_the_whole_document():
    spans = nlp.chunk_spans(1000, ChunkingParams(chunk_size_tokens=300, overlap_ratio=0.10))
    assert spans[0][0] == 0
    assert spans[-1][1] == 1000, "the last chunk must reach the end — no orphaned tail"
    assert all(end > start for start, end in spans), "no zero-length chunks"
    assert len(set(spans)) == len(spans), "no duplicate span at the tail"


@pytest.mark.network
def test_chunk_document_numbers_its_chunks_and_keeps_offsets_straight():
    doc = DocumentIn(
        title="policy",
        source="tests",
        text=("The support rota runs Monday to Thursday. " * 200),
    )
    params = ChunkingParams(chunk_size_tokens=300, overlap_ratio=0.20)
    chunks = nlp.chunk_document(doc, params)

    assert len(chunks) >= 2
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(c.text.strip() for c in chunks), "an empty chunk embeds to noise"
    assert all(c.end_token > c.start_token for c in chunks)
    assert chunks[0].start_token == 0


# ---------------------------------------------------------------------------
# 3. Embeddings
# ---------------------------------------------------------------------------
@pytest.mark.network
def test_embedding_dimension_matches_the_configured_width():
    # If this fails, the migration's vector(...) column and nlp.EMBEDDING_DIM
    # disagree with the model. Fix all three together and re-embed.
    assert nlp.embedding_dimension(nlp.EMBEDDING_MODEL) == nlp.EMBEDDING_DIM


@pytest.mark.network
def test_embed_texts_returns_one_vector_per_input():
    vecs = nlp.embed_texts(["first passage", "second passage"], nlp.EMBEDDING_MODEL)
    assert len(vecs) == 2
    assert len(vecs[0]) == len(vecs[1]) == nlp.EMBEDDING_DIM


@pytest.mark.network
def test_embed_texts_returns_unit_vectors():
    # Normalised at write time means a dot product IS cosine similarity
    # everywhere downstream. Un-normalised means your top-k is ranked partly by
    # passage length.
    (vec,) = nlp.embed_texts(["a passage about escalation approvals"], nlp.EMBEDDING_MODEL)
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-3, f"expected unit length, got {norm:.4f}"


@pytest.mark.network
def test_embed_texts_of_nothing_is_nothing():
    assert nlp.embed_texts([], nlp.EMBEDDING_MODEL) == []


def test_cosine_similarity_of_identical_vectors_is_one():
    assert pytest.approx(1.0, abs=1e-6) == nlp.cosine_similarity([1.0, 0.0], [1.0, 0.0])


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    assert pytest.approx(0.0, abs=1e-6) == nlp.cosine_similarity([1.0, 0.0], [0.0, 1.0])


def test_cosine_similarity_ignores_magnitude():
    # This is the whole point of dividing by both norms. [3, 0] and [1, 0] point
    # the same way; a bare dot product would score them 3.0 and call the longer
    # passage more relevant.
    assert pytest.approx(1.0, abs=1e-6) == nlp.cosine_similarity([3.0, 0.0], [1.0, 0.0])


def test_cosine_similarity_rejects_a_dimension_mismatch():
    with pytest.raises(ValueError):
        nlp.cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


def test_cosine_similarity_of_a_zero_vector_is_zero_not_a_crash():
    assert nlp.cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# 4. Retrieval
#
#    These patch the two things vector_search depends on — the embedder and the
#    database — so the contract can be checked with no model and no cloud. The
#    deployed path must go through api.db.match_chunks; that is what the patch
#    below asserts by standing in for it.
# ---------------------------------------------------------------------------
def test_vector_search_ranks_results_closest_first(monkeypatch):
    from api import db

    monkeypatch.setattr(
        nlp, "embed_texts", lambda texts, model_name=None: [[1.0] + [0.0] * 383]
    )
    monkeypatch.setattr(
        db,
        "match_chunks",
        lambda embedding, k=5, corpus="retrieval": [
            {
                "chunk_id": 7,
                "document_id": 1,
                "document_title": "rota",
                "content": "Requests after 16:00 are queued.",
                "similarity": 0.81,
            },
            {
                "chunk_id": 9,
                "document_id": 2,
                "document_title": "escalation",
                "content": "Escalation requires two approvers.",
                "similarity": 0.42,
            },
        ][:k],
    )

    hits = nlp.vector_search("when are late requests handled?", k=2)
    assert [h.rank for h in hits] == [1, 2]
    assert hits[0].similarity >= hits[1].similarity
    assert hits[0].chunk_id == 7
    assert hits[0].text


def test_vector_search_on_an_empty_store_returns_nothing(monkeypatch):
    from api import db

    monkeypatch.setattr(
        nlp, "embed_texts", lambda texts, model_name=None: [[1.0] + [0.0] * 383]
    )
    monkeypatch.setattr(db, "match_chunks", lambda embedding, k=5, corpus="retrieval": [])

    assert nlp.vector_search("anything at all", k=5) == []


# ---------------------------------------------------------------------------
# 5. Prompting and generation
# ---------------------------------------------------------------------------
def _passages():
    return [
        RetrievedChunk(
            chunk_id=7,
            document_id=1,
            document_title="rota",
            text="Requests received after 16:00 are queued for the next working day.",
            similarity=0.81,
            rank=1,
        ),
        RetrievedChunk(
            chunk_id=9,
            document_id=2,
            document_title="escalation",
            text="Escalation requires two named approvers.",
            similarity=0.42,
            rank=2,
        ),
    ]


def test_build_prompt_contains_the_question_and_every_passage():
    prompt = nlp.build_prompt("when are late requests handled?", _passages())
    assert "when are late requests handled?" in prompt
    for p in _passages():
        assert p.text in prompt


def test_build_prompt_numbers_the_passages_so_they_can_be_cited():
    prompt = nlp.build_prompt("when are late requests handled?", _passages())
    assert "1" in prompt and "2" in prompt, "passages need labels the model can cite"


def test_build_prompt_with_no_passages_is_the_control_condition():
    prompt = nlp.build_prompt("who approves an escalation?", [])
    assert "who approves an escalation?" in prompt
    # No smuggled context. Same function, same template, no passages — otherwise
    # the with/without comparison is comparing two different prompts.
    assert "Escalation requires two named approvers" not in prompt


@pytest.mark.network
def test_generate_returns_only_the_new_text():
    prompt = "Question: what colour is the sky on a clear day?\nAnswer:"
    out = nlp.generate(prompt, max_new_tokens=16)
    assert isinstance(out, str) and out.strip()
    assert not out.startswith(prompt), "strip the prompt off the continuation"


def test_answer_question_without_retrieval_retrieves_nothing(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("use_retrieval=False must not touch the vector store")

    monkeypatch.setattr(nlp, "vector_search", _boom)
    monkeypatch.setattr(nlp, "generate", lambda prompt, **kw: "an unconditioned answer")

    result = nlp.answer_question("who approves an escalation?", k=5, use_retrieval=False)
    assert result.retrieval_used is False
    assert result.retrieved == []
    assert result.cited_chunk_ids == []
    assert result.embedding_model is None
    assert result.answer


def test_answer_question_with_retrieval_reports_what_it_retrieved(monkeypatch):
    monkeypatch.setattr(
        nlp, "vector_search", lambda question, k=5, corpus="retrieval": _passages()
    )
    monkeypatch.setattr(
        nlp, "generate", lambda prompt, **kw: "Requests are queued to the next day [1]."
    )

    result = nlp.answer_question("when are late requests handled?", k=2)
    assert result.retrieval_used is True
    assert len(result.retrieved) == 2
    assert result.retrieved[0].rank == 1
    assert result.generator_model
    assert result.embedding_model
    # query_id is filled in by api/main.py after logging, not here.
    assert result.query_id is None


def test_answer_question_records_which_passages_it_cited(monkeypatch):
    monkeypatch.setattr(
        nlp, "vector_search", lambda question, k=5, corpus="retrieval": _passages()
    )
    monkeypatch.setattr(
        nlp, "generate", lambda prompt, **kw: "Queued to the next working day [1]."
    )

    result = nlp.answer_question("when are late requests handled?", k=2)
    # The answer cites passage 1, which is chunk 7. A retrieved-but-uncited
    # passage and a never-retrieved passage are different failures; this field is
    # how you tell them apart later.
    assert result.cited_chunk_ids == [7]


# ---------------------------------------------------------------------------
# 6. Perplexity — on the HELD-OUT split of the fine-tuning corpus
# ---------------------------------------------------------------------------
@pytest.mark.network
def test_perplexity_is_a_finite_number_above_one():
    report = nlp.perplexity(
        [
            "The support rota runs from Monday to Thursday each week.",
            "Escalations are approved by two people who are not the requester.",
        ],
        nlp.GENERATOR_MODEL,
    )
    assert report.split == "held_out"
    assert report.document_count == 2
    assert report.token_count > 0
    assert math.isfinite(report.perplexity)
    assert report.perplexity > 1.0, (
        "perplexity below 1 is impossible; you are averaging or masking wrongly"
    )


# ---------------------------------------------------------------------------
# 7. The disjointness check — the rule the whole comparison rests on
# ---------------------------------------------------------------------------
def test_normalise_for_comparison_erases_case_and_punctuation():
    a = nlp.normalise_for_comparison("The Board met on 3 March.")
    b = nlp.normalise_for_comparison("the  board met on 3 march")
    assert a == b


def test_shingles_are_overlapping_word_ngrams():
    out = nlp.shingles("the cat sat on the mat", size=3)
    assert len(out) == 4
    assert "the cat sat" in out


def test_shingles_of_a_short_text_is_empty_not_padded():
    assert nlp.shingles("two words", size=8) == set()


def test_disjoint_corpora_report_disjoint():
    report = nlp.check_corpus_disjointness(
        ["Annual rainfall in the northern catchment fell by nine percent."],
        ["The support rota runs from Monday to Thursday each week."],
    )
    assert report.disjoint is True
    assert report.exact_duplicate_count == 0
    assert report.overlapping_pairs == []


def test_an_identical_document_on_both_sides_raises():
    shared = "The escalation policy was revised in the third quarter of the year."
    with pytest.raises(nlp.CorpusOverlapError):
        nlp.check_corpus_disjointness([shared], [shared])


def test_a_reformatted_copy_is_caught_too():
    # Different hash, same document. This is the case that actually happens: a
    # PDF and its HTML version, or the same note with a header stripped.
    original = (
        "The support rota runs from Monday to Thursday each week. Requests "
        "received after sixteen hundred are queued for the following working "
        "day, and the queue is cleared before any new request is accepted."
    )
    reformatted = "THE SUPPORT ROTA RUNS FROM MONDAY TO THURSDAY EACH WEEK.\n\n" + (
        "Requests received after sixteen hundred are queued for the following "
        "working day, and the queue is cleared before any new request is accepted."
    )
    assert nlp.sha256_text(original) != nlp.sha256_text(reformatted)
    with pytest.raises(nlp.CorpusOverlapError):
        nlp.check_corpus_disjointness([original], [reformatted])


def test_the_check_can_report_instead_of_raising():
    shared = "The escalation policy was revised in the third quarter of the year."
    report = nlp.check_corpus_disjointness([shared], [shared], raise_on_overlap=False)
    assert report.disjoint is False
    assert report.exact_duplicate_count == 1
    assert report.overlapping_pairs
    assert report.overlapping_pairs[0].reason


def test_the_overlap_error_says_what_collided():
    shared = "The escalation policy was revised in the third quarter of the year."
    with pytest.raises(nlp.CorpusOverlapError) as exc:
        nlp.check_corpus_disjointness([shared], [shared])
    message = str(exc.value)
    # A person reading a 409 in the Streamlit tab has to know which file to
    # delete. "Corpora overlap" is not a message.
    assert any(ch.isdigit() for ch in message), (
        "name the offending document indices and the score in the error"
    )
