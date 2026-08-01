"""
THE REQUIRED TEST OF THIS TOPIC: the NumPy reward head must reproduce the
PyTorch + scikit-learn score, end to end, from a raw string.

Everything else in this repository can be right while this is wrong, and there
is no symptom. A reward model that disagrees with the one you trained still
returns a plausible float for every input; `/score` still answers 200; the
Streamlit tab still renders a number. What changes is that the number is not
the model's, and nothing anywhere reports an error.

WHAT IS COMPARED
----------------
The WHOLE CHAIN, not the matrix multiplies:

    training tier :  str -> sklearn.TfidfVectorizer -> torch.nn.Sequential -> float
    serving tier  :  str -> api.reward.tfidf_vector -> NumPy forward pass  -> float

That matters, because in this topic the featuriser is a bigger risk than the
weights. A transposed matrix is a familiar bug with a familiar fix; a
tokeniser that lowercases on one side and not the other, or an IDF formula
that uses `smooth_idf=False`, produces a subtly different vector for every
input and is invisible to any test that starts from a feature vector rather
than from a string.

TOLERANCE: 1e-4 on the maximum absolute difference in the assigned reward,
over 96 probe texts including adversarial ones.

That number is chosen, not copied.

  * `train/export.py` writes float32 weights and float32 IDF; `api/reward.py`
    evaluates in float64. Over a 74-to-2000 dimensional dot product followed by
    a 64-unit hidden layer, the accumulated rounding is of order 1e-6 on a
    reward whose scale is O(1) to O(10).
  * scikit-learn computes the TF-IDF vector in float64 and the training-side
    torch module then casts to float32, so the two sides also differ by one
    float32 round-trip of the input vector — another ~1e-6.
  * 1e-4 is therefore two orders of magnitude above the noise floor.
  * It is several orders BELOW any real bug. A transposed hidden layer, a
    dropped bias, `smooth_idf` flipped, `norm=None`, or a tokeniser that keeps
    punctuation each move the score by 0.1 or more — usually by whole units.

A tolerance of 1e-1 would pass while broken. A tolerance of 1e-9 would fail on
a different CPU, and someone would then "fix" it by loosening it to 1e-1.

WHEN IT FAILS, in order of how often each is the cause:

    1. A TOKENISER MISMATCH. Both sides must go through
       `shared.preprocess.TOKEN_PATTERN` — the training side by passing it to
       `TfidfVectorizer(token_pattern=...)`, the serving side by calling
       `shared.preprocess.tokenise`. Print the two token lists for one probe
       text and diff them; this takes thirty seconds and is right most of the
       time.
    2. A VECTORISER SETTING. `smooth_idf`, `sublinear_tf`, `norm`, `binary`.
       `train/reward_model.py::fit_tfidf` passes all four explicitly so that a
       scikit-learn default change cannot move the deployed arithmetic; if you
       changed one there, change `api/reward.py::tfidf_vector` to match.
    3. A TRANSPOSED WEIGHT MATRIX. PyTorch stores nn.Linear.weight as
       (out_features, in_features) and `export_reward_head` writes it
       UNCHANGED, because `api/reward.py` evaluates `W @ x + b`.
    4. A MISSING BIAS — a layer built with `bias=False`, so `b{i}` is absent,
       so the loader stops early and deploys a shorter network in silence.
    5. Only then, precision.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from api.reward import RewardHead

TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def reference(tmp_path_factory, request):
    """Train and export a head in a SUBPROCESS, and return its PyTorch scores.

    The subprocess is the point — see `run_torch_script` in conftest.py. This
    test process must remain free of torch or `tests/test_no_torch.py` becomes
    a lie about the deployment budget.
    """
    request.getfixturevalue("requires_torch")
    from tests.conftest import run_torch_script

    out = tmp_path_factory.mktemp("equivalence")
    run_torch_script(
        f"""
        from train.reward_model import dump_reference
        print(json.dumps(dump_reference({str(out)!r}, seed=0)))
        """
    )
    payload = json.loads((out / "reference.json").read_text())
    z = np.load(out / "equivalence_head.npz", allow_pickle=False)
    layers = []
    i = 0
    while f"W{i}" in z.files:
        layers.append((np.asarray(z[f"W{i}"], dtype=np.float64),
                       np.asarray(z[f"b{i}"], dtype=np.float64)))
        i += 1
    head = RewardHead(
        vocab=np.asarray(z["vocab"]).astype(str),
        idf=np.asarray(z["idf"], dtype=np.float64),
        layers=layers,
    )
    return {"head": head, "texts": payload["texts"], "scores": np.asarray(payload["scores"])}


def test_numpy_reproduces_the_torch_reward(reference):
    head, texts, theirs = reference["head"], reference["texts"], reference["scores"]
    ours = np.asarray([head.score(t)[0] for t in texts], dtype=np.float64)

    assert ours.shape == theirs.shape

    diffs = np.abs(ours - theirs)
    worst = int(np.argmax(diffs))
    max_abs_diff = float(diffs[worst])
    assert max_abs_diff < TOLERANCE, (
        f"max |NumPy - PyTorch| = {max_abs_diff:.3e}, tolerance {TOLERANCE:.0e}.\n"
        f"Worst probe text ({len(texts[worst])} chars): {texts[worst][:160]!r}\n"
        f"  NumPy   = {ours[worst]:.6f}\n"
        f"  PyTorch = {theirs[worst]:.6f}\n"
        "Check, in this order: (1) the tokeniser — print\n"
        "    shared.preprocess.tokenise(text)\n"
        "and compare it against the vectoriser's analyzer output; (2) the\n"
        "TfidfVectorizer settings in train/reward_model.py::fit_tfidf against\n"
        "the replication in api/reward.py::tfidf_vector; (3) a transposed W.\n"
        "See the header of this file for the full diagnostic order."
    )
    # Printed so the measured number lands in the report rather than only the
    # verdict. "It passed" is weaker evidence than "it agreed to 4e-8".
    print(f"max |NumPy - PyTorch| = {max_abs_diff:.3e} over {len(texts)} texts "
          f"(tolerance {TOLERANCE:.0e})")


def test_the_scores_are_not_all_the_same(reference):
    """A separate assertion, because agreement about nothing is not agreement.

    If the export wrote zeros, both implementations would return `b1` for every
    input and the difference test above would pass perfectly. This asserts the
    head actually discriminates between the probe texts before the agreement
    means anything.
    """
    head, texts = reference["head"], reference["texts"]
    ours = np.asarray([head.score(t)[0] for t in texts], dtype=np.float64)
    assert np.all(np.isfinite(ours)), "a non-finite reward is a 500 waiting to happen"
    assert float(ours.std()) > 1e-3, (
        "every probe text got the same score, so the equivalence test above "
        "agreed about nothing. Did the export write zeros?"
    )


def test_a_transposed_export_is_actually_caught(reference):
    """Deliberately break it the usual way, and confirm the test would notice.

    A test that has never failed is a test you cannot trust. This corrupts the
    weights exactly as a wrong export would and asserts that the comparison
    above would have caught it — which is what licenses the claim that a green
    suite means the deployed head is the head you trained.

    The hidden layer is square (64 x 64 is not the shape here, but W1 is
    1 x hidden and W0 is hidden x V), so a genuine transpose of W0 would fail
    on shape. What DOES survive a shape check, and is the realistic silent bug,
    is a permuted vocabulary: same shapes, same arithmetic, wrong answers. That
    is what this rolls.
    """
    head, texts, theirs = reference["head"], reference["texts"], reference["scores"]
    broken = RewardHead(
        vocab=head.vocab,
        idf=head.idf,
        # Roll the first layer's columns by one: every term's weight is now
        # attributed to its neighbour in the vocabulary. Shapes unchanged.
        layers=[(np.roll(head.layers[0][0], 1, axis=1), head.layers[0][1])]
        + list(head.layers[1:]),
    )
    ours = np.asarray([broken.score(t)[0] for t in texts], dtype=np.float64)
    max_abs_diff = float(np.max(np.abs(ours - theirs)))
    assert max_abs_diff > TOLERANCE, (
        "misaligning the vocabulary by one column did not move the score by "
        "more than the tolerance, which means the tolerance is too loose to "
        "catch the bug it exists for"
    )


def test_the_deployed_artifact_loads_and_scores(client, trained_head_path):
    """The committed artifact must satisfy the layout the forward pass assumes.

    Distinct from the tests above: those check a freshly trained head, this one
    checks the file actually sitting in `policies/` that the service will load
    in production.
    """
    body = client.get("/policies").json()
    heads = [p for p in body["policies"] if p["kind"] == "reward-head"]
    assert heads, "no artifact of kind 'reward-head' is registered; /score cannot work"
    assert heads[0]["obs_dim"] and heads[0]["obs_dim"] > 0
    assert trained_head_path.exists()

    r = client.post("/score", json={"text": "specific measured evidence with a tested baseline"})
    assert r.status_code == 200, r.text
    assert np.isfinite(r.json()["reward"])
