"""
train/data.py — the preference dataset, loaded once and persisted to `preferences`.

TRAINING TIER. Never imported by api/ or ui/.

    python -m train.data                     # the real path: Hugging Face hub
    python -m train.data --offline           # the deterministic fallback
    python -m train.data --offline --pairs 2400 --inspect 5

WHICH PATH YOU ARE ON, AND WHY IT MATTERS
-----------------------------------------
The required dataset is `trl-lib/ultrafeedback_binarized`: roughly 63,000
comparisons, each a prompt with a chosen and a rejected response, binarised
from UltraFeedback's GPT-4 ratings. `load_real()` below is the real loading
path and it is what you must run for the graded product.

`load_synthetic()` is an OFFLINE FALLBACK for environments with no network
access to the hub. It is a deterministic generator over a small artificial
vocabulary. **It is not a substitute for the real data and no result computed
on it belongs in your report as though it were.** What it is good for is
exactly one thing: making every downstream stage — embedding, both reward
heads, DPO, generation, scoring, the reward-hacking sweep — runnable and
debuggable end to end before you spend an hour of GPU time. Every row it writes
carries `source = 'synthetic-offline'`, so a `group by source` will always tell
you which numbers came from where. If a number in your README came from a row
with that source, say so in the sentence that quotes it.

The synthetic generator is not a random string generator. It has a latent
structure that mirrors the three properties of real preference data this
product needs to be able to see:

  1. A LEXICAL QUALITY SIGNAL. Some tokens indicate a careful answer, some
     indicate a vague one. This is what makes the comparison learnable at all,
     and it is why both reward heads should clear the 50% baseline clearly. If
     one of yours sits at chance on this data, the training code is wrong —
     the signal is genuinely there.

  2. A LENGTH CONFOUND. Better answers are drawn slightly longer, as they are
     in real preference corpora, where annotators reward thoroughness and
     length rides along. This is what makes the length-bias regression in
     `train/reward_model.py` a real measurement rather than a formality.

  3. A QUALITY SIGNAL THE REWARD MODEL CANNOT SEE. Preferences are labelled by
     `true_quality`, which penalises REPETITION as well as rewarding good
     tokens. Every training pair is non-degenerate, so the reward model never
     encounters a repetitive response and never learns to dislike one. That
     gap is the mechanism behind the reward hacking demonstrated in
     `train/reward_hacking.py`: it is not injected, it is a consequence of
     what the comparisons did and did not contain, which is exactly how the
     failure arises in practice.

Point 3 is the one to understand. A demonstration of Goodhart's law in which
you hand-code the divergence between proxy and truth demonstrates nothing. Here
the reward model is fit honestly to comparisons that were themselves generated
honestly from `true_quality`; the divergence appears because the comparisons
under-determine the target, which is the actual claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from shared.preprocess import response_length, tokenise

SYNTHETIC_SOURCE = "synthetic-offline"
REAL_SOURCE = "trl-lib/ultrafeedback_binarized"


# ===========================================================================
# The synthetic corpus
# ===========================================================================

# Tokens that carry latent quality. Kept short and readable so that you can
# eyeball a generated response and judge it yourself — which is what Step 2 of
# the build ("inspect several examples by hand") asks you to do, and which is
# much harder to do honestly on real UltraFeedback rows at 3am.
GOOD_TOKENS = (
    "specific measured stepwise verified evidence reproducible caveat quantified "
    "tested benchmark citation baseline assumption limitation tradeoff concrete"
).split()

BAD_TOKENS = (
    "vague handwave probably whatever unclear guessing obviously trivially hype "
    "magic effortless guaranteed flawless"
).split()

# Neutral filler. Carries no weight in `true_quality`, which is the point: it
# is the mass of ordinary language that both a good and a bad answer contain,
# and it is what stops the tf-idf head's job from being trivial.
FILLER_TOKENS = (
    "the model needs data output result system approach method under given "
    "process value case input change effect state above below where before "
    "during between within across return list report note number scale range "
    "table figure column row sample batch step run part item field level"
).split()

TOPIC_WORDS = (
    "scheduling routing inventory forecasting pricing staffing logistics "
    "compliance onboarding retention throughput latency uptime accuracy "
    "coverage triage escalation capacity"
).split()

PROMPT_TEMPLATES = (
    "how should we approach {t} for a small team",
    "explain the tradeoffs in our {t} process",
    "what would you change about the {t} pipeline",
    "write a short brief on {t} for a non technical reader",
    "our {t} numbers moved last quarter what should we check",
    "draft the first paragraph of a {t} proposal",
)


def true_quality(text: str) -> float:
    """The latent quality the SYNTHETIC labeller uses. The reward model never sees it.

    Two factors, multiplied:

        density   = (good tokens - bad tokens) / tokens
        diversity = distinct tokens / tokens

    The density term is the part a bag-of-words model can learn — it is a
    linear function of the token counts, which is precisely what a TF-IDF head
    computes. The diversity term is the part it cannot, because every training
    comparison is between two non-degenerate responses whose diversity is
    similar, so the comparisons contain almost no information about it.

    That asymmetry is the whole reward-hacking result. A policy pushed hard
    enough will find the response that maximises density at the cost of
    diversity — the same good token, repeated — and the reward model will
    score it highly because nothing in its training data ever punished that.
    """
    toks = tokenise(text)
    if not toks:
        return 0.0
    good = sum(1 for t in toks if t in _GOOD_SET)
    bad = sum(1 for t in toks if t in _BAD_SET)
    density = (good - bad) / len(toks)
    diversity = len(set(toks)) / len(toks)
    return float(density * diversity)


_GOOD_SET = set(GOOD_TOKENS)
_BAD_SET = set(BAD_TOKENS)


@dataclass
class SyntheticCorpus:
    """A deterministic generator of prompts and responses of controllable quality.

    Shared by `train/data.py` (to build comparisons) and `train/dpo.py` (to
    build the vocabulary the small language model is trained over, and to
    supply held-out prompts for generation). One generator rather than two
    means the DPO policy and the reward model are looking at the same language,
    which is the arrangement the real pipeline has and a second generator would
    quietly break.
    """

    seed: int = 0
    n_prompts: int = 300
    prompts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        seen: list[str] = []
        # Templates x topics gives 6 x 18 = 108 distinct prompts; beyond that we
        # append a discriminating word so prompt ids stay unique. Duplicated
        # prompt TEXT with distinct ids would look fine here and then split
        # across train and test, which is the leak this loop exists to avoid.
        for i in range(self.n_prompts):
            t = TOPIC_WORDS[int(rng.integers(len(TOPIC_WORDS)))]
            tmpl = PROMPT_TEMPLATES[int(rng.integers(len(PROMPT_TEMPLATES)))]
            base = tmpl.format(t=t)
            extra = FILLER_TOKENS[i % len(FILLER_TOKENS)]
            seen.append(f"{base} regarding {extra}")
        self.prompts = seen

    # -- response generation -------------------------------------------------

    def sample_response(self, quality: float, rng: np.random.Generator) -> str:
        """Draw a response whose expected `true_quality` increases with `quality`.

        `quality` is in [0, 1]. Note the LENGTH COUPLING on the first line: the
        mean length rises with quality. That is the length confound, put there
        on purpose because real preference corpora have one — annotators reward
        thoroughness, thoroughness correlates with length, and a reward model
        fit to the result can score length instead of quality and lose nothing
        on the training objective. Remove this line and the length-bias
        experiment in `train/reward_model.py` measures nothing.
        """
        mean_len = 12.0 + 10.0 * quality
        n = int(np.clip(rng.poisson(mean_len), 4, 60))

        p_good = 0.06 + 0.30 * quality
        p_bad = 0.02 + 0.26 * (1.0 - quality)
        toks: list[str] = []
        for _ in range(n):
            u = rng.random()
            if u < p_good:
                toks.append(GOOD_TOKENS[int(rng.integers(len(GOOD_TOKENS)))])
            elif u < p_good + p_bad:
                toks.append(BAD_TOKENS[int(rng.integers(len(BAD_TOKENS)))])
            else:
                toks.append(FILLER_TOKENS[int(rng.integers(len(FILLER_TOKENS)))])
        return " ".join(toks)

    def vocabulary(self) -> list[str]:
        """Every token the generator can emit, plus the prompt words.

        Used by `train/dpo.py` to size the small language model's softmax. It
        is a closed vocabulary of a few hundred types, which is what makes an
        exact KL divergence from the reference policy computable rather than
        estimated — see the note there.
        """
        vocab = set(GOOD_TOKENS) | set(BAD_TOKENS) | set(FILLER_TOKENS)
        for p in self.prompts:
            vocab |= set(tokenise(p))
        return sorted(vocab)


# ===========================================================================
# Building comparisons
# ===========================================================================


@dataclass
class PreferenceDataset:
    train: list[dict]
    test: list[dict]
    source: str

    def summary(self) -> dict:
        def stats(rows: list[dict]) -> dict:
            if not rows:
                return {"pairs": 0}
            cl = np.asarray([r["chosen_len"] for r in rows], dtype=np.float64)
            rl = np.asarray([r["rejected_len"] for r in rows], dtype=np.float64)
            # The label-length correlation of the DATA, which is the reference
            # point every length-bias number in this product is judged against.
            # A head whose reward correlates with length more strongly than the
            # labels themselves do has invented a preference nobody expressed.
            lens = np.concatenate([cl, rl])
            labels = np.concatenate([np.ones_like(cl), np.zeros_like(rl)])
            r = float(np.corrcoef(lens, labels)[0, 1]) if lens.size > 2 else float("nan")
            return {
                "pairs": len(rows),
                "prompts": len({x["prompt_id"] for x in rows}),
                "mean_chosen_len": float(cl.mean()),
                "mean_rejected_len": float(rl.mean()),
                "longer_chosen_rate": float((cl > rl).mean()),
                "label_length_pearson_r": r,
            }

        return {"source": self.source, "train": stats(self.train), "test": stats(self.test)}


def _prompt_id(prompt: str) -> str:
    """A STABLE key derived from the prompt text.

    Not a row number. `completions.prompt_id` joins to this, and a surrogate
    key that changes when you re-import the dataset would silently re-point
    every completion at a different prompt — a corruption with no error
    message, discovered when a stakeholder notices the answers do not match the
    questions.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def load_synthetic(
    n_pairs: int = 2400,
    seed: int = 0,
    test_fraction: float = 0.2,
    n_prompts: int = 300,
    label_noise: float = 0.08,
) -> PreferenceDataset:
    """The offline fallback. Deterministic given `seed`.

    THE SPLIT IS BY PROMPT, NOT BY PAIR. Several comparisons share a prompt, so
    a pair-level random split would put comparisons about the same prompt on
    both sides and the held-out accuracy would be measuring memorisation of the
    prompt's vocabulary. Splitting by prompt is the honest version and it is
    also what the real dataset's own split does.

    `label_noise` flips 8% of labels. Without it the Bayes-optimal accuracy is
    100%, a reward head that reaches 99% looks excellent, and you learn nothing
    about whether your training loop is right — every bug looks like "not quite
    converged". With it the ceiling is around 92% and a head stuck at 60% is
    visibly broken.
    """
    corpus = SyntheticCorpus(seed=seed, n_prompts=n_prompts)
    rng = np.random.default_rng(seed + 1)

    n_test_prompts = max(1, int(round(test_fraction * n_prompts)))
    order = rng.permutation(n_prompts)
    test_prompt_idx = set(int(i) for i in order[:n_test_prompts])

    train_rows: list[dict] = []
    test_rows: list[dict] = []
    for i in range(n_pairs):
        pi = i % n_prompts
        prompt = corpus.prompts[pi]

        # Two quality levels, drawn to overlap. Non-overlapping draws would make
        # every comparison easy and the accuracy ceiling meaningless.
        q_hi = float(rng.uniform(0.45, 0.95))
        q_lo = float(rng.uniform(0.05, 0.55))
        a = corpus.sample_response(q_hi, rng)
        b = corpus.sample_response(q_lo, rng)

        # The LABEL comes from true_quality, not from the q parameters. The
        # sampled response is what the labeller sees, and sampling noise means
        # the higher-q draw is sometimes genuinely worse. Labelling from the
        # parameter instead would be labelling from information no annotator has.
        if true_quality(a) >= true_quality(b):
            chosen, rejected = a, b
        else:
            chosen, rejected = b, a
        if rng.random() < label_noise:
            chosen, rejected = rejected, chosen

        row = {
            "prompt_id": _prompt_id(prompt),
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "split": "test" if pi in test_prompt_idx else "train",
            "chosen_len": response_length(chosen),
            "rejected_len": response_length(rejected),
            "source": SYNTHETIC_SOURCE,
        }
        (test_rows if row["split"] == "test" else train_rows).append(row)

    return PreferenceDataset(train=train_rows, test=test_rows, source=SYNTHETIC_SOURCE)


def load_real(
    n_pairs: int = 4000,
    seed: int = 0,
    dataset_name: str = REAL_SOURCE,
) -> PreferenceDataset:
    """The required path: a binarised preference dataset from the Hugging Face hub.

    `datasets` is a TRAINING-TIER dependency and is imported inside the
    function, not at module scope, for the same reason `train/export.py` imports
    torch inside `export_torch_mlp`: this module's other functions must remain
    importable in an environment that has neither.

    The row shape of `trl-lib/ultrafeedback_binarized` is
    `{"prompt": [...], "chosen": [...], "rejected": [...]}` where each value is
    a CHAT-FORMATTED LIST OF MESSAGES, not a string. `_flatten` below takes the
    last assistant turn. Read a few rows by hand before trusting that — the
    binarised TRL variants have changed shape at least once, and a silent
    `str(list_of_dicts)` would give you a dataset of Python repr strings that
    trains perfectly well and means nothing.

    The dataset ships its own `train_prefs` / `test_prefs` splits. Use them.
    Re-splitting a dataset that has a canonical split makes your numbers
    incomparable with every published number on it, for no benefit.
    """
    from datasets import load_dataset  # training tier only

    def _flatten(value) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value:
            last = value[-1]
            if isinstance(last, dict):
                return str(last.get("content", ""))
            return str(last)
        return str(value)

    out: dict[str, list[dict]] = {"train": [], "test": []}
    for split_name, our_name in (("train", "train"), ("test", "test")):
        ds = load_dataset(dataset_name, split=split_name)
        take = min(n_pairs if our_name == "train" else max(n_pairs // 4, 200), len(ds))
        ds = ds.shuffle(seed=seed).select(range(take))
        for rec in ds:
            prompt = _flatten(rec.get("prompt", ""))
            chosen = _flatten(rec.get("chosen", ""))
            rejected = _flatten(rec.get("rejected", ""))
            if not (prompt and chosen and rejected):
                continue  # a malformed row is a dropped row, not a crash
            out[our_name].append(
                {
                    "prompt_id": _prompt_id(prompt),
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "split": our_name,
                    "chosen_len": response_length(chosen),
                    "rejected_len": response_length(rejected),
                    "source": dataset_name,
                }
            )
    return PreferenceDataset(train=out["train"], test=out["test"], source=dataset_name)


def load_preferences(offline: bool, **kwargs) -> PreferenceDataset:
    """One switch. Everything downstream takes a `PreferenceDataset` and does not care.

    Written as an explicit flag rather than a try/except around `load_real`.
    A silent fallback is the single worst option here: you would run the graded
    experiment, the hub would time out, the synthetic generator would answer,
    and the numbers in your report would be about a corpus that does not exist.
    Failing loudly and requiring `--offline` to be typed is the point.
    """
    if offline:
        return load_synthetic(**kwargs)
    real_kwargs = {k: v for k, v in kwargs.items() if k in {"n_pairs", "seed", "dataset_name"}}
    return load_real(**real_kwargs)


def persist(dataset: PreferenceDataset, store=None) -> int:
    """Write both splits to `preferences`. Returns the row count."""
    if store is None:
        from shared.store import get_store

        store = get_store()
    rows = dataset.train + dataset.test
    return store.insert_preferences(rows)


def inspect(dataset: PreferenceDataset, n: int = 3) -> str:
    """Print `n` comparisons in full, for the by-hand inspection Step 2 requires.

    Deliberately prints WHOLE responses rather than truncating them. The point
    of the exercise is to notice something about the data — that the chosen
    response is often merely longer, that the "rejected" one is sometimes
    fine — and a 60-character preview hides exactly that.
    """
    lines = []
    for row in dataset.train[:n]:
        lines.append("-" * 72)
        lines.append(f"prompt_id {row['prompt_id']}  split={row['split']}")
        lines.append(f"PROMPT   : {row['prompt']}")
        lines.append(f"CHOSEN   ({row['chosen_len']:3d} tok): {row['chosen']}")
        lines.append(f"REJECTED ({row['rejected_len']:3d} tok): {row['rejected']}")
    lines.append("-" * 72)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Load a preference dataset and persist it.")
    ap.add_argument("--offline", action="store_true",
                    help="use the deterministic synthetic generator instead of the hub")
    ap.add_argument("--pairs", type=int, default=2400,
                    help="comparisons to load; the build step requires at least 2,000")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inspect", type=int, default=3,
                    help="how many comparisons to print in full for by-hand inspection")
    args = ap.parse_args(argv)

    ds = load_preferences(args.offline, n_pairs=args.pairs, seed=args.seed)
    n = persist(ds)
    summary = ds.summary() | {"rows_written": n}

    if args.inspect:
        print(inspect(ds, args.inspect))
    print(json.dumps(summary, indent=2))
    if ds.source == SYNTHETIC_SOURCE:
        print(
            "\nNOTE: these rows carry source='synthetic-offline'. They exercise the "
            "pipeline; they are NOT the dataset the product is graded on. Re-run "
            "without --offline before you quote a number from them."
        )
    return summary


if __name__ == "__main__":  # pragma: no cover - a CLI entry point
    main()
