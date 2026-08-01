"""
train/train.py — the one command that produces every artifact this product ships.

TRAINING TIER (imports torch, transitively). Runs on your laptop or in Colab.

    python -m train.train --offline --quick     # the whole pipeline in ~2 minutes
    python -m train.train --offline             # the offline pipeline at report budget
    python -m train.train                       # the REAL pipeline (hub + GPU)

What it does, in the order the build steps require:

    1. train.data        load the preference dataset, split by prompt, persist
                         to `preferences`
    2. train.embed       embed every response once with a FROZEN encoder, cache
                         the vectors, report elapsed time and cache size
    3. train.reward_model  fit BOTH heads on the same Bradley-Terry loss, export
                         and register both, measure accuracy and length bias
    4. train.dpo         SFT the reference, run DPO at each beta, generate
                         completions offline, score them through the DEPLOYED
                         endpoint, write `alignment_runs` and `completions`
    5. train.reward_hacking  chart proxy against target against KL, locate the
                         decoupling point
    6. train.multiagent  the three multi-agent experiments plus matching pennies

ORDERING IS NOT ARBITRARY. Step 4 scores completions with the head exported in
step 3, through the real `POST /score` handler, so the reward numbers in
`completions` are the numbers the deployed service returns. Running step 4
before step 3 would score them with whatever stale artifact happened to be in
`policies/`, and nothing would report an error.

WHAT `--quick` IS AND IS NOT
----------------------------
`--quick` exists so that a fresh fork completes end to end in about two minutes
and you find your bugs before you spend an hour on them. At those budgets the
beta sweep has three points, DPO has run for 200 steps, and the multi-agent
curves have not converged. **Nothing produced under `--quick` is a result.**
Say in your Quantitative Analysis which command produced the numbers you quote.

Every stage is also a module with its own `main()`, so you can re-run one
without re-running the pipeline — which is what you will actually do while
debugging. `python -m train.reward_model --offline` is the fastest loop.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

REPORTS = pathlib.Path("reports")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the whole Alignment Lab pipeline.")
    # `--offline` rather than `--online`, and no automatic fallback. See the
    # note in train/data.py::load_preferences: a silent fallback would let the
    # graded run quietly become a synthetic one when the hub times out.
    ap.add_argument("--offline", action="store_true",
                    help="use the deterministic offline generators throughout")
    ap.add_argument("--quick", action="store_true",
                    help="sandbox budget: ~2 minutes end to end. NOT a result.")
    ap.add_argument("--pairs", type=int, default=None,
                    help="comparison pairs; the build step requires at least 2,000")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--betas", type=float, nargs="+", default=None)
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["data", "embed", "reward", "dpo", "hacking", "multiagent"],
                    help="stages to skip; useful when re-running one of them")
    return ap.parse_args(argv)


def budgets(quick: bool) -> dict:
    """The two budgets, in one place, so the README table cannot drift from the code."""
    if quick:
        return {
            "pairs": 2400,          # NOT reduced: below ~2,000 the held-out accuracy
                                    # has a standard error of 0.03 and the two heads
                                    # stop being separable, which defeats the purpose
            # Identical to the report budget. The reward-head comparison is the
            # headline result and it must not change between the two budgets, or
            # a student debugs against one set of numbers and reports another.
            "epochs": 25,
            "sft_steps": 200,
            "dpo_steps": 200,
            "gen_prompts": 60,      # the build step requires at least 50
            "betas": (0.05, 0.1, 0.5),
            "ma_episodes": 800,
            "ma_coop_episodes": 800,
            "pennies_steps": 20000,
        }
    return {
        "pairs": 2400,
        "epochs": 25,
        "sft_steps": 300,
        "dpo_steps": 600,
        "gen_prompts": 60,
        # The three the build step requires, PLUS four smaller values. The
        # decoupling in this setup sits below 0.05, so a sweep that stops at
        # the required three shows a monotone curve and no phenomenon.
        "betas": (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5),
        "ma_episodes": 3000,
        "ma_coop_episodes": 4000,
        "pennies_steps": 60000,
    }


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    b = budgets(args.quick)
    if args.pairs:
        b["pairs"] = args.pairs
    if args.betas:
        b["betas"] = tuple(args.betas)

    from shared.config import get_settings
    from shared.store import get_store

    store = get_store()
    summary: dict = {"offline": args.offline, "quick": args.quick, "budgets": b, "stages": {}}
    t_start = time.perf_counter()

    def stage(name: str):
        print(f"\n=== {name} " + "=" * (60 - len(name)), flush=True)
        return time.perf_counter()

    # One `experiments` row for the whole pipeline, written FIRST so that every
    # artifact and every alignment run can point at it. The registry's foreign
    # key is what makes "which run produced the thing we deployed" answerable
    # with a join rather than with a guess.
    experiment_id = store.insert_experiment(
        {
            "algorithm": "dpo+reward-model",
            "env_id": "preference-corpus",
            "seed": args.seed,
            "hyperparameters": {"offline": args.offline, "quick": args.quick, **{
                k: (list(v) if isinstance(v, tuple) else v) for k, v in b.items()
            }},
            "git_sha": get_settings().git_sha,
        }
    )
    summary["experiment_id"] = experiment_id

    # -- 1 & 2: data and embeddings ----------------------------------------
    t = stage("1/6  preference data")
    from train.data import load_preferences, persist

    dataset = load_preferences(args.offline, n_pairs=b["pairs"], seed=args.seed)
    if "data" not in args.skip:
        persist(dataset, store)
    summary["stages"]["data"] = dataset.summary() | {"seconds": round(time.perf_counter() - t, 1)}
    print(json.dumps(summary["stages"]["data"], indent=2))

    t = stage("2/6  offline embedding (frozen encoder)")
    from train.embed import embed_dataset, make_encoder

    encoder = make_encoder(args.offline)
    embeddings, embed_stats = embed_dataset(dataset, encoder)
    summary["stages"]["embed"] = {
        "encoder": encoder.name,
        "total_seconds": round(sum(s["seconds"] for s in embed_stats.values()), 3),
        "cache_megabytes": round(
            sum(s["cache_bytes"] for s in embed_stats.values()) / 1e6, 3
        ),
        "dim": next(iter(embed_stats.values()))["dim"],
    }
    print(json.dumps(summary["stages"]["embed"], indent=2))

    # -- 3: the two reward heads -------------------------------------------
    if "reward" not in args.skip:
        t = stage("3/6  two reward heads, one Bradley-Terry loss")
        from train.reward_model import HeadConfig, train_both

        heads = train_both(
            dataset,
            embeddings,
            HeadConfig(seed=args.seed, epochs=b["epochs"]),
            experiment_id=experiment_id,
        )
        summary["stages"]["reward_heads"] = {
            name: {
                "held_out_accuracy": round(heads[name]["held_out_accuracy"], 4),
                "accuracy_stderr": round(heads[name]["accuracy_stderr"], 4),
                "length_bias_r": round(heads[name]["length_bias"]["pearson_r"], 4),
                "length_decodability_r2": round(heads[name]["length_decodability_r2"], 4),
                "length_matched_accuracy": round(
                    heads[name]["length_matched"]["accuracy"], 4
                ),
                "artifact_bytes": heads[name]["artifact"]["bytes"],
            }
            for name in ("tfidf", "embedding")
        } | {
            "deployed": heads["deployed"],
            # Paired, not two independent error bars — see
            # train/reward_model.py::paired_accuracy_difference.
            "tfidf_vs_embedding_paired": {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in heads["tfidf_vs_embedding"].items()
            },
            "seconds": round(time.perf_counter() - t, 1),
        }
        print(json.dumps(summary["stages"]["reward_heads"], indent=2))

        # The service tier caches artifacts at import time. The pipeline just
        # overwrote them, so the in-process handler that step 4 scores through
        # would otherwise still be holding the PREVIOUS run's weights — and
        # every reward in `completions` would belong to an artifact that is no
        # longer on disk. One line, and it is not optional.
        from api.main import POLICIES

        POLICIES.reload()

    # -- 4: alignment -------------------------------------------------------
    if "dpo" not in args.skip:
        t = stage("4/6  DPO alignment + offline generation + scoring")
        from train.dpo import run_alignment

        alignment = run_alignment(
            dataset,
            betas=tuple(b["betas"]),
            sft_steps=b["sft_steps"],
            dpo_steps=b["dpo_steps"],
            n_gen_prompts=b["gen_prompts"],
            seed=args.seed,
            store=store,
            experiment_id=experiment_id,
            notes=("tiny-lm offline fallback" if args.offline else "trl DPOTrainer"),
        )
        summary["stages"]["alignment"] = {
            "base_mean_reward": alignment["reference"]["mean_reward_model_score"],
            "base_mean_true_quality": alignment["reference"]["mean_true_quality"],
            "dpo_loss_monotone_in_margin": alignment["loss_monotonicity_check"][
                "monotone_decreasing"
            ],
            "runs": [
                {k: v for k, v in r.items() if k not in ("history", "experiment_id")}
                for r in alignment["runs"]
            ],
            "seconds": round(time.perf_counter() - t, 1),
        }
        print(json.dumps(summary["stages"]["alignment"], indent=2, default=str))

    # -- 5: reward hacking --------------------------------------------------
    if "hacking" not in args.skip:
        t = stage("5/6  reward hacking: proxy vs target vs KL")
        from train.reward_hacking import REPORTS as RH_REPORTS
        from train.reward_hacking import collect, find_decoupling, plot

        data = collect(store)
        # The base row is carried at beta = infinity so that `plot` can put it
        # at the correct end of the x-axis. It is excluded from the decoupling
        # search, which is a statement about the SWEEP: the base model is the
        # point the sweep departs from, not a configuration in it.
        import numpy as _np

        data_rows = data["rows"]
        finite = [r for r in data_rows if _np.isfinite(r["beta"])]
        decoupling = find_decoupling(
            [r["beta"] for r in finite],
            [r["mean_reward_model_score"] for r in finite],
            [r["mean_true_quality"] for r in finite],
        )
        plot(data["rows"], decoupling, RH_REPORTS / "reward_hacking.png")
        (RH_REPORTS / "reward_hacking.json").write_text(
            json.dumps({"rows": data["rows"], "decoupling": decoupling}, indent=2, default=str)
        )
        summary["stages"]["reward_hacking"] = {
            k: v for k, v in decoupling.items()
            if k not in ("betas_decreasing", "proxy", "truth")
        } | {"seconds": round(time.perf_counter() - t, 1)}
        print(json.dumps(summary["stages"]["reward_hacking"], indent=2))

    # -- 6: multi-agent -----------------------------------------------------
    if "multiagent" not in args.skip:
        t = stage("6/6  multi-agent experiments")
        from train.multiagent import (
            cooperative_independent_learners,
            iterated_prisoners_dilemma,
            matching_pennies,
            nonstationarity_experiment,
            plot_all,
        )

        ma = {
            "ipd": iterated_prisoners_dilemma(episodes=b["ma_episodes"], seed=args.seed),
            "nonstationarity": nonstationarity_experiment(
                episodes=b["ma_episodes"], seed=args.seed
            ),
            "cooperative": cooperative_independent_learners(
                episodes=b["ma_coop_episodes"], seed=args.seed
            ),
            "matching_pennies": matching_pennies(steps=b["pennies_steps"], seed=args.seed),
        }
        plot_all(ma)
        summary["stages"]["multiagent"] = {
            "ipd_final_reward": [
                round(ma["ipd"]["final_reward_a"], 3), round(ma["ipd"]["final_reward_b"], 3)
            ],
            "ipd_cooperation_rate": round(ma["ipd"]["final_cooperation_rate"], 3),
            "ipd_mutual_defection_payoff": ma["ipd"]["mutual_defection_payoff"],
            "nonstationarity_verdict": {
                k: round(v, 3) for k, v in ma["nonstationarity"]["verdict"].items()
            },
            "cooperative_final": round(ma["cooperative"]["final_team_reward"], 3),
            "cooperative_random_floor": round(ma["cooperative"]["random_policy_return"], 3),
            "pennies_time_average": [
                round(ma["matching_pennies"]["time_average_a"], 3),
                round(ma["matching_pennies"]["time_average_b"], 3),
            ],
            "pennies_late_over_early_std": round(
                ma["matching_pennies"]["late_over_early_std_a"], 3
            ),
            "seconds": round(time.perf_counter() - t, 1),
        }
        print(json.dumps(summary["stages"]["multiagent"], indent=2))

    summary["total_seconds"] = round(time.perf_counter() - t_start, 1)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "pipeline.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 68)
    print(f"pipeline complete in {summary['total_seconds']}s")
    print(f"artifacts  -> policies/    charts -> {REPORTS}/")
    if args.offline:
        print(
            "\nNOTE: --offline. Every row this run wrote is labelled with a synthetic\n"
            "source and every chart came from the fallback generators. The pipeline\n"
            "is exercised; the RESULT is not the one the product is graded on."
        )
    if not get_settings().data_tier_configured:
        print(
            "\nNOTE: SUPABASE_URL is unset, so every row went to the in-process\n"
            "fallback store and vanished when this process exited. The .npz files and\n"
            "the PNGs on disk are real; the telemetry is not. Fill in .env before the\n"
            "run you intend to report."
        )
    return summary


if __name__ == "__main__":  # pragma: no cover - a CLI entry point
    main()
