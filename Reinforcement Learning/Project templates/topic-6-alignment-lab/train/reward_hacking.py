"""
train/reward_hacking.py — the evidence that optimising the proxy stopped
improving the product.

TRAINING TIER. Reads `alignment_runs` and `completions`, writes a chart.

    python -m train.reward_hacking --offline
    python -m train.reward_hacking --offline --from-store    # use rows already written

WHAT THIS FILE IS FOR
---------------------
Build step 8 asks for one chart — mean reward-model score and KL from the
reference against beta, on the same axes — and one claim: the point at which
the proxy kept rising while quality did not. The chart is easy. The claim is
the assignment, and it is only defensible if there is a series in it that the
reward model NEVER SAW.

That series is `completions.true_quality`. On the synthetic path it is exact:
`train.data.true_quality` is the function the preference labeller used, and the
reward model was fitted to COMPARISONS drawn from it, never to the function
itself. On the real path it is your own hand ratings of at least twenty
completions, and there is no way around collecting them — a chart with only
the proxy on it shows a number going up, which is not evidence of anything.

WHY THE DECOUPLING HAPPENS HERE, MECHANICALLY
---------------------------------------------
`true_quality` is `good-token density x token diversity`. Every training
comparison is between two non-degenerate responses whose diversity is similar,
so the comparisons carry almost no information about the diversity term. The
reward model therefore learns the density term and nothing else — an honest fit
to what it was shown.

A policy pushed far from the reference (small beta) discovers that the density
term is maximised by repeating the single highest-weighted token. Density goes
up, diversity collapses, `true_quality` falls, and the reward model — which
never learned to dislike repetition, because it was never shown any — scores
the result higher than before.

That is Goodhart's law with its mechanism exposed: the proxy and the target
agreed on the distribution the proxy was fitted on, and the optimiser's job is
to leave that distribution.

WHERE THE KL PENALTY COMES IN
-----------------------------
The KL term does not fix this and cannot. It BUYS TIME: it makes leaving the
reference distribution expensive, so the policy reaches the degenerate region
later, at a larger beta. Set beta high enough and the policy never gets there —
and also never improves, because pi = pi_ref in the limit. The whole exercise
is finding the beta at which you have bought the improvement and not yet bought
the collapse, and the honest answer is that the location of that point depends
on a quantity (`true_quality`) that you cannot measure at deployment time.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

REPORTS = pathlib.Path("reports")

# The beta sweep. The three the build step REQUIRES are 0.05, 0.1 and 0.5;
# the smaller values exist because the decoupling lives below 0.05 in this
# setup and a sweep that stops at the required three shows a monotone curve
# and no phenomenon. Report all of them and say which three were required.
DEFAULT_BETAS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5)
REQUIRED_BETAS = (0.05, 0.1, 0.5)


def find_decoupling(betas, proxy, truth) -> dict:
    """The beta at which the proxy and the target stop moving together.

    Definition, stated because "eyeball the chart" is not a definition: order
    the runs by DECREASING beta (increasing optimisation pressure) and find the
    first step at which the proxy rises and the target does not. The decoupling
    point is the beta BEFORE that step — the last configuration at which the
    two still agreed, which is the one you would ship.

    Also reported: the Pearson correlation between proxy and target over the
    low-pressure half and the high-pressure half of the sweep separately. DQ 5
    asks for exactly this split, and it is a better summary than the single
    breakpoint because it survives one noisy run. A correlation near +1 in the
    first half and near 0 or negative in the second is the signature; if both
    halves are positive, say so and do not claim a decoupling you did not see.
    """
    order = np.argsort(-np.asarray(betas, dtype=np.float64))   # decreasing beta
    b = np.asarray(betas, dtype=np.float64)[order]
    p = np.asarray(proxy, dtype=np.float64)[order]
    t = np.asarray(truth, dtype=np.float64)[order]

    point = None
    for i in range(1, len(b)):
        if p[i] > p[i - 1] and t[i] <= t[i - 1]:
            point = float(b[i - 1])
            break

    half = max(len(b) // 2, 2)

    def corr(x, y):
        # None, not NaN, when there are too few points. Two reasons: NaN is not
        # valid JSON — `json.dumps` writes a bare `NaN` that strict parsers
        # reject, and this dict is written to reports/ and read by the UI — and
        # a missing correlation should render as "not computed" rather than as
        # a number-shaped hole. With the required three betas alone this branch
        # fires, which is itself the argument for sweeping more than three.
        if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    # The magnitudes, past the decoupling point. Reported because the two
    # correlations above are computed over three or four points each and are
    # not stable across seeds — a correlation from four noisy points is a
    # number, not a finding. These two are differences between measured means
    # and say the thing the chart is claiming: from the last configuration at
    # which proxy and target agreed, to the most heavily optimised one, the
    # proxy went UP by this much and the target went DOWN by this much.
    i_peak = int(np.argmax(t))
    proxy_after = float(p[-1] - p[i_peak])
    true_after = float(t[-1] - t[i_peak])

    return {
        "decoupling_beta": point,
        "peak_true_quality_beta": float(b[int(np.argmax(t))]),
        "peak_proxy_beta": float(b[int(np.argmax(p))]),
        "proxy_change_past_peak": proxy_after,
        "true_quality_change_past_peak": true_after,
        "goodhart_observed": bool(proxy_after >= 0.0 > true_after),
        "corr_low_pressure_half": corr(p[:half], t[:half]),
        "corr_high_pressure_half": corr(p[half:], t[half:]),
        "betas_decreasing": b.tolist(),
        "proxy": p.tolist(),
        "truth": t.tolist(),
    }


def repetition_rate(texts: list[str]) -> float:
    """1 - (distinct tokens / tokens), averaged. The mechanism, as one number.

    Reported alongside the reward because it is what makes the claim concrete.
    "The proxy rose while quality fell" is an assertion; "the proxy rose while
    the repetition rate went from 0.10 to 0.28 and here are two completions"
    is evidence a reader can check.
    """
    from shared.preprocess import tokenise

    vals = []
    for t in texts:
        toks = tokenise(t)
        if toks:
            vals.append(1.0 - len(set(toks)) / len(toks))
    return float(np.mean(vals)) if vals else float("nan")


def collect(store) -> dict:
    """Join `alignment_runs` and `completions` into the series the chart needs.

    The base model is carried as the beta -> infinity point, because with an
    infinite KL penalty the optimal policy IS the reference. That is where it
    belongs conceptually and it is why the row sorts to the low-pressure end.

    It does NOT appear on the chart: the x-axis is log beta and infinity has no
    position on it. It appears in the printed table and in
    `reports/reward_hacking.json`, which is where the "compared to doing
    nothing" numbers come from. Quote it — the aligned variants are all far
    better than the base, and a chart that shows only the aligned ones makes
    the whole sweep look like a failure.
    """
    runs = {float(r["beta"]): r for r in store.alignment_runs(200)}
    comps = store.completions(limit=5000)

    rows = []
    base = [c for c in comps if c.get("model_variant") == "base"]
    if base:
        rows.append(
            {
                "beta": float("inf"),
                "label": "base (beta -> inf)",
                "kl_from_reference": 0.0,
                "mean_reward_model_score": float(
                    np.mean([c["reward_score"] for c in base if c["reward_score"] is not None])
                ),
                "mean_true_quality": float(
                    np.mean([c["true_quality"] for c in base if c["true_quality"] is not None])
                ),
                "repetition_rate": repetition_rate([c["text"] for c in base]),
                "mean_tokens": float(np.mean([c["tokens"] or 0 for c in base])),
            }
        )
    for beta, run in sorted(runs.items()):
        sub = [c for c in comps if c.get("beta") is not None and float(c["beta"]) == beta]
        rows.append(
            {
                "beta": beta,
                "label": f"beta={beta:g}",
                "kl_from_reference": run.get("kl_from_reference"),
                "mean_reward_model_score": run.get("mean_reward_model_score"),
                "mean_true_quality": run.get("mean_true_quality"),
                "implicit_reward_accuracy": run.get("implicit_reward_accuracy"),
                "repetition_rate": repetition_rate([c["text"] for c in sub]),
                "mean_tokens": float(np.mean([c["tokens"] or 0 for c in sub])) if sub else None,
            }
        )
    return {"rows": rows}


def plot(rows: list[dict], decoupling: dict, path: pathlib.Path) -> None:
    """Mean reward-model score and KL against beta, ON THE SAME AXES.

    Two y-axes, because the two series have different units and forcing them
    onto one would mean rescaling one of them and inviting the reader to
    compare rescaled magnitudes. The x-axis is LOG beta: the sweep spans
    0.002 to 0.5, and on a linear axis every point except the last is a smear
    at the origin.

    `true_quality` goes on the reward axis, because it is the series the
    reward score is a proxy FOR and the whole point of the chart is that the
    two come apart. Plotting it on its own axis would let a reader rescale the
    divergence away.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite = [r for r in rows if np.isfinite(r["beta"])]
    b = np.asarray([r["beta"] for r in finite], dtype=np.float64)
    proxy = np.asarray([r["mean_reward_model_score"] for r in finite], dtype=np.float64)
    truth = np.asarray([r["mean_true_quality"] for r in finite], dtype=np.float64)
    kl = np.asarray([r["kl_from_reference"] for r in finite], dtype=np.float64)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.set_xscale("log")
    ax.invert_xaxis()          # left-to-right = increasing optimisation pressure
    ax.plot(b, proxy, "o-", color="tab:blue", label="mean reward-model score (the PROXY)")
    # true_quality is O(0.4) and the proxy is O(5); scaling the target onto the
    # proxy's axis by a stated constant keeps both readable while making the
    # scaling visible in the legend rather than hidden in the code.
    scale = float(np.nanmax(proxy) / max(np.nanmax(truth), 1e-9))
    ax.plot(b, truth * scale, "s--", color="tab:green",
            label=f"mean true quality (x{scale:.1f}, the TARGET)")
    ax.set_xlabel("beta  (KL coefficient; decreasing to the right = more optimisation pressure)")
    ax.set_ylabel("reward-model score")

    ax2 = ax.twinx()
    ax2.plot(b, kl, "^:", color="tab:red", label="KL from reference (nats)")
    ax2.set_ylabel("KL( pi || pi_ref )  [nats]")

    d = decoupling.get("decoupling_beta")
    if d:
        ax.axvline(d, color="k", lw=1.4, ls="-.")
        # Anchored in AXES coordinates with an arrow to the line, rather than
        # offset from a data point. An offset annotation lands on top of
        # whichever series happens to be highest, which changes run to run —
        # the label would be readable on the figure you checked and illegible
        # on the one you submitted.
        ax.annotate(
            f"decoupling: beta = {d:g}\nproxy still rising, target falling\n"
            f"proxy {decoupling.get('proxy_change_past_peak', 0):+.3f}, "
            f"target {decoupling.get('true_quality_change_past_peak', 0):+.3f} past this point",
            xy=(d, 0.03),
            xycoords=("data", "axes fraction"),
            xytext=(0.62, 0.12),
            textcoords="axes fraction",
            fontsize=9,
            ha="left",
            arrowprops={"arrowstyle": "->", "lw": 1.0},
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.6", "alpha": 0.9},
        )

    lines, labels = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lb2, loc="lower left", fontsize=9)
    ax.set_title("Reward hacking: the proxy and the target come apart as the KL constraint loosens")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Sweep beta and chart the proxy against the target.")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--pairs", type=int, default=2400)
    ap.add_argument("--betas", type=float, nargs="+", default=list(DEFAULT_BETAS))
    ap.add_argument("--sft-steps", type=int, default=300)
    ap.add_argument("--dpo-steps", type=int, default=600)
    ap.add_argument("--gen-prompts", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-store", action="store_true",
                    help="chart the rows already in the data tier instead of re-running the sweep")
    args = ap.parse_args(argv)

    from shared.store import get_store

    store = get_store()

    if not args.from_store:
        from train.data import load_preferences
        from train.dpo import run_alignment

        ds = load_preferences(args.offline, n_pairs=args.pairs, seed=args.seed)
        run_alignment(
            ds,
            betas=tuple(args.betas),
            sft_steps=args.sft_steps,
            dpo_steps=args.dpo_steps,
            n_gen_prompts=args.gen_prompts,
            seed=args.seed,
            store=store,
        )

    data = collect(store)
    if not data["rows"]:
        # `--from-store` in a fresh process with no Supabase credentials reads
        # an EMPTY MemoryStore, because that store lives in the process that
        # wrote it. Say so rather than drawing an empty chart: an empty chart
        # and a broken query look identical, which is the same failure the
        # `degraded` flag exists to prevent in the UI.
        raise SystemExit(
            "no rows in the data tier. Either drop --from-store so this command "
            "runs the sweep itself, or configure SUPABASE_URL so that the rows "
            "written by `python -m train.train` persist beyond that process."
        )
    finite = [r for r in data["rows"] if np.isfinite(r["beta"])]
    decoupling = find_decoupling(
        [r["beta"] for r in finite],
        [r["mean_reward_model_score"] for r in finite],
        [r["mean_true_quality"] for r in finite],
    )
    plot(data["rows"], decoupling, REPORTS / "reward_hacking.png")

    out = {
        "rows": data["rows"],
        "decoupling": decoupling,
        "required_betas": list(REQUIRED_BETAS),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "reward_hacking.json").write_text(json.dumps(out, indent=2, default=str))

    print(f"{'beta':>10} {'KL':>8} {'proxy':>8} {'true':>8} {'repeat':>8} {'tokens':>7}")
    for r in data["rows"]:
        print(
            f"{r['label']:>10} "
            f"{(r['kl_from_reference'] or 0):8.2f} "
            f"{(r['mean_reward_model_score'] or 0):8.3f} "
            f"{(r['mean_true_quality'] or 0):8.3f} "
            f"{(r['repetition_rate'] or 0):8.3f} "
            f"{(r['mean_tokens'] or 0):7.1f}"
        )
    print(json.dumps({k: v for k, v in decoupling.items()
                      if k not in ("betas_decreasing", "proxy", "truth")}, indent=2))
    return out


if __name__ == "__main__":  # pragma: no cover - a CLI entry point
    main()
