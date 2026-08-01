"""
The multi-agent experiments behave the way the theory says they will.

Pure NumPy, so these run in the same process as the rest of the suite and in
CI — `train/multiagent.py` deliberately imports no framework. Tabular
Q-learning is an array update; the whole point of writing it out is that it
needs nothing.

Each test asserts a THEORETICAL PREDICTION rather than a number this template
happened to produce. A test that pins `final_reward == 1.57` fails the moment
anyone changes the exploration schedule and teaches nothing when it does; a
test that asserts "independent learners land nearer mutual defection than
mutual cooperation" is a statement about the game.
"""

from __future__ import annotations

import numpy as np

from train.multiagent import (
    IPD_PAYOFF,
    cooperative_independent_learners,
    iterated_prisoners_dilemma,
    matching_pennies,
    nonstationarity_experiment,
)


def test_the_payoff_table_matches_the_serving_tier_copy():
    """`envs/` keeps its own copy so it never imports train/. Check they agree.

    Duplication is the right call there — the serving tier must not import the
    training tier for a four-entry dict — but duplication that nothing verifies
    is duplication that drifts.
    """
    from envs import PAYOFF

    assert PAYOFF == IPD_PAYOFF


def test_the_payoffs_are_actually_a_prisoners_dilemma():
    """T > R > P > S and 2R > T + S. Both inequalities, both load-bearing.

    The first makes defection the dominant strategy in the stage game, which is
    what makes the outcome a dilemma. The second makes mutual cooperation beat
    alternating exploitation, which is what makes cooperation worth sustaining
    in the iterated game. Change a payoff so that either fails and you are
    studying a different game with the same name.
    """
    T = IPD_PAYOFF[(1, 0)][0]     # I defect, you cooperate
    R = IPD_PAYOFF[(0, 0)][0]     # both cooperate
    P = IPD_PAYOFF[(1, 1)][0]     # both defect
    S = IPD_PAYOFF[(0, 1)][0]     # I cooperate, you defect
    assert T > R > P > S
    assert 2 * R > T + S


def test_independent_learners_drift_toward_defection_on_the_ipd():
    """The theoretical prediction, and the reason multi-agent RL is its own topic.

    Neither agent's update contains any term for the effect of its action on
    the OTHER agent's future behaviour, and that is the only channel through
    which reciprocity could pay. So the pair should land much nearer mutual
    defection (1.0 per step) than mutual cooperation (3.0), even though both
    would prefer the latter. That is the two-agent tragedy of the commons.

    Asserted as "below the midpoint", not as an exact number: the exact landing
    point depends on the exploration schedule, and pinning it would make this a
    change-detector rather than a test.
    """
    res = iterated_prisoners_dilemma(episodes=1500, seed=0)
    midpoint = (res["mutual_cooperation_payoff"] + res["mutual_defection_payoff"]) / 2
    for who in ("final_reward_a", "final_reward_b"):
        assert res[who] < midpoint, (
            f"{who} = {res[who]:.3f}, above the midpoint {midpoint:.1f} between "
            "mutual defection and mutual cooperation. Independent Q-learners "
            "have no mechanism for sustaining reciprocity; if yours found one, "
            "check whether the Q-table was optimistically initialised or "
            "whether argmax's leftmost tie-break is picking cooperate."
        )
    assert res["final_cooperation_rate"] < 0.5


def test_a_frozen_opponent_makes_the_problem_stationary_again():
    """Experiment (b): the same agent, the same algorithm, one thing changed.

    Against a FROZEN opponent the environment is a genuine MDP and tabular
    Q-learning converges: the table stops moving and the greedy policy stops
    switching. Against a LEARNING opponent neither is guaranteed, because the
    target agent A is regressing onto moves whenever B updates.

    The assertions are about STABILITY, not about return. The two arms play
    different opponents, so their returns are not comparable and comparing them
    would be comparing two different games.
    """
    res = nonstationarity_experiment(episodes=1500, seed=0)
    frozen, both = res["frozen_opponent"], res["both_learning"]

    assert frozen["q_drift"] < both["q_drift"], (
        f"agent A's Q-table drifted {frozen['q_drift']:.2e} per episode against a "
        f"frozen opponent and {both['q_drift']:.2e} against a learning one. The "
        "frozen arm must be the settled one — if it is not, the 'frozen' agent "
        "is still updating."
    )
    assert frozen["policy_switches"] <= both["policy_switches"]
    assert frozen["late_reward_std"] < both["late_reward_std"], (
        "non-stationarity shows up as persistent VARIANCE in the late training "
        "reward, not as a lower mean. If the frozen arm is the noisier one, "
        "check that the frozen opponent's policy is actually fixed."
    )
    # And the frozen arm should genuinely SETTLE, not merely drift less. Stated
    # as an order-of-magnitude ratio rather than as an absolute drift value:
    # the absolute number depends on where the epsilon schedule has got to by
    # the last third of training, so a fixed bound would pass at 3,000 episodes
    # and fail at 1,500 for reasons that have nothing to do with the claim.
    assert frozen["policy_switches"] == 0
    assert both["q_drift"] > 10 * frozen["q_drift"], (
        f"Q-table drift against a learning opponent ({both['q_drift']:.2e}) is not "
        f"an order of magnitude above the frozen arm's ({frozen['q_drift']:.2e}). "
        "The frozen arm should be converging and the other should not be."
    )


def test_matching_pennies_cycles_around_the_mixed_equilibrium():
    """DQ 6(b): a game with no deterministic Nash equilibrium.

    Two things must hold, and they say different things:

      * the TIME AVERAGE of each agent's action frequency is near 0.5 — the
        cycle is centred on the mixed equilibrium;
      * the WINDOWED frequency keeps moving — the late windows are as spread
        out as the early ones. That is what distinguishes a cycle from
        convergence with noise, and it is the actual prediction. A pair that
        converged would show a shrinking spread.
    """
    res = matching_pennies(steps=40000, seed=0)
    assert abs(res["time_average_a"] - 0.5) < 0.06, (
        f"agent A's time-averaged action frequency is {res['time_average_a']:.3f}; "
        "the unique Nash equilibrium of matching pennies is 0.5"
    )
    assert abs(res["time_average_b"] - 0.5) < 0.06
    assert res["late_over_early_std_a"] > 0.5, (
        f"the spread of A's windowed action frequency shrank by "
        f"{1 - res['late_over_early_std_a']:.2f} over training, which would mean "
        "the learners converged. They cannot: no deterministic policy is a Nash "
        "equilibrium here, so a converging pair means the exploration rate is "
        "decaying and hiding the cycle."
    )


def test_independent_learners_solve_the_small_cooperative_task():
    """Experiment (c): they beat a random policy, which is the floor that matters.

    A rising team-reward curve is not evidence on its own — epsilon decay alone
    raises it. The comparison against a random policy on the same environment
    is what makes "it learned" a claim rather than a shape.

    Note the scale this works at: two agents on a 5x5 grid, 625 joint states.
    Independent learning degrades as agents are added, because the shared
    reward gives each agent the same number regardless of its own
    contribution — that credit-assignment problem is the reason centralised
    training with decentralised execution exists, and it is what your 200-300
    words should be about.
    """
    res = cooperative_independent_learners(episodes=1500, seed=0)
    assert res["final_team_reward"] > res["random_policy_return"], (
        f"team reward {res['final_team_reward']:.2f} did not beat the random "
        f"policy's {res['random_policy_return']:.2f}"
    )
    assert res["improvement_over_random"] > 1.0


def test_the_frozen_opponent_environment_is_a_real_environment():
    """`envs/make_env()` must satisfy the contract `POST /rollout` calls.

    It exists to make the point that a FROZEN opponent yields a genuine MDP —
    so it had better actually behave like one, including returning `truncated`
    rather than `terminated` at the horizon.
    """
    from envs import make_env

    env = make_env(opponent="tit_for_tat", max_steps=5)
    obs, _ = env.reset(seed=0)
    assert obs == 4, "the first observation must be the distinct 'no history' state"

    rewards = []
    for step in range(5):
        obs, reward, terminated, truncated, _ = env.step(0)   # always cooperate
        rewards.append(reward)
        assert terminated is False, "the IPD has no terminal state"
        assert truncated is (step == 4)
    # Tit-for-tat opens with cooperation and copies; cooperating throughout
    # yields mutual cooperation on every step.
    assert rewards == [IPD_PAYOFF[(0, 0)][0]] * 5

    env.reset(seed=0)
    # Against always_cooperate the best response is always to defect, for T.
    env2 = make_env(opponent="always_cooperate", max_steps=3)
    env2.reset(seed=0)
    assert env2.step(1)[1] == IPD_PAYOFF[(1, 0)][0]


def test_the_q_learner_breaks_ties_at_random():
    """An all-zero table must not make every agent cooperate.

    `np.argmax` returns the LEFTMOST maximum, so with a zero-initialised table
    both agents would deterministically choose action 0 — cooperate — for as
    long as the table stayed flat. The 'emergent cooperation' in the first
    thousand steps would be an artefact of NumPy's tie-breaking rule, and it is
    exactly the kind of result that survives into a report.
    """
    from train.multiagent import QLearner

    agent = QLearner(1, 2, epsilon=0.0, seed=0)
    actions = [agent.act(0) for _ in range(400)]
    assert 0.3 < float(np.mean(actions)) < 0.7, (
        "with a flat Q-table and no exploration, the agent picked the same "
        "action almost every time — the tie-break is not random"
    )
