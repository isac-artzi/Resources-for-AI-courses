"""
train/multiagent.py — what happens when the environment is another learner.

TRAINING TIER. Pure NumPy: no torch, no framework. Tabular Q-learning is an
array update, and writing it out is the point.

    python -m train.multiagent --offline
    python -m train.multiagent --offline --experiment ipd

Three experiments, matching build step 7:

  (a) INDEPENDENT TABULAR Q-LEARNING on the iterated prisoner's dilemma, with
      each agent's reward plotted.
  (b) NON-STATIONARITY, demonstrated rather than asserted: freeze one agent and
      train the other (a stationary MDP), then let both learn at once (not an
      MDP at all), and compare.
  (c) INDEPENDENT LEARNERS ON A COOPERATIVE TASK, with team reward plotted.

Plus `matching_pennies()`, which is DQ 6(b) and is here because it is the
cleanest possible demonstration of the failure: a game whose only equilibrium
is mixed, where two independent Q-learners provably cannot converge and instead
cycle forever.

WHY NON-STATIONARITY IS A THEOREM, NOT A NUISANCE
--------------------------------------------------
Every algorithm in Topics 1 to 5 assumed a Markov decision process, and an MDP
requires a FIXED transition function P(s' | s, a). From agent i's point of
view, the environment includes agent j, so the effective transition is

    P_i(s' | s, a_i) = sum_{a_j} pi_j(a_j | s) * P(s' | s, a_i, a_j)

which depends on pi_j. While agent j is learning, pi_j changes, so P_i changes
with time. The stationarity assumption is violated by construction, and with it
go the convergence guarantees of Q-learning: the target agent i is regressing
onto is moving because of something agent i cannot observe.

Experiment (b) is that statement, measured. Same agent, same algorithm, same
hyperparameters; the only difference is whether the opponent is frozen.

THE PETTINGZOO VERSION NOTE (read before you `pip install`)
-----------------------------------------------------------
The multi-particle environments (MPE) MOVED OUT OF PettingZoo in release
1.26.0 and now live in a separate `mpe2` package. `pip install
'pettingzoo[mpe]'` no longer resolves — that extra does not exist any more —
and every tutorial written before the split tells you to run it. The correct
installs are:

    pip install mpe2                     # simple_spread_v3 and friends
    pip install 'pettingzoo[butterfly]'  # pistonball_v6

and the correct import for the cooperative task is

    from mpe2 import simple_spread_v3

`cooperative_real()` below does exactly that, and `CooperativeGridworld` is a
self-contained fallback with the same structure — N agents, N landmarks, a
shared team reward for covering them and a penalty for colliding — so this file
runs offline. The fallback is a gridworld and simple_spread is continuous; the
qualitative result (independent learners solve the small instance, and the
credit-assignment problem is visible in the variance) is the same, and the
quantitative numbers are not comparable. Say which one produced your figure.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass

import numpy as np

REPORTS = pathlib.Path("reports")

# Prisoner's dilemma payoffs, in the standard (T > R > P > S) ordering:
#   T(emptation) = 5 > R(eward) = 3 > P(unishment) = 1 > S(ucker) = 0
# and 2R > T + S, so mutual cooperation beats alternating exploitation. Both
# inequalities matter: the first makes defection dominant in the one-shot game,
# the second is what makes cooperation worth sustaining in the iterated one.
IPD_PAYOFF = {
    (0, 0): (3.0, 3.0),   # C, C
    (0, 1): (0.0, 5.0),   # C, D
    (1, 0): (5.0, 0.0),   # D, C
    (1, 1): (1.0, 1.0),   # D, D
}
COOPERATE, DEFECT = 0, 1


# ===========================================================================
# The learner
# ===========================================================================


@dataclass
class QLearner:
    """Tabular Q-learning with epsilon-greedy exploration. One per agent.

    "Independent" means exactly this: the agent's table is indexed by the state
    and its OWN action. It has no representation of the other agent, no access
    to the other agent's parameters and no term for the other agent's next
    action. That is the assumption experiment (b) breaks.
    """

    n_states: int
    n_actions: int
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 0.2
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.9995
    seed: int = 0
    frozen: bool = False
    # Boltzmann (softmax) exploration instead of epsilon-greedy, when set.
    #
    # Used only by `matching_pennies`, and for a specific reason: an
    # epsilon-greedy policy is a STEP FUNCTION of the Q-table, so its action
    # probability can only take two values and its phase plot is four corners.
    # A Boltzmann policy moves continuously with the Q-values, so the orbit DQ
    # 6(d) asks you to plot is actually visible. It is also the theoretically
    # right choice: the continuous-time limit of Boltzmann Q-learning is the
    # replicator dynamics, whose trajectories in matching pennies are closed
    # cycles around the mixed equilibrium — which is exactly the prediction the
    # experiment is checking.
    temperature: float | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        # Optimistic initialisation at zero rather than at the max payoff. With
        # an optimistic start both IPD agents cooperate for thousands of steps
        # purely because they have not yet been disappointed, and you would
        # report emergent reciprocity that is actually an initialisation
        # artefact. Zero is the honest starting point here.
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

    def act(self, state: int) -> int:
        if self.temperature is not None and not self.frozen:
            return int(self.rng.choice(self.n_actions, p=self.action_probabilities(state)))
        if not self.frozen and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        row = self.Q[state]
        # Ties broken at RANDOM, not by argmax's leftmost rule. With an
        # all-zero table argmax always returns action 0, so both agents would
        # cooperate on every tie and the "emergent cooperation" in the first
        # thousand steps would be an artefact of NumPy's tie-breaking.
        best = np.flatnonzero(row == row.max())
        return int(self.rng.choice(best))

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        if self.frozen:
            return          # a frozen agent is a fixed policy, i.e. part of the environment
        target = reward + self.gamma * float(self.Q[next_state].max())
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

    def decay(self) -> None:
        if not self.frozen:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def action_probabilities(self, state: int) -> np.ndarray:
        """The BEHAVIOUR policy at `state`, as a distribution.

        Two branches, because the two exploration schemes give genuinely
        different objects. The Boltzmann policy is a smooth function of the
        Q-values and is what the matching-pennies phase plot traces; the
        epsilon-greedy policy is a step function of them and can take only two
        values per action.
        """
        if self.temperature is not None:
            # Max-subtracted, as everywhere else in this repository. Without it
            # a Q-value of 80 at temperature 0.1 exponentiates to inf and the
            # whole distribution becomes NaN — and matching pennies drives
            # Q-values apart indefinitely, so this is reachable, not theoretical.
            z = (self.Q[state] - self.Q[state].max()) / self.temperature
            e = np.exp(z)
            return e / e.sum()
        p = np.full(self.n_actions, self.epsilon / self.n_actions)
        best = np.flatnonzero(self.Q[state] == self.Q[state].max())
        p[best] += (1.0 - self.epsilon) / len(best)
        return p


# ===========================================================================
# (a) The iterated prisoner's dilemma
# ===========================================================================


def _ipd_state(last: tuple[int, int] | None) -> int:
    """Memory-1 state: the previous joint action, plus a distinct start state.

    State 4 is "no history yet". Folding it into (C, C) would tell both agents
    that the game opens from mutual cooperation, which is a different game and
    is exactly the assumption tit-for-tat needs to work.
    """
    return 4 if last is None else last[0] * 2 + last[1]


def iterated_prisoners_dilemma(
    episodes: int = 3000,
    steps: int = 20,
    seed: int = 0,
    freeze_agent_b: bool = False,
    b_policy: np.ndarray | None = None,
    alpha: float = 0.1,
    gamma: float = 0.95,
) -> dict:
    """Two independent Q-learners playing memory-1 IPD.

    Returns per-episode mean reward for each agent, the cooperation rate, and
    both final Q-tables.

    `freeze_agent_b` is experiment (b)'s first arm: agent B plays a FIXED
    policy, so agent A faces a genuine stationary MDP and Q-learning's
    convergence guarantee applies. Everything else is identical.

    WHAT THE THEORY PREDICTS, so you can check it rather than admire the plot:
    defection strictly dominates in the stage game, and an independent
    Q-learner maximises its own discounted return against whatever the other
    agent is currently doing. Nothing in either agent's update represents the
    effect of its action on the OTHER agent's future behaviour, which is the
    only mechanism by which reciprocity could pay. So mutual defection —
    per-step reward 1.0 against the 3.0 available from mutual cooperation — is
    the expected outcome, and this is the two-agent form of the tragedy of the
    commons rather than a bug in the learner.
    """
    a = QLearner(5, 2, alpha=alpha, gamma=gamma, seed=seed)
    b = QLearner(5, 2, alpha=alpha, gamma=gamma, seed=seed + 1000, frozen=freeze_agent_b)
    if b_policy is not None:
        # Encode a fixed policy as a Q-table with a large gap, so `act` picks it
        # greedily. Cleaner than special-casing `act`, and it means the frozen
        # agent goes through exactly the same code path as a learning one.
        b.Q = np.zeros((5, 2))
        b.Q[np.arange(5), b_policy] = 1.0

    rewards_a, rewards_b, coop = [], [], []
    for _ in range(episodes):
        last: tuple[int, int] | None = None
        ra = rb = 0.0
        n_coop = 0
        for _ in range(steps):
            s = _ipd_state(last)
            act_a, act_b = a.act(s), b.act(s)
            pa, pb = IPD_PAYOFF[(act_a, act_b)]
            last = (act_a, act_b)
            s2 = _ipd_state(last)
            a.update(s, act_a, pa, s2)
            b.update(s, act_b, pb, s2)
            ra += pa
            rb += pb
            n_coop += int(act_a == COOPERATE) + int(act_b == COOPERATE)
        a.decay()
        b.decay()
        rewards_a.append(ra / steps)
        rewards_b.append(rb / steps)
        coop.append(n_coop / (2 * steps))

    tail = max(episodes // 10, 1)
    return {
        "reward_a": rewards_a,
        "reward_b": rewards_b,
        "cooperation_rate": coop,
        "final_reward_a": float(np.mean(rewards_a[-tail:])),
        "final_reward_b": float(np.mean(rewards_b[-tail:])),
        "final_cooperation_rate": float(np.mean(coop[-tail:])),
        "mutual_defection_payoff": IPD_PAYOFF[(DEFECT, DEFECT)][0],
        "mutual_cooperation_payoff": IPD_PAYOFF[(COOPERATE, COOPERATE)][0],
        "Q_a": a.Q.tolist(),
        "Q_b": b.Q.tolist(),
        "frozen_opponent": freeze_agent_b,
    }


# ===========================================================================
# (b) Non-stationarity, measured
# ===========================================================================


def nonstationarity_experiment(episodes: int = 3000, steps: int = 20, seed: int = 0) -> dict:
    """Freeze one agent and train the other; then let both learn. Compare.

    The comparison is NOT "which arm got more reward" — the two arms play
    against different opponents, so their returns are not comparable and a
    chart of the two returns together is a chart of two different games.

    The comparison is about STABILITY, which is what non-stationarity actually
    costs you. Three measurements, all on agent A:

      * `policy_switches` — how many times A's greedy action changes after the
        curve has nominally converged. Against a frozen opponent this should
        fall to near zero; against a learning one it does not, because A's
        target keeps moving.
      * `late_reward_std` — the standard deviation of A's per-episode reward
        over the last third of training. Non-stationarity shows up as
        persistent variance, not as a lower mean.
      * `q_drift` — the mean absolute change in A's Q-table per episode over
        the last third. A converging tabular learner's drift goes to zero; a
        learner chasing a moving target's does not.

    Against a frozen opponent Q-learning's convergence guarantee applies and
    all three settle. Against a learning opponent none of them need to, and
    that is the assumption violation made visible.
    """
    # Tit-for-tat as the frozen opponent: cooperate first, then copy A's last
    # move. Indexing follows _ipd_state, where state = a_prev * 2 + b_prev.
    #   state 0 = (C,C) -> A cooperated -> C      state 1 = (C,D) -> C
    #   state 2 = (D,C) -> A defected   -> D      state 3 = (D,D) -> D
    #   state 4 = start                          -> C
    tit_for_tat = np.array([COOPERATE, COOPERATE, DEFECT, DEFECT, COOPERATE])

    frozen = _instrumented_ipd(episodes, steps, seed, b_policy=tit_for_tat)
    both = _instrumented_ipd(episodes, steps, seed, b_policy=None)
    return {
        "frozen_opponent": frozen,
        "both_learning": both,
        "verdict": {
            "policy_switch_ratio": (
                both["policy_switches"] / max(frozen["policy_switches"], 1)
            ),
            "late_reward_std_ratio": (
                both["late_reward_std"] / max(frozen["late_reward_std"], 1e-9)
            ),
            "q_drift_ratio": both["q_drift"] / max(frozen["q_drift"], 1e-12),
        },
    }


def _instrumented_ipd(episodes: int, steps: int, seed: int, b_policy) -> dict:
    """The IPD loop again, with the stability instrumentation experiment (b) needs.

    Deliberately a separate function rather than a flag on
    `iterated_prisoners_dilemma`. That one is the readable reference
    implementation students are meant to follow; adding drift tracking and
    policy-switch counting to it would bury the four lines of Q-learning that
    are the actual lesson.
    """
    a = QLearner(5, 2, seed=seed)
    b = QLearner(5, 2, seed=seed + 1000, frozen=b_policy is not None)
    if b_policy is not None:
        b.Q = np.zeros((5, 2))
        b.Q[np.arange(5), b_policy] = 1.0

    rewards, drifts, switches = [], [], 0
    prev_greedy = a.Q.argmax(axis=1).copy()
    start_measuring = 2 * episodes // 3

    for ep in range(episodes):
        q_before = a.Q.copy()
        last = None
        ra = 0.0
        for _ in range(steps):
            s = _ipd_state(last)
            act_a, act_b = a.act(s), b.act(s)
            pa, pb = IPD_PAYOFF[(act_a, act_b)]
            last = (act_a, act_b)
            s2 = _ipd_state(last)
            a.update(s, act_a, pa, s2)
            b.update(s, act_b, pb, s2)
            ra += pa
        a.decay()
        b.decay()
        rewards.append(ra / steps)

        if ep >= start_measuring:
            drifts.append(float(np.abs(a.Q - q_before).mean()))
            greedy = a.Q.argmax(axis=1)
            switches += int((greedy != prev_greedy).sum())
            prev_greedy = greedy.copy()

    late = np.asarray(rewards[start_measuring:], dtype=np.float64)
    return {
        "reward": rewards,
        "mean_late_reward": float(late.mean()),
        "late_reward_std": float(late.std()),
        "q_drift": float(np.mean(drifts)) if drifts else 0.0,
        "policy_switches": switches,
        "opponent": "frozen tit-for-tat" if b_policy is not None else "learning",
    }


# ===========================================================================
# DQ 6(b): matching pennies — the game with no deterministic equilibrium
# ===========================================================================


def matching_pennies(steps: int = 60000, seed: int = 0, temperature: float = 0.1,
                     alpha: float = 0.02, stride: int = 5) -> dict:
    """Two independent Q-learners in a zero-sum game with a unique MIXED equilibrium.

    Payoff: the matcher wins when the two coins agree, the mismatcher when they
    differ. There is NO deterministic Nash equilibrium — for any pure pair, one
    player strictly gains by switching — and the unique equilibrium is both
    players randomising 50/50.

    Q-learning cannot represent a mixed policy. Its greedy action is a
    deterministic function of the table, so the pair CYCLES: the matcher learns
    to play heads, the mismatcher best-responds with tails, the matcher
    switches, and around it goes. The empirical action frequencies orbit
    (0.5, 0.5) without settling on it.

    WHAT IS PLOTTED, and why it is not the empirical action frequency. The
    series recorded below is each agent's POLICY — `P(agent plays tails)` read
    off its own Q-table — sampled every `stride` steps. Two alternatives were
    tried and rejected:

      * The GREEDY action is a step function, so its phase plot is four
        corners with lines between them.
      * The empirical frequency in a sliding window averages the cycle away
        unless the window is shorter than the period, and the period here is
        about 27 steps. A window that short has a standard error of 0.1 and
        the "orbit" is mostly sampling noise. (Measure it yourself: the greedy
        action flips roughly every 27 steps at these hyperparameters.)

    Hence Boltzmann exploration, which makes the policy a smooth function of
    the Q-values. That is also the theoretically right choice: the
    continuous-time limit of Boltzmann Q-learning is the replicator dynamics,
    and in matching pennies the replicator trajectories are CLOSED ORBITS
    around the mixed equilibrium — never converging to it, never leaving it.
    The phase plot is a picture of that.

    WHAT TO CHECK, because "the plot looks like a spiral" is not a result:
      * the TIME AVERAGE of each agent's action probability is close to 0.5 —
        the orbit is centred on the equilibrium even though no iterate is at it;
      * the SPREAD does not shrink — the second half of training is as spread
        out as the first. That is what distinguishes an orbit from convergence
        with noise, and a pair that converged would fail it.

    A single-state game, so gamma is 0 and there is no bootstrapping; this
    isolates the equilibrium-selection failure from anything to do with credit
    assignment over time.
    """
    a = QLearner(1, 2, alpha=alpha, gamma=0.0, temperature=temperature, seed=seed)
    b = QLearner(1, 2, alpha=alpha, gamma=0.0, temperature=temperature, seed=seed + 7)
    # A tiny asymmetric nudge off the exact equilibrium. Starting both tables at
    # exactly zero puts the pair ON the fixed point of the dynamics, where it
    # would stay but for sampling noise — and the resulting plot would show a
    # random walk rather than an orbit, which is a different phenomenon.
    a.Q[0, 1] += 0.02
    b.Q[0, 1] -= 0.02

    prob_a, prob_b = [], []
    for t in range(steps):
        act_a, act_b = a.act(0), b.act(0)
        r_a = 1.0 if act_a == act_b else -1.0     # A is the matcher
        a.update(0, act_a, r_a, 0)
        b.update(0, act_b, -r_a, 0)
        if t % stride == 0:
            prob_a.append(float(a.action_probabilities(0)[1]))
            prob_b.append(float(b.action_probabilities(0)[1]))

    fa, fb = np.asarray(prob_a), np.asarray(prob_b)
    half = len(fa) // 2
    return {
        "stride": stride,
        "temperature": temperature,
        "freq_a": prob_a,
        "freq_b": prob_b,
        "time_average_a": float(fa.mean()),
        "time_average_b": float(fb.mean()),
        "policy_std_a": float(fa.std()),
        "policy_std_b": float(fb.std()),
        "policy_range_a": [float(fa.min()), float(fa.max())],
        # A converging pair would have a small late spread; an orbiting pair's
        # late spread matches its early one. The ratio is the test.
        "late_over_early_std_a": float(fa[half:].std() / max(fa[:half].std(), 1e-9)),
        "nash_action_probability": 0.5,
    }


# ===========================================================================
# (c) A cooperative task
# ===========================================================================


class CooperativeGridworld:
    """N agents, N landmarks, one SHARED team reward. The offline stand-in for simple_spread.

    Reward, per step, shared identically by every agent:

        -sum over landmarks of (distance from the nearest agent)   [coverage]
        -collision_penalty for each pair of agents on the same cell

    That is `simple_spread_v3`'s reward structure on a grid. The shared reward
    is the point: it creates the CREDIT-ASSIGNMENT problem that independent
    learners have no mechanism for. An agent that moves usefully and an agent
    that stands still receive the same number, so each has to infer its own
    contribution from the noise of the other's behaviour. That is the concrete
    reason independent learning degrades as the number of agents grows, and it
    is the reason centralised training with decentralised execution exists.

    State for each agent is the JOINT position (own cell, other's cell), which
    keeps the task Markov for a tabular learner. Note what that costs: the
    state space is |cells|^n_agents, so this scales to two agents on a 5x5 grid
    and to nothing else. A real independent learner sees only its own local
    observation and the task is then partially observed as well as
    non-stationary — worth stating in your 200-300 words.
    """

    def __init__(self, size: int = 5, n_agents: int = 2, collision_penalty: float = 1.0,
                 seed: int = 0) -> None:
        self.size = size
        self.n_agents = n_agents
        self.collision_penalty = collision_penalty
        self.rng = np.random.default_rng(seed)
        # Fixed landmarks. Randomising them per episode would make the task
        # partially observed for an agent whose state is only positions, and
        # the tabular learner would then be failing for a reason that has
        # nothing to do with multi-agent learning.
        self.landmarks = [(0, 0), (size - 1, size - 1)][:n_agents]
        self.n_cells = size * size
        self.n_states = self.n_cells ** n_agents
        self.n_actions = 5              # stay, up, down, left, right

    def reset(self) -> list[tuple[int, int]]:
        self.pos = [
            (int(self.rng.integers(self.size)), int(self.rng.integers(self.size)))
            for _ in range(self.n_agents)
        ]
        return self.pos

    def state_index(self, agent: int) -> int:
        """Joint state, with THIS agent's own cell first.

        The reordering matters: agent 0's state 37 and agent 1's state 37 then
        mean "I am here and the other is there" for both, so the two agents'
        tables are indexed compatibly and a symmetric task produces symmetric
        tables. Without it, the two learners are solving mirror-image problems
        and their curves are not comparable.
        """
        order = [agent] + [i for i in range(self.n_agents) if i != agent]
        idx = 0
        for i in order:
            r, c = self.pos[i]
            idx = idx * self.n_cells + (r * self.size + c)
        return idx

    def step(self, actions: list[int]) -> float:
        moves = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        for i, a in enumerate(actions):
            dr, dc = moves[a]
            r, c = self.pos[i]
            # Clipped at the walls rather than wrapped or penalised. A wrapping
            # grid makes the distance metric wrong; a wall penalty adds a second
            # objective to a task that is meant to have one.
            self.pos[i] = (int(np.clip(r + dr, 0, self.size - 1)),
                           int(np.clip(c + dc, 0, self.size - 1)))

        reward = 0.0
        for lm in self.landmarks:
            reward -= min(abs(p[0] - lm[0]) + abs(p[1] - lm[1]) for p in self.pos)
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if self.pos[i] == self.pos[j]:
                    reward -= self.collision_penalty
        return reward


def cooperative_independent_learners(
    episodes: int = 4000, steps: int = 20, size: int = 5, seed: int = 0
) -> dict:
    """Independent tabular Q-learners on the cooperative gridworld. Plots team reward.

    Every agent receives the SAME reward and updates its OWN table. No
    communication, no shared parameters, no centralised critic. This is the
    baseline that centralised training with decentralised execution is measured
    against, and running it first is what makes the CTDE discussion in your
    write-up concrete.
    """
    env = CooperativeGridworld(size=size, seed=seed)
    agents = [
        QLearner(env.n_states, env.n_actions, alpha=0.15, gamma=0.9,
                 epsilon=0.3, epsilon_decay=0.999, seed=seed + i)
        for i in range(env.n_agents)
    ]

    team_rewards = []
    for _ in range(episodes):
        env.reset()
        total = 0.0
        for _ in range(steps):
            states = [env.state_index(i) for i in range(env.n_agents)]
            actions = [agents[i].act(states[i]) for i in range(env.n_agents)]
            reward = env.step(actions)
            next_states = [env.state_index(i) for i in range(env.n_agents)]
            for i, ag in enumerate(agents):
                ag.update(states[i], actions[i], reward, next_states[i])
            total += reward
        for ag in agents:
            ag.decay()
        team_rewards.append(total / steps)

    tail = max(episodes // 10, 1)
    # The random-policy floor, so "it learned" is a comparison rather than a
    # shape. A team-reward curve that rises is not evidence on its own: the
    # epsilon decay alone raises it.
    floor = _random_policy_return(env, steps, seed=seed + 99)
    return {
        "team_reward": team_rewards,
        "final_team_reward": float(np.mean(team_rewards[-tail:])),
        "random_policy_return": floor,
        "improvement_over_random": float(np.mean(team_rewards[-tail:]) - floor),
        "n_agents": env.n_agents,
        "n_states": env.n_states,
        "environment": "CooperativeGridworld (offline fallback for simple_spread_v3)",
    }


def _random_policy_return(env: CooperativeGridworld, steps: int, episodes: int = 200,
                          seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    totals = []
    for _ in range(episodes):
        env.reset()
        total = 0.0
        for _ in range(steps):
            total += env.step([int(rng.integers(env.n_actions)) for _ in range(env.n_agents)])
        totals.append(total / steps)
    return float(np.mean(totals))


def cooperative_real(episodes: int = 200, seed: int = 0):  # pragma: no cover - needs mpe2
    """The required path: `simple_spread_v3` from the `mpe2` package.

    THE INSTALL IS `pip install mpe2`, NOT `pip install 'pettingzoo[mpe]'`.
    That extra was removed in PettingZoo 1.26.0 when the MPE environments were
    split into their own package, and every tutorial older than that release
    tells you to use it.

    `parallel_env` rather than the AEC API: independent learners all act at
    once, and the AEC turn-based wrapper would serialise them into an
    interleaving that is not the game. `continuous_actions=False` because the
    learner here is tabular; with continuous actions you need Topic 4's
    machinery, which is a different product.

    The observation is a float vector, so a tabular learner needs a
    discretisation — `shared.preprocess.discretise` per component is the
    course-standard way, and the number of bins is a hyperparameter you must
    report because it changes the state space by orders of magnitude.
    """
    from mpe2 import simple_spread_v3

    return simple_spread_v3.parallel_env(
        N=2,
        local_ratio=0.5,          # split between team reward and own-collision term
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
    )


# ===========================================================================
# Plots and the CLI
# ===========================================================================


def _smooth(x, k: int = 50) -> np.ndarray:
    """Moving average. Raw per-episode curves in these games are unreadable.

    Report the window. A curve smoothed over 50 episodes and a curve smoothed
    over 500 tell different stories about how stable the run was, and a figure
    that does not say which one it is cannot be checked.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="valid")


def plot_all(results: dict, reports: pathlib.Path = REPORTS) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # (a) IPD, each agent's reward
    ipd = results["ipd"]
    ax = axes[0][0]
    ax.plot(_smooth(ipd["reward_a"]), label="agent A")
    ax.plot(_smooth(ipd["reward_b"]), label="agent B")
    ax.axhline(ipd["mutual_cooperation_payoff"], ls="--", c="tab:green",
               label="mutual cooperation (3.0)")
    ax.axhline(ipd["mutual_defection_payoff"], ls="--", c="tab:red",
               label="mutual defection (1.0)")
    ax.set_title("(a) Iterated prisoner's dilemma, independent Q-learners")
    ax.set_xlabel("episode (smoothed, window 50)")
    ax.set_ylabel("mean reward per step")
    ax.legend(fontsize=8)

    # (b) non-stationarity
    ns = results["nonstationarity"]
    ax = axes[0][1]
    ax.plot(_smooth(ns["frozen_opponent"]["reward"]), label="A vs FROZEN opponent")
    ax.plot(_smooth(ns["both_learning"]["reward"]), label="A vs LEARNING opponent")
    ax.set_title(
        "(b) Non-stationarity: Q drift ratio "
        f"{ns['verdict']['q_drift_ratio']:.1f}x, "
        f"late-reward sd ratio {ns['verdict']['late_reward_std_ratio']:.1f}x"
    )
    ax.set_xlabel("episode (smoothed, window 50)")
    ax.set_ylabel("agent A mean reward per step")
    ax.legend(fontsize=8)

    # (c) cooperative task
    co = results["cooperative"]
    ax = axes[1][0]
    ax.plot(_smooth(co["team_reward"]), color="tab:purple", label="team reward")
    ax.axhline(co["random_policy_return"], ls="--", c="k", label="random policy")
    ax.set_title("(c) Cooperative task, independent learners, shared reward")
    ax.set_xlabel("episode (smoothed, window 50)")
    ax.set_ylabel("team reward per step")
    ax.legend(fontsize=8)

    # DQ 6(d): the phase plot
    mp = results["matching_pennies"]
    ax = axes[1][1]
    # The LAST 600 samples only. The full trace is ~12,000 samples and several
    # hundred orbits, which overplot into a solid rectangle — technically the
    # same information, visually none of it. A short LATE window shows
    # individual loops AND makes the point the statistics make: the orbits are
    # still full-width at the end of training, so nothing converged.
    tail_a, tail_b = mp["freq_a"][-600:], mp["freq_b"][-600:]
    ax.plot(tail_a, tail_b, lw=0.8, alpha=0.85, color="tab:blue",
            label="last 600 samples of the policy trace")
    ax.plot(0.5, 0.5, "r*", ms=14, label="mixed Nash (0.5, 0.5)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("P(agent A plays tails)")
    ax.set_ylabel("P(agent B plays tails)")
    ax.set_title(
        "Matching pennies: policy orbit, time-average "
        f"({mp['time_average_a']:.2f}, {mp['time_average_b']:.2f})"
    )
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(reports / "multiagent.png", dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Multi-agent experiments (a), (b) and (c).")
    ap.add_argument("--offline", action="store_true",
                    help="use the built-in gridworld instead of mpe2's simple_spread_v3")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--coop-episodes", type=int, default=4000)
    ap.add_argument("--pennies-steps", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--experiment", choices=["all", "ipd", "nonstationarity",
                                             "cooperative", "pennies"], default="all")
    args = ap.parse_args(argv)

    results: dict = {}
    if args.experiment in ("all", "ipd"):
        results["ipd"] = iterated_prisoners_dilemma(episodes=args.episodes, seed=args.seed)
    if args.experiment in ("all", "nonstationarity"):
        results["nonstationarity"] = nonstationarity_experiment(
            episodes=args.episodes, seed=args.seed
        )
    if args.experiment in ("all", "cooperative"):
        results["cooperative"] = cooperative_independent_learners(
            episodes=args.coop_episodes, seed=args.seed
        )
    if args.experiment in ("all", "pennies"):
        results["matching_pennies"] = matching_pennies(steps=args.pennies_steps, seed=args.seed)

    if args.experiment == "all":
        plot_all(results)

    REPORTS.mkdir(parents=True, exist_ok=True)
    # The raw curves are megabytes of JSON and nothing reads them from here —
    # the chart is the artifact. Persist the SUMMARY, and keep the curves in
    # `episodes` if you want them queryable.
    slim = {
        k: {kk: vv for kk, vv in v.items()
            if not isinstance(vv, list) or len(vv) <= 32}
        for k, v in results.items()
    }
    (REPORTS / "multiagent.json").write_text(json.dumps(slim, indent=2, default=str))
    print(json.dumps(slim, indent=2, default=str))
    return results


if __name__ == "__main__":  # pragma: no cover - a CLI entry point
    main()
