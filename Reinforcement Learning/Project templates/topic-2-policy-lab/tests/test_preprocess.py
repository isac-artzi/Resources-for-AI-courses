"""
Unit tests for every function in shared/preprocess.py.

"Every function" is the standard the build steps set, and it is worth meeting
literally rather than approximately. Preprocessing is the layer where a bug is
silent: nothing raises, the learner trains, the numbers look plausible, and the
deployed agent behaves differently from the one you measured. The pairs that
are supposed to invert each other are tested as round trips, because a
transform whose inverse you have never run is a transform whose convention you
do not actually know.
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.preprocess import (
    bin_centre,
    clip_reward,
    dense_model,
    discounted_returns,
    discretise,
    first_visit_indices,
    from_one_hot,
    index_to_state,
    normalise_observation,
    normalise_returns,
    one_hot,
    state_index,
)


# -- one_hot / from_one_hot -------------------------------------------------


@pytest.mark.parametrize("index", [0, 3, 24])
def test_one_hot_round_trips(index):
    assert from_one_hot(one_hot(index, 25)) == index


def test_one_hot_has_exactly_one_hot_entry():
    v = one_hot(7, 25)
    assert v.shape == (25,) and v.sum() == 1.0 and v[7] == 1.0


def test_one_hot_rejects_an_out_of_range_index():
    with pytest.raises(ValueError):
        one_hot(25, 25)


def test_from_one_hot_rejects_a_probability_vector():
    """A softmax output is not a one-hot vector, and argmaxing it here would
    turn a caller's mistake into a plausible-looking wrong answer."""
    with pytest.raises(ValueError):
        from_one_hot(np.array([0.7, 0.2, 0.1]))


# -- state_index / index_to_state -------------------------------------------


@pytest.mark.parametrize("row,col", [(0, 0), (2, 3), (4, 4)])
def test_grid_coordinates_round_trip(row, col):
    assert index_to_state(state_index(row, col, 5), 5) == (row, col)


def test_state_index_is_row_major():
    """Row-major is load-bearing: every value function here is reshaped to
    (rows, cols) for a heat map, and column-major would render a transpose."""
    assert state_index(1, 0, 5) == 5
    assert state_index(0, 1, 5) == 1


def test_state_index_rejects_a_column_outside_the_grid():
    with pytest.raises(ValueError):
        state_index(0, 5, 5)


# -- discretise / bin_centre ------------------------------------------------


def test_discretise_bins_are_uniform_and_cover_the_range():
    assert discretise(0.0, 0.0, 1.0, 10) == 0
    assert discretise(0.99, 0.0, 1.0, 10) == 9
    assert discretise(0.5, 0.0, 1.0, 10) == 5


def test_discretise_clips_rather_than_raising():
    """An out-of-range observation is a normal deployment event; a crash is a
    worse answer than the nearest bin."""
    assert discretise(-99.0, 0.0, 1.0, 10) == 0
    assert discretise(99.0, 0.0, 1.0, 10) == 9


@pytest.mark.parametrize("value", [0.02, 0.37, 0.5, 0.99])
def test_bin_centre_recovers_the_value_to_within_half_a_bin(value):
    bins, low, high = 10, 0.0, 1.0
    recovered = bin_centre(discretise(value, low, high, bins), low, high, bins)
    assert abs(recovered - value) <= 0.5 * (high - low) / bins + 1e-12


def test_bin_centre_rejects_an_out_of_range_bin():
    with pytest.raises(ValueError):
        bin_centre(10, 0.0, 1.0, 10)


def test_discretise_rejects_a_degenerate_range():
    with pytest.raises(ValueError):
        discretise(0.5, 1.0, 1.0, 10)


# -- clip_reward ------------------------------------------------------------


def test_clip_reward_bounds_both_tails():
    assert clip_reward(50.0) == 1.0
    assert clip_reward(-50.0) == -1.0
    assert clip_reward(0.3) == pytest.approx(0.3)


def test_clip_reward_destroys_the_ordering_it_was_asked_to_bound():
    """Documented behaviour, not a bug: after clipping, 5 and 50 are the same
    reward, so an agent trained on clipped rewards cannot prefer the larger."""
    assert clip_reward(5.0) == clip_reward(50.0)


# -- normalise_returns ------------------------------------------------------


def test_normalise_returns_is_zero_mean_unit_variance():
    out = normalise_returns(np.array([1.0, 2.0, 3.0, 4.0]))
    assert out.mean() == pytest.approx(0.0, abs=1e-9)
    assert out.std() == pytest.approx(1.0, abs=1e-6)


def test_normalise_returns_on_a_batch_of_one_is_zero_not_nan():
    """A batch statistic computed from one sample is meaningless; returning
    zeros beats propagating a NaN into the gradient."""
    assert normalise_returns(np.array([7.0])).tolist() == [0.0]


# -- normalise_observation --------------------------------------------------


def test_normalise_observation_uses_the_statistics_it_is_given():
    obs = np.array([2.0, 4.0])
    out = normalise_observation(obs, mean=np.array([1.0, 2.0]), std=np.array([1.0, 2.0]))
    assert out.tolist() == pytest.approx([1.0, 1.0])


# -- discounted_returns -----------------------------------------------------


def test_discounted_returns_matches_the_hand_computed_value():
    # G_2 = 3, G_1 = 2 + 0.5*3 = 3.5, G_0 = 1 + 0.5*3.5 = 2.75
    assert discounted_returns([1.0, 2.0, 3.0], 0.5).tolist() == pytest.approx([2.75, 3.5, 3.0])


def test_discounted_returns_at_gamma_one_is_the_reward_suffix_sum():
    assert discounted_returns([1.0, 1.0, 1.0], 1.0).tolist() == [3.0, 2.0, 1.0]


def test_discounted_returns_on_an_empty_episode():
    assert discounted_returns([], 0.9).size == 0


def test_discounted_returns_rejects_a_gamma_outside_the_unit_interval():
    with pytest.raises(ValueError):
        discounted_returns([1.0], 1.5)


def test_discounted_returns_does_not_underflow_on_a_long_episode():
    """The naive forward implementation accumulates gamma**t, which reaches
    zero in double precision well before this episode ends and silently
    truncates the return. The backward recursion does not."""
    g = discounted_returns([1.0] * 5000, 0.99)
    assert g[0] == pytest.approx(1.0 / (1.0 - 0.99), rel=1e-9)


# -- first_visit_indices ----------------------------------------------------


def test_first_visit_indices_keeps_only_the_earliest_occurrence():
    assert first_visit_indices([4, 1, 4, 2, 1]) == {4: 0, 1: 1, 2: 3}


def test_first_visit_indices_on_an_episode_with_no_repeats_keeps_everything():
    assert first_visit_indices([0, 1, 2]) == {0: 0, 1: 1, 2: 2}


# -- dense_model ------------------------------------------------------------


def _tiny_model():
    """Two states, one action. s0 -> s1 with reward 1; s1 terminal."""
    return {
        0: {0: [(1.0, 1, 1.0, True)]},
        1: {0: [(1.0, 1, 0.0, True)]},
    }


def test_dense_model_shapes_and_row_sums():
    T, R, B = dense_model(_tiny_model(), 2, 1)
    assert T.shape == (2, 1, 2) and R.shape == (2, 1) and B.shape == (2, 1, 2)
    assert T.sum(axis=2).tolist() == [[1.0], [1.0]]


def test_dense_model_marks_terminal_successors_as_non_bootstrappable():
    """B is what stops value leaking through a terminal transition. Fold
    `terminated` into T instead and the rows stop summing to one."""
    _, R, B = dense_model(_tiny_model(), 2, 1)
    assert R[0, 0] == 1.0
    assert B[0, 0, 1] == 0.0


def test_dense_model_averages_the_reward_over_successors():
    P = {0: {0: [(0.5, 0, 0.0, False), (0.5, 1, 4.0, True)]}, 1: {0: [(1.0, 1, 0.0, True)]}}
    _, R, _ = dense_model(P, 2, 1)
    assert R[0, 0] == pytest.approx(2.0)


def test_dense_model_rejects_a_model_whose_rows_do_not_sum_to_one():
    """The usual cause is a terminal state left out of P, and the error message
    says so — a planner that silently accepts it returns a wrong value function
    that still looks smooth on a heat map."""
    broken = {0: {0: [(0.5, 1, 1.0, True)]}, 1: {0: [(1.0, 1, 0.0, True)]}}
    with pytest.raises(ValueError, match="sum to 1"):
        dense_model(broken, 2, 1)


def test_dense_model_matches_the_environments_own_model():
    from envs import make_env

    core = make_env(time_limit=False)
    T, R, B = dense_model(core.P, core.n_states, core.n_actions)
    assert np.allclose(T.sum(axis=2), 1.0)
    # Every terminal state is absorbing with zero reward, so it must contribute
    # neither reward nor bootstrapped value.
    for s in core.terminal_states:
        assert np.allclose(R[s], 0.0)
        assert np.allclose(B[s, :, s], 0.0)
