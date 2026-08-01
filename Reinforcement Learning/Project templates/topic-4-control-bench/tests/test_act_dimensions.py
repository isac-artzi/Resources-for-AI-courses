"""
The dimensionality-mismatch test, which in THIS product is not a contrived case.

Three agents share one `/act` endpoint. CartPole's observation is 4 numbers,
Acrobot's is 6 and Pendulum's is 3. Every one of those vectors is a perfectly
well-formed `ActRequest` — Pydantic cannot reject any of them, because it does
not know which artifact `policy_name` will resolve to. So the only thing
standing between a caller and a 500 is the explicit check in `api/main.act`, and
the only thing standing between that check and quietly rotting is this file.

The required contract: **422 with both numbers in the message, never a stack
trace.** A caller who receives a 500 cannot tell whether to retry, to reshape,
or to give up. A caller who receives "observation has dimension 4, but policy
'ppo_acrobot' expects 6 (trained on Acrobot-v1)" fixes it in ten seconds without
opening your source.

Everything here goes through the HTTP test client. Half of the contract — status
codes, validation, serialisation of a NumPy integer into JSON — exists only at
that boundary, and calling the handler function directly tests the half that was
never in doubt.
"""

from __future__ import annotations

import pytest

# One plausible observation per environment, written out with names rather than
# as bare lists. A reader has no way to know which of six numbers is Acrobot's
# second joint velocity, and a test whose fixtures cannot be read is a test
# nobody will maintain.
CARTPOLE_STATE = [0.01, 0.0, 0.03, 0.0]              # cart pos, cart vel, angle, ang vel
ACROBOT_STATE = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]       # cos θ1, sin θ1, cos θ2, sin θ2, ω1, ω2
PENDULUM_STATE = [1.0, 0.0, 0.0]                     # cos θ, sin θ, ω

STATES = {
    "a2c_cartpole": CARTPOLE_STATE,
    "ppo_acrobot": ACROBOT_STATE,
    "sac_pendulum": PENDULUM_STATE,
}


@pytest.mark.parametrize("policy_name", sorted(STATES))
def test_each_policy_accepts_its_own_observation(client, policy_name):
    """The control case. Without it, a service that 422s everything would pass."""
    r = client.post("/act", json={"state": STATES[policy_name], "policy_name": policy_name})
    assert r.status_code == 200, r.text


@pytest.mark.parametrize(
    ("policy_name", "wrong_state", "wrong_dim", "right_dim"),
    [
        # Every off-diagonal pair of the three agents. Each row is a mistake a
        # caller genuinely makes: they had one policy working, switched the
        # policy_name, and forgot that the observation changed shape too.
        ("a2c_cartpole", ACROBOT_STATE, 6, 4),
        ("a2c_cartpole", PENDULUM_STATE, 3, 4),
        ("ppo_acrobot", CARTPOLE_STATE, 4, 6),
        ("ppo_acrobot", PENDULUM_STATE, 3, 6),
        ("sac_pendulum", CARTPOLE_STATE, 4, 3),
        ("sac_pendulum", ACROBOT_STATE, 6, 3),
    ],
)
def test_a_mismatched_observation_is_a_readable_422_not_a_stack_trace(
    client, policy_name, wrong_state, wrong_dim, right_dim
):
    r = client.post("/act", json={"state": wrong_state, "policy_name": policy_name})

    assert r.status_code == 422, (
        f"expected 422, got {r.status_code}. A 500 here means the request reached "
        f"the matmul and NumPy raised a shape error deep inside the forward pass. "
        f"The check belongs in api/main.act, before the policy is called.\n{r.text}"
    )
    detail = str(r.json()["detail"])
    assert str(wrong_dim) in detail and str(right_dim) in detail, (
        f"the 422 must name BOTH dimensions so the caller can fix it without "
        f"reading your source: {detail!r}"
    )
    assert "Traceback" not in detail and "numpy" not in detail.lower(), (
        f"the message leaks an implementation detail: {detail!r}"
    )


def test_the_error_names_the_environment_too(client):
    """Not required, and worth doing anyway.

    "expects 6" tells the caller the shape. "trained on Acrobot-v1" tells them
    WHY, which is the difference between fixing the call and fixing the
    misunderstanding behind it.
    """
    r = client.post("/act", json={"state": CARTPOLE_STATE, "policy_name": "ppo_acrobot"})
    assert "Acrobot-v1" in str(r.json()["detail"])


def test_an_unknown_policy_is_404_and_lists_what_exists(client):
    """A different failure from a mismatch, and it must be a different status.

    404 for "no such policy", 422 for "wrong shape for a policy that exists".
    Collapsing the two into one code would leave the caller unable to tell a
    typo from a shape bug.
    """
    r = client.post("/act", json={"state": CARTPOLE_STATE, "policy_name": "a2c_cartpol"})
    assert r.status_code == 404
    detail = str(r.json()["detail"])
    assert "a2c_cartpole" in detail, "a 404 should list the names that do exist"


def test_the_continuous_policy_returns_a_vector_and_the_discrete_ones_an_int(client):
    """One contract, two action types, and the caller must be able to tell which.

    `GET /policies` publishes `action_space` for exactly this reason. A caller
    that assumes `action` is an int will crash on the Pendulum agent, and the
    only thing that stops that is the field being there and being correct.
    """
    spaces = {p["name"]: p["action_space"] for p in client.get("/policies").json()["policies"]}
    assert spaces["a2c_cartpole"] == "discrete"
    assert spaces["sac_pendulum"] == "continuous"

    discrete = client.post(
        "/act", json={"state": CARTPOLE_STATE, "policy_name": "a2c_cartpole"}
    ).json()
    assert isinstance(discrete["action"], int), (
        "a numpy.int64 does not serialise to JSON — cast at the boundary, in "
        "api/policy.py, not in the caller"
    )

    continuous = client.post(
        "/act", json={"state": PENDULUM_STATE, "policy_name": "sac_pendulum"}
    ).json()
    assert isinstance(continuous["action"], list) and len(continuous["action"]) == 1
    assert abs(continuous["action"][0]) <= 2.0 + 1e-9, (
        "a tanh-squashed action scaled by 2 cannot leave [-2, 2]; if it did, "
        "either the squash or the scale is not being applied at serving time"
    )


def test_rollout_uses_the_environment_named_in_the_artifact(client):
    """The caller picks the policy; the policy picks the world.

    If `/rollout` took an env_id from the request it would be possible to
    evaluate the Pendulum actor on CartPole, which fails deep inside Gymnasium
    with a shape error rather than at the boundary.
    """
    r = client.post(
        "/rollout",
        json={"policy_name": "sac_pendulum", "episodes": 2, "max_steps": 50, "seed": 0},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["env_id"] == "Pendulum-v1"
    assert body["random_baseline"] is not None, (
        "a return is not interpretable without its floor; /rollout returns the "
        "random baseline alongside the result so a reader cannot forget it"
    )
    assert len(body["trajectory"][0]["state"]) == 3
