"""
The serving path must not import a framework — including the ones that ARE
allowed to be installed.

`tests/test_no_torch.py` guards the 490 MB case. This file guards the subtler
one: gymnasium IS in `requirements-serve.txt` (POST /rollout runs episodes
server-side and needs an environment), so the no-torch guard cannot notice a
search agent that quietly grew a gymnasium dependency. The claim in
`requirements-serve.txt`'s header — "the search agents need nothing here beyond
NumPy" — is only worth writing down if something checks it.

It is checked in a SUBPROCESS with a clean interpreter, because `sys.modules` is
per-process and by the time this file runs some other test has already called
`make_env()`. Asserting on this process would measure the test suite's import
history rather than the application's import graph.

This one caught a real defect during development: `envs/connect_four.py` had a
`try: import gymnasium` at module scope, wrapped in a `except ImportError` that
made it look optional. A defensive import still imports.
"""

from __future__ import annotations

import subprocess
import sys

PROBE = """
import sys, json
import api.main                      # the whole service tier
import search.agents, search.net     # every agent, and the learned evaluator
print(json.dumps({
    "torch": "torch" in sys.modules,
    "gymnasium": any(m == "gymnasium" or m.startswith("gymnasium.")
                     for m in sys.modules),
    "pandas": "pandas" in sys.modules,
    "modules": len(sys.modules),
}))
"""


def _probe() -> dict:
    import json

    proc = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True
    )
    assert proc.returncode == 0, f"the probe failed:\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_importing_the_service_does_not_load_a_training_framework():
    result = _probe()
    assert result["torch"] is False


def test_importing_the_service_does_not_load_gymnasium():
    """The search path is NumPy only, as requirements-serve.txt claims.

    If this fails, find the import with:

        python -X importtime -c "import api.main" 2>&1 | grep -i gymnasium

    The fix is to move the import inside the function that constructs the
    environment — `envs/__init__.py::make_env` is the model.
    """
    result = _probe()
    assert result["gymnasium"] is False, (
        "importing api.main pulled in gymnasium. The Gymnasium wrapper belongs "
        "in envs/gym_env.py, imported lazily by make_env(); envs/connect_four.py "
        "must stay pure."
    )


def test_make_env_still_works_and_is_the_thing_that_pays_for_gymnasium():
    """The lazy import must not be a broken import.

    A guard that is satisfied because the module no longer works is worse than
    no guard, so this asserts the environment is still constructible and still
    honours the alternating-move contract.
    """
    from envs import make_env

    env = make_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == (43,)
    assert info["action_mask"].sum() == 7
    obs, reward, terminated, truncated, info = env.step(3)
    assert reward == 0.0 and not terminated and not truncated
    # The reward is from the point of view of the player who JUST MOVED, and
    # the observation now belongs to the other player. That sign convention is
    # the thing people get wrong; pinning it here means a change to it fails
    # loudly rather than producing a learner that trains towards losing.
    assert info["player"] == -1
