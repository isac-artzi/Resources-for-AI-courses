"""
The required node-budget test: the budget is ENFORCED and it is REPORTED.

Why this is a required test rather than a nice one. Every other product in this
course serves a fixed-cost forward pass — an observation in, a few matrix
multiplies, an action out — and the work is the same whatever the observation
was. A tree search is not like that: the same 43-number board with `depth: 9`
instead of `depth: 4` is roughly two thousand times the work. An unbounded
search endpoint is therefore a denial-of-service vector against your own
free-tier instance, reachable by anyone who can type a number into a JSON body,
including your own Streamlit app with a slider on it.

Four properties, and each one fails differently:

  1. the search never expands more nodes than the budget      (the bound holds)
  2. the response says how many it actually expanded          (it is observable)
  3. an exhausted budget still returns a LEGAL move           (it degrades)
  4. a budget above the ceiling is a 422 before any work      (it is cheap)

Property 3 is the one people leave out, and it is the one a human notices: a
500 in the middle of their game is a worse outcome than a slightly weaker move.
"""

from __future__ import annotations

import pytest

from envs.connect_four import Position, encode_state
from shared.schemas import MAX_NODE_BUDGET


def midgame_state() -> list[float]:
    """A position with a real tree under it, so a small budget actually binds."""
    position = Position()
    for col in [3, 3, 2, 4, 4, 2, 5, 1]:
        position.push(col)
    return encode_state(position)


@pytest.mark.parametrize("budget", [1, 5, 50, 500, 5_000])
def test_the_search_never_exceeds_its_node_budget(client, budget):
    r = client.post(
        "/act",
        json={
            "state": midgame_state(),
            "agent": "exhaustive",
            "depth": 8,                 # far more than the budget can fund
            "node_budget": budget,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["nodes_expanded"] <= budget, (
        f"expanded {body['nodes_expanded']} nodes under a budget of {budget}. "
        "The counter is charged before descending in search/minimax.py; an "
        "overshoot of one node per root child means the root loop stopped "
        "checking the return value of `spend`."
    )
    assert body["node_budget"] == budget


@pytest.mark.parametrize("budget", [1, 5, 50, 500])
def test_an_exhausted_budget_is_reported_and_still_returns_a_legal_move(client, budget):
    body = client.post(
        "/act",
        json={"state": midgame_state(), "agent": "exhaustive", "depth": 8,
              "node_budget": budget},
    ).json()
    assert body["budget_exhausted"] is True, (
        "a search that could not finish must SAY so. An agent that is silently "
        "weaker under load is worse than one that admits it."
    )
    assert body["action"] in body["legal_moves"]


def test_a_generous_budget_completes_without_truncation(client):
    body = client.post(
        "/act",
        json={"state": midgame_state(), "agent": "heuristic", "depth": 4,
              "node_budget": 200_000},
    ).json()
    assert body["budget_exhausted"] is False
    assert 0 < body["nodes_expanded"] < 200_000
    assert body["search_depth"] >= 1


def test_the_budget_also_binds_mcts(client):
    """MCTS counts nodes differently from alpha-beta — one per selection step
    plus one per expansion — and the budget has to cover both or a 5,000-
    simulation request walks straight past it."""
    body = client.post(
        "/act",
        json={"state": midgame_state(), "agent": "mcts", "node_budget": 40},
    ).json()
    assert body["nodes_expanded"] <= 40
    assert body["action"] in body["legal_moves"]


def test_a_budget_above_the_ceiling_is_rejected_before_any_work(client):
    """422 from the validator, not a five-minute request that eventually 500s."""
    r = client.post(
        "/act",
        json={"state": midgame_state(), "agent": "exhaustive",
              "node_budget": MAX_NODE_BUDGET + 1},
    )
    assert r.status_code == 422
    assert "node_budget" in r.text


def test_a_depth_above_the_ceiling_is_rejected(client):
    r = client.post(
        "/act",
        json={"state": midgame_state(), "agent": "exhaustive", "depth": 99},
    )
    assert r.status_code == 422


def test_more_budget_buys_a_deeper_search(client):
    """The budget must actually govern the work, not merely be echoed back.

    A response that reports the budget while ignoring it would pass every test
    above except this one.
    """
    small = client.post(
        "/act",
        json={"state": midgame_state(), "agent": "exhaustive", "depth": 8,
              "node_budget": 100},
    ).json()
    large = client.post(
        "/act",
        json={"state": midgame_state(), "agent": "exhaustive", "depth": 8,
              "node_budget": 20_000},
    ).json()
    assert large["nodes_expanded"] > small["nodes_expanded"]
    assert small["nodes_expanded"] <= 100 and large["nodes_expanded"] <= 20_000
    # NOTE what is deliberately NOT asserted: that the larger budget reached a
    # greater DEPTH. It does not, and expecting it to is the natural mistake.
    # Depth-first search descends the leftmost line to the horizon before it
    # expands anything else, so a budget of 100 already touches ply 8 — it just
    # touches almost nothing else. Depth reached is a poor proxy for work done;
    # the node count is the honest one, which is why the budget counts nodes.
    assert small["search_depth"] == large["search_depth"] == 8


def test_the_budget_does_not_leak_between_requests(client):
    """The registry is process-global; a per-request override must not stick.

    `_scoped_agent` copies before mutating precisely because of this. A leak
    here only shows up under concurrency — which is to say, in production, on
    the day of the demonstration.
    """
    client.post(
        "/act",
        json={"state": midgame_state(), "agent": "exhaustive", "depth": 8,
              "node_budget": 10},
    )
    body = client.post(
        "/act",
        json={"state": midgame_state(), "agent": "exhaustive", "depth": 4},
    ).json()
    assert body["budget_exhausted"] is False
    assert body["node_budget"] == 200_000
