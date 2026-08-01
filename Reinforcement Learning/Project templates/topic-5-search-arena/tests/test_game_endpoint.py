"""
`POST /game`, `GET /agents`, and the `/act` search route — the contract the
Streamlit "Play" and "Tournament" tabs are built on.

Everything here goes through the HTTP test client rather than calling the
handler functions, because half the contract — status codes, validation,
serialisation — only exists at the HTTP boundary. A handler that returns a
`Position` object works fine in a unit test and 500s on `json.dumps`.
"""

from __future__ import annotations

from envs.connect_four import COLS, ROWS, Position, encode_state


def test_agents_endpoint_lists_the_playable_agents(client):
    body = client.get("/agents").json()
    names = {a["name"] for a in body["agents"]}
    assert {"random", "exhaustive", "heuristic", "mcts", "mcts_v2"} <= names
    assert body["count"] == len(body["agents"])
    for a in body["agents"]:
        assert a["description"], f"agent '{a['name']}' has no description for the UI"
        assert len(a["config_sha256"]) == 64


def test_act_with_an_agent_returns_a_legal_column_and_its_telemetry(client):
    body = client.post(
        "/act", json={"state": encode_state(Position()), "agent": "heuristic"}
    ).json()
    assert body["action"] in range(COLS)
    assert body["agent"] == "heuristic"
    assert body["nodes_expanded"] > 0
    assert body["search_depth"] >= 1
    assert body["legal_moves"] == list(range(COLS))
    # The identity of a search agent is its CONFIGURATION, since it has no
    # weights. Still 64 hex characters, so the audit log's column does not have
    # to know which kind of agent produced the row.
    assert len(body["policy_sha256"]) == 64


def test_act_on_an_unknown_agent_is_404_and_lists_what_exists(client):
    r = client.post(
        "/act", json={"state": encode_state(Position()), "agent": "alphabeta_deluxe"}
    )
    assert r.status_code == 404
    assert "heuristic" in str(r.json()["detail"])


def test_act_rejects_a_malformed_board_with_a_readable_422(client):
    r = client.post("/act", json={"state": [0.0] * 20, "agent": "heuristic"})
    assert r.status_code == 422
    detail = str(r.json()["detail"])
    assert "43" in detail, f"the 422 must say what shape it wanted: {detail}"


def test_act_refuses_to_move_in_a_finished_game(client):
    """A 422, not a 500 out of `search_root`. The UI calls this every time a
    human clicks, including the click after the game ended."""
    position = Position()
    for col in [3, 0, 4, 1, 5, 2, 6]:
        position.push(col)
    assert position.is_terminal()
    r = client.post("/act", json={"state": encode_state(position), "agent": "mcts"})
    assert r.status_code == 422
    assert "already over" in str(r.json()["detail"])


def test_game_plays_a_complete_bounded_game(client):
    body = client.post(
        "/game",
        json={"agent_a": "heuristic", "agent_b": "random", "seed": 5,
              "log_to_store": False},
    ).json()
    assert body["result"] in ("agent_a", "agent_b", "draw")
    assert 0 < body["moves"] <= ROWS * COLS
    assert len(body["move_records"]) == body["moves"]
    assert len(body["final_board"]) == ROWS and len(body["final_board"][0]) == COLS
    # Per-agent costs, reported separately. A combined total would hide exactly
    # the difference the tournament exists to measure.
    assert set(body["nodes_expanded"]) == {"heuristic", "random"}
    assert body["nodes_expanded"]["heuristic"] > body["nodes_expanded"]["random"]
    assert body["node_budget"] == 200_000


def test_game_respects_the_per_decision_node_budget(client):
    body = client.post(
        "/game",
        json={"agent_a": "exhaustive", "agent_b": "random", "seed": 1,
              "node_budget": 60, "log_to_store": False},
    ).json()
    per_move = [m for m in body["move_records"] if m["agent"] == "exhaustive"]
    assert per_move, "the searching agent made no moves"
    assert all(m["nodes_expanded"] <= 60 for m in per_move)
    assert body["budget_exhausted_moves"] >= 1, (
        "a 60-node budget on a depth-4 search must truncate, and the response "
        "has to say so — a tournament of truncated agents is a different "
        "experiment from the one you meant to run"
    )


def test_game_is_reproducible_from_its_seed(client):
    """Two calls with the same seed must produce the same game.

    Without the reseeding in `/game`, the process-global registry's RNG state
    makes the Nth call depend on the N-1 before it, and a tournament table
    stops being reproducible from the numbers printed next to it.
    """
    payload = {"agent_a": "mcts", "agent_b": "random", "seed": 42,
               "log_to_store": False}
    first = client.post("/game", json=payload).json()
    second = client.post("/game", json=payload).json()
    assert [m["column"] for m in first["move_records"]] == \
           [m["column"] for m in second["move_records"]]
    assert first["result"] == second["result"]


def test_game_on_an_unknown_agent_is_404(client):
    r = client.post("/game", json={"agent_a": "heuristic", "agent_b": "nope"})
    assert r.status_code == 404


def test_game_logs_a_row_the_tournament_view_can_read(client):
    from shared.store import get_store

    store = get_store()
    before = len(store.recent_games(limit=10_000))
    body = client.post(
        "/game",
        json={"agent_a": "heuristic", "agent_b": "random", "seed": 9},
    ).json()
    assert body["logged"] is True
    rows = store.recent_games(limit=10_000)
    assert len(rows) == before + 1
    row = next(r for r in rows if r["game_id"] == body["game_id"])
    # `result` is from `agent`'s point of view — always. The API's own
    # vocabulary ("agent_a") must not leak into the column, or the SQL view's
    # check constraint rejects the insert against a real project.
    assert row["result"] in ("win", "loss", "draw")
    assert row["agent"] == "heuristic" and row["opponent"] == "random"
    assert row["agent_played_first"] is True
    # Both halves of the cost pair, so that a /game row and a train/benchmark.py
    # row are indistinguishable to a query. The win-rate matrix view reads each
    # game from both directions and needs the opponent's columns to do it.
    assert row["nodes_expanded"] > row["opponent_nodes_expanded"]
    assert row["opponent_search_depth"] == 0    # the random agent does not search


def test_tournament_and_scalability_endpoints_answer_and_declare_their_source(client):
    for path in ("/tournament", "/scalability"):
        body = client.get(path).json()
        assert body["source"] in ("data-tier", "checked-in-report", "none")
        # `degraded` must be true whenever the numbers did not come from a live
        # query, so the UI can never present a checked-in file as current.
        assert body["degraded"] == (body["source"] != "data-tier")


def test_rollout_with_an_incompatible_artifact_is_a_readable_422(client):
    """A shape mismatch between the environment and an artifact must be a 422.

    This is a real case in this topic, not a hypothetical: the environment emits
    the 43-float board encoding while `alphazero_c4.npz` consumes the 84-float
    canonical planes. It was an unhandled 500 until someone tried it, which is
    the usual life cycle of an error path nobody wrote a test for.

    The 422 also has to point at the endpoint the caller probably meant. "The
    shapes do not match" tells them what went wrong; "/game with
    agent_a='alphazero'" tells them what to do instead.
    """
    r = client.post(
        "/rollout", json={"policy_name": "alphazero_c4", "episodes": 1, "seed": 0}
    )
    assert r.status_code == 422, r.text
    detail = str(r.json()["detail"])
    assert "84" in detail and "43" in detail, f"name both dimensions: {detail}"
    assert "/game" in detail
