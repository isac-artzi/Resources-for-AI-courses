"""
ui/service.py — the single switch between the two ways of reaching the service tier.

SERVICE_MODE=inprocess  -> import the FastAPI handlers and call them directly.
                           This is what runs on Streamlit Community Cloud: one
                           process, one host, no network hop, no cold start.
SERVICE_MODE=http       -> call SERVICE_URL over HTTP. This is what you run
                           locally under uvicorn, and what you screen-capture
                           for submission.

The contract is identical in both directions because both go through the same
Pydantic models. That is the property that makes the "lift the service onto
its own host" exercise at the 300 level a small change rather than a rewrite.

This module contains NO policy code and NO training code — it is a client.
"""

from __future__ import annotations

from typing import Any

from shared.config import get_settings

_settings = get_settings()


class ServiceError(RuntimeError):
    """Raised with a message the UI can show a stakeholder verbatim."""


def _inprocess(path: str, payload: dict[str, Any] | None, method: str) -> dict[str, Any]:
    # Imported lazily so that a UI running in http mode never touches the
    # policy artifacts at all.
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app, raise_server_exceptions=False)
    r = client.request(method, path, json=payload)
    if r.status_code >= 400:
        raise ServiceError(_readable(r.status_code, r.json()))
    return r.json()


def _http(path: str, payload: dict[str, Any] | None, method: str) -> dict[str, Any]:
    import httpx

    url = _settings.service_url.rstrip("/") + path
    try:
        r = httpx.request(method, url, json=payload, timeout=30.0)
    except Exception as exc:
        raise ServiceError(
            f"could not reach the service tier at {url}. Is uvicorn running? ({exc})"
        ) from exc
    if r.status_code >= 400:
        raise ServiceError(_readable(r.status_code, r.json()))
    return r.json()


def call(path: str, payload: dict[str, Any] | None = None, method: str = "POST") -> dict[str, Any]:
    if _settings.service_mode == "http":
        return _http(path, payload, method)
    return _inprocess(path, payload, method)


def _readable(status: int, body: Any) -> str:
    detail = body.get("detail", body) if isinstance(body, dict) else body
    if status == 422:
        return f"The service rejected the request as invalid (422): {detail}"
    if status == 404:
        return f"Not found (404): {detail}"
    if status == 503:
        return (
            "The service is running but a dependency is not. This usually means "
            f"the database project is paused. ({detail})"
        )
    return f"Service error {status}: {detail}"


# -- convenience wrappers used by ui/app.py ---------------------------------


def act(state: list[float], policy_name: str = "default", deterministic: bool = True):
    return call("/act", {"state": state, "policy_name": policy_name,
                         "deterministic": deterministic})


def rollout(policy_name: str = "default", episodes: int = 20, seed: int | None = 0):
    return call("/rollout", {"policy_name": policy_name, "episodes": episodes, "seed": seed})


def runs(limit: int = 50):
    return call(f"/runs?limit={limit}", None, method="GET")


def policies():
    return call("/policies", None, method="GET")


def healthz():
    return call("/healthz", None, method="GET")


# -- Topic 5: the search endpoints ------------------------------------------
#
# Thin wrappers, deliberately. Every one of them is a single `call()` with the
# argument names spelled out, so that the Streamlit tier never constructs a
# JSON body by hand. The moment a tab starts assembling its own payload, the
# contract in shared/schemas.py stops being the single source of truth and a
# renamed field becomes a runtime KeyError in front of a stakeholder instead of
# an import error in CI.


def agents():
    return call("/agents", None, method="GET")


def act_agent(state: list[float], agent: str, node_budget: int = 200_000,
              depth: int | None = None):
    """One move from a named search agent. `state` is the 43-float board."""
    payload = {"state": state, "agent": agent, "node_budget": node_budget}
    if depth is not None:
        payload["depth"] = depth
    return call("/act", payload)


def game(agent_a: str, agent_b: str, node_budget: int = 200_000,
         seed: int | None = 0, log_to_store: bool = True):
    return call("/game", {"agent_a": agent_a, "agent_b": agent_b,
                          "node_budget": node_budget, "seed": seed,
                          "log_to_store": log_to_store})


def tournament():
    return call("/tournament", None, method="GET")


def scalability():
    return call("/scalability", None, method="GET")
