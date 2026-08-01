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


def act(state: list[float], policy_name: str = "default", deterministic: bool = True,
        policy_source: str | None = None):
    """Ask one of the two agents for an action.

    The UI passes `policy_source` and never an artifact filename. That is the
    whole reason the field exists: the presentation tier should know that there
    is a planner and a learner, which is a product fact, and should not know
    that they happen to be stored as `value_iteration.npz` and
    `monte_carlo.npz`, which is a deployment detail that will change.
    """
    return call("/act", {"state": state, "policy_name": policy_name,
                         "deterministic": deterministic,
                         "policy_source": policy_source})


def rollout(policy_name: str = "default", episodes: int = 20, seed: int | None = 0,
            policy_source: str | None = None):
    return call("/rollout", {"policy_name": policy_name, "episodes": episodes,
                             "seed": seed, "policy_source": policy_source})


def value_map():
    """Both value functions and their difference, for the heat maps."""
    return call("/value_map", None, method="GET")


def convergence():
    """RMSE against episode budget with confidence bands."""
    return call("/convergence", None, method="GET")


def runs(limit: int = 50):
    return call(f"/runs?limit={limit}", None, method="GET")


def policies():
    return call("/policies", None, method="GET")


def healthz():
    return call("/healthz", None, method="GET")
