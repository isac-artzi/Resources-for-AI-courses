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


# -- Topic 6 ----------------------------------------------------------------
# Note what these are NOT: there is no `generate()`. The Streamlit tier cannot
# ask the service to produce text because the service cannot produce text. It
# reads completions the training tier generated offline. See the architecture
# note in the README — this absence is the design.


def score(text: str, policy_name: str = "reward_tfidf"):
    return call("/score", {"text": text, "policy_name": policy_name})


def compare(text_a: str, text_b: str, policy_name: str = "reward_tfidf"):
    return call("/compare", {"text_a": text_a, "text_b": text_b, "policy_name": policy_name})


def completions(prompt_id: str | None = None, limit: int = 200):
    # Query string built here rather than passed as a payload: this is a GET,
    # and a GET with a JSON body is legal, widely mishandled, and uncacheable.
    q = f"/completions?limit={limit}" + (f"&prompt_id={prompt_id}" if prompt_id else "")
    return call(q, None, method="GET")


def alignment_runs(limit: int = 50):
    return call(f"/alignment_runs?limit={limit}", None, method="GET")
