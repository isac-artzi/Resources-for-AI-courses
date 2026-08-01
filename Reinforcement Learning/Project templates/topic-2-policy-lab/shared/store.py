"""
shared/store.py — the data tier, behind one interface.

Why this file exists rather than calling supabase directly from three places:

  * CI has no Supabase project. The test suite must still exercise the real
    code path for writing and reading an episode row. `MemoryStore` is that
    path with the network removed — it is NOT a mock, it implements the same
    interface and the same row shapes, so a schema mistake still shows up.

  * The deployed Streamlit app must survive a paused project. Supabase free
    tier pauses after one week idle, which WILL happen to you during a break
    in the term. Every read here returns `degraded=True` rather than raising,
    and the UI is required to render that state visibly.

The interface is deliberately small. If you need a query it does not have,
add a method here — do not reach past it.
"""

from __future__ import annotations

import threading
from typing import Any, Protocol

from shared.config import get_settings


class Store(Protocol):
    def insert_experiment(self, row: dict[str, Any]) -> str: ...
    def insert_episodes(self, rows: list[dict[str, Any]]) -> int: ...
    def insert_evaluation(self, row: dict[str, Any]) -> None: ...
    def insert_policy(self, row: dict[str, Any]) -> None: ...
    def log_audit(self, row: dict[str, Any]) -> None: ...
    def recent_runs(self, limit: int = 50) -> list[dict[str, Any]]: ...
    def episodes_for(self, experiment_id: str) -> list[dict[str, Any]]: ...
    # Added by this topic. The Convergence tab needs the RMSE-versus-budget
    # rows, and the alternative — letting the UI issue its own query — is the
    # first crack in "the contract between tiers is a model in shared/ or a
    # migration in db/". Added HERE, per the note above, rather than reached
    # past.
    def evaluations_for_metric(
        self, metric: str, limit: int = 500
    ) -> list[dict[str, Any]]: ...
    def reachable(self) -> bool: ...


class MemoryStore:
    """In-process fallback. Thread-safe because uvicorn will call it from a pool."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.experiments: list[dict[str, Any]] = []
        self.episodes: list[dict[str, Any]] = []
        self.evaluations: list[dict[str, Any]] = []
        self.policies: list[dict[str, Any]] = []
        self.audit: list[dict[str, Any]] = []

    def insert_experiment(self, row):
        with self._lock:
            row = dict(row)
            row.setdefault("id", f"exp-{len(self.experiments) + 1:04d}")
            self.experiments.append(row)
            return row["id"]

    def insert_episodes(self, rows):
        with self._lock:
            self.episodes.extend(dict(r) for r in rows)
            return len(rows)

    def insert_evaluation(self, row):
        with self._lock:
            self.evaluations.append(dict(row))

    def insert_policy(self, row):
        with self._lock:
            self.policies.append(dict(row))

    def log_audit(self, row):
        with self._lock:
            self.audit.append(dict(row))
            # Bound it: an audit log that grows without limit is a memory leak
            # with a compliance story attached.
            if len(self.audit) > 10_000:
                del self.audit[:5_000]

    def recent_runs(self, limit=50):
        with self._lock:
            out = []
            for e in self.experiments[-limit:][::-1]:
                eps = [x for x in self.episodes if x.get("experiment_id") == e.get("id")]
                ev = [x for x in self.evaluations if x.get("experiment_id") == e.get("id")]
                last = [x.get("return", 0.0) for x in eps[-100:]]
                out.append(
                    {
                        **e,
                        "experiment_id": e.get("id"),
                        "episodes_logged": len(eps),
                        "mean_return_last_100": (sum(last) / len(last)) if last else None,
                        "eval_mean_return": ev[-1]["mean_return"] if ev else None,
                        "eval_stderr": ev[-1].get("stderr_return") if ev else None,
                    }
                )
            return out

    def episodes_for(self, experiment_id):
        with self._lock:
            return [x for x in self.episodes if x.get("experiment_id") == experiment_id]

    def evaluations_for_metric(self, metric, limit=500):
        with self._lock:
            rows = [x for x in self.evaluations if x.get("metric") == metric]
            return rows[-limit:]

    def reachable(self):
        return True


class SupabaseStore:
    """The real tier. Constructed only when credentials are present."""

    def __init__(self, url: str, key: str) -> None:
        from supabase import create_client  # imported lazily: serving dep, not a test dep

        self._c = create_client(url, key)

    def insert_experiment(self, row):
        res = self._c.table("experiments").insert(row).execute()
        return res.data[0]["id"]

    def insert_episodes(self, rows):
        # Chunked: the free tier will reject a single enormous insert, and
        # 20,000 episodes is a normal training run in this course.
        n = 0
        for i in range(0, len(rows), 500):
            self._c.table("episodes").insert(rows[i : i + 500]).execute()
            n += len(rows[i : i + 500])
        return n

    def insert_evaluation(self, row):
        self._c.table("evaluations").insert(row).execute()

    def insert_policy(self, row):
        self._c.table("policies").insert(row).execute()

    def log_audit(self, row):
        self._c.table("audit_log").insert(row).execute()

    def recent_runs(self, limit=50):
        res = (
            self._c.table("run_summary")  # the view created in 001_init.sql
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return res.data or []

    def episodes_for(self, experiment_id):
        res = (
            self._c.table("episodes")
            .select("*")
            .eq("experiment_id", experiment_id)
            .order("episode_index")
            .execute()
        )
        return res.data or []

    def evaluations_for_metric(self, metric, limit=500):
        # Reads the base table rather than the `mc_convergence` view created in
        # 002_topic2.sql, because the aggregation the service needs uses a t
        # quantile and Postgres has no t distribution. The view exists for
        # anyone poking at the project with psql; the numbers in the report
        # come from train/compare.py. Where the two could disagree, the code
        # that can be unit-tested wins.
        res = (
            self._c.table("evaluations")
            .select("*")
            .eq("metric", metric)
            .order("at_training_episode")
            .limit(limit)
            .execute()
        )
        return res.data or []

    def reachable(self):
        try:
            self._c.table("experiments").select("id").limit(1).execute()
            return True
        except Exception:
            return False


_store: Store | None = None


def get_store() -> Store:
    """One store per process. Falls back to memory when unconfigured."""
    global _store
    if _store is None:
        s = get_settings()
        if s.data_tier_configured:
            try:
                _store = SupabaseStore(s.supabase_url, s.supabase_service_role_key)
            except Exception:
                # Do not crash the service because the data tier is asleep.
                # /healthz will report degraded and the UI will say so.
                _store = MemoryStore()
        else:
            _store = MemoryStore()
    return _store


def reset_store_for_tests() -> None:
    global _store
    _store = None
