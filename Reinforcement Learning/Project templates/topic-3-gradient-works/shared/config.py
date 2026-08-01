"""
shared/config.py — every secret and every switch, read from the environment.

The one rule that matters: the SERVICE-ROLE key is read here but is only ever
used by the service tier (api/). The Streamlit tier uses the ANON key and only
for read-only views. If you find yourself wanting the service-role key in
ui/app.py, the query you are writing belongs behind an endpoint instead.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    supabase_url: str = Field(default="", alias="SUPABASE_URL")
    supabase_anon_key: str = Field(default="", alias="SUPABASE_ANON_KEY")
    supabase_service_role_key: str = Field(default="", alias="SUPABASE_SERVICE_ROLE_KEY")

    service_mode: str = Field(default="inprocess", alias="SERVICE_MODE")
    service_url: str = Field(default="http://127.0.0.1:8000", alias="SERVICE_URL")

    policy_dir: str = Field(default="policies", alias="POLICY_DIR")

    # Which artifact `policy_name="default"` resolves to. This topic ships two
    # `.npz` files (the base template's smoke-test table and the network you
    # trained), so "the only one" is no longer a rule that can pick. The default
    # value below is the name train/train.py writes, which means a fresh fork
    # resolves correctly with no .env at all; point it somewhere else when you
    # promote a different run's artifact.
    default_policy: str = Field(default="vpg_cartpole", alias="DEFAULT_POLICY")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")
    git_sha: str = Field(default=os.environ.get("GIT_SHA", "local"), alias="GIT_SHA")

    @property
    def data_tier_configured(self) -> bool:
        """False in CI and on a fresh clone — the store falls back to memory."""
        return bool(self.supabase_url and self.supabase_service_role_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
