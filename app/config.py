from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# override=True: the project's .env wins over variables already set in the shell. A stale
# LLM_BASE_URL left in a terminal (for example from copying an example command) otherwise
# silently replaces the real value. Hosts without a .env file are unaffected.
load_dotenv(PROJECT_ROOT / ".env", override=True)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: str) -> Path:
    raw = os.getenv(name, "").strip() or default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True, slots=True)
class Settings:
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "").strip())
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "").strip())
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "").strip())
    llm_timeout_seconds: float = field(
        default_factory=lambda: _env_float("LLM_TIMEOUT_SECONDS", 60.0)
    )
    llm_json_mode: bool = field(default_factory=lambda: _env_bool("LLM_JSON_MODE", False))
    patient_max_tokens: int = field(
        default_factory=lambda: _env_int("LLM_PATIENT_MAX_TOKENS", 180)
    )
    evaluator_max_tokens: int = field(
        default_factory=lambda: _env_int("LLM_EVALUATOR_MAX_TOKENS", 2000)
    )
    max_student_turns: int = 12
    app_api_key: str = field(default_factory=lambda: os.getenv("APP_API_KEY", "").strip())
    session_ttl_minutes: int = field(
        default_factory=lambda: _env_int("SESSION_TTL_MINUTES", 120)
    )
    max_active_sessions: int = field(
        default_factory=lambda: _env_int("MAX_ACTIVE_SESSIONS", 200)
    )
    scenario_dir: Path = PROJECT_ROOT / "scenarios"
    prompt_dir: Path = PROJECT_ROOT / "prompts"
    static_dir: Path = PROJECT_ROOT / "app" / "static"
    guidelines_path: Path = field(default_factory=lambda: _env_path("GUIDELINES_PATH", "guidelines.json"))
    log_dir: Path = PROJECT_ROOT / "logs" / "transcripts"
    whatsapp_verify_token: str = field(
        default_factory=lambda: os.getenv("WHATSAPP_VERIFY_TOKEN", "").strip()
    )
    whatsapp_access_token: str = field(
        default_factory=lambda: os.getenv("WHATSAPP_ACCESS_TOKEN", "").strip()
    )
    whatsapp_phone_number_id: str = field(
        default_factory=lambda: os.getenv("WHATSAPP_PHONE_NUMBER_ID", "").strip()
    )
    whatsapp_app_secret: str = field(
        default_factory=lambda: os.getenv("WHATSAPP_APP_SECRET", "").strip()
    )
    whatsapp_api_version: str = field(
        default_factory=lambda: os.getenv("WHATSAPP_API_VERSION", "v23.0").strip() or "v23.0"
    )
    whatsapp_timeout_seconds: float = field(
        default_factory=lambda: _env_float("WHATSAPP_TIMEOUT_SECONDS", 30.0)
    )

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_base_url and self.llm_api_key and self.llm_model)

    def llm_config_issues(self) -> list[str]:
        """Human-readable configuration problems, without ever exposing the API key."""
        issues: list[str] = []
        if not self.llm_base_url:
            issues.append("LLM_BASE_URL is not set.")
        elif not self.llm_base_url.lower().startswith(("http://", "https://")):
            issues.append("LLM_BASE_URL must start with http:// or https://.")
        if not self.llm_api_key:
            issues.append("LLM_API_KEY is not set.")
        elif self.llm_api_key.lower() in {"replace-me", "changeme", "your-api-key"}:
            issues.append("LLM_API_KEY still contains the example placeholder value.")
        if not self.llm_model:
            issues.append("LLM_MODEL is not set.")
        return issues

    @property
    def whatsapp_configured(self) -> bool:
        # The app secret is required: without it, forged webhook payloads would be
        # accepted and replies sent to any "from" number.
        return bool(
            self.whatsapp_verify_token
            and self.whatsapp_access_token
            and self.whatsapp_phone_number_id
            and self.whatsapp_app_secret
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
