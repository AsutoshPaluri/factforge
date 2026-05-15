"""Application settings loaded from environment (.env)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Loaded from `.env` or environment.

    Free-tier ceilings are baked in as defaults; tighten via `.env` if needed.
    """

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),  # backend/.env or repo-root .env
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Required keys ---
    gemini_api_key: str = ""             # Vision (image-input decomposition)
    groq_api_key: str = ""               # Text LLM (decompose + summarize)
    supabase_url: str = ""
    supabase_publishable_key: str = ""   # Frontend-safe (formerly "anon")
    supabase_secret_key: str = ""        # Backend-only, bypasses RLS (formerly "service_role")

    # --- Optional keys ---
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    sentry_dsn: str = ""
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""

    # --- Daily budget caps (kill-switch thresholds) ---
    daily_gemini_tokens_cap: int = 800_000  # 80% of 1M free tier
    daily_ddg_searches_cap: int = 1_000

    # --- Rate limits ---
    anon_claims_per_day: int = 2
    user_claims_per_day: int = 10

    # --- Model config ---
    nli_model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    nli_device: str = "cpu"  # cpu | mps | cuda
    gemini_model: str = "gemini-2.5-flash"
    # Default Groq model — 8b-instant has 500k TPD on the free tier
    # (vs 100k for 70b), giving us headroom for both real traffic and eval.
    # Override to 'llama-3.3-70b-versatile' via env if you want higher
    # accuracy on the decomposer (uses more daily token budget).
    groq_model: str = "llama-3.1-8b-instant"
    # Summarizer model (separate slot so we can A/B test independently).
    groq_summary_model: str = "llama-3.1-8b-instant"

    # --- App ---
    frontend_url: str = "http://localhost:3000"
    log_level: str = "INFO"


settings = Settings()
