import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    off_country: str = os.getenv("OFF_COUNTRY", "world")
    off_page_size: int = int(os.getenv("OFF_PAGE_SIZE", "20"))
    off_max_retries: int = int(os.getenv("OFF_MAX_RETRIES", "2"))
    debug_log: bool = _as_bool(os.getenv("DEBUG_LOG", "0"))
    usda_api_key: str = os.getenv("USDA_API_KEY", "")
    usda_page_size: int = int(os.getenv("USDA_PAGE_SIZE", "12"))
    meli_site_id: str = os.getenv("MELI_SITE_ID", "MPE")
    meli_fallback_sites: str = os.getenv("MELI_FALLBACK_SITES", "MLA,MLB")
    meli_access_token: str = os.getenv("MELI_ACCESS_TOKEN", "")
    meli_items_limit: int = int(os.getenv("MELI_ITEMS_LIMIT", "20"))
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "12"))
    gradio_server_name: str = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    gradio_server_port: int = int(os.getenv("GRADIO_SERVER_PORT", "7860"))


settings = Settings()
