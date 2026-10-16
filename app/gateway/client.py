from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, OpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

from app.config import settings

# Portkey routing strategy:
#   - Primary/fallback logic lives in a Portkey saved config (required when
#     block_inline_config is enabled on the workspace).
#   - We reference that config via the x-portkey-config-id header.
#   - The inline config dict approach is disabled for this account, so all
#     retry/fallback/cache behavior must be configured inside the Portkey UI.


def _make_headers(feature: str = "rag") -> dict:
    """Build Portkey headers that reference the primary saved config by ID."""
    # If a saved Portkey config ID is not provided (local dev), return
    # an empty header set so the application can start without external
    # configuration. Users should populate `PORTKEY_PRIMARY_CONFIG_ID`
    # for production usage.
    if not settings.PORTKEY_PRIMARY_CONFIG_ID:
        return {}

    return createHeaders(
        api_key=settings.PORTKEY_API_KEY,
        config_id=settings.PORTKEY_PRIMARY_CONFIG_ID,
        metadata={
            "feature": feature,
            "_user": "rag-system",
            "environment": "production",
        },
    )


# OpenAI-compatible client routed through Portkey.
# We use the OpenAI SDK directly because the native Portkey SDK does not
# surface a first-class config_id constructor parameter; the header-based
# approach works reliably with block_inline_config enabled.
if settings.PORTKEY_API_KEY:
    portkey_client = OpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        default_headers=_make_headers(),
    )
else:
    # Local dev: do not instantiate OpenAI client when no credentials are provided.
    portkey_client = None


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI - a drop-in for LangChain nodes.

    Why ChatOpenAI:
      Portkey is a proxy. It exposes an OpenAI-compatible endpoint at PORTKEY_GATEWAY_URL.
      ChatOpenAI supports base_url (points at Portkey) and default_headers (passes Portkey
      auth + saved-config reference). The @slug/model-name format is Portkey-specific - the
      upstream provider's own client does not understand it. Portkey is just in the middle.
    """
    # If DeepSeek is configured, prefer it for direct LLM calls.
    if settings.DEEPSEEK_API_KEY:
        return ChatOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL or "https://api.deepseek.com",
            model=settings.DEEPSEEK_MODEL,
        )

    # Default: route through Portkey gateway (OpenAI-compatible)
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=f"@{settings.PORTKEY_PRIMARY_SLUG}/gpt-5-mini",
        default_headers=_make_headers(feature),
    )


def get_async_openai_client(feature: str = "rag") -> AsyncOpenAI:
    """
    Returns an async OpenAI client that routes through the Portkey gateway.
    Use this for non-LangChain async LLM calls (e.g. async FastAPI endpoints).
    """
    return AsyncOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        default_headers=_make_headers(feature),
    )


def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the response.

    The OpenAI SDK does not expose raw headers on parsed responses, so cache
    hit/miss tracking is best-effort. We inspect common attribute paths and
    fall back to 'MISS'.
    """
    for attr in ("_raw_response", "_response", "_http_response", "headers"):
        raw = getattr(response, attr, None)
        if raw is not None:
            headers = getattr(raw, "headers", None)
            if headers is not None:
                status = headers.get("x-portkey-cache-status", "")
                if status:
                    return status.upper()
    return "MISS"
