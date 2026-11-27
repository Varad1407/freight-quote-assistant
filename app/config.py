"""Settings shared by the API, the mailbox worker and scripts.

- load_dotenv(): reads KEY=VALUE lines from .env into the environment (no extra library needed).
  Values already set in the environment win, so `OLLAMA_MODEL=qwen2.5:7b python ...` still works.
- build_llm(): picks the LLM provider from LLM_PROVIDER.
"""

import os
from functools import lru_cache
from pathlib import Path

from app.extractor import (
    AnthropicLLM,
    GroqLLM,
    LLMClient,
    LLMUnavailableError,
    OllamaLLM,
)

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_OLLAMA_MODEL = "qwen2.5:14b"  # 30/30 dev, 12/12 held-out (v0.9); 7b is faster but less accurate
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Load simple KEY=VALUE lines. Ignores blank lines and # comments. Never overrides existing variables."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@lru_cache
def build_llm() -> LLMClient:
    """Pick the LLM provider from LLM_PROVIDER: 'ollama' (free, local, default), 'groq' (free, hosted) or 'anthropic'."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "ollama":
        return OllamaLLM(
            model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        )
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMUnavailableError("ANTHROPIC_API_KEY is not set")
        return AnthropicLLM(api_key=api_key, model=os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL))
    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise LLMUnavailableError("GROQ_API_KEY is not set")
        return GroqLLM(api_key=api_key, model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL))
    raise LLMUnavailableError(f"Unknown LLM_PROVIDER '{provider}'. Use 'ollama', 'anthropic' or 'groq'.")