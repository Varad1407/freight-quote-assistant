"""FastAPI application: API endpoints plus a one-page demo UI."""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from app.extractor import (
    AnthropicLLM,
    ExtractionError,
    LLMClient,
    LLMUnavailableError,
    OllamaLLM,
)
from app.models import QuoteDraft, QuoteRequestIn
from app.pipeline import process_email

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")

APP_DIR = Path(__file__).parent
SAMPLES_DIR = APP_DIR / "samples"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
DEFAULT_OLLAMA_MODEL = "qwen2.5:14b"  # 30/30 dev, 12/12 held-out (v0.9); 7b is faster but less accurate

app = FastAPI(
    title="Freight Quote Assistant",
    description="Turns messy freight quote-request emails into structured data and a draft reply.",
    version="0.5.0",
)


@lru_cache
def _build_llm() -> LLMClient:
    """Pick the LLM provider from LLM_PROVIDER: 'ollama' (free, local, default) or 'anthropic'."""
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
    raise LLMUnavailableError(f"Unknown LLM_PROVIDER '{provider}'. Use 'ollama' or 'anthropic'.")


def get_llm() -> LLMClient:
    """FastAPI dependency. Tests replace it with a fake via app.dependency_overrides."""
    try:
        return _build_llm()
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/samples")
def list_samples() -> list[dict]:
    return [
        {"name": path.stem, "email_text": path.read_text(encoding="utf-8")}
        for path in sorted(SAMPLES_DIR.glob("*.txt"))
    ]


@app.post("/api/quote-requests", response_model=QuoteDraft)
def process_quote_request(payload: QuoteRequestIn, llm: Annotated[LLMClient, Depends(get_llm)]) -> QuoteDraft:
    try:
        return process_email(payload.email_text, llm)
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ExtractionError as exc:
        raise HTTPException(status_code=502, detail="Could not extract shipment details reliably.") from exc


@app.get("/", response_class=HTMLResponse)
def demo_page() -> str:
    return (APP_DIR / "static" / "index.html").read_text(encoding="utf-8")
