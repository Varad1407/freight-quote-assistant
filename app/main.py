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
    extract_shipment,
)
from app.grounding import check_grounding
from app.models import QuoteDraft, QuoteRequestIn, missing_fields
from app.normalize import to_shipment
from app.reply import build_warnings, draft_reply
from app.safety import notes_are_suspicious, sanitize_email

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")

APP_DIR = Path(__file__).parent
SAMPLES_DIR = APP_DIR / "samples"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"

app = FastAPI(
    title="Freight Quote Assistant",
    description="Turns messy freight quote-request emails into structured data and a draft reply.",
    version="0.3.0",
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
    # Layer 1: strip lines that look like instructions to an AI before the LLM sees them.
    clean_email, removed_lines = sanitize_email(payload.email_text)
    try:
        extraction, attempts = extract_shipment(clean_email, llm)
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ExtractionError as exc:
        raise HTTPException(status_code=502, detail="Could not extract shipment details reliably.") from exc

    # Every number must appear in the email; invented numbers are dropped.
    extraction, grounding_warnings = check_grounding(extraction, clean_email)

    # Python, not the LLM, converts units and multiplies by piece count.
    shipment = to_shipment(extraction)

    safety_warnings = list(grounding_warnings)
    if removed_lines:
        safety_warnings.append(
            f"Possible prompt injection: {removed_lines} line(s) in this email looked like instructions "
            "to an AI and were ignored. Check every field against the original email."
        )
    # Layer 2: free-text notes may still carry injected text or commitments we never made.
    if notes_are_suspicious(shipment.notes):
        shipment = shipment.model_copy(update={"notes": None})
        safety_warnings.append("Notes removed: they contained instruction-like or commitment wording.")

    missing = missing_fields(shipment)
    subject, body = draft_reply(shipment, missing)
    if safety_warnings:
        status = "needs_review"
    elif missing:
        status = "needs_info"
    else:
        status = "ready_to_quote"
    return QuoteDraft(
        status=status,
        missing=missing,
        warnings=safety_warnings + build_warnings(shipment),
        shipment=shipment,
        reply_subject=subject,
        reply_body=body,
        attempts=attempts,
    )


@app.get("/", response_class=HTMLResponse)
def demo_page() -> str:
    return (APP_DIR / "static" / "index.html").read_text(encoding="utf-8")
