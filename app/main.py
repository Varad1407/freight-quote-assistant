"""FastAPI application: API endpoints plus a one-page demo UI."""

import logging
import os
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from app.config import build_llm as _build_llm
from app.extractor import (
    ExtractionError,
    LLMClient,
    LLMUnavailableError,
)
from app.models import QuoteDraft, QuoteRequestIn
from app.pipeline import process_email
from app.ratelimit import RateLimiter, client_ip

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")

APP_DIR = Path(__file__).parent
SAMPLES_DIR = APP_DIR / "samples"

# Public-demo protection. Set a value to 0 to switch that limit off (e.g. for local work).
limiter = RateLimiter(
    per_ip_limit=int(os.getenv("RATE_LIMIT_PER_HOUR", "10")),
    window_s=3600,
    global_daily_limit=int(os.getenv("RATE_LIMIT_PER_DAY_TOTAL", "300")),
)

app = FastAPI(
    title="Freight Quote Assistant",
    description="Turns messy freight quote-request emails into structured data and a draft reply.",
    version="1.1.0",
)


def get_llm() -> LLMClient:
    """FastAPI dependency. Tests replace it with a fake via app.dependency_overrides."""
    try:
        return _build_llm()
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def enforce_rate_limit(request: Request) -> None:
    """FastAPI dependency: reject the request with 429 when a limit is reached."""
    ip = client_ip(request.headers.get("x-forwarded-for"), request.client.host if request.client else None)
    decision = limiter.check(ip)
    if decision.allowed:
        return
    if decision.reason == "global":
        detail = "The demo has reached its daily limit. Please try again tomorrow."
    else:
        minutes = max(1, round(decision.retry_after_s / 60))
        detail = f"Demo limit reached: try again in about {minutes} minute(s)."
    raise HTTPException(status_code=429, detail=detail, headers={"Retry-After": str(decision.retry_after_s)})


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/samples")
def list_samples() -> list[dict]:
    return [
        {"name": path.stem, "email_text": path.read_text(encoding="utf-8")}
        for path in sorted(SAMPLES_DIR.glob("*.txt"))
    ]


@app.post("/api/quote-requests", response_model=QuoteDraft, dependencies=[Depends(enforce_rate_limit)])
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
