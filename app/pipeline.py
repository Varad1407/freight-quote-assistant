"""The full email -> QuoteDraft pipeline, independent of FastAPI.

Keeping this separate from the web layer means the same logic runs in the API,
in the evaluation script (eval/run_eval.py) and in tests, so what we measure is
exactly what users get.
"""

from app.extractor import LLMClient, extract_shipment
from app.grounding import check_grounding
from app.models import QuoteDraft, missing_fields
from app.normalize import to_shipment
from app.reply import build_warnings, draft_reply
from app.safety import notes_are_suspicious, sanitize_email


def process_email(email_text: str, llm: LLMClient) -> QuoteDraft:
    """Run every step. Raises LLMUnavailableError or ExtractionError from the extractor."""
    # Layer 1: strip lines that look like instructions to an AI before the LLM sees them.
    clean_email, removed_lines = sanitize_email(email_text)

    extraction, attempts = extract_shipment(clean_email, llm)

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
