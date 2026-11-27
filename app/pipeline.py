"""The full email -> QuoteDraft pipeline, independent of FastAPI.

Keeping this separate from the web layer means the same logic runs in the API,
in the evaluation script (eval/run_eval.py) and in tests, so what we measure is
exactly what users get.
"""

from app.extractor import ExtractionError, LLMClient, extract_shipment
from app.grounding import check_grounding
from app.models import QuoteDraft, TransportMode, missing_fields
from app.normalize import to_shipment
from app.reply import build_warnings, draft_reply
from app.rules import find_explicit_mode, find_piece_count
from app.safety import notes_are_suspicious, sanitize_email


def process_email(email_text: str, llm: LLMClient) -> QuoteDraft:
    """Run every step. Raises LLMUnavailableError or ExtractionError from the extractor."""
    # Layer 1: strip lines that look like instructions to an AI before the LLM sees them.
    clean_email, removed_lines = sanitize_email(email_text)

    extraction, attempts = extract_shipment(clean_email, llm)

    # Every number must appear in the email; invented numbers are dropped.
    checked, grounding_warnings = check_grounding(extraction, clean_email)
    if grounding_warnings:
        # Self-correction: tell the model exactly which numbers were not in the email, ask once more.
        feedback = (
            "\n".join(grounding_warnings)
            + "\nCopy every number and unit exactly as written in the email. Do not convert or multiply."
        )
        try:
            retry, retry_attempts = extract_shipment(clean_email, llm, feedback=feedback)
            attempts += retry_attempts
            retry_checked, retry_warnings = check_grounding(retry, clean_email)
            if len(retry_warnings) < len(grounding_warnings):
                checked, grounding_warnings = retry_checked, retry_warnings
        except ExtractionError:
            pass  # keep the first result, with the bad numbers already dropped
    extraction = checked

    # Rule-based backup: fill the piece count from patterns like "12 pallets" if the model skipped it.
    if extraction.pieces is None:
        extraction = extraction.model_copy(update={"pieces": find_piece_count(clean_email)})

    # Rule-based backup: if the email names exactly one transport mode ("Rail:", "by truck"), use it.
    explicit_mode = find_explicit_mode(clean_email)
    if explicit_mode is not None and extraction.mode.value != explicit_mode:
        extraction = extraction.model_copy(update={"mode": TransportMode(explicit_mode)})

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
    subject, body = draft_reply(shipment, missing, needs_review=bool(safety_warnings))
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
