"""Grounding check: every number the LLM returns must actually appear in the email.

LLMs sometimes invent or calculate numbers (e.g. a volume of 40,776 cbm for 12
pallets). If a number is not in the email text, we cannot trust it, so we drop it
and flag the email for human review. Python then works out what it can from the
numbers that ARE grounded (e.g. volume from dimensions x pieces).
"""

import re

from app.models import RawExtraction

_THOUSANDS_SEP = re.compile(r"(?<=\d)[,\s](?=\d{3}\b)")  # "12,400" or "12 400" -> "12400"
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def numbers_in(text: str) -> set[float]:
    return {float(n) for n in _NUMBER.findall(_THOUSANDS_SEP.sub("", text))}


def _grounded(value: float, found: set[float]) -> bool:
    return any(abs(value - n) < 1e-6 for n in found)


def check_grounding(raw: RawExtraction, email_text: str) -> tuple[RawExtraction, list[str]]:
    """Drop numeric fields whose numbers are not in the email. Returns (raw, warnings)."""
    found = numbers_in(email_text)
    updates: dict = {}
    warnings: list[str] = []

    def reject(field: str, shown: str) -> None:
        updates[field] = None
        warnings.append(f"AI returned {field} '{shown}', which does not appear in the email: ignored.")

    if raw.pieces is not None and not _grounded(raw.pieces, found):
        reject("pieces", str(raw.pieces))
    if raw.weight is not None and not _grounded(raw.weight.value, found):
        reject("weight", f"{raw.weight.value:g} {raw.weight.unit}")
    if raw.volume is not None and not _grounded(raw.volume.value, found):
        reject("volume", f"{raw.volume.value:g} {raw.volume.unit}")
    if raw.dimensions is not None:
        d = raw.dimensions
        if not all(_grounded(v, found) for v in (d.length, d.width, d.height)):
            reject("dimensions", f"{d.length:g} x {d.width:g} x {d.height:g} {d.unit}")

    return raw.model_copy(update=updates), warnings
