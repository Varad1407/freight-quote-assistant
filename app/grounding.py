"""Grounding check: every number the LLM returns must actually appear in the email,
with the same unit written next to it.

LLMs sometimes invent or calculate numbers (e.g. a volume of 40,776 cbm for 12
pallets). If a number is not in the email text, we cannot trust it, so we drop it
and flag the email for human review. Python then works out what it can from the
numbers that ARE grounded (e.g. volume from dimensions x pieces).
"""

import re

from app.models import RawExtraction

_THOUSANDS_SEP = re.compile(r"(?<=\d)[,\s](?=\d{3}\b)")  # "12,400" or "12 400" -> "12400"
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_NUMBER_WORDS = {
    word: float(i)
    for i, word in enumerate(
        ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
    )
}
_WORD = re.compile(r"[a-z]+")


def numbers_in(text: str) -> set[float]:
    """All numbers in the text, including written ones like "two pallets"."""
    found = {float(n) for n in _NUMBER.findall(_THOUSANDS_SEP.sub("", text))}
    found |= {_NUMBER_WORDS[w] for w in _WORD.findall(text.lower()) if w in _NUMBER_WORDS}
    return found


def _grounded(value: float, found: set[float]) -> bool:
    return any(abs(value - n) < 1e-6 for n in found)


# How each unit can be written in an email. Used to check the unit next to a number.
UNIT_PATTERNS = {
    "kg": r"kgs?|kilos?|kilograms?",
    "lb": r"lbs?|pounds?",
    "t": r"t|tons?|tonnes?|mt|metric\s+tons?",
    "cbm": r"cbm|m3|m³|cubic\s+met(?:er|re)s?",
    "cuft": r"cu\.?\s*ft|cubic\s+f(?:ee|oo)t|cft|ft3|ft³",
}
WEIGHT_UNITS = ("kg", "lb", "t")
VOLUME_UNITS = ("cbm", "cuft")


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def units_written_after(value: float, email_text: str, candidates: tuple[str, ...]) -> set[str]:
    """Which of the candidate units are written directly after this number in the email."""
    text = _THOUSANDS_SEP.sub("", email_text).lower()
    found = set()
    number = re.escape(_format_number(value))
    for match in re.finditer(rf"(?<![\d.]){number}(?![\d.])", text):
        after = text[match.end() : match.end() + 25]
        for unit in candidates:
            if re.match(rf"\s*-?\s*(?:{UNIT_PATTERNS[unit]})\b", after):
                found.add(unit)
    return found


def _unit_mismatch(value: float, unit: str, email_text: str, candidates: tuple[str, ...]) -> str | None:
    """Return the unit the email actually uses, if it clearly differs from the claimed one."""
    written = units_written_after(value, email_text, candidates)
    if written and unit not in written:
        return min(written)
    return None


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
    if raw.weight is not None:
        w = raw.weight
        if not _grounded(w.value, found):
            reject("weight", f"{w.value:g} {w.unit}")
        elif actual := _unit_mismatch(w.value, w.unit, email_text, WEIGHT_UNITS):
            updates["weight"] = None
            warnings.append(f"AI returned weight unit '{w.unit}' for {w.value:g}, but the email says {actual}: ignored.")
    if raw.volume is not None:
        v = raw.volume
        if not _grounded(v.value, found):
            reject("volume", f"{v.value:g} {v.unit}")
        elif actual := _unit_mismatch(v.value, v.unit, email_text, VOLUME_UNITS):
            updates["volume"] = None
            warnings.append(f"AI returned volume unit '{v.unit}' for {v.value:g}, but the email says {actual}: ignored.")
    if raw.dimensions is not None:
        d = raw.dimensions
        if not all(_grounded(v, found) for v in (d.length, d.width, d.height)):
            reject("dimensions", f"{d.length:g} x {d.width:g} x {d.height:g} {d.unit}")

    return raw.model_copy(update=updates), warnings
