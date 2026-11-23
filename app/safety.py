"""Prompt-injection defences.

Defence in depth, because no single layer is perfect:
1. Before the LLM: remove lines that look like instructions aimed at an AI.
2. After the LLM: drop free-text notes that look like instructions or commitments.
3. Any hit sets status "needs_review", so a human checks the email.
4. (In reply.py) notes never go into the customer reply.

Honest limitation: pattern matching catches obvious attacks only. A determined
attacker can phrase things differently, which is why a human stays in the loop.
"""

import re

INJECTION_PATTERNS = [
    r"\bignore\b.{0,40}\b(instructions?|prompts?|rules)\b",
    r"\bdisregard\b.{0,40}\b(instructions?|prompts?|above|previous)\b",
    r"\b(system|developer|admin)\s+(note|prompt|message|instruction|override)s?\b",
    r"\byou\s+are\s+now\b",
    r"\bact\s+as\b.{0,30}\b(ai|assistant|model|system)\b",
    r"\b(weight_kg|volume_cbm|cargo_type|ready_date|customer_name)\b",  # our internal field names (with underscores; customers never type these)
]

COMMITMENT_PATTERNS = [
    r"\bapproved\b",
    r"\bfree\s+(shipping|of\s+charge)\b",
    r"\bdiscount\b",
    r"\bwaive[ds]?\b",
    r"\bguarantee[ds]?\b",
]

_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)
_COMMITMENT_RE = re.compile("|".join(COMMITMENT_PATTERNS), re.IGNORECASE)

REDACTION = "[line removed: looked like instructions to an AI]"


def sanitize_email(text: str) -> tuple[str, int]:
    """Replace suspicious lines. Returns (clean_text, number_of_lines_removed)."""
    removed = 0
    lines = []
    for line in text.splitlines():
        if _INJECTION_RE.search(line):
            lines.append(REDACTION)
            removed += 1
        else:
            lines.append(line)
    return "\n".join(lines), removed


def notes_are_suspicious(notes: str | None) -> bool:
    return bool(notes) and bool(_INJECTION_RE.search(notes) or _COMMITMENT_RE.search(notes))
