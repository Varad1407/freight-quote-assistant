"""Rule-based extraction for patterns regular enough that plain Python reads them reliably.

Hybrid design: the LLM handles messy language; simple rules fill in well-structured
details the LLM sometimes skips. Evaluation showed qwen2.5:7b leaving "pieces" empty
for emails that clearly said "6 crates" or "33 pallets".
"""

import re

PACKAGE_WORDS = (
    r"pallets?|boxes|box|crates?|cartons?|drums?|bags?|pieces|pcs|packages?|pkgs?|cases|bundles?|rolls?|skids?"
)
_PIECES = re.compile(rf"\b(\d+)\s*(?:x\s*)?(?:{PACKAGE_WORDS})\b", re.IGNORECASE)


def find_piece_count(email_text: str) -> int | None:
    """Return the piece count if the email states exactly one, e.g. "12 pallets". Otherwise None."""
    counts = {int(n) for n in _PIECES.findall(email_text) if int(n) > 0}
    return counts.pop() if len(counts) == 1 else None
