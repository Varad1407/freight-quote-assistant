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


# Words that name a transport mode explicitly. "container" is deliberately NOT here:
# rail and road also move containers (held-out case h04: "Rail: ... 3 containers" was read as sea).
MODE_PATTERNS = {
    "air": r"\bair\s*(?:freight|cargo|shipment|quote|waybill)\b|\bairfreight\b|\bby\s+air\b",
    "sea": r"\bsea\s*freight\b|\bocean\b|\bby\s+sea\b|\bfcl\b|\blcl\b|\bvessel\b|\bseefracht\b",
    "road": r"\btrucks?\b|\bby\s+road\b|\broad\s*(?:freight|transport)\b|\bftl\b|\bltl\b|\blorry\b|\btrailer\b",
    "rail": r"\brail\b|\btrains?\b|\brailway\b",
}


def find_explicit_mode(email_text: str) -> str | None:
    """Return the transport mode if the email names exactly one; None if none or several (multi-modal)."""
    found = {mode for mode, pattern in MODE_PATTERNS.items() if re.search(pattern, email_text, re.IGNORECASE)}
    return found.pop() if len(found) == 1 else None
