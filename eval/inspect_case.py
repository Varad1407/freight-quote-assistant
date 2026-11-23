"""Show exactly what happened for one evaluation case, step by step.

Usage:
    python -m eval.inspect_case 01
    python -m eval.inspect_case 22 --model qwen2.5:14b

Prints the raw text the model returned (every attempt), what the grounding check
rejected, the final shipment, and which expected fields were wrong. Look at the
data before deciding on a fix.
"""

import argparse
import json
import sys

from app.extractor import ExtractionError, LLMUnavailableError, extract_shipment
from app.grounding import check_grounding
from app.pipeline import process_email
from app.safety import sanitize_email
from eval.run_eval import build_llm, load_cases, score_case


class RecordingLLM:
    """Wraps a real LLM and remembers every raw response."""

    def __init__(self, inner):
        self.inner = inner
        self.responses: list[str] = []

    def complete(self, system: str, user: str) -> str:
        text = self.inner.complete(system, user)
        self.responses.append(text)
        return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", help="text that appears in the case id, e.g. 01")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "anthropic"])
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    cases = load_cases(args.case)
    if len(cases) != 1:
        sys.exit(f"'{args.case}' matches {len(cases)} cases; be more specific.")
    case = cases[0]
    llm, model = build_llm(args.provider, args.model)
    recorder = RecordingLLM(llm)

    print(f"=== {case['id']}  ({args.provider}:{model}) ===\n")
    print("--- EMAIL ---")
    print(case["email"])

    clean_email, _ = sanitize_email(case["email"])
    try:
        extraction, _ = extract_shipment(clean_email, recorder)
    except LLMUnavailableError as exc:
        sys.exit(f"Stopped: {exc}")
    except ExtractionError as exc:  # still show the raw responses below
        extraction = None
        print(f"\nExtraction failed: {exc}")

    for i, text in enumerate(recorder.responses, start=1):
        print(f"\n--- RAW MODEL RESPONSE, attempt {i} ---")
        print(text)

    if extraction is None:
        return

    _, grounding_warnings = check_grounding(extraction, clean_email)
    print("\n--- GROUNDING ---")
    print("\n".join(grounding_warnings) if grounding_warnings else "all numbers found in the email")

    draft = process_email(case["email"], llm)  # same pipeline the app uses
    print("\n--- FINAL SHIPMENT ---")
    print(json.dumps(draft.shipment.model_dump(mode="json"), indent=2, ensure_ascii=False))
    print(f"\nstatus: {draft.status}   missing: {draft.missing}")

    print("\n--- SCORE ---")
    for field, outcome in score_case(draft, case["expected"]).items():
        mark = "ok " if outcome["ok"] else "BAD"
        print(f"  {mark} {field}: expected {outcome['expected']!r}, got {outcome['actual']!r}")


if __name__ == "__main__":
    main()
