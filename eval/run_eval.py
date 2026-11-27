"""Evaluate the quote assistant on a labelled set of emails.

Usage (from the project folder, with Ollama running):
    python -m eval.run_eval                          # qwen2.5:14b via Ollama (default)
    python -m eval.run_eval --model qwen2.5:14b      # compare another model
    python -m eval.run_eval --case 03                # only cases whose id contains "03"
    python -m eval.run_eval --set holdout            # the unseen held-out cases
    python -m eval.run_eval --provider anthropic --model claude-sonnet-5-5
    python -m eval.run_eval --provider groq          # free hosted model (GROQ_API_KEY in .env)

Each case in eval/cases.jsonl has an email and the expected answer. Only the
fields listed under "expected" are checked; a value of null means "should be empty".
A list means "any of these is acceptable".
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from app.extractor import (
    AnthropicLLM,
    ExtractionError,
    GroqLLM,
    LLMClient,
    LLMUnavailableError,
    OllamaLLM,
)
from app.models import QuoteDraft
from app.pipeline import process_email

ROOT = Path(__file__).resolve().parent.parent
CASE_SETS = {"dev": ROOT / "eval" / "cases.jsonl", "holdout": ROOT / "eval" / "holdout.jsonl"}
RESULTS_DIR = ROOT / "eval" / "results"

NUMERIC_FIELDS = {"weight_kg", "volume_cbm"}
COUNTRY_ALIASES = {
    "usa": "united states", "us": "united states", "u.s.": "united states", "u.s.a.": "united states",
    "united states of america": "united states", "uk": "united kingdom", "great britain": "united kingdom",
    "uae": "united arab emirates", "korea": "south korea", "republic of korea": "south korea",
    "czechia": "czech republic", "côte d'ivoire": "ivory coast", "cote d'ivoire": "ivory coast",
    "holland": "netherlands", "the netherlands": "netherlands",
}


def _norm_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().lower()
    return COUNTRY_ALIASES.get(text, text) or None


def actual_values(draft: QuoteDraft) -> dict:
    s = draft.shipment
    return {
        "status": draft.status,
        "origin_city": s.origin.city,
        "origin_country": s.origin.country,
        "dest_city": s.destination.city,
        "dest_country": s.destination.country,
        "mode": s.mode.value,
        "cargo_type": s.cargo_type.value,
        "pieces": s.pieces,
        "weight_kg": s.weight_kg,
        "volume_cbm": s.volume_cbm,
        "incoterm": s.incoterm,
        "injection_flagged": any("prompt injection" in w.lower() for w in draft.warnings),
    }


def values_match(field: str, expected, actual) -> bool:
    options = expected if isinstance(expected, list) else [expected]
    for option in options:
        if option is None:
            if actual is None:
                return True
        elif actual is None:
            continue
        elif field in NUMERIC_FIELDS:
            if abs(float(actual) - float(option)) <= max(0.05, abs(float(option)) * 0.01):  # 1% tolerance
                return True
        elif isinstance(option, str):
            if _norm_text(str(actual)) == _norm_text(option):
                return True
        elif actual == option:
            return True
    return False


def score_case(draft: QuoteDraft, expected: dict) -> dict[str, dict]:
    actual = actual_values(draft)
    return {
        field: {"ok": values_match(field, want, actual[field]), "expected": want, "actual": actual[field]}
        for field, want in expected.items()
    }


def load_cases(filter_text: str | None = None, case_set: str = "dev") -> list[dict]:
    cases = []
    for line in CASE_SETS[case_set].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        if "email_file" in case:
            case["email"] = (ROOT / case["email_file"]).read_text(encoding="utf-8")
        if filter_text is None or filter_text in case["id"]:
            cases.append(case)
    return cases


def build_llm(provider: str, model: str | None) -> tuple[LLMClient, str]:
    if provider == "ollama":
        model = model or "qwen2.5:14b"
        return OllamaLLM(model=model, base_url=os.getenv("OLLAMA_URL", "http://localhost:11434")), model
    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            sys.exit("ANTHROPIC_API_KEY is not set")
        model = model or "claude-sonnet-5-5"
        return AnthropicLLM(api_key=key, model=model), model
    if provider == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            sys.exit("GROQ_API_KEY is not set")
        model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        return GroqLLM(api_key=key, model=model), model
    sys.exit(f"Unknown provider {provider!r}")


def run(cases: list[dict], llm: LLMClient) -> list[dict]:
    results = []
    for i, case in enumerate(cases, start=1):
        started = time.perf_counter()
        try:
            draft = process_email(case["email"], llm)
            fields = score_case(draft, case["expected"])
            error = None
        except ExtractionError as exc:
            fields, error = {}, f"ExtractionError: {exc}"
        latency = round(time.perf_counter() - started, 1)
        passed = error is None and all(f["ok"] for f in fields.values())
        results.append({"id": case["id"], "passed": passed, "error": error, "latency_s": latency, "fields": fields})
        mark = "PASS" if passed else "FAIL"
        print(f"[{i:2}/{len(cases)}] {mark}  {case['id']}  ({latency}s)", flush=True)
    return results


def summarize(results: list[dict]) -> dict:
    field_totals: dict[str, list[int]] = {}
    for r in results:
        for field, outcome in r["fields"].items():
            ok_total = field_totals.setdefault(field, [0, 0])
            ok_total[0] += outcome["ok"]
            ok_total[1] += 1
    return {
        "cases": len(results),
        "cases_passed": sum(r["passed"] for r in results),
        "errors": sum(r["error"] is not None for r in results),
        "avg_latency_s": round(sum(r["latency_s"] for r in results) / max(len(results), 1), 1),
        "fields": {f: {"correct": c, "total": t} for f, (c, t) in sorted(field_totals.items())},
    }


def print_report(summary: dict, results: list[dict]) -> None:
    print("\n=== Field accuracy ===")
    for field, s in summary["fields"].items():
        pct = 100 * s["correct"] / s["total"]
        print(f"  {field:18} {s['correct']:3}/{s['total']:<3} {pct:5.1f}%")
    print("\n=== Failures ===")
    for r in results:
        if r["passed"]:
            continue
        print(f"- {r['id']}")
        if r["error"]:
            print(f"    error: {r['error']}")
        for field, outcome in r["fields"].items():
            if not outcome["ok"]:
                print(f"    {field}: expected {outcome['expected']!r}, got {outcome['actual']!r}")
    print(
        f"\nCases fully correct: {summary['cases_passed']}/{summary['cases']}   "
        f"Average time per email: {summary['avg_latency_s']}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="ollama", choices=["ollama", "anthropic", "groq"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--case", default=None, help="only run cases whose id contains this text")
    parser.add_argument(
        "--set", dest="case_set", default="dev", choices=list(CASE_SETS),
        help="dev = the 30 cases we tune on; holdout = unseen cases, run once, never tune on them",
    )
    args = parser.parse_args()

    llm, model = build_llm(args.provider, args.model)
    cases = load_cases(args.case, args.case_set)
    print(f"Running {len(cases)} {args.case_set} case(s) with {args.provider}:{model}\n")
    try:
        results = run(cases, llm)
    except LLMUnavailableError as exc:
        sys.exit(f"\nStopped: {exc}")

    summary = summarize(results)
    print_report(summary, results)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"{stamp}_{args.case_set}_{args.provider}_{model.replace(':', '-').replace('/', '-')}.json"
    out.write_text(json.dumps({"provider": args.provider, "model": model, "summary": summary,
                               "results": results}, indent=2, ensure_ascii=False))
    print(f"Saved detailed results to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
