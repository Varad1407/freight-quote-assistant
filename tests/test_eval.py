"""Tests for the evaluation harness itself, and a sanity check of the dataset."""

import json

import pytest

from eval.run_eval import load_cases, run, score_case, summarize, values_match
from tests.test_core import FakeLLM

KNOWN_FIELDS = {
    "status", "origin_city", "origin_country", "dest_city", "dest_country", "mode", "cargo_type",
    "pieces", "weight_kg", "volume_cbm", "incoterm", "injection_flagged",
}


@pytest.mark.parametrize(("case_set", "count"), [("dev", 30), ("holdout", 12)])
def test_dataset_is_valid(case_set, count):
    cases = load_cases(case_set=case_set)
    assert len(cases) == count
    assert len({c["id"] for c in cases}) == count
    for case in cases:
        assert len(case["email"]) >= 20, case["id"]
        assert set(case["expected"]) <= KNOWN_FIELDS, case["id"]
        assert "status" in case["expected"], case["id"]


@pytest.mark.parametrize(
    ("field", "expected", "actual", "ok"),
    [
        ("weight_kg", 3538.0, 3538.0, True),
        ("weight_kg", 3538.0, 3560.0, True),     # within 1%
        ("weight_kg", 3538.0, 8148.0, False),
        ("weight_kg", None, None, True),
        ("weight_kg", None, 100.0, False),
        ("dest_country", "United States", "USA", True),
        ("origin_city", ["Gdansk", "Gdańsk"], "gdańsk", True),
        ("origin_city", "Shanghai", None, False),
        ("injection_flagged", True, True, True),
        ("injection_flagged", False, True, False),
    ],
)
def test_values_match(field, expected, actual, ok):
    assert values_match(field, expected, actual) is ok


def test_full_run_with_fake_llm_scores_correctly():
    case = {
        "id": "x",
        "email": "Air freight Chicago to Singapore. 12 pallets, each 120 x 100 x 110 cm, 650 lbs per pallet. April 2.",
        "expected": {"mode": "air", "weight_kg": 3538.0, "volume_cbm": 15.84, "incoterm": None},
    }
    raw = {
        "origin": {"city": "Chicago", "country": "USA"},
        "destination": {"city": "Singapore", "country": None},
        "mode": "air",
        "cargo_description": "pallets",
        "pieces": 12,
        "weight": {"value": 650, "unit": "lb", "basis": "per_piece"},
        "dimensions": {"length": 120, "width": 100, "height": 110, "unit": "cm"},
        "ready_date": "April 2",
    }
    results = run([case], FakeLLM([json.dumps(raw)]))
    assert results[0]["passed"], results[0]
    summary = summarize(results)
    assert summary["cases_passed"] == 1
    assert summary["fields"]["weight_kg"] == {"correct": 1, "total": 1}


def test_score_case_reports_wrong_field():
    from app.pipeline import process_email

    raw = {"mode": "sea", "origin": {"city": "Oslo"}}
    draft = process_email("Sea freight from Oslo please, details to follow.", FakeLLM([json.dumps(raw)]))
    scores = score_case(draft, {"mode": "air", "origin_city": "Oslo"})
    assert scores["origin_city"]["ok"] is True
    assert scores["mode"] == {"ok": False, "expected": "air", "actual": "sea"}


def test_holdout_does_not_overlap_dev():
    dev = {c["email"] for c in load_cases(case_set="dev")}
    assert not dev & {c["email"] for c in load_cases(case_set="holdout")}
