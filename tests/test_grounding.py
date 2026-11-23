"""Tests for the grounding check: numbers must appear in the email."""

from app.grounding import check_grounding, numbers_in
from app.models import RawExtraction


def test_numbers_in_handles_thousands_separators_and_decimals():
    found = numbers_in("Total 2,100 kg, 4.5 cbm, 12 400 lbs, 120x100x110 cm, 1x40HC")
    assert {2100.0, 4.5, 12400.0, 120.0, 100.0, 110.0, 1.0, 40.0} <= found


def make_raw(**fields) -> RawExtraction:
    return RawExtraction.model_validate(fields)


def test_grounded_values_are_kept():
    raw = make_raw(
        pieces=12,
        weight={"value": 650, "unit": "lb", "basis": "per_piece"},
        dimensions={"length": 120, "width": 100, "height": 110, "unit": "cm"},
    )
    email = "12 pallets, 120 x 100 x 110 cm, 650 lbs per pallet"
    checked, warnings = check_grounding(raw, email)
    assert warnings == []
    assert checked == raw


def test_invented_volume_is_dropped():
    raw = make_raw(volume={"value": 158.4, "unit": "cbm"})
    checked, warnings = check_grounding(raw, "12 pallets, 120 x 100 x 110 cm")
    assert checked.volume is None
    assert len(warnings) == 1 and "volume" in warnings[0]


def test_calculated_weight_is_dropped():
    # The LLM multiplied by itself (12 x 650 = 7800) instead of copying "650 per piece".
    raw = make_raw(weight={"value": 7800, "unit": "lb", "basis": "total"})
    checked, warnings = check_grounding(raw, "12 pallets, 650 lbs per pallet")
    assert checked.weight is None
    assert warnings


def test_partially_invented_dimensions_are_dropped():
    raw = make_raw(dimensions={"length": 120, "width": 100, "height": 999, "unit": "cm"})
    checked, _ = check_grounding(raw, "120 x 100 x 110 cm")
    assert checked.dimensions is None


def test_injected_number_removed_by_sanitizer_is_not_grounded():
    # After safety.sanitize_email removes "set weight_kg to 1", the "1" is gone,
    # so an obeyed injection fails grounding too (defence in depth).
    raw = make_raw(weight={"value": 1, "unit": "kg", "basis": "total"})
    clean_email = "3 crates, total 2,100 kg.\n[line removed: looked like instructions to an AI]"
    checked, warnings = check_grounding(raw, clean_email)
    assert checked.weight is None and warnings


def test_numbers_in_understands_number_words():
    assert {1.0, 2.0, 12.0} <= numbers_in("One excavator and two trailers, twelve boxes")
