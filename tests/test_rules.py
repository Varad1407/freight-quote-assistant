"""Tests for rule-based extraction (app/rules.py)."""

import json

import pytest

from app.pipeline import process_email
from app.rules import find_explicit_mode, find_piece_count
from tests.test_core import COMPLETE_RAW, FakeLLM


@pytest.mark.parametrize(
    ("email", "expected"),
    [
        ("6 crates of auto parts", 6),
        ("33 pallets of bottled water", 33),
        ("80 drums x 25 kg = 2,000 kg", 80),
        ("12 pallets, each pallet 120 x 100 x 110 cm", 12),
        ("2 x 40ft containers of LED lights", None),   # containers are not pieces
        ("Pallets of tiles, 900 kg per pallet", None),  # no count stated
        ("3 boxes now and 5 boxes next week", None),    # ambiguous: two different counts
    ],
)
def test_find_piece_count(email, expected):
    assert find_piece_count(email) == expected


def test_pipeline_fills_pieces_when_model_skips_them():
    email = "Air freight Frankfurt to Toronto. 4 boxes, 35 kg each, 60 x 40 x 50 cm each. June 10."
    raw = {**COMPLETE_RAW, "pieces": None, "weight": {"value": 35, "unit": "kg", "basis": "per_piece"},
           "volume": None, "dimensions": {"length": 60, "width": 40, "height": 50, "unit": "cm"}}
    draft = process_email(email, FakeLLM([json.dumps(raw)]))
    assert draft.shipment.pieces == 4
    assert draft.shipment.weight_kg == 140.0   # per-piece weight now computable
    assert draft.shipment.volume_cbm == 0.48


@pytest.mark.parametrize(
    ("email", "expected"),
    [
        ("Rail: Almaty to Lodz, 3 containers 40HC", "rail"),  # held-out h04: containers are not sea
        ("Air quote Paris, France to Dubai", "air"),          # dev case 21
        ("Truck needed from Milan to Munich", "road"),
        ("ocean freight Savannah to Rotterdam", "sea"),
        ("1 x 40ft reefer container, Santos to Jeddah", None),          # mode not named: model decides
        ("Sea freight to Rotterdam, then truck to Duisburg", None),     # multi-modal: model decides
        ("Please send me your rates for shipping to Europe", None),
        ("Delivery to the airport warehouse", None),                    # "airport" is not air freight
    ],
)
def test_find_explicit_mode(email, expected):
    assert find_explicit_mode(email) == expected


def test_pipeline_overrides_mode_when_email_names_it():
    email = "Rail: Almaty, Kazakhstan to Lodz, Poland, 3 containers, total gross 54 t, 200 m3. Early January."
    raw = {**COMPLETE_RAW, "mode": "sea", "weight": {"value": 54, "unit": "t", "basis": "total"},
           "volume": {"value": 200, "unit": "m3"}}
    draft = process_email(email, FakeLLM([json.dumps(raw)]))
    assert draft.shipment.mode.value == "rail"
