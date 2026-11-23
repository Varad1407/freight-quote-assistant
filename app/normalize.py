"""Turn the LLM's raw extraction into a ShipmentRequest.

All unit conversion and multiplication happens here, in plain Python, so it is
exact and testable. The LLM only reads numbers; it never calculates them.
"""

from app.models import (
    CargoType,
    RawDimensions,
    RawExtraction,
    RawVolume,
    RawWeight,
    ShipmentRequest,
)

KG_PER_UNIT = {"kg": 1.0, "lb": 0.45359237, "t": 1000.0}
METRES_PER_UNIT = {"cm": 0.01, "m": 1.0, "mm": 0.001, "in": 0.0254}
CBM_PER_UNIT = {"cbm": 1.0, "cuft": 0.0283168466}


def total_weight_kg(weight: RawWeight | None, pieces: int | None) -> float | None:
    """Total gross weight in kg, or None if it can't be worked out safely."""
    if weight is None:
        return None
    kg = weight.value * KG_PER_UNIT[weight.unit]
    if weight.basis == "per_piece":
        if pieces is None:
            return None  # per-piece weight without a piece count: ask the customer
        kg *= pieces
    return round(kg, 1)


def total_volume_cbm(
    volume: RawVolume | None, dimensions: RawDimensions | None, pieces: int | None
) -> float | None:
    """Total volume in cubic metres: stated volume first, otherwise from dimensions x pieces."""
    if volume is not None:
        return round(volume.value * CBM_PER_UNIT[volume.unit], 3)
    if dimensions is not None and pieces is not None:
        m = METRES_PER_UNIT[dimensions.unit]
        per_piece = (dimensions.length * m) * (dimensions.width * m) * (dimensions.height * m)
        return round(per_piece * pieces, 3)
    return None


def dimensions_label(dimensions: RawDimensions | None) -> str | None:
    if dimensions is None:
        return None
    d = dimensions
    return f"{d.length:g} x {d.width:g} x {d.height:g} {d.unit} per piece"


# Largest piece a standard 40ft high-cube container takes (metres, approximate inner size).
CONTAINER_MAX_M = (2.69, 2.35, 12.0)  # smallest side, middle side, longest side


def exceeds_container(dimensions: RawDimensions | None) -> bool:
    """True if one piece cannot fit a standard container in any orientation."""
    if dimensions is None:
        return False
    m = METRES_PER_UNIT[dimensions.unit]
    sides = sorted(v * m for v in (dimensions.length, dimensions.width, dimensions.height))
    return any(side > limit for side, limit in zip(sides, sorted(CONTAINER_MAX_M), strict=True))


def to_shipment(raw: RawExtraction) -> ShipmentRequest:
    cargo_type = raw.cargo_type
    if cargo_type in (CargoType.GENERAL, CargoType.UNKNOWN) and exceeds_container(raw.dimensions):
        cargo_type = CargoType.OVERSIZED
    return ShipmentRequest(
        customer_name=raw.customer_name,
        company=raw.company,
        origin=raw.origin,
        destination=raw.destination,
        mode=raw.mode,
        cargo_description=raw.cargo_description,
        cargo_type=cargo_type,
        weight_kg=total_weight_kg(raw.weight, raw.pieces),
        volume_cbm=total_volume_cbm(raw.volume, raw.dimensions, raw.pieces),
        pieces=raw.pieces,
        dimensions=dimensions_label(raw.dimensions),
        ready_date=raw.ready_date,
        incoterm=raw.incoterm,
        notes=raw.notes,
    )
