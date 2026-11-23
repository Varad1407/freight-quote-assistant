"""Data models: the contract between the LLM, the API and the reply drafter.

Pydantic validates everything the LLM returns. If the model invents a negative
weight or an unknown transport mode, validation fails and the extractor retries.
"""

from collections.abc import Callable
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransportMode(str, Enum):
    SEA = "sea"
    AIR = "air"
    ROAD = "road"
    RAIL = "rail"
    UNKNOWN = "unknown"


class CargoType(str, Enum):
    GENERAL = "general"
    HAZARDOUS = "hazardous"
    REFRIGERATED = "refrigerated"
    OVERSIZED = "oversized"
    UNKNOWN = "unknown"


class Location(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city: str | None = None
    country: str | None = None

    def is_known(self) -> bool:
        return bool(self.city or self.country)

    def label(self) -> str:
        return ", ".join(p for p in (self.city, self.country) if p) or "unknown"


VALID_INCOTERMS = {"EXW", "FCA", "CPT", "CIP", "DAP", "DPU", "DDP", "FAS", "FOB", "CFR", "CIF"}


class ShipmentRequest(BaseModel):
    """Structured details of one freight quote request."""

    model_config = ConfigDict(extra="forbid")

    customer_name: str | None = None
    company: str | None = None
    origin: Location = Field(default_factory=Location)
    destination: Location = Field(default_factory=Location)
    mode: TransportMode = TransportMode.UNKNOWN
    cargo_description: str | None = None
    cargo_type: CargoType = CargoType.UNKNOWN
    weight_kg: float | None = Field(default=None, gt=0, le=1_000_000)
    volume_cbm: float | None = Field(default=None, gt=0, le=100_000)
    pieces: int | None = Field(default=None, gt=0, le=1_000_000)
    dimensions: str | None = None
    ready_date: str | None = None
    incoterm: str | None = None
    notes: str | None = None

    @field_validator("incoterm")
    @classmethod
    def normalise_incoterm(cls, value: str | None) -> str | None:
        if not value:
            return None
        words = value.strip().upper().split()
        if not words:
            return None
        code = words[0]
        if code in VALID_INCOTERMS:
            return code
        return None


# --- Raw extraction: what the LLM is allowed to produce -------------------------
# The LLM copies numbers and units exactly as written. It never converts or
# multiplies: Python does that in normalize.py, because LLMs are bad at arithmetic.

class RawWeight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float = Field(gt=0)
    unit: Literal["kg", "lb", "t"]
    basis: Literal["total", "per_piece"] = "total"


class RawVolume(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float = Field(gt=0)
    unit: Literal["cbm", "cuft"]


class RawDimensions(BaseModel):
    """Size of ONE piece, as written in the email."""

    model_config = ConfigDict(extra="forbid")

    length: float = Field(gt=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    unit: Literal["cm", "m", "mm", "in"]


class RawExtraction(BaseModel):
    """Exactly what the LLM must return. Validated before anything else happens."""

    model_config = ConfigDict(extra="forbid")

    customer_name: str | None = None
    company: str | None = None
    origin: Location = Field(default_factory=Location)
    destination: Location = Field(default_factory=Location)
    mode: TransportMode = TransportMode.UNKNOWN
    cargo_description: str | None = None
    cargo_type: CargoType = CargoType.UNKNOWN
    pieces: int | None = Field(default=None, gt=0, le=1_000_000)
    weight: RawWeight | None = None
    volume: RawVolume | None = None
    dimensions: RawDimensions | None = None
    ready_date: str | None = None
    incoterm: str | None = None
    notes: str | None = None

    # Lenient parsing: a vague or empty field becomes "unknown" instead of failing
    # the whole email. (Evaluation case 02 failed completely because of one field.)
    @field_validator("origin", "destination", mode="before")
    @classmethod
    def null_location_is_empty(cls, value):
        return {} if value is None else value

    @field_validator("mode", "cargo_type", mode="before")
    @classmethod
    def null_enum_is_unknown(cls, value):
        return "unknown" if value is None else value

    @field_validator("weight", "volume", "dimensions", mode="before")
    @classmethod
    def incomplete_quantity_is_unknown(cls, value):
        if isinstance(value, dict) and any(v is None for v in value.values()):
            return None
        return value


# Field checks in priority order: (label shown to customer, check function).
REQUIRED_FIELDS: list[tuple[str, Callable[["ShipmentRequest"], bool]]] = [
    ("pickup location (city and country)", lambda r: r.origin.is_known()),
    ("delivery location (city and country)", lambda r: r.destination.is_known()),
    ("preferred transport mode (sea, air, road or rail)", lambda r: r.mode != TransportMode.UNKNOWN),
    ("description of the goods", lambda r: bool(r.cargo_description)),
    ("total gross weight", lambda r: r.weight_kg is not None),
    ("volume or package dimensions", lambda r: r.volume_cbm is not None or bool(r.dimensions)),
    ("cargo ready date", lambda r: bool(r.ready_date)),
]


def missing_fields(request: ShipmentRequest) -> list[str]:
    """Return human-readable labels for every required detail that is missing."""
    return [label for label, is_present in REQUIRED_FIELDS if not is_present(request)]


class QuoteRequestIn(BaseModel):
    email_text: str = Field(min_length=20, max_length=20_000)


class QuoteDraft(BaseModel):
    status: Literal["ready_to_quote", "needs_info", "needs_review"]
    missing: list[str]
    warnings: list[str]
    shipment: ShipmentRequest
    reply_subject: str
    reply_body: str
    attempts: int
