"""Reply drafting.

Deliberately template-based, not LLM-generated: the reply only ever states facts
that passed validation, and it never invents a price. A human sets the rate.
"""

from app.models import CargoType, ShipmentRequest, TransportMode


def build_warnings(request: ShipmentRequest) -> list[str]:
    """Internal flags for the forwarder's team (not sent to the customer)."""
    warnings = []
    if request.cargo_type == CargoType.HAZARDOUS:
        warnings.append("Dangerous goods: request the MSDS/SDS and check carrier DG acceptance.")
    if request.cargo_type == CargoType.REFRIGERATED:
        warnings.append("Temperature-controlled cargo: confirm required temperature range.")
    if request.cargo_type == CargoType.OVERSIZED:
        warnings.append("Oversized cargo: may need special equipment (flat rack, open top, permits).")
    if request.mode == TransportMode.AIR and request.weight_kg and request.weight_kg > 10_000:
        warnings.append("Heavy air shipment (>10 t): consider charter or sea/air option.")
    if not request.incoterm:
        warnings.append("No Incoterm stated: confirm who pays which leg before quoting.")
    return warnings


def _greeting(request: ShipmentRequest) -> str:
    return f"Dear {request.customer_name}," if request.customer_name else "Hello,"


def _summary_lines(request: ShipmentRequest) -> list[str]:
    lines = []
    if request.origin.is_known() or request.destination.is_known():
        lines.append(f"- Route: {request.origin.label()} → {request.destination.label()}")
    if request.mode != TransportMode.UNKNOWN:
        lines.append(f"- Mode: {request.mode.value}")
    if request.cargo_description:
        lines.append(f"- Cargo: {request.cargo_description}")
    if request.pieces:
        lines.append(f"- Pieces: {request.pieces}")
    if request.weight_kg:
        lines.append(f"- Gross weight: {request.weight_kg:,.0f} kg")
    if request.volume_cbm:
        lines.append(f"- Volume: {request.volume_cbm:g} cbm")
    if request.dimensions:
        lines.append(f"- Dimensions: {request.dimensions}")
    if request.ready_date:
        lines.append(f"- Ready date: {request.ready_date}")
    if request.incoterm:
        lines.append(f"- Incoterm: {request.incoterm}")
    return lines


def draft_reply(request: ShipmentRequest, missing: list[str]) -> tuple[str, str]:
    """Return (subject, body) for the customer reply."""
    route = f"{request.origin.label()} → {request.destination.label()}"
    summary = "\n".join(_summary_lines(request)) or "- (no details could be confirmed yet)"

    if missing:
        asks = "\n".join(f"{i}. {item}" for i, item in enumerate(missing, start=1))
        subject = f"Re: Quote request {route} – a few details needed"
        body = (
            f"{_greeting(request)}\n\n"
            "Thank you for your enquiry. Here is what we have so far:\n\n"
            f"{summary}\n\n"
            "To prepare an accurate quote, could you please confirm:\n\n"
            f"{asks}\n\n"
            "As soon as we have these, we will send your rates.\n\n"
            "Best regards"
        )
    else:
        subject = f"Re: Quote request {route}"
        body = (
            f"{_greeting(request)}\n\n"
            "Thank you for your enquiry. We have all the details we need:\n\n"
            f"{summary}\n\n"
            "Our team is preparing your quote and will send it shortly.\n\n"
            "Best regards"
        )
    return subject, body
