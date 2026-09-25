"""Output schema: what the model must return for every document.

Every extracted field carries its own evidence (the text it read and where it
read it). A field that is not present on the document must be ``null`` -- a
model that invents a value for a missing field is penalised by the evaluator.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The fields we extract, in a fixed order. The descriptions are reused in the
# model prompt so the schema and the prompt can never drift apart.
FIELD_DESCRIPTIONS: dict[str, str] = {
    "claim_number": "The insurer's claim number (e.g. CLM-2025-004817). Not an invoice number.",
    "policy_number": "The insurance policy number.",
    "claimant_name": "Full name of the claimant / policyholder / patient being billed.",
    "date_of_loss": "Date of the incident, or the date of service on an invoice. Not a date of birth, report date or invoice date.",
    "provider_name": "Name of the repair shop, contractor or clinic providing the service.",
    "incident_type": "Type of incident as written on a claim form (e.g. 'Auto collision').",
    "total_amount": "Final total amount claimed or billed (after tax). Not a subtotal or line item.",
}
FIELD_NAMES: list[str] = list(FIELD_DESCRIPTIONS)

DOC_TYPES = ("claim_form", "invoice")


class BBox(BaseModel):
    """Axis-aligned box in pixel coordinates of the original image: [x0, y0, x1, y1]."""

    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_list(cls, xs: list[float]) -> "BBox":
        if len(xs) != 4:
            raise ValueError("bbox must have 4 numbers [x0, y0, x1, y1]")
        x0, y0, x1, y1 = (float(v) for v in xs)
        return cls(x0=min(x0, x1), y0=min(y0, y1), x1=max(x0, x1), y1=max(y0, y1))

    def to_list(self) -> list[float]:
        return [round(self.x0, 1), round(self.y0, 1), round(self.x1, 1), round(self.y1, 1)]


class ExtractedField(BaseModel):
    """One extracted value plus the evidence that supports it."""

    model_config = ConfigDict(extra="ignore")

    value: str
    evidence_text: Optional[str] = None  # verbatim text as it appears on the page
    page: int = 1
    bbox: Optional[list[float]] = None  # [x0, y0, x1, y1]

    @field_validator("value", mode="before")
    @classmethod
    def _coerce_value(cls, v):
        if v is None:
            raise ValueError("value is null -- the whole field should be null instead")
        return str(v)

    @field_validator("bbox", mode="before")
    @classmethod
    def _check_bbox(cls, v):
        if v is None:
            return None
        return BBox.from_list(list(v)).to_list()


class ClaimExtraction(BaseModel):
    """The full model output for one document. Missing fields are null."""

    model_config = ConfigDict(extra="ignore")

    claim_number: Optional[ExtractedField] = None
    policy_number: Optional[ExtractedField] = None
    claimant_name: Optional[ExtractedField] = None
    date_of_loss: Optional[ExtractedField] = None
    provider_name: Optional[ExtractedField] = None
    incident_type: Optional[ExtractedField] = None
    total_amount: Optional[ExtractedField] = None

    def fields(self) -> dict[str, Optional[ExtractedField]]:
        return {name: getattr(self, name) for name in FIELD_NAMES}


def example_output() -> dict:
    """A small example used in the prompt so the model sees the exact format."""
    return {
        "claim_number": {
            "value": "CLM-2025-004817",
            "evidence_text": "CLM-2025-004817",
            "page": 1,
            "bbox": [212, 188, 391, 210],
        },
        "date_of_loss": {
            "value": "2025-03-14",
            "evidence_text": "03/14/2025",
            "page": 1,
            "bbox": [212, 402, 318, 424],
        },
        "provider_name": None,
    }
