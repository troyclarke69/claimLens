"""Synthetic claim document generator.

Produces realistic-looking (but entirely fake) claim forms and provider
invoices as images, together with *exact* ground truth:

* the canonical value of every field (or null when it is not on the page),
* the text exactly as printed and the pixel box where it was printed,
* a layout of every piece of text on the page (useful for grounding checks
  and, later, for a text-only version of the task),
* metadata used for error analysis (handwritten? scan quality? skew?) and for
  fairness testing (the synthetic name group / gender of the claimant).

Because we draw every string ourselves, the labels are perfect -- no human
annotation, no real personal data.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .names import GENDERS, GROUPS, sample_name
from .schema import FIELD_NAMES

GENERATOR_VERSION = "1.0"

# Page size in pixels. Both are multiples of 28 so Qwen2.5-VL does not need to
# resize the image (which keeps predicted pixel boxes in the same coordinates).
PAGE_W, PAGE_H = 980, 1260

INK = (25, 25, 25)
PEN = (22, 45, 140)  # "handwritten" ink colour
GREY = (110, 110, 110)
RULE = (170, 170, 170)

# --------------------------------------------------------------------------
# Fonts: try project-local fonts first, then common system fonts, then
# Pillow's built-in font. The fonts actually used are recorded in
# dataset_info.json, because different fonts => a different dataset.
# --------------------------------------------------------------------------
_ASSET_FONTS = Path(__file__).resolve().parent.parent / "assets" / "fonts"
_FONT_CANDIDATES = {
    "regular": ["DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf"],
    "bold": ["DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "LiberationSans-Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf",
             "/System/Library/Fonts/Supplemental/Arial Bold.ttf"],
    "hand": ["DejaVuSans-Oblique.ttf", "ariali.ttf", "Arial Italic.ttf", "LiberationSans-Italic.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", "C:/Windows/Fonts/ariali.ttf",
             "/System/Library/Fonts/Supplemental/Arial Italic.ttf"],
}
_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
FONTS_USED: dict[str, str] = {}


def get_font(kind: str, size: int):
    key = (kind, size)
    if key in _font_cache:
        return _font_cache[key]
    candidates = []
    local = _ASSET_FONTS / f"{kind}.ttf"
    if local.exists():
        candidates.append(str(local))
    candidates += _FONT_CANDIDATES[kind]
    for cand in candidates:
        try:
            font = ImageFont.truetype(cand, size)
            FONTS_USED[kind] = Path(cand).name
            _font_cache[key] = font
            return font
        except OSError:
            continue
    font = ImageFont.load_default(size=size)  # Pillow >= 10.1
    FONTS_USED[kind] = "pillow-default"
    _font_cache[key] = font
    return font


def _stable_seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:12], 16)


# --------------------------------------------------------------------------
# Content vocabularies (all fictional)
# --------------------------------------------------------------------------
INSURERS = ["Northgate Mutual Insurance", "Maple Shield Insurance Co.", "Harbourline Assurance",
            "Granite Peak Insurance Group"]
INCIDENTS = {
    "auto": ["Auto collision", "Windshield damage", "Vehicle theft", "Hail damage"],
    "property": ["Water damage", "Fire damage", "Wind damage", "Burglary"],
    "health": ["Slip and fall injury", "Physiotherapy treatment", "Dental injury", "Emergency room visit"],
}
DESCRIPTIONS = {
    "Auto collision": "Rear-ended at a red light; damage to rear bumper and trunk lid.",
    "Windshield damage": "Stone chip on highway spread into a long crack across the windshield.",
    "Vehicle theft": "Vehicle taken from driveway overnight; police report filed next morning.",
    "Hail damage": "Severe hail storm caused dents across hood, roof and trunk.",
    "Water damage": "Burst pipe under kitchen sink flooded kitchen and part of basement.",
    "Fire damage": "Small kitchen fire damaged cabinets, range hood and ceiling.",
    "Wind damage": "High winds tore shingles off roof and damaged the rear fence.",
    "Burglary": "Rear door forced open; electronics and jewellery taken.",
    "Slip and fall injury": "Slipped on icy walkway and injured left wrist and lower back.",
    "Physiotherapy treatment": "Ongoing physiotherapy for shoulder strain following accident.",
    "Dental injury": "Fractured front tooth after fall; emergency dental repair required.",
    "Emergency room visit": "Treated in emergency for a deep cut to the hand requiring stitches.",
}
PROVIDERS = {
    "auto": ["Precision Auto Body", "Lakeshore Collision Centre", "Kingsway Auto Glass", "Summit Motor Repair"],
    "property": ["BlueRidge Restoration Ltd.", "DryPro Water Damage Services", "Cornerstone Contractors Inc.",
                 "Allied Home Repair"],
    "health": ["Riverside Physiotherapy Clinic", "Bayview Family Health", "Northern Lights Dental",
               "Crestview Medical Centre"],
}
LINE_ITEMS = {
    "auto": [("Rear bumper replacement", 1, 640.0), ("Paint and refinish", 1, 380.0), ("Labour (hours)", 6, 95.0),
             ("Tail lamp assembly", 1, 215.0), ("Windshield replacement", 1, 520.0), ("Wheel alignment", 1, 129.0)],
    "property": [("Water extraction", 1, 850.0), ("Drywall removal and replacement", 1, 1420.0),
                 ("Dehumidifier rental (days)", 5, 45.0), ("Mould remediation", 1, 1100.0),
                 ("Flooring replacement (sq ft)", 180, 6.5), ("Debris removal", 1, 275.0)],
    "health": [("Initial assessment", 1, 120.0), ("Treatment session", 6, 85.0), ("X-ray imaging", 1, 150.0),
               ("Orthotic brace", 1, 210.0), ("Follow-up consultation", 1, 95.0), ("Prescription dressing kit", 1, 42.5)],
}
STREETS = ["Maple Ave", "King St W", "Lakeshore Blvd", "Queen St E", "Elm Dr", "Bloor St", "Oak Cres", "Main St"]
CITIES = ["Toronto, ON", "Mississauga, ON", "Ottawa, ON", "Hamilton, ON", "Buffalo, NY", "Rochester, NY"]
DATE_FORMATS = ["%m/%d/%Y", "%B %d, %Y", "%Y-%m-%d", "%d %b %Y"]
AMOUNT_FORMATS = ["dollar_comma", "comma", "dollar_plain", "cad"]


def fmt_amount(x: float, style: str) -> str:
    if style == "dollar_comma":
        return f"${x:,.2f}"
    if style == "comma":
        return f"{x:,.2f}"
    if style == "dollar_plain":
        return f"${x:.2f}"
    return f"CAD {x:,.2f}"


# --------------------------------------------------------------------------
# The record: everything about one document *before* it is drawn.
# --------------------------------------------------------------------------
@dataclass
class ClaimRecord:
    doc_id: str
    doc_type: str  # claim_form | invoice
    lob: str  # auto | property | health
    group: str
    gender: str
    first_name: str
    last_name: str
    claim_number: str
    policy_number: str
    invoice_number: str
    insurer: str
    provider_name: Optional[str]
    incident_type: str
    date_of_loss: date
    date_reported: date
    date_of_birth: date
    invoice_date: date
    street: str
    city: str
    phone: str
    line_items: list = field(default_factory=list)
    tax_rate: float = 0.13
    claimed_amount: float = 0.0
    deductible: float = 500.0
    show_claim_ref: bool = True
    show_policy_on_invoice: bool = False
    date_format: str = "%m/%d/%Y"
    amount_format: str = "dollar_comma"
    handwritten: bool = False
    quality: str = "clean"  # clean | moderate | poor
    skew_deg: float = 0.0
    render_seed: int = 0
    set_id: Optional[str] = None  # fairness set this document belongs to

    @property
    def claimant_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    @property
    def subtotal(self) -> float:
        return round(sum(q * p for _, q, p in self.line_items), 2)

    @property
    def tax(self) -> float:
        return round(self.subtotal * self.tax_rate, 2)

    @property
    def invoice_total(self) -> float:
        return round(self.subtotal + self.tax, 2)


def _rand_date(rng: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=rng.randint(0, (end - start).days))


def make_record(doc_id: str, seed: int, doc_type: Optional[str] = None,
                group: Optional[str] = None, gender: Optional[str] = None) -> ClaimRecord:
    rng = random.Random(seed)
    doc_type = doc_type or rng.choice(["claim_form", "invoice"])
    lob = rng.choice(list(INCIDENTS))
    group = group or rng.choice(GROUPS)
    gender = gender or rng.choice(GENDERS)
    first, last = sample_name(rng, group, gender)
    loss = _rand_date(rng, date(2024, 1, 1), date(2025, 12, 31))

    claim_fmt = rng.choice(["CLM-{y}-{n:06d}", "C-{n8:08d}", "{yy}-{n4:04d}-{lob}"])
    claim_number = claim_fmt.format(y=loss.year, yy=str(loss.year)[2:], n=rng.randint(1, 999999),
                                    n8=rng.randint(10**7, 10**8 - 1), n4=rng.randint(1, 9999),
                                    lob={"auto": "AU", "property": "PR", "health": "HL"}[lob])
    policy_fmt = rng.choice(["POL-{a:03d}-{b:04d}-{c}", "NG{n:07d}", "{c}{c2}-{n6:06d}"])
    policy_number = policy_fmt.format(a=rng.randint(100, 999), b=rng.randint(0, 9999),
                                      c=rng.choice("ABCDEFGHJK"), c2=rng.choice("MNPRSTUVWX"),
                                      n=rng.randint(10**6, 10**7 - 1), n6=rng.randint(0, 999999))
    invoice_number = rng.choice(["INV-{n:05d}", "{y}-{n4:04d}", "No. {n:05d}"]).format(
        n=rng.randint(1000, 99999), y=loss.year, n4=rng.randint(1, 9999))

    items = rng.sample(LINE_ITEMS[lob], rng.randint(2, 5))
    items = [(d, q, round(p * rng.uniform(0.8, 1.35), 2)) for d, q, p in items]

    is_form = doc_type == "claim_form"
    quality = rng.choices(["clean", "moderate", "poor"], weights=[0.45, 0.35, 0.20])[0]
    skew_max = {"clean": 0.4, "moderate": 1.5, "poor": 3.0}[quality]

    rec = ClaimRecord(
        doc_id=doc_id, doc_type=doc_type, lob=lob, group=group, gender=gender,
        first_name=first, last_name=last, claim_number=claim_number, policy_number=policy_number,
        invoice_number=invoice_number, insurer=rng.choice(INSURERS),
        provider_name=rng.choice(PROVIDERS[lob]) if (not is_form or rng.random() < 0.7) else None,
        incident_type=rng.choice(INCIDENTS[lob]),
        date_of_loss=loss,
        date_reported=loss + timedelta(days=rng.randint(1, 21)),
        date_of_birth=_rand_date(rng, date(1950, 1, 1), date(2004, 12, 31)),
        invoice_date=loss + timedelta(days=rng.randint(1, 12)),
        street=f"{rng.randint(12, 9800)} {rng.choice(STREETS)}", city=rng.choice(CITIES),
        phone=f"({rng.randint(200, 989)}) {rng.randint(200, 989)}-{rng.randint(0, 9999):04d}",
        line_items=items,
        tax_rate=rng.choice([0.13, 0.13, 0.05, 0.08]),
        claimed_amount=round(rng.uniform(300, 25000), 2),
        deductible=rng.choice([250.0, 500.0, 1000.0]),
        show_claim_ref=rng.random() < 0.8,
        show_policy_on_invoice=rng.random() < 0.3,
        date_format=rng.choice(DATE_FORMATS),
        amount_format=rng.choice(AMOUNT_FORMATS),
        handwritten=is_form and rng.random() < 0.35,
        quality=quality,
        skew_deg=round(rng.uniform(-skew_max, skew_max), 2),
        render_seed=rng.randint(0, 2**31 - 1),
    )
    return rec


def ground_truth_values(rec: ClaimRecord) -> dict[str, Optional[str]]:
    """Canonical value per field (None = not present on this document)."""
    is_form = rec.doc_type == "claim_form"
    vals = {
        "claim_number": rec.claim_number if (is_form or rec.show_claim_ref) else None,
        "policy_number": rec.policy_number if (is_form or rec.show_policy_on_invoice) else None,
        "claimant_name": rec.claimant_name,
        "date_of_loss": rec.date_of_loss.isoformat(),
        "provider_name": rec.provider_name,
        "incident_type": rec.incident_type if is_form else None,
        "total_amount": f"{(rec.claimed_amount if is_form else rec.invoice_total):.2f}",
    }
    # Stored in a readable canonical form (ISO dates, plain 2-dp amounts, names
    # as printed). The evaluator normalises before comparing.
    return vals


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
class _Canvas:
    """Draws text and remembers exactly where every string landed."""

    def __init__(self, rec: ClaimRecord):
        self.rec = rec
        self.img = Image.new("RGB", (PAGE_W, PAGE_H), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.img)
        self.layout: list[dict] = []
        self.fields: dict[str, dict] = {}

    def _rng(self, key: str) -> random.Random:
        # One RNG per text item, so changing one string (e.g. the name in a
        # fairness variant) does not change the random jitter of any other.
        return random.Random(_stable_seed(self.rec.render_seed, key))

    def text(self, xy, s: str, size=18, kind="regular", fill=INK, role="other",
             field_name: Optional[str] = None, hand: bool = False, key: Optional[str] = None):
        key = key or f"{role}:{field_name}:{xy}"
        if hand:
            box = self._hand_text(xy, s, size, key)
        else:
            font = get_font(kind, size)
            self.draw.text(xy, s, font=font, fill=fill)
            box = list(self.draw.textbbox(xy, s, font=font))
        seg = {"text": s, "bbox": box, "role": role, "field": field_name}
        self.layout.append(seg)
        if field_name and role == "value":
            self.fields[field_name] = {"evidence_text": s, "bbox": box}
        return box

    def _hand_text(self, xy, s, size, key):
        rng = self._rng(key)
        x, y = xy
        x0 = y0 = math.inf
        x1 = y1 = -math.inf
        for ch in s:
            sz = size + rng.randint(-1, 1)
            font = get_font("hand", sz)
            dy = rng.uniform(-1.2, 1.2)
            self.draw.text((x, y + dy), ch, font=font, fill=PEN)
            if ch.strip():
                bx = self.draw.textbbox((x, y + dy), ch, font=font)
                x0, y0, x1, y1 = min(x0, bx[0]), min(y0, bx[1]), max(x1, bx[2]), max(y1, bx[3])
            x += self.draw.textlength(ch, font=font) + rng.uniform(-0.3, 1.2)
        return [x0, y0, x1, y1]

    def hline(self, x0, x1, y, fill=RULE, width=1):
        self.draw.line([(x0, y), (x1, y)], fill=fill, width=width)

    def box(self, xy, outline=RULE, width=1):
        self.draw.rectangle(xy, outline=outline, width=width)


def _labelled(c: _Canvas, x, y, label, value, field_name=None, hand=False, value_dx=175, size=18,
              role_if_no_field="distractor", distractor_of=None):
    c.text((x, y), label, size=16, kind="bold", role="label", key=f"lbl:{label}")
    vx = x + value_dx
    role = "value" if field_name else role_if_no_field
    fname = field_name or distractor_of
    c.text((vx, y - 1), value, size=size, role=role, field_name=fname, hand=hand,
           key=f"val:{label}")
    c.hline(vx - 4, vx + 250, y + 24)


def _wrap(text: str, font, max_w: int, draw) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _render_claim_form(rec: ClaimRecord, c: _Canvas):
    d = rec.date_format
    hw = rec.handwritten
    c.text((60, 50), rec.insurer, size=28, kind="bold", role="header")
    c.text((60, 92), f"PROOF OF LOSS / CLAIM FORM - {rec.lob.upper()}", size=18, kind="bold", fill=GREY,
           role="header")
    c.text((690, 56), "Form CL-100 Rev. 03/2024", size=13, fill=GREY, role="other")
    c.hline(60, 920, 128, fill=INK, width=2)

    y = 160
    c.text((60, y), "1. POLICYHOLDER INFORMATION", size=17, kind="bold", role="header")
    y += 42
    c.text((60, y), "Policyholder name:", size=16, kind="bold", role="label")
    c.text((248, y - 1), rec.claimant_name, size=19, role="value", field_name="claimant_name", hand=hw,
           key="val:name")
    c.hline(244, 700, y + 24)
    y += 46
    _labelled(c, 60, y, "Policy number:", rec.policy_number, "policy_number", hw)
    _labelled(c, 520, y, "Date of birth:", rec.date_of_birth.strftime(d), None, hw, value_dx=140,
              distractor_of="date_of_loss")
    y += 46
    _labelled(c, 60, y, "Phone:", rec.phone, None, hw, role_if_no_field="other")
    c.text((520, y), "Address:", size=16, kind="bold", role="label")
    c.text((660, y - 1), rec.street, size=17, role="other", hand=hw, key="val:street")
    c.text((660, y + 24), rec.city, size=17, role="other", hand=hw, key="val:city")

    y += 90
    c.text((60, y), "2. CLAIM DETAILS", size=17, kind="bold", role="header")
    y += 42
    _labelled(c, 60, y, "Claim number:", rec.claim_number, "claim_number", hw)
    _labelled(c, 520, y, "Date reported:", rec.date_reported.strftime(d), None, hw, value_dx=140,
              distractor_of="date_of_loss")
    y += 46
    _labelled(c, 60, y, "Date of loss:", rec.date_of_loss.strftime(d), "date_of_loss", hw)
    _labelled(c, 520, y, "Incident type:", rec.incident_type, "incident_type", hw, value_dx=140)
    y += 50
    c.text((60, y), "Description of loss:", size=16, kind="bold", role="label")
    y += 30
    font = get_font("hand" if hw else "regular", 17)
    for i, line in enumerate(_wrap(DESCRIPTIONS[rec.incident_type], font, 820, c.draw)):
        c.text((80, y), line, size=17, role="other", hand=hw, key=f"desc:{i}")
        c.hline(76, 920, y + 24)
        y += 32

    y += 36
    c.text((60, y), "3. SERVICE PROVIDER / REPAIR FACILITY", size=17, kind="bold", role="header")
    y += 42
    c.text((60, y), "Provider name:", size=16, kind="bold", role="label")
    if rec.provider_name:
        c.text((248, y - 1), rec.provider_name, size=18, role="value", field_name="provider_name", hand=hw,
               key="val:provider")
    c.hline(244, 700, y + 24)

    y += 80
    c.text((60, y), "4. AMOUNT CLAIMED", size=17, kind="bold", role="header")
    y += 42
    _labelled(c, 60, y, "Deductible:", fmt_amount(rec.deductible, rec.amount_format), None, hw,
              distractor_of="total_amount")
    _labelled(c, 520, y, "Total claimed:", fmt_amount(rec.claimed_amount, rec.amount_format), "total_amount",
              hw, value_dx=140)

    y += 110
    c.text((60, y), "I declare that the information above is true and complete.", size=15, fill=GREY,
           role="other")
    y += 60
    c.text((60, y), "Signature:", size=16, kind="bold", role="label")
    rng = c._rng("signature")
    pts = [(200 + i * 12, y + 12 + rng.uniform(-9, 9)) for i in range(22)]
    c.draw.line(pts, fill=PEN, width=2)
    c.hline(196, 500, y + 26)
    c.text((560, y), "Date:", size=16, kind="bold", role="label")
    c.text((630, y - 1), rec.date_reported.strftime(d), size=17, role="distractor", field_name="date_of_loss",
           hand=hw, key="val:sigdate")


def _render_invoice(rec: ClaimRecord, c: _Canvas):
    d = rec.date_format
    af = rec.amount_format
    c.text((60, 50), rec.provider_name, size=28, kind="bold", role="value", field_name="provider_name")
    c.text((60, 92), f"{rec._prov_street}", size=15, fill=GREY, role="other")
    c.text((60, 114), f"{rec._prov_city}  |  Tel {rec._prov_phone}", size=15, fill=GREY, role="other")
    c.text((720, 48), "INVOICE", size=34, kind="bold", role="header")
    c.hline(60, 920, 150, fill=INK, width=2)

    y = 175
    c.text((60, y), "BILL TO", size=15, kind="bold", fill=GREY, role="label")
    c.text((60, y + 26), rec.claimant_name, size=19, role="value", field_name="claimant_name", key="val:name")
    c.text((60, y + 52), rec.street, size=16, role="other")
    c.text((60, y + 74), rec.city, size=16, role="other")

    rx = 560
    c.text((rx, y), "Invoice #:", size=16, kind="bold", role="label")
    c.text((rx + 150, y), rec.invoice_number, size=16, role="distractor", field_name="claim_number",
           key="val:invno")
    c.text((rx, y + 28), "Invoice date:", size=16, kind="bold", role="label")
    c.text((rx + 150, y + 28), rec.invoice_date.strftime(d), size=16, role="distractor",
           field_name="date_of_loss", key="val:invdate")
    c.text((rx, y + 56), "Date of service:", size=16, kind="bold", role="label")
    c.text((rx + 150, y + 56), rec.date_of_loss.strftime(d), size=16, role="value", field_name="date_of_loss",
           key="val:dos")
    ry = y + 84
    if rec.show_claim_ref:
        c.text((rx, ry), "Claim ref:", size=16, kind="bold", role="label")
        c.text((rx + 150, ry), rec.claim_number, size=16, role="value", field_name="claim_number",
               key="val:claim")
        ry += 28
    if rec.show_policy_on_invoice:
        c.text((rx, ry), "Policy #:", size=16, kind="bold", role="label")
        c.text((rx + 150, ry), rec.policy_number, size=16, role="value", field_name="policy_number",
               key="val:policy")

    y = 340
    c.draw.rectangle([60, y, 920, y + 34], fill=(236, 236, 236))
    for x, h in [(72, "Description"), (560, "Qty"), (650, "Unit price"), (800, "Amount")]:
        c.text((x, y + 8), h, size=15, kind="bold", role="label")
    y += 46
    for i, (desc, q, p) in enumerate(rec.line_items):
        c.text((72, y), desc, size=16, role="other", key=f"li:{i}:d")
        c.text((560, y), str(q), size=16, role="other", key=f"li:{i}:q")
        c.text((650, y), fmt_amount(p, af), size=16, role="other", key=f"li:{i}:p")
        c.text((800, y), fmt_amount(q * p, af), size=16, role="distractor", field_name="total_amount",
               key=f"li:{i}:a")
        y += 34
        c.hline(60, 920, y - 8)

    y += 24
    c.text((620, y), "Subtotal:", size=16, kind="bold", role="label")
    c.text((800, y), fmt_amount(rec.subtotal, af), size=16, role="distractor", field_name="total_amount",
           key="val:subtotal")
    y += 30
    c.text((620, y), f"Tax ({rec.tax_rate * 100:.0f}%):", size=16, kind="bold", role="label")
    c.text((800, y), fmt_amount(rec.tax, af), size=16, role="distractor", field_name="total_amount",
           key="val:tax")
    y += 36
    c.hline(610, 920, y - 6, fill=INK, width=2)
    c.text((620, y), "TOTAL:", size=19, kind="bold", role="label")
    c.text((800, y), fmt_amount(rec.invoice_total, af), size=19, kind="bold", role="value",
           field_name="total_amount", key="val:total")

    y += 120
    c.text((60, y), "Payment due within 30 days. Please reference the invoice number with payment.", size=14,
           fill=GREY, role="other")
    c.text((60, y + 24), "Thank you for your business.", size=14, fill=GREY, role="other")


# ---- degradation ("make it look scanned") --------------------------------
def _rotate_point(x, y, cx, cy, deg):
    t = math.radians(deg)
    dx, dy = x - cx, y - cy
    # PIL rotates counter-clockwise on screen (y axis points down).
    return cx + dx * math.cos(t) + dy * math.sin(t), cy - dx * math.sin(t) + dy * math.cos(t)


def rotate_bbox(box, deg, w=PAGE_W, h=PAGE_H):
    if not deg:
        return box
    cx, cy = w / 2, h / 2
    x0, y0, x1, y1 = box
    pts = [_rotate_point(x, y, cx, cy, deg) for x, y in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [max(0.0, min(xs)), max(0.0, min(ys)), min(float(w), max(xs)), min(float(h), max(ys))]


def degrade(img: Image.Image, rec: ClaimRecord) -> Image.Image:
    rng = np.random.default_rng(_stable_seed(rec.render_seed, "degrade"))
    if rec.skew_deg:
        img = img.rotate(rec.skew_deg, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    if rec.quality == "clean":
        return img
    if rec.quality == "poor":
        small = img.resize((int(PAGE_W * 0.55), int(PAGE_H * 0.55)), Image.BILINEAR)
        img = small.resize((PAGE_W, PAGE_H), Image.BILINEAR)
    img = img.filter(ImageFilter.GaussianBlur({"moderate": 0.5, "poor": 0.9}[rec.quality]))
    arr = np.asarray(img).astype(np.float32)
    sigma = {"moderate": 7.0, "poor": 16.0}[rec.quality]
    arr += rng.normal(0, sigma, arr.shape[:2])[..., None]
    if rec.quality == "poor":
        arr = arr * 0.88 + 18  # washed-out contrast, greyish paper
        speck = rng.random(arr.shape[:2]) < 0.0015
        arr[speck] = 30
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def render(rec: ClaimRecord) -> tuple[Image.Image, dict]:
    """Draw the document; return (image, label dict)."""
    c = _Canvas(rec)
    if rec.doc_type == "claim_form":
        _render_claim_form(rec, c)
    else:
        prng = random.Random(_stable_seed(rec.render_seed, "provider-address"))
        rec._prov_street = f"{prng.randint(10, 999)} {prng.choice(STREETS)}"
        rec._prov_city = prng.choice(CITIES)
        rec._prov_phone = f"({prng.randint(200, 989)}) {prng.randint(200, 989)}-{prng.randint(0, 9999):04d}"
        _render_invoice(rec, c)
    img = degrade(c.img, rec)

    for seg in c.layout:
        seg["bbox"] = [round(v, 1) for v in rotate_bbox(seg["bbox"], rec.skew_deg)]

    gold = ground_truth_values(rec)
    fields: dict[str, Optional[dict]] = {}
    for name in FIELD_NAMES:
        if gold[name] is None:
            fields[name] = None
            continue
        drawn = c.fields[name]
        fields[name] = {
            "value": gold[name],
            "evidence_text": drawn["evidence_text"],
            "page": 1,
            "bbox": [round(v, 1) for v in rotate_bbox(drawn["bbox"], rec.skew_deg)],
        }

    meta = {
        "line_of_business": rec.lob,
        "group": rec.group,
        "gender": rec.gender,
        "handwritten": rec.handwritten,
        "quality": rec.quality,
        "skew_deg": rec.skew_deg,
        "date_format": rec.date_format,
        "amount_format": rec.amount_format,
        "set_id": rec.set_id,
        "generator_version": GENERATOR_VERSION,
    }
    label = {"doc_id": rec.doc_id, "doc_type": rec.doc_type, "width": PAGE_W, "height": PAGE_H,
             "fields": fields, "layout": c.layout, "meta": meta}
    return img, label


# --------------------------------------------------------------------------
# Dataset writer
# --------------------------------------------------------------------------
def _write_doc(split_dir: Path, rec: ClaimRecord) -> dict:
    img, label = render(rec)
    img_rel = f"images/{rec.doc_id}.jpg"
    lbl_rel = f"labels/{rec.doc_id}.json"
    img.save(split_dir / img_rel, quality=88)
    label["image"] = img_rel
    (split_dir / lbl_rel).write_text(json.dumps(label, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"doc_id": rec.doc_id, "doc_type": rec.doc_type, "image": img_rel, "label": lbl_rel, **label["meta"]}


def _sha256_dir(split_dir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted((split_dir / "labels").glob("*.json")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def generate_split(out: Path, split: str, n: int, seed: int) -> list[dict]:
    split_dir = out / split
    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "labels").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        rec = make_record(f"{split}-{i:05d}", _stable_seed(seed, split, i))
        rows.append(_write_doc(split_dir, rec))
    _write_manifest(split_dir, rows)
    return rows


def generate_fairness_split(out: Path, n_sets: int, variants: int, seed: int) -> list[dict]:
    """Counterfactual sets: identical documents except for the claimant's name.

    Each set picks `variants` different (group, gender) combinations and renders
    the same underlying document once per combination.
    """
    split_dir = out / "fairness"
    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "labels").mkdir(parents=True, exist_ok=True)
    combos = [(g, s) for g in GROUPS for s in GENDERS]
    rows = []
    for i in range(n_sets):
        base = make_record(f"fair-{i:04d}", _stable_seed(seed, "fairness", i))
        rng = random.Random(_stable_seed(seed, "fairness-combos", i))
        for group, gender in rng.sample(combos, min(variants, len(combos))):
            rec = copy.deepcopy(base)
            rec.group, rec.gender = group, gender
            rec.first_name, rec.last_name = sample_name(rng, group, gender)
            rec.doc_id = f"fair-{i:04d}-{group}-{gender}"
            rec.set_id = f"fair-{i:04d}"
            rows.append(_write_doc(split_dir, rec))
    _write_manifest(split_dir, rows)
    return rows


def _write_manifest(split_dir: Path, rows: list[dict]):
    with open(split_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def generate_dataset(out: str | Path, n_train=200, n_val=50, n_test=100, n_fair_sets=25, fair_variants=4,
                     seed=1234) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        if n > 0:
            counts[split] = len(generate_split(out, split, n, seed))
    if n_fair_sets > 0:
        counts["fairness"] = len(generate_fairness_split(out, n_fair_sets, fair_variants, seed))
    info = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "counts": counts,
        "page_size": [PAGE_W, PAGE_H],
        "fonts_used": dict(FONTS_USED),
        "split_hashes": {s: _sha256_dir(out / s) for s in counts},
        "fields": FIELD_NAMES,
    }
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
