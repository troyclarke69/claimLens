"""Value normalisation, so that '$1,240.55' and '1240.55' count as the same answer.

Documents print values in many formats; the ground truth stores one canonical
form. Normalising both the prediction and the label before comparing keeps the
evaluation about *reading the right thing*, not about formatting trivia.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Optional

_DATE_FORMATS = [
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%d/%m/%Y",  # tried after US format; ambiguous dates resolve US-first
    "%B %d, %Y",
    "%b %d, %Y",
    "%b. %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%Y/%m/%d",
    "%m/%d/%y",
]


def _clean(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_date(s: str) -> Optional[str]:
    s = _clean(s).rstrip(".")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def normalize_amount(s: str) -> Optional[str]:
    s = _clean(s)
    s = re.sub(r"(?i)\b(usd|cad|us|ca)\b", "", s)
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    m = re.fullmatch(r"-?\d+(\.\d+)?", s)
    if not m:
        return None
    return f"{float(s):.2f}"


def normalize_id(s: str) -> str:
    """Letters and digits only (evaluator 1.1): 'POL. 214-4955 F' == 'POL-214-4955-F'."""
    return re.sub(r"[^0-9A-Za-z]", "", _clean(s)).upper()


def normalize_text(s: str) -> str:
    s = _clean(s).casefold()
    s = re.sub(r"[.,;:]+$", "", s)
    return s


def normalize_value(field: str, value: Optional[str]) -> Optional[str]:
    """Canonical form of a value for a given field (None stays None)."""
    if value is None:
        return None
    value = str(value)
    if not value.strip():
        return None
    if field == "date_of_loss":
        return normalize_date(value) or normalize_text(value)
    if field == "total_amount":
        return normalize_amount(value) or normalize_text(value)
    if field in ("claim_number", "policy_number"):
        return normalize_id(value)
    return normalize_text(value)


def similarity(a: str, b: str) -> float:
    """Character-level similarity in [0, 1] (1 - normalised edit distance)."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))
