"""The extraction prompt. Versioned and hashed so every prediction in the audit
log can be traced back to the exact instructions the model was given.

Old versions are kept (never edited) so earlier runs stay reproducible.

v1 -> v2 (after the first Kaggle baseline): Qwen2.5-VL-3B often stopped in the
middle of its JSON (it emitted its end-of-turn token mid-bbox, no numerical
problem) and sometimes wrapped the object in a list. v2 shows a compact
example with every key present and one field per line, which is also ~40%
fewer output tokens.
"""

from __future__ import annotations

import hashlib
import json

from .schema import FIELD_DESCRIPTIONS, FIELD_NAMES, example_output

PROMPT_VERSION = "v2"


def _v1(width: int, height: int) -> str:
    fields = "\n".join(f"- {name}: {desc}" for name, desc in FIELD_DESCRIPTIONS.items())
    return f"""You are extracting information from a scanned insurance claim document.
The image is {width} x {height} pixels.

Return ONLY a JSON object with exactly these keys:
{fields}

For each key, return either null (if the information is NOT on the document) or an object:
  {{"value": ..., "evidence_text": ..., "page": 1, "bbox": [x0, y0, x1, y1]}}
where
  - "value" is the normalised value: dates as YYYY-MM-DD, amounts as a plain number like 1240.55,
    other text exactly as written;
  - "evidence_text" is the text copied exactly as it appears on the page;
  - "bbox" is the pixel box around that text (top-left x0,y0 and bottom-right x1,y1).

Never guess. If a field is not on the document, use null.

Example of the format (values are illustrative only):
{json.dumps(example_output(), indent=1)}
"""


_V2_EXAMPLE = {
    "claim_number": {"value": "CLM-2025-004817", "evidence_text": "CLM-2025-004817", "page": 1,
                     "bbox": [212, 188, 391, 210]},
    "claimant_name": {"value": "Jane Doe", "evidence_text": "Jane Doe", "page": 1, "bbox": [248, 142, 340, 164]},
    "date_of_loss": {"value": "2025-03-14", "evidence_text": "03/14/2025", "page": 1, "bbox": [212, 402, 318, 424]},
    "total_amount": {"value": "1240.55", "evidence_text": "$1,240.55", "page": 1, "bbox": [800, 610, 890, 632]},
}


def _compact_example() -> str:
    ex = _V2_EXAMPLE
    lines = []
    for name in FIELD_NAMES:
        v = ex.get(name)
        lines.append(f'  "{name}": {json.dumps(v, separators=(", ", ": ")) if v else "null"}')
    return "{\n" + ",\n".join(lines) + "\n}"


def _v2(width: int, height: int) -> str:
    fields = "\n".join(f"- {name}: {desc}" for name, desc in FIELD_DESCRIPTIONS.items())
    return f"""You are extracting information from a scanned insurance claim document.
The image is {width} x {height} pixels.

Fields to extract:
{fields}

Output rules:
1. Return ONE JSON object (not a list) with ALL {len(FIELD_NAMES)} keys above, in that order.
2. Each key is either null (the information is NOT on the document) or
   {{"value": ..., "evidence_text": ..., "page": 1, "bbox": [x0, y0, x1, y1]}}
   - value: dates as YYYY-MM-DD, amounts as a plain number like 1240.55, other text exactly as written
   - evidence_text: the text copied exactly as printed on the page
   - bbox: pixel box around the VALUE itself (not its label), top-left x0,y0 to bottom-right x1,y1
3. Write each field on ONE line, exactly like the example. Finish the whole object.
4. Never guess. If a field is not on the document, use null.

Example (values are illustrative only):
{_compact_example()}
"""


_BUILDERS = {"v1": _v1, "v2": _v2}


def build_prompt(width: int, height: int, version: str = PROMPT_VERSION) -> str:
    return _BUILDERS[version](width, height)


def prompt_hash(width: int = 980, height: int = 1260, version: str = PROMPT_VERSION) -> str:
    return hashlib.sha256(build_prompt(width, height, version).encode()).hexdigest()[:16]
