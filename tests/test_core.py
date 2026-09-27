"""Run with:  python -m pytest -q"""

import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from claimlens import audit
from claimlens.evaluation import compute_reward, iou, parse_prediction, score_document
from claimlens.generator import PAGE_H, PAGE_W, make_record, render, rotate_bbox
from claimlens.normalize import normalize_value
from claimlens.predictors import _rescale_bboxes
from claimlens.schema import FIELD_NAMES


# ---------------------------------------------------------------- normalisation
@pytest.mark.parametrize("raw,expected", [
    ("03/14/2025", "2025-03-14"), ("March 14, 2025", "2025-03-14"),
    ("14 Mar 2025", "2025-03-14"), ("2025-03-14", "2025-03-14"),
])
def test_dates(raw, expected):
    assert normalize_value("date_of_loss", raw) == expected


@pytest.mark.parametrize("raw", ["$1,240.55", "1,240.55", "1240.55", "CAD 1,240.55", "$1240.5500"])
def test_amounts(raw):
    assert normalize_value("total_amount", raw) == "1240.55"


# ---------------------------------------------------------------- generator
def test_rotated_bbox_contains_ink():
    """The rotated label box must still cover the drawn pixels."""
    img = Image.new("L", (PAGE_W, PAGE_H), 255)
    box = [300, 400, 520, 430]
    ImageDraw.Draw(img).rectangle(box, fill=0)
    deg = 2.7
    rot = np.array(img.rotate(deg, fillcolor=255))
    ys, xs = np.where(rot < 128)
    ink = [xs.min(), ys.min(), xs.max(), ys.max()]
    pred = rotate_bbox(box, deg)
    assert iou(pred, ink) > 0.9


def test_render_labels_are_consistent():
    for i, doc_type in enumerate(["claim_form", "invoice", "claim_form_b", "invoice_b"]):
        rec = make_record(f"t{i}", 42 + i, doc_type=doc_type)
        img, label = render(rec)
        assert img.size == (PAGE_W, PAGE_H)
        assert set(label["fields"]) == set(FIELD_NAMES)
        for name, f in label["fields"].items():
            if f is None:
                continue
            x0, y0, x1, y1 = f["bbox"]
            assert 0 <= x0 < x1 <= PAGE_W and 0 <= y0 < y1 <= PAGE_H
            assert normalize_value(name, f["evidence_text"]) == normalize_value(name, f["value"])


def test_generation_is_deterministic():
    a, b = make_record("d", 7), make_record("d", 7)
    assert render(a)[1] == render(b)[1]


# ---------------------------------------------------------------- evaluation
def _perfect_output(label):
    return json.dumps({k: v for k, v in label["fields"].items()})


def test_perfect_prediction_scores_one():
    _, label = render(make_record("p", 3, doc_type="invoice"))
    res = score_document(_perfect_output(label), label)
    assert res["all_correct"] and res["field_accuracy"] == 1.0
    assert compute_reward(_perfect_output(label), label) == pytest.approx(1.0)


def test_garbage_scores_zero():
    _, label = render(make_record("g", 3))
    assert compute_reward("I cannot read this document.", label) == 0.0
    assert parse_prediction("{not json")["json_valid"] is False


def test_subtotal_is_flagged_as_distractor():
    _, label = render(make_record("s", 11, doc_type="invoice"))
    # Amount distractors are drawn in order: line items..., subtotal, tax.
    subtotal_seg = [s for s in label["layout"] if s["role"] == "distractor" and s["field"] == "total_amount"][-2]
    out = json.loads(_perfect_output(label))
    out["total_amount"] = {"value": subtotal_seg["text"], "evidence_text": subtotal_seg["text"], "page": 1,
                           "bbox": subtotal_seg["bbox"]}
    res = score_document(json.dumps(out), label)
    assert res["fields"]["total_amount"]["error_type"] == "distractor_confusion"


def test_hallucination_detected():
    _, label = render(make_record("h", 5, doc_type="invoice"))
    assert label["fields"]["incident_type"] is None
    out = json.loads(_perfect_output(label))
    out["incident_type"] = {"value": "Fire damage", "evidence_text": "Fire damage", "page": 1, "bbox": [0, 0, 5, 5]}
    res = score_document(json.dumps(out), label)
    assert res["fields"]["incident_type"]["error_type"] == "hallucinated"


def test_wrong_citation_lowers_reward():
    _, label = render(make_record("c", 9, doc_type="claim_form"))
    good = _perfect_output(label)
    out = json.loads(good)
    for v in out.values():
        if v:
            v["bbox"] = [v["bbox"][0], v["bbox"][1] + 200, v["bbox"][2], v["bbox"][3] + 200]
    assert compute_reward(json.dumps(out), label) < compute_reward(good, label)


def test_bbox_rescale():
    raw = '```json\n{"total_amount": {"value": "10.00", "bbox": [10, 20, 30, 40]}, "claim_number": null}\n```'
    obj = json.loads(_rescale_bboxes(raw, 2.0, 0.5))
    assert obj["total_amount"]["bbox"] == [20.0, 10.0, 60.0, 20.0]


# ---------------------------------------------------------------- audit log
def test_audit_chain_detects_tampering(tmp_path):
    log = audit.AuditLog(tmp_path / "a.jsonl")
    for i in range(3):
        log.append({"doc_id": str(i), "raw_output": f"out{i}"})
    assert audit.verify(tmp_path / "a.jsonl")[0]
    lines = (tmp_path / "a.jsonl").read_text().splitlines()
    rec = json.loads(lines[1]); rec["raw_output"] = "edited"; lines[1] = json.dumps(rec)
    (tmp_path / "a.jsonl").write_text("\n".join(lines) + "\n")
    ok, msg = audit.verify(tmp_path / "a.jsonl")
    assert not ok and "line 2" in msg


def test_finish_json_blocks_early_stop():
    torch = pytest.importorskip("torch")
    from claimlens.predictors import _FinishJson

    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(int(i)) for i in ids)

    proc = _FinishJson(Tok(), prompt_len=0, eos_ids=[0])
    open_json = torch.tensor([[ord(c) for c in '{"a": [1,']])
    closed_json = torch.tensor([[ord(c) for c in '{"a": [1]}']])
    assert proc(open_json, torch.zeros(1, 5))[0, 0] == float("-inf")
    assert proc(closed_json, torch.zeros(1, 5))[0, 0] == 0


def test_ids_ignore_punctuation():
    assert normalize_value("policy_number", "POL. 214-4955 F") == normalize_value("policy_number", "POL-214-4955-F")
    assert normalize_value("claim_number", "C-60006147") != normalize_value("claim_number", "C-69686147")


def test_invoice_number_for_absent_claim_is_distractor():
    for seed in range(50):
        rec = make_record("inv", seed, doc_type="invoice")
        if not rec.show_claim_ref:
            break
    _, label = render(rec)
    assert label["fields"]["claim_number"] is None
    inv = next(s for s in label["layout"] if s["role"] == "distractor" and s["field"] == "claim_number")
    out = json.loads(_perfect_output(label))
    out["claim_number"] = {"value": inv["text"], "evidence_text": inv["text"], "page": 1, "bbox": inv["bbox"]}
    res = score_document(json.dumps(out), label)
    assert res["fields"]["claim_number"]["error_type"] == "distractor_for_absent"


def test_sft_targets_and_oversampling(tmp_path):
    from claimlens.generator import generate_dataset
    from claimlens.sft import build_examples, describe_examples

    generate_dataset(tmp_path, n_train=40, n_val=0, n_test=0, n_fair_sets=0)
    ex = build_examples(tmp_path, "train", poor_share=0.35)
    d = describe_examples(ex)
    assert d["unique_docs"] == 40 and 0.3 <= d["poor_share"] <= 0.4
    # every target is valid JSON in the schema, scores perfectly, and has explicit nulls
    label = json.loads((tmp_path / "train" / "labels" / f"{ex[0]['doc_id'].split('#')[0]}.json").read_text())
    res = score_document(ex[0]["target"], label)
    assert res["schema_valid"] and res["all_correct"]
    assert all(line.count('"') for line in ex[0]["target"].splitlines()[1:-1])  # one field per line


def test_holdout_split_uses_only_unseen_templates(tmp_path):
    from claimlens.generator import HOLDOUT_TYPES, generate_dataset
    info = generate_dataset(tmp_path, n_train=10, n_val=0, n_test=0, n_fair_sets=0, n_holdout=6)
    rows = [json.loads(l) for l in (tmp_path / "holdout" / "manifest.jsonl").read_text().splitlines()]
    assert {r["doc_type"] for r in rows} == set(HOLDOUT_TYPES)
    train = [json.loads(l) for l in (tmp_path / "train" / "manifest.jsonl").read_text().splitlines()]
    assert not {r["doc_type"] for r in train} & set(HOLDOUT_TYPES)
    for r in rows:  # perfect answers score perfectly on the new layouts too
        label = json.loads((tmp_path / "holdout" / r["label"]).read_text())
        assert score_document(_perfect_output(label), label)["all_correct"]


def test_randomised_templates_are_consistent_and_avoid_holdout_wording():
    from claimlens.generator import SYN
    holdout_words = ["our reference", "nature of loss", "insured / claimant", "loss date", "account no",
                     "balance due", "statement date", "treatment / service date", "amount of claim", "policy excess"]
    for syns in SYN.values():
        for w in syns:
            assert not any(h == w.lower() or h in w.lower() for h in holdout_words), w
    for i in range(10):
        _, label = render(make_record(f"r{i}", 700 + i, doc_type=["claim_form_r", "invoice_r"][i % 2]))
        assert score_document(_perfect_output(label), label)["all_correct"]


def test_costed_reward_orders_behaviours_sensibly():
    from claimlens.evaluation import costed_reward
    _, label = render(make_record("cr", 21, doc_type="claim_form"))
    perfect = json.loads(_perfect_output(label))
    miss = dict(perfect, claim_number=None)
    wrong = dict(perfect, claim_number=dict(perfect["claim_number"], value="C-00000000"))
    all_null = {k: None for k in perfect}
    r = lambda o: costed_reward(json.dumps(o), label)["reward"]
    assert r(perfect) > r(miss) > r(wrong)      # an honest null beats a wrong value
    assert r(all_null) < 0 < r(perfect)          # "say null everywhere" is not a winning strategy
    assert costed_reward("garbage", label)["reward"] == -1.0
