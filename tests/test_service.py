"""Service, registry and cost tests (CPU only, simulated predictor)."""

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from claimlens.evaluation import parse_prediction  # noqa: E402
from claimlens.generator import generate_dataset  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    generate_dataset(tmp_path / "data", n_train=0, n_val=0, n_test=3, n_fair_sets=0)
    monkeypatch.setenv("CLAIMLENS_PREDICTOR", "simulated")
    monkeypatch.setenv("CLAIMLENS_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("CLAIMLENS_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    from claimlens.service import app
    with TestClient(app) as c:
        c.data_dir = tmp_path / "data"
        yield c


def test_extract_returns_fields_review_and_receipt(client):
    img = client.data_dir / "test" / "images" / "test-00000.jpg"
    r = client.post("/v1/extract", files={"file": ("test-00000.jpg", img.read_bytes(), "image/jpeg")},
                    data={"doc_id": "test-00000"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["fields"]) >= {"claim_number", "total_amount"}
    assert "needs_review" in body["review"] and len(body["audit"]["record_hash"]) == 64
    assert client.get("/v1/audit/verify").json()["ok"]
    assert client.get("/v1/model").json()["name"] == "simulated-predictor"


def test_cors_allows_the_web_demo_only(client):
    ok = client.options("/v1/extract", headers={"Origin": "https://troyclarke69.github.io",
                                                "Access-Control-Request-Method": "POST"})
    assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == "https://troyclarke69.github.io"
    other = client.get("/v1/health", headers={"Origin": "https://example.com"})
    assert "access-control-allow-origin" not in other.headers


def test_extract_requires_doc_id_in_demo_mode(client):
    img = client.data_dir / "test" / "images" / "test-00001.jpg"
    r = client.post("/v1/extract", files={"file": ("x.jpg", img.read_bytes(), "image/jpeg")})
    assert r.status_code == 400


def test_review_flags():
    from claimlens.service import review_flags
    bad = parse_prediction('{"claimant_name": {"value": "Jane Doe", "evidence_text": "John Roe"}}')
    reasons = review_flags(bad)
    assert any("does not match its own evidence" in x for x in reasons)
    assert any("no citation box" in x for x in reasons)
    assert any("total_amount: required field not found" in x for x in reasons)
    assert review_flags(parse_prediction("not json")) == ["model output could not be parsed"]


def _fake_run(runs, run_id, split, fa, ga, hr):
    d = runs / run_id
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({
        "run_id": run_id, "split": split, "evaluator_version": "1.1",
        "summary": {"field_accuracy": fa, "grounded_accuracy": ga, "hallucination_rate": hr, "miss_rate": 0.0,
                    "doc_exact_match": 0.5, "mean_reward": 0.9, "latency_ms_p50": 1000},
        "fairness": {"counterfactual_sets": 0}}))


def test_registry_build_and_release_gate(tmp_path, monkeypatch):
    from claimlens import registry
    monkeypatch.chdir(tmp_path)
    ad = tmp_path / "adapters" / "good"
    ad.mkdir(parents=True)
    (ad / "training_card.json").write_text(json.dumps({"adapter_name": "good", "adapter_sha256": "abc",
                                                        "parent_adapter": {"name": "sft_v1", "sha256": "def"}}))
    runs = tmp_path / "runs"
    for split in ("test", "holdout"):
        _fake_run(runs, f"sft_v1_v2_greedy_{split}", split, 0.90, 0.85, 0.05)
        _fake_run(runs, f"good_v2_greedy_{split}", split, 0.92, 0.86, 0.03)
        _fake_run(runs, f"worse_v2_greedy_{split}", split, 0.80, 0.70, 0.10)
    reg = registry.build()
    assert reg["models"]["good"]["adapters"][0]["name"] == "sft_v1"
    assert registry.promote("sft_v1", "initial")[0]
    ok, rows = registry.promote("worse", "try")
    assert not ok and any(r["result"] == "FAIL" for r in rows)
    assert json.loads(registry.REGISTRY_PATH.read_text())["production"] == "sft_v1"
    assert registry.promote("good", "better on holdout")[0]
    reg = json.loads(registry.REGISTRY_PATH.read_text())
    assert reg["production"] == "good" and len(reg["changelog"]) == 2


def test_cost_from_audit_log(tmp_path):
    from claimlens.costs import run_cost
    d = tmp_path / "r"
    d.mkdir()
    with open(d / "audit.jsonl", "w") as f:
        for _ in range(4):
            f.write(json.dumps({"latency_ms": 36000, "input_tokens": 2000, "output_tokens": 400}) + "\n")
    c = run_cost(d, gpu_hourly=0.5)
    assert c["usd_per_doc"] == pytest.approx(0.005, rel=1e-3)
    c = run_cost(d, in_price=1.0, out_price=5.0)
    assert c["usd_per_doc"] == pytest.approx(0.004, rel=1e-3)
