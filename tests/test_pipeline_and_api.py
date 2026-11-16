import json

import pytest
from fastapi.testclient import TestClient


def test_pipeline_writes_outputs(trained):
    out = trained["out"]
    for f in ["artefacts/model.joblib", "reports/metrics.json", "reports/results.md",
              "reports/figures/cost_vs_threshold.png", "reports/figures/shap_summary.png"]:
        assert (out / f).exists(), f
    m = json.loads((out / "reports/metrics.json").read_text())
    assert m["rows"]["train"] > m["rows"]["validation"] > 0


def test_model_beats_chance_and_is_tuned_on_validation(trained):
    m = trained["metrics"]
    base_rate = m["test_raw"]["model_tuned"]["tp"] + m["test_raw"]["model_tuned"]["fn"]
    base_rate /= m["test_raw"]["model_tuned"]["transactions"]
    assert m["ranking"]["model_all"]["pr_auc"] > base_rate * 1.5
    assert 0 < m["model_threshold"] < 1
    assert 1 <= m["rules_k"] <= 7


@pytest.fixture
def client(trained, monkeypatch):
    import api.app as app_module

    monkeypatch.setattr(app_module, "MODEL_PATH", trained["out"] / "artefacts" / "model.joblib")
    app_module.load_artefact.cache_clear()
    yield TestClient(app_module.app)
    app_module.load_artefact.cache_clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_score_returns_decision_reasons_and_rules(client):
    tx = {"TransactionAmt": 1500.0, "ProductCD": "C", "P_emaildomain": "protonmail.com",
          "R_emaildomain": "gmail.com", "TransactionDT": 3600, "C1": 40, "D1": 0}
    r = client.post("/score", json={"transaction": tx})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["fraud_score"] <= 1
    assert body["decision"] in {"BLOCK", "APPROVE"}
    assert {"high_value", "anonymous_email", "email_mismatch"} <= set(body["rules_fired"])
    assert body["rules_decision"] == "BLOCK"


def test_score_handles_sparse_input(client):
    r = client.post("/score", json={"transaction": {"TransactionAmt": 12.5}})
    assert r.status_code == 200


def test_missing_model_gives_503(monkeypatch, tmp_path):
    import api.app as app_module

    monkeypatch.setattr(app_module, "MODEL_PATH", tmp_path / "none.joblib")
    app_module.load_artefact.cache_clear()
    assert TestClient(app_module.app).get("/health").status_code == 503
    app_module.load_artefact.cache_clear()
