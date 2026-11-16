import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_sample_data import make  # noqa: E402


@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    tx, ids = make(rows=12000, seed=1)
    tx.to_csv(d / "train_transaction.csv", index=False)
    ids.to_csv(d / "train_identity.csv", index=False)
    return d


@pytest.fixture(scope="session")
def trained(sample_dir, tmp_path_factory):
    """Run the whole pipeline once on synthetic data; outputs go to a temp dir."""
    from fraud.pipeline import run

    out = tmp_path_factory.mktemp("run")
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    cfg["paths"] = {"artefacts_dir": str(out / "artefacts"), "reports_dir": str(out / "reports")}
    cfg["model"]["n_estimators"] = 300
    cfg_path = out / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    metrics = run(config_path=cfg_path, data_dir=sample_dir, skip_logistic=True, shap_sample=300)
    return {"metrics": metrics, "out": out}
