import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spam_detector import external_eval
from spam_detector.external_data import prepare_cohorts, file_hash
from spam_detector.external_eval import evaluate_model, recall_interval, summarize, load_frozen_model
from test_demo import model_run


def protocol():
    value = json.loads((Path(__file__).resolve().parents[1] / "data/external/imc25/protocol.json").read_text())
    value["bootstrap"]["draws"] = 100
    return value


def cohorts():
    frame = pd.DataFrame([
        dict(text="FREE prize <URL>", language="English", time="2023", scam_type="spam"),
        dict(text="hello meet for lunch", language="English", time="2024", scam_type="others"),
        dict(text="!!!", language="English", time="2024", scam_type="others"),
        dict(text="unknown date cash offer", language="English", time="Today", scam_type="spam"),
    ])
    return prepare_cohorts(frame, ["unrelated reference message"])[0]


def test_group_bootstrap_matches_explicit_cluster_sampling():
    prediction = np.array([1, 1, 1, 0])
    groups = np.array(["a", "a", "a", "b"])
    rng = np.random.default_rng(7)
    samples = []
    for _ in range(200):
        chosen = rng.integers(2, size=2)
        samples.append(np.array([3, 0])[chosen].sum() / np.array([3, 1])[chosen].sum())
    result = recall_interval(prediction, groups, draws=200, seed=7)
    assert result["recall_lower"] == np.quantile(samples, .025)
    assert result["recall_upper"] == np.quantile(samples, .975)
    assert result["n_groups"] == 2
    assert result == recall_interval(prediction, groups, draws=200, seed=7)


@pytest.mark.parametrize("prediction,groups", [([], []), ([1, 0], ["a"]), ([2], ["a"]), ([1], [None])])
def test_invalid_interval_inputs(prediction, groups):
    with pytest.raises(ValueError, match="Expected"):
        recall_interval(prediction, groups)


def test_no_fit_no_dropped_oov_and_no_misleading_one_class_metrics(tmp_path, monkeypatch):
    model, _ = model_run(tmp_path)
    monkeypatch.setattr(model.pipeline, "fit", lambda *a, **k: pytest.fail("Evaluation must never fit"))
    before = model.threshold
    data = cohorts()
    ledger = evaluate_model(model, data)
    assert len(ledger) == 2 * sum(map(len, data.values()))
    assert (ledger.recognized_features == 0).sum() == 4  # OOV kept in both cohorts/views.
    primary = ledger[(ledger.cohort == "recent_2023_2024") & (ledger.view == "published")]
    np.testing.assert_array_equal(primary.validation_prediction, model.predict(data["recent_2023_2024"].text))
    np.testing.assert_array_equal(primary.default_prediction, model.pipeline.predict(data["recent_2023_2024"].text))
    assert model.threshold == before
    metrics, categories = summarize(ledger, data, protocol())
    assert len(metrics) == 8
    assert metrics[list(external_eval.UNAVAILABLE)].isna().all().all()
    assert (metrics.detected + metrics.missed == metrics.n_messages).all()
    assert (metrics.n_ham == 0).all()
    assert np.allclose(metrics.recall + metrics.miss_rate, 1)
    assert categories.groupby(["cohort", "view", "threshold_mode"]).n_messages.sum().tolist() == [4]*4 + [3]*4
    missing = ledger.drop(index=0)
    with pytest.raises(ValueError, match="incomplete"):
        summarize(missing, data, protocol())
    with pytest.raises(ValueError, match="duplicated"):
        summarize(pd.concat([ledger, ledger.iloc[:1]]), data, protocol())


def test_frozen_model_rejects_changed_artifact_before_loading(tmp_path, monkeypatch):
    model, metadata = model_run(tmp_path)
    metadata["source_sha256"] = {}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    frozen = dict(model_id=model.model_id, model_metadata_sha256=file_hash(tmp_path / "metadata.json"),
                  model_sha256=metadata["models"][model.model_id]["artifact_sha256"],
                  validation_threshold=model.threshold)
    assert load_frozen_model(frozen, tmp_path).threshold == model.threshold
    frozen["model_sha256"] = "0" * 64
    monkeypatch.setattr(external_eval, "load_verified_model", lambda _: pytest.fail("Loaded changed model"))
    with pytest.raises(ValueError, match="frozen"):
        load_frozen_model(frozen, tmp_path)


def test_publication_verifier_detects_changed_decisions_even_with_updated_file_hash(tmp_path, monkeypatch):
    from scripts import verify_external

    model, _ = model_run(tmp_path)
    data, config = cohorts(), protocol()
    config["validation_threshold"] = model.threshold
    ledger = evaluate_model(model, data)
    metrics, categories = summarize(ledger, data, config)
    cfg_path = tmp_path / "protocol.json"
    cfg_path.write_text(json.dumps(config))
    for module in [verify_external]:
        monkeypatch.setattr(module, "PROTOCOL", cfg_path)
        monkeypatch.setattr(module, "load_cohorts", lambda: (data, {}))
        monkeypatch.setattr(module, "check_memberships", lambda *a: None)
        monkeypatch.setattr(module, "load_frozen_model", lambda *a: model)
    for name, table in [("predictions", ledger), ("metrics", metrics), ("categories", categories)]:
        table.to_csv(tmp_path / f"{name}.csv", index=False)
    metadata = {k: config[k] for k in ["model_id", "model_sha256", "model_metadata_sha256",
                                      "validation_threshold", "no_fit", "no_threshold_selection"]}
    metadata.update(protocol_sha256=file_hash(cfg_path), source_sha256={},
                    output_sha256={f"{name}.csv": file_hash(tmp_path/f"{name}.csv")
                                   for name in ["predictions", "metrics", "categories"]})
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    assert verify_external.verify(tmp_path, tmp_path)["binary_decisions_checked"] == len(ledger)*2
    ledger.loc[0, "validation_prediction"] = 1 - ledger.loc[0, "validation_prediction"]
    ledger.to_csv(tmp_path / "predictions.csv", index=False)
    metadata["output_sha256"]["predictions.csv"] = file_hash(tmp_path / "predictions.csv")
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(AssertionError):
        verify_external.verify(tmp_path, tmp_path)
