import json

import numpy as np
import pandas as pd
import pytest

from spam_detector import ham_expansion as study
from spam_detector.evaluation import SpamModel


def test_ham_multiplier_changes_class_mass_without_breaking_source_balance():
    frame = pd.DataFrame({"label": [0, 0, 0, 1, 1, 1], "source": ["a", "a", "b", "a", "b", "b"],
                          "group_id": ["x", "x", "y", "z", "q", "r"]})
    weighted = frame.assign(weight=study.ham_weights(frame, 4))
    masses = weighted.groupby("label").weight.sum()
    assert masses[0] / masses[1] == pytest.approx(4)
    assert weighted.weight.mean() == pytest.approx(1)
    assert weighted.loc[:1, "weight"].sum() == pytest.approx(weighted.loc[2, "weight"])


@pytest.mark.parametrize("factor", [0, -1, np.nan, np.inf])
def test_ham_multiplier_rejects_invalid_weights(factor):
    with pytest.raises(ValueError):
        study.ham_weights(pd.DataFrame(), factor)


class DevelopmentOnlyPipeline:
    classes_ = [0, 1]

    def fit(self, texts, labels, **kwargs):
        assert all(text.startswith("train") for text in texts)
        assert len(kwargs["clf__sample_weight"]) == len(texts)
        return self

    def decision_function(self, texts):
        assert all(text.startswith("validation") for text in texts)
        return np.array([1.0 if "spam" in text else -1.0 for text in texts])


def test_fit_and_threshold_only_control_never_see_test_messages(tmp_path, monkeypatch):
    rows = []
    for split in ["train", "validation", "test"]:
        for label in [0, 1]:
            key = f"{split}_{label}"
            rows.append({"source": "fixture", "source_row": len(rows), "split": split, "label": label,
                         "text": f"{split} {'spam' if label else 'ham'}", "text_sha256": key, "group_id": key})
    monkeypatch.setattr(study, "load_frozen", lambda: (pd.DataFrame(rows), {}))
    monkeypatch.setattr(study, "candidate_pipeline", lambda *args: DevelopmentOnlyPipeline())
    monkeypatch.setattr(study, "controls", lambda p: {
        name: SpamModel(DevelopmentOnlyPipeline(), 0, name) for name in ["baseline", "previous"]})
    selected = study.fit(tmp_path / "run")
    assert selected["model"] == "ham_word_char_c0.5_w1"
    assert len(pd.read_csv(tmp_path / "run/candidates.csv")) == 6
    assert len(pd.read_csv(tmp_path / "run/validation_scores.csv")) == 14
    assert json.loads((tmp_path / "run/training_metadata.json").read_text())["test_predictions_during_selection"] == 0
    assert not (tmp_path / "run/test_scores.csv").exists()
    with pytest.raises(FileExistsError):
        study.fit(tmp_path / "run")


def test_recall_loss_blocks_promotion_despite_lower_false_positive_rate():
    table = pd.DataFrame({"model": ["candidate", "baseline"], "fpr": [0.001, 0.002]})
    intervals = {name: {"delta_fpr": {"upper": -0.01}, "delta_recall": {"lower": -0.029}}
                 for name in ["baseline", "previous"]}
    assert study.promotion_gate(table, intervals)["passed"]
    intervals["previous"]["delta_recall"]["lower"] = -0.031
    assert not study.promotion_gate(table, intervals)["passed"]


def test_test_cannot_be_rescored_after_inspection(tmp_path):
    (tmp_path / "test_scores.csv").write_text("already inspected")
    with pytest.raises(FileExistsError, match="already evaluated"):
        study.evaluate(tmp_path)
