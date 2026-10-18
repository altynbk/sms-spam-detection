import json

import numpy as np
import pandas as pd
import pytest

from spam_detector import adaptation


def test_source_and_class_balance_preserves_equal_campaign_mass():
    frame = pd.DataFrame({"label": [0, 0, 0, 1, 1, 1],
                          "source": ["a", "a", "b", "a", "b", "b"],
                          "group_id": ["x", "x", "y", "z", "q", "r"]})
    weighted = frame.assign(weight=adaptation.sample_weights(frame))
    assert weighted.groupby("label").weight.sum().to_dict() == {0: 3.0, 1: 3.0}
    assert weighted.groupby(["label", "source"]).weight.sum().eq(1.5).all()
    assert weighted.loc[0, "weight"] == weighted.loc[1, "weight"] == 0.75
    assert weighted.loc[2, "weight"] == 1.5


def test_threshold_enforces_small_source_constraint_independently():
    # A pooled 1% budget would allow the one new-source false positive.
    frame = pd.DataFrame({"source": ["old"] * 200 + ["new", "new", "old"],
                          "label": [0] * 200 + [0, 1, 1]})
    scores = np.array([-1.] * 200 + [0.8, 0.9, 0.7])
    result = adaptation.validation_choice(frame, scores)
    assert result["threshold"] == 0.9
    assert result["macro_recall"] == 0.5
    assert result["max_source_fpr"] == 0


def test_threshold_matches_exhaustive_source_aware_search():
    rng = np.random.default_rng(19)
    frame = pd.DataFrame({"source": ["a"] * 12 + ["b"] * 12 + ["c"] * 6,
                          "label": [0] * 6 + [1] * 6 + [0] * 6 + [1] * 12})
    scores = np.round(rng.normal(size=len(frame)), 1)
    choices = []
    for threshold in [np.nextafter(scores.max(), np.inf), *np.unique(scores)]:
        pred = scores >= threshold
        recalls, fprs = [], []
        for source in frame.source.unique():
            for label, destination in [(1, recalls), (0, fprs)]:
                mask = frame.source.eq(source) & frame.label.eq(label)
                if mask.any():
                    destination.append(pred[mask].mean())
        if max(fprs) <= 0.2:
            choices.append((np.mean(recalls), -np.mean(fprs), threshold))
    expected = max(choices)
    actual = adaptation.validation_choice(frame, scores, max_fpr=0.2)
    assert (actual["macro_recall"], -actual["macro_fpr"], actual["threshold"]) == expected


@pytest.mark.parametrize("scores", [[0.1], [np.nan, 0.2], [np.inf, 0.1]])
def test_bad_validation_scores_are_rejected(scores):
    frame = pd.DataFrame({"source": ["a", "a"], "label": [0, 1]})
    with pytest.raises(ValueError):
        adaptation.validation_choice(frame, scores)


def test_paired_bootstrap_identical_models_have_zero_delta():
    frame = pd.DataFrame({"group_id": ["a", "a", "b", "c", "c", "d"],
                          "label": [0, 0, 0, 1, 1, 1]})
    pred = np.array([0, 1, 0, 1, 0, 1])
    intervals = adaptation.paired_intervals(frame, pred, pred, draws=100)
    for metric in ["f1", "fpr", "precision", "recall"]:
        assert intervals[f"delta_{metric}"] == {"lower": 0, "upper": 0}


class GuardPipeline:
    def fit(self, texts, labels, **kwargs):
        assert all(text.startswith("train") for text in texts)
        assert len(kwargs["clf__sample_weight"]) == len(texts)
        return self

    def decision_function(self, texts):
        assert all(text.startswith("validation") for text in texts)
        return np.array([1. if "spam" in text else -1. for text in texts])


def test_candidate_selection_never_fits_or_scores_test_texts(tmp_path, monkeypatch):
    rows = []
    for split in ["train", "validation", "test"]:
        for label in [0, 1]:
            key = f"{split}_{label}"
            rows.append({"source": "fixture", "source_row": len(rows), "split": split,
                         "label": label, "text": f"{split} {'spam' if label else 'ham'}",
                         "text_sha256": key, "group_id": key})
    monkeypatch.setattr(adaptation, "load_frozen", lambda: (pd.DataFrame(rows), {}))
    monkeypatch.setattr(adaptation, "candidate_pipeline", lambda *args: GuardPipeline())
    chosen = adaptation.fit(tmp_path / "run")
    assert chosen["model"] == "adapted_word_c0.5"
    metadata = json.loads((tmp_path / "run/training_metadata.json").read_text())
    assert metadata["test_predictions_during_selection"] == 0
    assert not (tmp_path / "run/test_scores.csv").exists()
    with pytest.raises(FileExistsError):
        adaptation.fit(tmp_path / "run")


def test_checked_run_rejects_changed_training_evidence(tmp_path):
    adaptation.write_json(tmp_path / "training_metadata.json", {
        "source_sha256": {}, "versions": {}, "output_sha256": {"selection.json": "wrong"}, "artifacts": {}})
    (tmp_path / "selection.json").write_text("{}")
    with pytest.raises(ValueError, match="Training output changed"):
        adaptation.checked_run(tmp_path)
