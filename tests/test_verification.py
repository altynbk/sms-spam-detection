"""A saved report must reject plausible-looking, incomplete or altered tables."""

from itertools import product
import json

import pandas as pd
import pytest

from scripts.verify_results import require_keys, verify_cv, verify_summary


@pytest.mark.parametrize("replacement", [("leet", 10), ("unicode", 10), ("leet", 99)])
def test_manifest_count_cannot_hide_duplicate_or_unexpected_draw(replacement):
    expected = list(product(["leet", "separators"], [10, 11]))
    frame = pd.DataFrame([*expected[:-1], replacement], columns=["attack", "seed"])
    assert len(frame) == len(expected)
    with pytest.raises(ValueError, match="keys"):
        require_keys(frame, ["attack", "seed"], expected, "attack manifest")


def test_valid_manifest_accepts_any_row_order():
    frame = pd.DataFrame({"attack": ["leet", "leet"], "seed": [11, 10]})
    require_keys(frame, ["attack", "seed"], [("leet", 10), ("leet", 11)], "attack manifest")


@pytest.mark.parametrize("field,value", [("mean_cv_f1", 0.99), ("std_cv_f1", 0.1), ("rank", 2)])
def test_cv_report_rejects_altered_aggregation(field, value):
    frame = pd.DataFrame([{"model": "model", "params": json.dumps({"C": 1}, sort_keys=True),
                           "fold_0_f1": 0.5, "fold_1_f1": 1.0, "mean_cv_f1": 0.75,
                           "std_cv_f1": 0.25, "rank": 1}])
    meta = {"config": {"cv_folds": 2}, "search_spaces": {"model": {"C": [1]}},
            "models": {"model": {"best_params": {"C": 1}, "best_cv_f1": 0.75}}}
    verify_cv(frame, meta)
    frame.loc[0, field] = value
    with pytest.raises(ValueError, match="CV"):
        verify_cv(frame, meta)


def test_robustness_summary_rejects_edited_mean():
    keys = {"model": "model", "feature_type": "word", "preprocessing": "gentle",
            "threshold_mode": "validation", "attack_type": "leet", "intensity": 0.3}
    names = ["precision", "recall", "f1", "roc_auc", "average_precision", "asr"]
    runs = pd.DataFrame([{**keys, **dict.fromkeys(names, value)} for value in [0, 1]])
    summary = runs.groupby(list(keys))[names].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    saved = summary.reset_index()
    verify_summary(runs, saved)
    saved.loc[0, "asr_mean"] = 0.1
    with pytest.raises(AssertionError):
        verify_summary(runs, saved)
