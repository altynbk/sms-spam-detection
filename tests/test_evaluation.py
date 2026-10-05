import numpy as np
import pytest

from spam_detector.evaluation import attack_success_rate, metrics, predictions, select_threshold


def test_counts_ranking_metrics_and_fpr():
    result = metrics([0, 0, 1, 1], [0.1, 0.8, 0.2, 0.9], 0.5)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["fp"] == result["fn"] == 1
    assert result["fpr"] == result["recall"] == result["f1"] == 0.5
    assert result["roc_auc"] == 0.75
    assert result["average_precision"] == pytest.approx(5 / 6)


def test_native_default_ties_and_tuned_threshold_are_distinct():
    # An equally probable two-class Random Forest prediction chooses ham.
    native = metrics([0, 1], [0.5, 0.9], 0.5, y_pred=[0, 1])
    tuned = metrics([0, 1], [0.5, 0.9], 0.5)
    assert native["fp"] == 0 and tuned["fp"] == 1
    assert native["roc_auc"] == tuned["roc_auc"]
    with pytest.raises(ValueError, match="aligned binary"):
        metrics([0, 1], [0.5, 0.9], 0.5, y_pred=[0])


def test_asr_conditions_on_clean_true_positives():
    result = attack_success_rate([1, 1, 1, 0], [1, 1, 0, 1], [0, 1, 0, 0])
    assert result == {"asr": 0.5, "asr_numerator": 1, "asr_denominator": 2}
    assert attack_success_rate([1, 0], [0, 1], [0, 1]) == {
        "asr": None, "asr_numerator": 0, "asr_denominator": 0}


@pytest.mark.parametrize("max_fpr", [0, 0.01, 0.2, 1])
def test_threshold_matches_exhaustive_search_and_handles_ties(max_fpr):
    y = np.array([0, 1, 0, 1, 0, 0, 1])
    scores = np.array([-1, -0.2, -0.2, 0.4, 0.3, 0.8, 0.8])
    choice = select_threshold(y, scores, max_fpr)
    candidates = [np.nextafter(max(scores), np.inf), *np.unique(scores)]
    feasible = [(t, metrics(y, scores, t)) for t in candidates]
    feasible = [(t, m) for t, m in feasible if m["fpr"] <= max_fpr]
    expected, _ = max(feasible, key=lambda tm: (tm[1]["recall"], -tm[1]["fp"], tm[0]))
    assert choice["threshold"] == expected
    assert choice["n_ham"] == 4
    assert choice["validation_metrics"]["fpr"] <= max_fpr


def test_threshold_can_reject_all_and_rejects_invalid_input():
    choice = select_threshold([0, 1], [0.5, 0.5], 0)
    assert np.isfinite(choice["threshold"])
    assert predictions([0.5, 0.5], choice["threshold"]).tolist() == [0, 0]
    for y, scores, fpr in [([0, 0], [0.1, 0.2], 0.01), ([0, 1], [0.1, np.nan], 0.01),
                           ([0, 1], [0.1, 0.2], -1)]:
        with pytest.raises(ValueError):
            select_threshold(y, scores, fpr)
