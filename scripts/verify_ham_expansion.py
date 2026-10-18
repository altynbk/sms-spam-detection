"""Recount predictions, exhaustively select thresholds and bootstrap by rows."""

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.verify_adaptation import bootstrap_by_rows, brute_threshold, recount
from spam_detector.adaptation import checked_run
from spam_detector.external_data import file_hash
from spam_detector.ham_data import DIRECTORY, load_frozen
from spam_detector.ham_expansion import controls


def verify(run):
    metadata = checked_run(run)
    protocol = json.loads((DIRECTORY / "protocol.json").read_text())
    frame, _ = load_frozen()
    validation = frame[frame.split.eq("validation")].reset_index(drop=True)
    test = frame[frame.split.eq("test")].reset_index(drop=True)
    fixed = controls(protocol)
    candidates = pd.read_csv(run / "candidates.csv", float_precision="round_trip")
    scores = pd.read_csv(run / "validation_scores.csv", float_precision="round_trip")
    val_metrics = pd.read_csv(run / "validation_metrics.csv", float_precision="round_trip")
    choices = []
    control = json.loads((run / "threshold_control.json").read_text())
    expected_choices = candidates.to_dict("records") + [{"model": "threshold_only", **control}]
    for row in expected_choices:
        name = row["model"]
        model = fixed["previous"] if name == "threshold_only" else joblib.load(run / "models" / f"{name}.joblib")
        ledger = scores[scores.model.eq(name)].reset_index(drop=True)
        pd.testing.assert_frame_equal(ledger[validation.drop(columns="text").columns],
                                      validation.drop(columns="text"), check_dtype=False)
        direct = model.pipeline.decision_function(validation.text.tolist())
        np.testing.assert_allclose(direct, ledger.score, rtol=0, atol=1e-12)
        recall, negative_fpr, threshold = brute_threshold(validation, direct)
        np.testing.assert_allclose([recall, -negative_fpr, threshold],
                                   [row["macro_recall"], row["macro_fpr"], row["threshold"]], rtol=0, atol=1e-12)
        if name != "threshold_only":
            assert model.threshold == threshold
            choices.append((recall, negative_fpr, name))
        for source, part in ledger.groupby("source"):
            expected = recount(part.label, (part.score >= threshold).astype(int))
            recorded = val_metrics[val_metrics.source.eq(source) & val_metrics.model.eq(name)].iloc[0]
            for key, value in expected.items():
                if value is None:
                    assert pd.isna(recorded[key])
                else:
                    np.testing.assert_allclose(recorded[key], value, rtol=0, atol=1e-12)
    selected = max(choices, key=lambda x: x[:2])[2]
    assert selected == metadata["selected_model"]
    evaluation = json.loads((run / "evaluation_metadata.json").read_text())
    assert evaluation["training_metadata_sha256"] == file_hash(run / "training_metadata.json")
    for path, digest in evaluation["output_sha256"].items():
        assert file_hash(run / path) == digest
    ledger = pd.read_csv(run / "test_scores.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(ledger[test.drop(columns="text").columns], test.drop(columns="text"), check_dtype=False)
    test_metrics = pd.read_csv(run / "test_metrics.csv", float_precision="round_trip").set_index("model")
    models = {**fixed, "threshold_only": fixed["previous"],
              "candidate": joblib.load(run / "models" / f"{selected}.joblib")}
    for name, model in models.items():
        threshold = control["threshold"] if name == "threshold_only" else model.threshold
        direct = model.pipeline.decision_function(test.text.tolist())
        np.testing.assert_allclose(direct, ledger[f"{name}_score"], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(direct >= threshold, ledger[f"{name}_prediction"])
        expected = recount(test.label, ledger[f"{name}_prediction"])
        expected.update(roc_auc=roc_auc_score(test.label, direct), average_precision=average_precision_score(test.label, direct))
        for key, value in expected.items():
            np.testing.assert_allclose(test_metrics.loc[name, key], value, rtol=0, atol=1e-12)
        assert test_metrics.loc[name, "threshold"] == threshold
    intervals = json.loads((run / "test_intervals.json").read_text())
    for name in ["baseline", "previous", "threshold_only"]:
        pair = ledger[["group_id", "label", "candidate_prediction"]].assign(baseline_prediction=ledger[f"{name}_prediction"])
        independent = bootstrap_by_rows(pair, protocol["uncertainty"]["draws"], protocol["uncertainty"]["seed"])
        for metric, bounds in independent.items():
            for bound, value in bounds.items():
                np.testing.assert_allclose(intervals[name][metric][bound], value, rtol=0, atol=1e-12)
    # Recompute gate without calling the training/evaluation implementation.
    passed = (test_metrics.loc["candidate", "fpr"] <= min(0.01, test_metrics.loc["baseline", "fpr"])
              and intervals["previous"]["delta_fpr"]["upper"] < 0
              and all(intervals[name]["delta_recall"]["lower"] >= -0.03 for name in ["baseline", "previous"]))
    assert bool(passed) == evaluation["promotion_gate"]["passed"]
    result = {"status": "passed", "fitted_candidates_checked": len(candidates),
              "threshold_only_control_checked": True, "validation_scores_checked": len(scores),
              "validation_metric_rows_checked": len(val_metrics), "test_messages_checked": len(test),
              "test_score_values_checked": 4 * len(test), "test_metric_rows_checked": len(test_metrics),
              "thresholds_reselected_by_exhaustive_search": True,
              "paired_intervals_recomputed_by_rows": True,
              "bootstrap_draws_per_comparison": protocol["uncertainty"]["draws"],
              "promotion_gate_recomputed": True, "output_sha256": evaluation["output_sha256"],
              "verifier_sha256": file_hash(__file__),
              "shared_verifier_sha256": file_hash(ROOT / "scripts/verify_adaptation.py")}
    (run / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    verify(parser.parse_args().run)
