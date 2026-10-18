"""Independently reconstruct adaptation decisions and metrics without fitting."""

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

from spam_detector.adaptation import checked_run
from spam_detector.adaptation_protocol import load_frozen
from spam_detector.external_data import file_hash


def recount(labels, predictions):
    y, p = np.asarray(labels), np.asarray(predictions)
    tn, fp, fn, tp = np.bincount(2 * y + p, minlength=4)
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "n_ham": int(tn + fp), "n_spam": int(tp + fn),
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
            "fpr": float(fp / (fp + tn)) if fp + tn else None,
            "accuracy": float((tp + tn) / len(y))}


def brute_threshold(frame, scores):
    choices = []
    masks = [(source, frame.source.eq(source).to_numpy(), frame.label.to_numpy())
             for source in sorted(frame.source.unique())]
    for threshold in [np.nextafter(scores.max(), np.inf), *np.unique(scores)]:
        predicted = scores >= threshold
        recalls, fprs = [], []
        for _, source_mask, labels in masks:
            for label, destination in [(1, recalls), (0, fprs)]:
                mask = source_mask & (labels == label)
                if mask.any():
                    destination.append(predicted[mask].mean())
        if max(fprs) <= 0.01:
            choices.append((float(np.mean(recalls)), -float(np.mean(fprs)), float(threshold)))
    return max(choices)


def bootstrap_by_rows(part, draws, seed):
    """Independent implementation resamples row indices through whole groups."""
    part = part.reset_index(drop=True)
    groups = [[g.index.to_numpy() for _, g in part[part.label.eq(label)].groupby("group_id", sort=True)]
              for label in [0, 1]]
    y = part.label.to_numpy()
    predictions = [part.baseline_prediction.to_numpy(), part.candidate_prediction.to_numpy()]
    rng = np.random.default_rng(seed)
    results = {f"{model}_{metric}": [] for model in ["baseline", "candidate", "delta"]
               for metric in ["f1", "recall", "fpr", "precision"]}
    for _ in range(draws):
        indices = np.concatenate([np.concatenate([group[i] for i in rng.integers(0, len(group), len(group))])
                                  for group in groups])
        m = [recount(y[indices], p[indices]) for p in predictions]
        for metric in ["f1", "recall", "fpr", "precision"]:
            for i, model in enumerate(["baseline", "candidate"]):
                results[f"{model}_{metric}"].append(m[i][metric])
            results[f"delta_{metric}"].append(m[1][metric] - m[0][metric])
    return {k: dict(zip(["lower", "upper"], np.quantile(v, [0.025, 0.975]).tolist())) for k, v in results.items()}


def verify(run):
    metadata = checked_run(run)
    frame, _ = load_frozen()
    validation = frame[frame.split.eq("validation")].reset_index(drop=True)
    saved = pd.read_csv(run / "validation_scores.csv", float_precision="round_trip")
    candidates = pd.read_csv(run / "candidates.csv", float_precision="round_trip")
    validation_metrics = pd.read_csv(run / "validation_metrics.csv", float_precision="round_trip")
    recomputed = []
    for row in candidates.itertuples():
        model = joblib.load(run / "models" / f"{row.model}.joblib")
        ledger = saved[saved.model.eq(row.model)].reset_index(drop=True)
        pd.testing.assert_frame_equal(ledger[validation.drop(columns="text").columns],
                                      validation.drop(columns="text"), check_dtype=False)
        direct = model.pipeline.decision_function(validation.text.tolist())
        np.testing.assert_allclose(direct, ledger.score, rtol=0, atol=1e-12)
        recall, negative_fpr, threshold = brute_threshold(validation, direct)
        np.testing.assert_allclose([recall, -negative_fpr, threshold],
                                   [row.macro_recall, row.macro_fpr, row.threshold], rtol=0, atol=1e-12)
        assert model.threshold == row.threshold
        for source, part in ledger.groupby("source"):
            recorded = validation_metrics[validation_metrics.source.eq(source) &
                                          validation_metrics.model.eq(row.model)].iloc[0]
            expected = recount(part.label, (part.score >= threshold).astype(int))
            for key, value in expected.items():
                if value is None:
                    assert pd.isna(recorded[key])
                else:
                    np.testing.assert_allclose(recorded[key], value, rtol=0, atol=1e-12)
        recomputed.append((recall, negative_fpr, row.model))
    selected = max(recomputed, key=lambda x: x[:2])[2]
    assert selected == metadata["selected_model"]
    evaluation = json.loads((run / "evaluation_metadata.json").read_text())
    assert evaluation["training_metadata_sha256"] == file_hash(run / "training_metadata.json")
    for name, digest in evaluation["output_sha256"].items():
        assert file_hash(run / name) == digest
    test = frame[frame.split.eq("test")].reset_index(drop=True)
    ledger = pd.read_csv(run / "test_scores.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(ledger[test.drop(columns="text").columns], test.drop(columns="text"), check_dtype=False)
    model_paths = {"baseline": ROOT / "results/full/models/linear_svc_word_gentle.joblib",
                   "candidate": run / "models" / f"{selected}.joblib"}
    protocol = json.loads((ROOT / "data/external/adaptation/protocol.json").read_text())
    assert file_hash(model_paths["baseline"]) == protocol["baseline_model_sha256"] == evaluation["baseline_sha256"]
    metrics = pd.read_csv(run / "test_metrics.csv", float_precision="round_trip")
    for name, path in model_paths.items():
        model = joblib.load(path)
        direct = model.pipeline.decision_function(test.text.tolist())
        np.testing.assert_allclose(direct, ledger[f"{name}_score"], rtol=0, atol=1e-12)
        predicted = (direct >= model.threshold).astype(int)
        np.testing.assert_array_equal(predicted, ledger[f"{name}_prediction"])
        for source, part in ledger.groupby("source"):
            recorded = metrics[metrics.source.eq(source) & metrics.model.eq(name)].iloc[0]
            expected = recount(part.label, part[f"{name}_prediction"])
            expected["roc_auc"] = roc_auc_score(part.label, part[f"{name}_score"])
            expected["average_precision"] = average_precision_score(part.label, part[f"{name}_score"])
            for key, value in expected.items():
                np.testing.assert_allclose(recorded[key], value, rtol=0, atol=1e-12)
    saved_intervals = json.loads((run / "test_intervals.json").read_text())
    for source, part in ledger.groupby("source", sort=True):
        recomputed = bootstrap_by_rows(part, protocol["uncertainty"]["draws"], protocol["uncertainty"]["seed"])
        for metric, interval in recomputed.items():
            for bound, value in interval.items():
                np.testing.assert_allclose(saved_intervals[source][metric][bound], value, rtol=0, atol=1e-12)
    passed = saved_intervals["sms_history"]["delta_f1"]["lower"] > 0
    for source, part in metrics.groupby("source"):
        fpr = part.set_index("model").fpr
        passed = passed and fpr["candidate"] <= min(0.01, fpr["baseline"])
    assert bool(passed) == evaluation["promotion_gate_passed"]
    result = {"status": "passed", "candidate_models_checked": len(candidates),
              "validation_scores_checked": len(saved), "test_messages_checked": len(test),
              "validation_metric_rows_checked": len(validation_metrics),
              "test_score_values_checked": 2 * len(test), "metric_rows_checked": len(metrics),
              "bootstrap_draws_per_source": protocol["uncertainty"]["draws"],
              "thresholds_reselected_by_exhaustive_search": True,
              "test_intervals_recomputed_by_row_resampling": True,
              "promotion_gate_recomputed": True,
              "output_sha256": evaluation["output_sha256"],
              "verifier_sha256": file_hash(__file__)}
    (run / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    verify(parser.parse_args().run)
