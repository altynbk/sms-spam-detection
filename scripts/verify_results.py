"""Recompute saved metrics without fitting or selecting text models.

Run from the repository root:
    python scripts/verify_results.py --run results/full
    python scripts/verify_results.py --run results/full --all-attacks
"""

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spam_detector.attacks import attack_messages
from spam_detector.data import assert_disjoint, load_data, make_cv_folds, split_data
from spam_detector.evaluation import (attack_success_rate, default_threshold, majority_baseline,
                                      metrics, predictions, select_threshold)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def equal_metrics(actual, row):
    for name, value in actual.items():
        if name == "confusion_matrix":
            check(value == json.loads(row[name]), f"Incorrect {name}")
        elif isinstance(value, str):
            check(value == row[name], f"Incorrect {name}")
        elif value is None:
            check(pd.isna(row[name]), f"Expected undefined {name}")
        else:
            check(np.isclose(value, row[name], rtol=0, atol=1e-14), f"Incorrect {name}: {value} != {row[name]}")


def require_keys(frame, keys, expected, description):
    """Counts alone cannot detect missing rows replaced with other scenarios/seeds."""
    check(not frame[keys].isna().any().any(), f"Missing key in {description}")
    check(not frame.duplicated(keys).any(), f"Duplicate keys in {description}")
    check(set(frame[keys].itertuples(index=False, name=None)) == set(expected),
          f"Missing or unexpected keys in {description}")


def verify_cv(cv, meta):
    expected = [(name, json.dumps(params, sort_keys=True))
                for name, grid in meta["search_spaces"].items() for params in ParameterGrid(grid)]
    require_keys(cv, ["model", "params"], expected, "CV results")
    folds = cv[[f"fold_{i}_f1" for i in range(meta["config"]["cv_folds"])]].to_numpy()
    check(np.isfinite(folds).all() and ((folds >= 0) & (folds <= 1)).all(), "Invalid fold scores")
    check(np.allclose(cv.mean_cv_f1, folds.mean(axis=1), rtol=0, atol=1e-14), "CV mean differs from folds")
    check(np.allclose(cv.std_cv_f1, folds.std(axis=1), rtol=0, atol=1e-14), "CV SD differs from folds")
    for name, info in meta["models"].items():
        part = cv[cv.model == name]
        check(np.array_equal(part["rank"], part.mean_cv_f1.rank(method="min", ascending=False)),
              "Incorrect CV ranks")
        best = part.loc[part.mean_cv_f1.idxmax()]
        check(json.loads(best.params) == info["best_params"], "Best CV parameters differ from metadata")
        check(best.mean_cv_f1 == info["best_cv_f1"], "Best CV score differs from metadata")


def verify_summary(runs, saved):
    keys = ["model", "feature_type", "preprocessing", "threshold_mode", "attack_type", "intensity"]
    computed = runs.groupby(keys)[["precision", "recall", "f1", "roc_auc", "average_precision", "asr"]].agg(["mean", "std"])
    computed.columns = ["_".join(c) for c in computed.columns]
    expected = computed.reset_index()
    require_keys(saved, keys, expected[keys].itertuples(index=False, name=None), "robustness summary")
    pd.testing.assert_frame_equal(saved.sort_values(keys).reset_index(drop=True)[expected.columns],
                                  expected.sort_values(keys).reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=0, atol=1e-14)


def verify(run_dir, data_path, all_attacks=False):
    out = Path(run_dir)
    meta = json.loads((out / "metadata.json").read_text())
    check(meta["status"] == "complete", "Experiment is incomplete")
    check(meta.get("evaluation_revision") == 2, "Expected native-default evaluation revision 2")
    check(json.loads((out / "config.json").read_text()) == meta["config"], "Config differs from metadata")
    for relative, digest in meta["source_sha256"].items():
        check(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, f"Source changed: {relative}")
    check(hashlib.sha256(Path(data_path).read_bytes()).hexdigest() == meta["dataset"]["sha256"], "Dataset changed")
    check(hashlib.sha256((out / "splits.csv").read_bytes()).hexdigest() == meta["split_manifest_sha256"], "Split file changed")
    frame, _ = load_data(data_path)
    cfg = meta["config"]
    generated = split_data(frame, cfg["seed"], cfg["validation_size"], cfg["test_size"])
    manifest = pd.read_csv(out / "splits.csv")
    check(generated.message_id.tolist() == manifest.message_id.tolist(), "Message identity/order changed")
    check(generated.split.tolist() == manifest.split.tolist(), "Saved split does not match seed/config")
    assert_disjoint(manifest)
    train = generated[generated.split == "train"]
    _, fold_ids = make_cv_folds(train, cfg["seed"], cfg["cv_folds"])
    check(np.array_equal(fold_ids, manifest.loc[manifest.split == "train", "cv_fold"]), "CV folds changed")
    check(manifest.loc[manifest.split != "train", "cv_fold"].isna().all(), "CV includes held-out data")
    for column in ["group_id", "label", "source_row", "source_rows"]:
        check(generated[column].tolist() == manifest[column].tolist(), f"Manifest {column} differs from data")
    test = generated[generated.split == "test"]
    validation = generated[generated.split == "validation"]
    clean = pd.read_csv(out / "clean_metrics.csv", float_precision="round_trip")
    validation_metrics = pd.read_csv(out / "validation_metrics.csv", float_precision="round_trip")
    baselines = pd.read_csv(out / "baseline_metrics.csv", float_precision="round_trip")
    require_keys(baselines, ["split"], [("validation",), ("test",)], "baseline results")
    for name, part in [("validation", validation), ("test", test)]:
        equal_metrics(majority_baseline(train.label, part.label), baselines[baselines.split == name].iloc[0])
    cv = pd.read_csv(out / "cv_results.csv", float_precision="round_trip")
    verify_cv(cv, meta)
    errors = pd.read_csv(out / "errors.csv", float_precision="round_trip", keep_default_na=False)
    error_keys = ["model", "threshold_mode", "message_id"]
    expected_errors = []
    runs = pd.read_csv(out / "robustness_runs.csv", float_precision="round_trip", na_values=["undefined"])
    attack_manifest = pd.read_csv(out / "attack_manifest.csv", float_precision="round_trip")
    keys = ["model", "attack_type", "intensity", "seed", "threshold_mode"]
    expected_seeds = [cfg["seed"] + 10000 + i for i in range(cfg["repeats"])]
    check(meta["attack_seeds"] == expected_seeds, "Attack seeds differ from configuration")
    require_keys(attack_manifest, ["attack_type", "intensity", "seed"],
                 product(cfg["attacks"], cfg["intensities"], expected_seeds), "attack manifest")
    require_keys(runs, keys, product(meta["models"], cfg["attacks"], cfg["intensities"],
                                     expected_seeds, ["default", "validation"]), "attack results")
    for name, table in [("clean results", clean), ("validation results", validation_metrics)]:
        require_keys(table, ["model", "threshold_mode"],
                     product(meta["models"], ["default", "validation"]), name)
    expected_datasets = len(cfg["attacks"]) * len(cfg["intensities"]) * cfg["repeats"]
    check(len(attack_manifest) == expected_datasets, "Wrong attack dataset count")
    check(len(runs) == expected_datasets * len(meta["models"]) * 2, "Wrong attack metric count")
    check(len(clean) == len(meta["models"]) * 2, "Wrong clean metric count")
    summary = pd.read_csv(out / "robustness_summary.csv", float_precision="round_trip", na_values=["undefined"])
    verify_summary(runs, summary)
    models, clean_pred = {}, {}
    for name, info in meta["models"].items():
        path = out / info["artifact"]
        check(path.is_file(), f"Missing model: {path}. Train into a new output directory and verify that run.")
        check(hashlib.sha256(path.read_bytes()).hexdigest() == info["artifact_sha256"], f"Artifact changed: {name}")
        bundle = joblib.load(path)
        models[name] = bundle
        check(bundle.model_id == name, "Artifact model identity differs from metadata")
        check(default_threshold(bundle.pipeline) == info["default_threshold"], "Incorrect native threshold")
        validation_scores = bundle.score(validation.sms)
        choice = select_threshold(validation.label, validation_scores, cfg["max_fpr"])
        check(choice["threshold"] == bundle.threshold == info["threshold_selection"]["threshold"], "Threshold changed")
        check(choice == info["threshold_selection"], "Threshold selection details differ from metadata")
        scores = bundle.score(test.sms)
        for mode, threshold in [("default", info["default_threshold"]), ("validation", bundle.threshold)]:
            rows = clean[(clean.model == name) & (clean.threshold_mode == mode)]
            check(len(rows) == 1, "Missing/duplicate clean result")
            row = rows.iloc[0]
            check(row.threshold == threshold, "Incorrect clean threshold")
            val = validation_metrics[(validation_metrics.model == name) &
                                     (validation_metrics.threshold_mode == mode)].iloc[0]
            check(val.threshold == threshold, "Incorrect validation threshold")
            val_pred = bundle.pipeline.predict(validation.sms) if mode == "default" else predictions(validation_scores, threshold)
            equal_metrics(metrics(validation.label, validation_scores, threshold, y_pred=val_pred), val)
            pred = bundle.pipeline.predict(test.sms) if mode == "default" else predictions(scores, threshold)
            clean_pred[name, mode] = pred
            equal_metrics(metrics(test.label, scores, threshold, y_pred=pred), row)
            for position in np.flatnonzero(pred != test.label.to_numpy()):
                sample = test.iloc[position]
                expected_errors.append({"model": name, "threshold_mode": mode, "message_id": sample.message_id,
                                        "threshold": threshold, "source_row": int(sample.source_row),
                                        "text": sample.sms if meta["dataset"]["error_text_exported"] else "[withheld: unrecognized dataset]",
                                        "label": int(sample.label), "prediction": int(pred[position]),
                                        "score": float(scores[position]), "score_kind": bundle.score_kind,
                                        "error_type": "FP" if pred[position] else "FN"})
            part = runs[(runs.model == name) & (runs.threshold_mode == mode)]
            check(part.threshold.eq(threshold).all(), "Threshold differs between clean and attacked test")
            check(part.fp.eq(row.fp).all(), "Spam-only attack changed FP count")
            check(part.asr_denominator.eq(row.tp).all(), "Wrong ASR denominator")
            if row.tp:
                check(np.allclose(part.asr, part.asr_numerator / row.tp, rtol=0, atol=1e-14), "Wrong ASR ratio")
            else:
                check(part.asr.isna().all(), "ASR with no clean TP must be undefined")
            for _, zero in part[part.intensity == 0].iterrows():
                equal_metrics(metrics(test.label, scores, threshold, y_pred=pred), zero)
    require_keys(errors, error_keys, [tuple(row[k] for k in error_keys) for row in expected_errors], "error results")
    indexed_errors = errors.set_index(error_keys)
    for row in expected_errors:
        equal_metrics({key: value for key, value in row.items() if key not in error_keys},
                      indexed_errors.loc[tuple(row[k] for k in error_keys)])
    # Check every attack dataset hash. Recompute all predictions only on request;
    # otherwise use one fixed seed per scenario/intensity for the expensive check.
    indexed = runs.set_index(keys)
    checked_rows = 0
    for attack in attack_manifest.itertuples(index=False):
        raw = attack_messages(test.sms, test.label, int(attack.seed), attack.intensity, attack.attack_type)
        digest = hashlib.sha256(json.dumps(raw, ensure_ascii=False).encode()).hexdigest()
        check(digest == attack.dataset_sha256, "Attack dataset is not reproducible")
        changed = np.array(raw) != test.sms.to_numpy()
        check(int(changed[test.label.to_numpy() == 0].sum()) == attack.changed_ham == 0, "Ham changed")
        check(int(changed[test.label.to_numpy() == 1].sum()) == attack.changed_spam, "Wrong changed-spam count")
        if not all_attacks and attack.seed != meta["attack_seeds"][0]:
            continue
        for name, bundle in models.items():
            scores = bundle.score(raw)
            for mode, threshold in [("default", meta["models"][name]["default_threshold"]),
                                    ("validation", bundle.threshold)]:
                pred = bundle.pipeline.predict(raw) if mode == "default" else predictions(scores, threshold)
                row = indexed.loc[(name, attack.attack_type, attack.intensity, attack.seed, mode)]
                equal_metrics(metrics(test.label, scores, threshold, y_pred=pred), row)
                equal_metrics(attack_success_rate(test.label, clean_pred[name, mode], pred), row)
                checked_rows += 1
    expected_choice = min(meta["models"], key=lambda n: (-meta["models"][n]["best_cv_f1"], n))
    selection = json.loads((out / "selection.json").read_text())
    check(selection["selected_model"] == meta["selected_model"] == expected_choice, "Choice does not follow CV rule")
    check(selection["rule"] == meta["cv"]["selection"], "Selection rule differs from metadata")
    check(selection["thresholds"] == {name: bundle.threshold for name, bundle in models.items()},
          "Frozen thresholds differ from artifacts")
    report = {"status": "passed", "evaluation_revision": 2, "models_loaded": len(models),
              "clean_metric_rows_recomputed": len(clean), "attack_rows_checked_for_invariants": len(runs),
              "attack_metric_rows_recomputed": checked_rows, "attack_dataset_hashes_recomputed": len(attack_manifest),
              "all_attacks": all_attacks, "source_and_model_hashes_match": True,
              "validation_thresholds_recomputed": True, "split_and_cv_membership_recomputed": True,
              "validation_metric_rows_recomputed": len(validation_metrics),
              "baseline_metric_rows_recomputed": len(baselines),
              "cv_rows_checked_against_folds_and_selection": len(cv),
              "error_rows_recomputed": len(errors), "summary_rows_recomputed": len(summary),
              "complete_scenario_key_sets_checked": True,
              "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "No text-model fitting, parameter changes, or model selection based on test results. The dummy reference uses training label counts only."}
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="results/full")
    parser.add_argument("--data", default="data/smshamspam.csv")
    parser.add_argument("--all-attacks", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.run, args.data, args.all_attacks), indent=2))
