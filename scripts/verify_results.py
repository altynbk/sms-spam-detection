"""Recompute saved metrics without fitting or selecting anything.

Run from the repository root:
    python scripts/verify_results.py --run results/full
    python scripts/verify_results.py --run results/full --all-attacks
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spam_detector.attacks import attack_messages
from spam_detector.data import assert_disjoint, load_data, make_cv_folds, split_data
from spam_detector.evaluation import attack_success_rate, metrics, predictions, select_threshold


def check(condition, message):
    if not condition:
        raise ValueError(message)


def equal_metrics(actual, row):
    for name, value in actual.items():
        if name == "confusion_matrix":
            check(value == json.loads(row[name]), f"Incorrect {name}")
        elif value is None:
            check(pd.isna(row[name]), f"Expected undefined {name}")
        else:
            check(np.isclose(value, row[name], rtol=0, atol=1e-14), f"Incorrect {name}: {value} != {row[name]}")


def verify(run_dir, data_path, all_attacks=False):
    out = Path(run_dir)
    meta = json.loads((out / "metadata.json").read_text())
    check(meta["status"] == "complete", "Experiment is incomplete")
    check(meta.get("evaluation_revision") == 2, "Expected native-default evaluation revision 2")
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
    test = generated[generated.split == "test"]
    validation = generated[generated.split == "validation"]
    clean = pd.read_csv(out / "clean_metrics.csv", float_precision="round_trip")
    runs = pd.read_csv(out / "robustness_runs.csv", float_precision="round_trip", na_values=["undefined"])
    attack_manifest = pd.read_csv(out / "attack_manifest.csv", float_precision="round_trip")
    keys = ["model", "attack_type", "intensity", "seed", "threshold_mode"]
    check(not runs.duplicated(keys).any(), "Duplicate attack result rows")
    expected_datasets = len(cfg["attacks"]) * len(cfg["intensities"]) * cfg["repeats"]
    check(len(attack_manifest) == expected_datasets, "Wrong attack dataset count")
    check(len(runs) == expected_datasets * len(meta["models"]) * 2, "Wrong attack metric count")
    check(len(clean) == len(meta["models"]) * 2, "Wrong clean metric count")
    models, clean_pred = {}, {}
    for name, info in meta["models"].items():
        path = out / info["artifact"]
        check(hashlib.sha256(path.read_bytes()).hexdigest() == info["artifact_sha256"], f"Artifact changed: {name}")
        bundle = joblib.load(path)
        models[name] = bundle
        choice = select_threshold(validation.label, bundle.score(validation.sms), cfg["max_fpr"])
        check(choice["threshold"] == bundle.threshold == info["threshold_selection"]["threshold"], "Threshold changed")
        scores = bundle.score(test.sms)
        for mode, threshold in [("default", info["default_threshold"]), ("validation", bundle.threshold)]:
            rows = clean[(clean.model == name) & (clean.threshold_mode == mode)]
            check(len(rows) == 1, "Missing/duplicate clean result")
            row = rows.iloc[0]
            pred = bundle.pipeline.predict(test.sms) if mode == "default" else predictions(scores, threshold)
            clean_pred[name, mode] = pred
            equal_metrics(metrics(test.label, scores, threshold, y_pred=pred), row)
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
    report = {"status": "passed", "evaluation_revision": 2, "models_loaded": len(models),
              "clean_metric_rows_recomputed": len(clean), "attack_rows_checked_for_invariants": len(runs),
              "attack_metric_rows_recomputed": checked_rows, "attack_dataset_hashes_recomputed": len(attack_manifest),
              "all_attacks": all_attacks, "source_and_model_hashes_match": True,
              "validation_thresholds_recomputed": True, "split_and_cv_membership_recomputed": True,
              "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "No fitting, parameter changes, or model selection based on test results."}
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="results/full")
    parser.add_argument("--data", default="data/smshamspam.csv")
    parser.add_argument("--all-attacks", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.run, args.data, args.all_attacks), indent=2))
