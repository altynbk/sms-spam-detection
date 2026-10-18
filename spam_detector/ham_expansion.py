"""Fit the frozen ham-expansion study and evaluate its untouched test once."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd

from .adaptation import (candidate_pipeline, checked_run, paired_intervals,
                         sample_weights, validation_choice, write_json)
from .evaluation import SpamModel, metrics
from .external_data import ROOT, file_hash
from .ham_data import DIRECTORY, load_frozen

SOURCE_FILES = ["spam_detector/ham_expansion.py", "spam_detector/ham_data.py",
                "spam_detector/adaptation.py", "spam_detector/adaptation_data.py",
                "spam_detector/adaptation_protocol.py", "spam_detector/modeling.py",
                "spam_detector/preprocessing.py", "spam_detector/evaluation.py",
                "data/external/ham_expansion/protocol.json",
                "data/external/ham_expansion/source.json",
                "results/ham_expansion/protocol/membership.csv"]


def ham_weights(frame, multiplier):
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Expected a finite positive ham multiplier")
    weights = sample_weights(frame) * np.where(frame.label.eq(0), multiplier, 1.0)
    return weights / weights.mean()


def controls(protocol):
    result = {}
    for name in ["baseline", "previous"]:
        spec = protocol[name]
        path = ROOT / spec["path"]
        if file_hash(path) != spec["sha256"]:
            raise ValueError(f"Changed {name} artifact")
        model = joblib.load(path)
        if not isinstance(model, SpamModel) or model.threshold != spec["threshold"]:
            raise ValueError(f"Changed {name} threshold")
        result[name] = model
    return result


def fit(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an unused run directory")
    protocol = json.loads((DIRECTORY / "protocol.json").read_text())
    fixed = controls(protocol)
    frame, _ = load_frozen()
    train = frame[frame.split.eq("train")].reset_index(drop=True)
    validation = frame[frame.split.eq("validation")].reset_index(drop=True)
    (output / "models").mkdir(parents=True)
    choices, metric_rows, score_rows, weight_rows, artifacts = [], [], [], [], {}

    def record(identity, scores, threshold):
        ledger = validation.drop(columns="text").assign(model=identity, score=scores)
        score_rows.append(ledger)
        for source, part in ledger.groupby("source"):
            metric_rows.append({"model": identity, "source": source,
                                **metrics(part.label, part.score, threshold)})

    for c in protocol["candidate_C"]:
        for multiplier in protocol["ham_weight_multipliers"]:
            identity = f"ham_word_char_c{c:g}_w{multiplier:g}"
            weights = ham_weights(train, multiplier)
            pipeline = candidate_pipeline("word_char", c, protocol)
            pipeline.fit(train.text.tolist(), train.label, clf__sample_weight=weights)
            scores = pipeline.decision_function(validation.text.tolist())
            choice = {"model": identity, "C": c, "ham_multiplier": multiplier,
                      **validation_choice(validation, scores, protocol["max_validation_fpr"])}
            path = output / "models" / f"{identity}.joblib"
            joblib.dump(SpamModel(pipeline, choice["threshold"], identity), path)
            artifacts[identity] = file_hash(path)
            choices.append(choice)
            record(identity, scores, choice["threshold"])
            summary = train.assign(weight=weights).groupby(["source", "label"]).agg(
                messages=("source_row", "size"), total_weight=("weight", "sum")).reset_index()
            weight_rows.append(summary.assign(model=identity))
            print(json.dumps(choice), flush=True)

    previous_scores = fixed["previous"].score(validation.text.tolist())
    calibration = validation_choice(validation, previous_scores, protocol["max_validation_fpr"])
    record("threshold_only", previous_scores, calibration["threshold"])
    best = max(choices, key=lambda r: (r["macro_recall"], -r["macro_fpr"]))
    pd.DataFrame(choices).to_csv(output / "candidates.csv", index=False)
    pd.DataFrame(metric_rows).drop(columns="confusion_matrix").to_csv(output / "validation_metrics.csv", index=False)
    pd.concat(score_rows, ignore_index=True).to_csv(output / "validation_scores.csv", index=False)
    pd.concat(weight_rows, ignore_index=True).to_csv(output / "training_weights.csv", index=False)
    write_json(output / "selection.json", best)
    write_json(output / "threshold_control.json", calibration)
    names = ["candidates.csv", "validation_metrics.csv", "validation_scores.csv", "selection.json",
             "training_weights.csv", "threshold_control.json"]
    write_json(output / "training_metadata.json", {
        "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ["scikit-learn", "numpy", "pandas", "scipy", "joblib"]},
        "selected_model": best["model"], "source_sha256": {p: file_hash(ROOT / p) for p in SOURCE_FILES},
        "artifacts": artifacts, "output_sha256": {p: file_hash(output / p) for p in names},
        "train_rows": len(train), "validation_rows": len(validation),
        "test_predictions_during_selection": 0,
    })
    return best


def promotion_gate(table, intervals):
    indexed = table.set_index("model")
    tests = {
        "fpr_at_most_one_percent": bool(indexed.loc["candidate", "fpr"] <= 0.01),
        "fpr_not_above_original": bool(indexed.loc["candidate", "fpr"] <= indexed.loc["baseline", "fpr"]),
        "fpr_reduction_vs_previous_upper_bound_below_zero": intervals["previous"]["delta_fpr"]["upper"] < 0,
        "recall_vs_original_lower_bound_at_least_minus_3pp": intervals["baseline"]["delta_recall"]["lower"] >= -0.03,
        "recall_vs_previous_lower_bound_at_least_minus_3pp": intervals["previous"]["delta_recall"]["lower"] >= -0.03,
    }
    return {"passed": bool(all(tests.values())), "criteria": tests}


def evaluate(output):
    output = Path(output)
    if (output / "test_scores.csv").exists():
        raise FileExistsError("Test already evaluated; do not retune using this holdout")
    metadata = checked_run(output)
    protocol = json.loads((DIRECTORY / "protocol.json").read_text())
    models = controls(protocol)
    choice = json.loads((output / "selection.json").read_text())
    calibration = json.loads((output / "threshold_control.json").read_text())
    models["threshold_only"] = SpamModel(models["previous"].pipeline, calibration["threshold"], "threshold_only")
    candidate = joblib.load(output / "models" / f"{metadata['selected_model']}.joblib")
    if candidate.model_id != metadata["selected_model"] or candidate.threshold != choice["threshold"]:
        raise ValueError("Candidate identity or threshold differs from frozen selection")
    models["candidate"] = candidate
    frame, _ = load_frozen()
    test = frame[frame.split.eq("test")].reset_index(drop=True)
    rows = []
    for name, model in models.items():
        scores = model.score(test.text.tolist())
        test[f"{name}_score"] = scores
        test[f"{name}_prediction"] = (scores >= model.threshold).astype(int)
        rows.append({"model": name, "threshold": model.threshold,
                     **metrics(test.label, scores, model.threshold)})
    test.drop(columns="text").to_csv(output / "test_scores.csv", index=False)
    table = pd.DataFrame(rows).drop(columns="confusion_matrix")
    table.to_csv(output / "test_metrics.csv", index=False)
    intervals = {name: paired_intervals(test, test[f"{name}_prediction"].to_numpy(),
                                        test.candidate_prediction.to_numpy(),
                                        draws=protocol["uncertainty"]["draws"], seed=protocol["uncertainty"]["seed"])
                 for name in ["baseline", "previous", "threshold_only"]}
    write_json(output / "test_intervals.json", intervals)
    gate = promotion_gate(table, intervals)
    write_json(output / "evaluation_metadata.json", {
        "training_metadata_sha256": file_hash(output / "training_metadata.json"),
        "selected_model": candidate.model_id, "promotion_gate": gate,
        "deployment_status": "research candidate; historical demo unchanged",
        "test_rows": len(test), "new_test_sources": 1,
        "test_scope": "Untouched components from the same author corpus as added development data; not unseen-provider validation",
        "output_sha256": {p: file_hash(output / p) for p in ["test_scores.csv", "test_metrics.csv", "test_intervals.json"]},
    })
    print(table.to_string(index=False))
    print(json.dumps(gate, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit", "evaluate"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (fit if args.stage == "fit" else evaluate)(args.output)


if __name__ == "__main__":
    main()
