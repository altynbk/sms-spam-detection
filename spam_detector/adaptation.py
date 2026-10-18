"""Refit on expanded development data, then evaluate one frozen candidate."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd

from .adaptation_data import DIRECTORY
from .adaptation_protocol import load_frozen
from .evaluation import SpamModel, metrics
from .external_data import ROOT, file_hash
from .modeling import make_pipeline

SOURCE_FILES = ["spam_detector/adaptation.py", "spam_detector/adaptation_data.py",
                "spam_detector/adaptation_protocol.py", "spam_detector/modeling.py",
                "spam_detector/preprocessing.py", "spam_detector/evaluation.py",
                "data/external/adaptation/protocol.json", "results/adaptation/protocol/membership.csv"]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sample_weights(frame):
    """Equal class, source-within-class, and group-within-cell mass."""
    if set(frame.label) != {0, 1} or frame.groupby("group_id").label.nunique().max() != 1:
        raise ValueError("Expected two classes and homogeneous campaign groups")
    result = pd.Series(0.0, index=frame.index)
    for label, labeled in frame.groupby("label"):
        n_sources = labeled.source.nunique()
        for source, part in labeled.groupby("source"):
            sizes = part.groupby("group_id").group_id.transform("size")
            result.loc[part.index] = len(frame) / (2 * n_sources * part.group_id.nunique() * sizes)
    return result.to_numpy()


def validation_choice(frame, scores, max_fpr=0.01):
    """Select using validation alone, with a separate ham constraint per source."""
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(frame),) or not np.isfinite(scores).all() or not len(frame):
        raise ValueError("Expected aligned finite validation scores")
    if set(frame.label) != {0, 1} or not 0 <= max_fpr <= 1:
        raise ValueError("Expected both classes and an FPR limit in [0, 1]")
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered_scores)), len(scores) - 1]
    thresholds = np.r_[np.nextafter(scores.max(), np.inf), ordered_scores[ends]]
    labels = frame.label.to_numpy()[order]
    sources = frame.source.to_numpy()[order]
    recalls, fprs = [], []
    for source in sorted(set(sources)):
        for label, destination in [(1, recalls), (0, fprs)]:
            mask = (sources == source) & (labels == label)
            if mask.any():
                destination.append(np.r_[0, np.cumsum(mask)[ends]] / mask.sum())
    macro_recall = np.mean(recalls, axis=0)
    macro_fpr = np.mean(fprs, axis=0)
    feasible = np.flatnonzero(np.all(np.asarray(fprs) <= max_fpr, axis=0))
    best = max(feasible, key=lambda i: (macro_recall[i], -macro_fpr[i], thresholds[i]))
    return {"threshold": float(thresholds[best]), "macro_recall": float(macro_recall[best]),
            "macro_fpr": float(macro_fpr[best]), "max_source_fpr": float(np.max(fprs, axis=0)[best])}


def candidate_pipeline(feature, c, protocol):
    model = make_pipeline("linear_svc", feature, "gentle", seed=protocol["seed"])
    settings = {"clf__C": c, "clf__max_iter": 20000, "clf__class_weight": None}
    if feature == "word_char":
        settings.update(features__union__word__max_features=protocol["word_features"],
                        features__union__char__max_features=protocol["char_features"])
    else:
        settings["features__max_features"] = protocol[f"{feature}_features"]
    return model.set_params(**settings)


def fit(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an unused run directory")
    protocol = json.loads((DIRECTORY / "protocol.json").read_text())
    frame, _ = load_frozen()
    train = frame[frame.split.eq("train")].reset_index(drop=True)
    validation = frame[frame.split.eq("validation")].reset_index(drop=True)
    weights = sample_weights(train)
    (output / "models").mkdir(parents=True)
    choices, metric_rows, score_rows, artifacts = [], [], [], {}
    for feature in protocol["candidate_features"]:
        for c in protocol["candidate_C"]:
            identity = f"adapted_{feature}_c{c:g}"
            pipeline = candidate_pipeline(feature, c, protocol)
            pipeline.fit(train.text.tolist(), train.label, clf__sample_weight=weights)
            scores = pipeline.decision_function(validation.text.tolist())
            choice = {"model": identity, "feature": feature, "C": c,
                      **validation_choice(validation, scores)}
            wrapped = SpamModel(pipeline, choice["threshold"], identity)
            path = output / "models" / f"{identity}.joblib"
            joblib.dump(wrapped, path)
            artifacts[identity] = file_hash(path)
            choices.append(choice)
            scored = validation.drop(columns="text").assign(model=identity, score=scores)
            score_rows.append(scored)
            for source, part in scored.groupby("source"):
                metric_rows.append({"model": identity, "source": source,
                                    **metrics(part.label, part.score, choice["threshold"])})
            print(json.dumps(choice), flush=True)
    # Stable ordering is the predeclared feature/C tie break.
    best = max(choices, key=lambda row: (row["macro_recall"], -row["macro_fpr"]))
    pd.DataFrame(choices).to_csv(output / "candidates.csv", index=False)
    pd.DataFrame(metric_rows).drop(columns="confusion_matrix").to_csv(output / "validation_metrics.csv", index=False)
    pd.concat(score_rows, ignore_index=True).to_csv(output / "validation_scores.csv", index=False)
    write_json(output / "selection.json", best)
    summary = train.assign(weight=weights).groupby(["source", "label"]).agg(
        messages=("source_row", "size"), total_weight=("weight", "sum")).reset_index()
    summary.to_csv(output / "training_weights.csv", index=False)
    names = ["candidates.csv", "validation_metrics.csv", "validation_scores.csv", "selection.json", "training_weights.csv"]
    write_json(output / "training_metadata.json", {
        "python": platform.python_version(),
        "versions": {package: importlib.metadata.version(package) for package in ["scikit-learn", "numpy", "pandas", "scipy", "joblib"]},
        "selected_model": best["model"], "source_sha256": {p: file_hash(ROOT / p) for p in SOURCE_FILES},
        "artifacts": artifacts, "output_sha256": {p: file_hash(output / p) for p in names},
        "train_rows": len(train), "validation_rows": len(validation),
        "test_predictions_during_selection": 0,
    })
    return best


def checked_run(output):
    output = Path(output)
    metadata = json.loads((output / "training_metadata.json").read_text())
    for path, digest in metadata["source_sha256"].items():
        if file_hash(ROOT / path) != digest:
            raise ValueError(f"Training source changed: {path}")
    for package, version in metadata["versions"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Runtime version changed: {package}")
    for path, digest in metadata["output_sha256"].items():
        if file_hash(output / path) != digest:
            raise ValueError(f"Training output changed: {path}")
    for model, digest in metadata["artifacts"].items():
        if file_hash(output / "models" / f"{model}.joblib") != digest:
            raise ValueError(f"Artifact changed: {model}")
    return metadata


def paired_intervals(frame, baseline, candidate, draws=5000, seed=20261018):
    """Conditional paired class-stratified group bootstrap, with identical draws."""
    if draws < 1 or frame.groupby("group_id").label.nunique().max() != 1:
        raise ValueError("Expected positive draws and homogeneous groups")
    frame = frame.reset_index(drop=True)
    matrices = []
    for label in [0, 1]:
        groups = []
        for _, part in frame[frame.label.eq(label)].groupby("group_id", sort=True):
            positions = part.index.to_numpy()
            values = []
            for prediction in [baseline, candidate]:
                positive = int(np.asarray(prediction)[positions].sum())
                values.extend([len(part) - positive, positive, 0, 0] if label == 0
                              else [0, 0, len(part) - positive, positive])
            groups.append(values)
        if not groups:
            raise ValueError("Each evaluation source must have both classes")
        matrices.append(np.asarray(groups))
    rng = np.random.default_rng(seed)
    values = {f"{model}_{metric}": [] for model in ["baseline", "candidate", "delta"]
              for metric in ["f1", "recall", "fpr", "precision"]}
    for _ in range(draws):
        counts = sum(m[rng.integers(0, len(m), len(m))].sum(axis=0) for m in matrices).reshape(2, 4)
        estimates = []
        for tn, fp, fn, tp in counts:
            estimates.append({"f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
                              "recall": tp / (tp + fn), "fpr": fp / (fp + tn),
                              "precision": tp / (tp + fp) if tp + fp else 0})
        for metric in estimates[0]:
            for i, model in enumerate(["baseline", "candidate"]):
                values[f"{model}_{metric}"].append(float(estimates[i][metric]))
            values[f"delta_{metric}"].append(float(estimates[1][metric] - estimates[0][metric]))
    return {key: {"lower": float(np.quantile(v, 0.025)), "upper": float(np.quantile(v, 0.975))}
            for key, v in values.items()}


def evaluate(output):
    output = Path(output)
    if (output / "test_scores.csv").exists():
        raise FileExistsError("This run already has test predictions; do not re-tune on this test")
    metadata = checked_run(output)
    protocol = json.loads((DIRECTORY / "protocol.json").read_text())
    frame, _ = load_frozen()
    test = frame[frame.split.eq("test")].copy()
    baseline_path = ROOT / "results/full/models/linear_svc_word_gentle.joblib"
    if file_hash(baseline_path) != protocol["baseline_model_sha256"]:
        raise ValueError("Frozen baseline artifact differs")
    baseline = joblib.load(baseline_path)
    candidate = joblib.load(output / "models" / f"{metadata['selected_model']}.joblib")
    choice = json.loads((output / "selection.json").read_text())
    if baseline.threshold != protocol["baseline_threshold"] or candidate.threshold != choice["threshold"]:
        raise ValueError("Saved threshold mismatch")
    if candidate.model_id != metadata["selected_model"]:
        raise ValueError("Candidate identity mismatch")
    test["baseline_score"] = baseline.score(test.text.tolist())
    test["candidate_score"] = candidate.score(test.text.tolist())
    test["baseline_prediction"] = (test.baseline_score >= baseline.threshold).astype(int)
    test["candidate_prediction"] = (test.candidate_score >= candidate.threshold).astype(int)
    test.drop(columns="text").to_csv(output / "test_scores.csv", index=False)
    rows, intervals = [], {}
    for source, part in test.groupby("source", sort=True):
        for name, threshold in [("baseline", baseline.threshold), ("candidate", candidate.threshold)]:
            rows.append({"source": source, "model": name, "threshold": threshold,
                         **metrics(part.label, part[f"{name}_score"], threshold)})
        intervals[source] = paired_intervals(part, part.baseline_prediction.to_numpy(),
                                            part.candidate_prediction.to_numpy(),
                                            **{k: protocol["uncertainty"][k] for k in ["draws", "seed"]})
    table = pd.DataFrame(rows).drop(columns="confusion_matrix")
    table.to_csv(output / "test_metrics.csv", index=False)
    write_json(output / "test_intervals.json", intervals)
    limits = []
    for source, part in table.groupby("source"):
        a = part[part.model.eq("candidate")].iloc[0]
        b = part[part.model.eq("baseline")].iloc[0]
        limits.append(bool(a.fpr <= b.fpr and a.fpr <= 0.01))
    promote = all(limits) and intervals["sms_history"]["delta_f1"]["lower"] > 0
    write_json(output / "evaluation_metadata.json", {
        "training_metadata_sha256": file_hash(output / "training_metadata.json"),
        "baseline_sha256": file_hash(baseline_path), "selected_model": candidate.model_id,
        "promotion_gate_passed": bool(promote),
        "deployment_status": "research_candidate; demo unchanged",
        "test_rows": len(test), "output_sha256": {p: file_hash(output / p) for p in
            ["test_scores.csv", "test_metrics.csv", "test_intervals.json"]},
    })
    print(table.to_string(index=False))
    print(f"Promotion gate passed: {promote}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit", "evaluate"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (fit if args.stage == "fit" else evaluate)(args.output)


if __name__ == "__main__":
    main()
