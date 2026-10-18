"""Evaluate the frozen historical model on predeclared positive-only SMS cohorts."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .demo import artifact_identity, load_verified_model
from .external_data import (COHORTS, PROTOCOL, ROOT, SOURCE, file_hash,
                            load_cohorts, without_placeholders)
from .data import sha256_text

MEMBERSHIPS = ROOT / "results/external/imc25/cohorts"
UNAVAILABLE = ("precision", "f1", "accuracy", "false_positive_rate", "roc_auc", "average_precision")


def recall_interval(prediction, groups, *, draws=5000, seed=42, confidence=.95):
    prediction, groups = np.asarray(prediction), np.asarray(groups)
    if (prediction.ndim != 1 or groups.ndim != 1 or not len(prediction)
            or len(prediction) != len(groups) or not np.isin(prediction, [0, 1]).all()
            or pd.isna(groups).any()):
        raise ValueError("Expected nonempty aligned binary predictions and nonmissing groups")
    if not isinstance(draws, int) or isinstance(draws, bool) or draws < 100:
        raise ValueError("draws must be an integer of at least 100")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    unique, inverse = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inverse)
    detected = np.bincount(inverse, weights=prediction)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    # Batches bound memory for the supplementary 15k-group cohort.
    for start in range(0, draws, 64):
        end = min(start + 64, draws)
        choice = rng.integers(len(unique), size=(end - start, len(unique)))
        samples[start:end] = detected[choice].sum(axis=1) / sizes[choice].sum(axis=1)
    alpha = (1 - confidence) / 2
    return dict(recall_lower=float(np.quantile(samples, alpha)),
                recall_upper=float(np.quantile(samples, 1 - alpha)),
                n_groups=len(unique), draws=draws, seed=seed, confidence=confidence)


def load_frozen_model(protocol, model_run):
    _, metadata, meta_hash, artifact_hash = artifact_identity(model_run)
    if (meta_hash != protocol["model_metadata_sha256"]
            or artifact_hash != protocol["model_sha256"]
            or metadata["selected_model"] != protocol["model_id"]):
        raise ValueError("Model differs from the artifact frozen before external scoring")
    # Also guard the importable preprocessing/inference code used by the artifact.
    for path, expected in metadata["source_sha256"].items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f"Historical training/inference source changed: {path}")
    model = load_verified_model(model_run)
    if model.threshold != protocol["validation_threshold"]:
        raise ValueError("Model threshold differs from the frozen protocol")
    return model


def check_memberships(cohorts, audit):
    if audit != json.loads((MEMBERSHIPS / "audit.json").read_text()):
        raise ValueError("Cohort audit differs from the version fixed before scoring")
    for name, frame in cohorts.items():
        expected = (MEMBERSHIPS / f"{name}_membership.csv").read_text()
        if frame.empty or frame.drop(columns="text").to_csv(index=False) != expected:
            raise ValueError(f"Cohort membership changed: {name}")


def evaluate_model(model, cohorts):
    """No fit calls, UI abstention, length truncation or external threshold search."""
    ledgers = []
    for name, frame in cohorts.items():
        for view in ["published", "placeholders_removed"]:
            texts = frame.text if view == "published" else frame.text.map(without_placeholders)
            score = model.score(texts)
            if not np.isfinite(score).all():
                raise ValueError("Nonfinite model scores")
            features = model.pipeline[:-1].transform(texts)
            ledger = pd.DataFrame(dict(cohort=name, view=view, source_row=frame.source_row,
                input_sha256=texts.map(sha256_text), score=score,
                validation_prediction=(score >= model.threshold).astype(int),
                default_prediction=model.pipeline.predict(texts),
                recognized_features=np.diff(features.tocsr().indptr)))
            ledgers.append(ledger)
    return pd.concat(ledgers, ignore_index=True)


def summarize(ledger, cohorts, protocol):
    rows, categories = [], []
    bootstrap = {k: protocol["bootstrap"][k] for k in ["draws", "seed", "confidence"]}
    for cohort in COHORTS:
        frame = cohorts[cohort].set_index("source_row")
        for view in protocol["input_views"]:
            part = ledger[(ledger.cohort == cohort) & (ledger.view == view)]
            if (part.source_row.duplicated().any() or set(part.source_row) != set(frame.index)
                    or not np.isfinite(part.score).all()):
                raise ValueError("Prediction ledger is incomplete, duplicated or invalid")
            aligned = frame.loc[part.source_row]
            for mode in protocol["decision_rules"]:
                prediction = part[f"{mode}_prediction"].to_numpy()
                interval = recall_interval(prediction, aligned.template_sha256, **bootstrap)
                n, tp = len(part), int(prediction.sum())
                rows.append(dict(cohort=cohort, view=view, threshold_mode=mode,
                    n_messages=n, n_ham=0, detected=tp, missed=n-tp, recall=tp/n, miss_rate=(n-tp)/n,
                    zero_feature_messages=int(part.recognized_features.eq(0).sum()), **interval,
                    **{metric: None for metric in UNAVAILABLE}))
                for category in sorted(aligned.category.unique()):
                    mask = aligned.category.to_numpy() == category
                    count, found = int(mask.sum()), int(prediction[mask].sum())
                    categories.append(dict(cohort=cohort, view=view, threshold_mode=mode,
                        category=category, n_messages=count, detected=found, missed=count-found,
                        recall=found/count))
    return pd.DataFrame(rows), pd.DataFrame(categories)


def run(output, model_run):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an unused output directory")
    protocol = json.loads(PROTOCOL.read_text())
    cohorts, audit = load_cohorts()
    check_memberships(cohorts, audit)
    model = load_frozen_model(protocol, model_run)
    ledger = evaluate_model(model, cohorts)
    metrics, categories = summarize(ledger, cohorts, protocol)
    sources = [SOURCE, PROTOCOL, MEMBERSHIPS / "audit.json"]
    sources += [MEMBERSHIPS / f"{c}_membership.csv" for c in COHORTS]
    sources += [ROOT / p for p in ["spam_detector/external_data.py", "spam_detector/external_eval.py",
                                   "spam_detector/demo.py", "scripts/verify_external.py"]]
    metadata = dict(protocol_sha256=file_hash(PROTOCOL), model_id=model.model_id,
        model_sha256=protocol["model_sha256"], model_metadata_sha256=protocol["model_metadata_sha256"],
        validation_threshold=model.threshold, native_decision="pipeline.predict",
        source_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in sources},
        versions={p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]},
        no_fit=True, no_threshold_selection=True, primary_result=protocol["primary_result"],
        scope="Recall on publisher-labeled positive reports. No ham, precision, F1, accuracy or FPR estimate. "
              "Pseudonymized convenience sample; dates/annotations not independently verified. "
              "Bootstrap conditions on fixed model and observed heuristic groups.")
    output.mkdir(parents=True, exist_ok=True)
    for name, table in [("predictions", ledger), ("metrics", metrics), ("categories", categories)]:
        table.to_csv(output / f"{name}.csv", index=False)
    metadata["output_sha256"] = {f"{name}.csv": file_hash(output / f"{name}.csv")
                                for name in ["predictions", "metrics", "categories"]}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(metrics[["cohort", "view", "threshold_mode", "n_messages", "detected", "missed", "recall",
                   "recall_lower", "recall_upper"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-run", type=Path, default=ROOT / "results/full")
    args = parser.parse_args()
    run(args.output, args.model_run)


if __name__ == "__main__":
    main()
