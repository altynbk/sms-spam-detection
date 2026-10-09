"""One-command experiment: python -m spam_detector.train --mode full."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid

from .attacks import ATTACK_TYPES, attack_messages
from .data import load_data, make_cv_folds, split_data, template_overlap
from .evaluation import (SpamModel, attack_success_rate, default_threshold, majority_baseline, metrics,
                         predictions, score_kind, select_threshold, spam_scores)
from .modeling import experiments

DATA_SOURCE = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
PUBLIC_DATA_SHA256 = "bbcb13af6558d89e007094a1d62a982ce2b03ce679f89cd88625de3c71cea76a"


@dataclass
class Config:
    mode: str = "full"
    data: str = "data/smshamspam.csv"
    output: str = "results/full"
    grouping: str = "normalized"
    seed: int = 42
    validation_size: float = 0.2
    test_size: float = 0.2
    cv_folds: int = 5
    max_fpr: float = 0.01
    jobs: int = 1
    intensities: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    repeats: int = 20
    attacks: tuple = ATTACK_TYPES
    export_error_text: bool = False


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    default=_json_default, allow_nan=False) + "\n", encoding="utf-8")


def _revision(root):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
        return {"commit": commit, "working_tree_dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "working_tree_dirty": None}


def _source_hashes(root):
    files = sorted((root / "spam_detector").glob("*.py")) + [root / "requirements.txt"]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}


def _row(spec):
    return {"model": spec.model_id, "classifier": spec.classifier,
            "feature_type": spec.feature_type, "preprocessing": spec.preprocessing}


def plot_robustness(runs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(exist_ok=True)
    names = list(runs.model.unique())
    colors = {name: plt.get_cmap("tab20")(i) for i, name in enumerate(names)}
    for attack_type, attack in runs.groupby("attack_type", sort=False):
        fig, axes = plt.subplots(2, 3, figsize=(17, 10))
        for row, mode in enumerate(["default", "validation"]):
            for name, part in attack[attack.threshold_mode == mode].groupby("model", sort=False):
                grouped = part.groupby("intensity")[["recall", "f1", "asr"]].agg(["mean", "std"])
                for ax, metric in zip(axes[row], ["recall", "f1", "asr"]):
                    x = grouped.index.to_numpy(float)
                    mean = grouped[(metric, "mean")].to_numpy(float)
                    std = grouped[(metric, "std")].fillna(0).to_numpy(float)
                    ax.plot(x, mean, marker=".", label=name, linewidth=1.2, color=colors[name])
                    ax.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                                    color=colors[name], alpha=0.07)
                    decision = "Native default" if mode == "default" else "Fixed validation threshold"
                    ax.set(title=f"{decision}: {metric.upper()}", xlabel="Intensity",
                           ylabel=metric.upper(), ylim=(-0.02, 1.02))
                    ax.grid(alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
        fig.suptitle(f"{attack_type}: both decision settings; mean ± SD over attack seeds only")
        fig.tight_layout(rect=(0, 0.12, 1, 0.95))
        fig.savefig(output / f"{attack_type}.png", dpi=150)
        plt.close(fig)


def run(config):
    if config.mode not in {"quick", "full"}:
        raise ValueError("mode must be quick or full")
    if not np.isfinite(config.max_fpr) or not 0 <= config.max_fpr <= 1:
        raise ValueError("max_fpr must be between 0 and 1")
    if config.repeats < 1 or config.cv_folds != 5:
        raise ValueError("Use positive repeats and the predefined 5-fold CV protocol")
    if not config.attacks or not config.intensities:
        raise ValueError("At least one attack and intensity are required")
    if len(set(config.attacks)) != len(config.attacks) or len(set(config.intensities)) != len(config.intensities):
        raise ValueError("Attack types and intensities must not contain duplicates")
    for attack in config.attacks:
        for intensity in config.intensities:
            attack_messages(["check"], [1], config.seed, intensity, attack)
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    out = Path(config.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an experiment: {out}. Choose a new --output directory.")
    frame, audit = load_data(config.data, grouping=config.grouping)
    frame = split_data(frame, config.seed, config.validation_size, config.test_size)
    train = frame[frame.split == "train"].copy()
    validation = frame[frame.split == "validation"].copy()
    test = frame[frame.split == "test"].copy()
    folds, fold_ids = make_cv_folds(train, config.seed, config.cv_folds)
    frame["cv_fold"] = pd.Series(index=frame.index, dtype="Int64")
    frame.loc[train.index, "cv_fold"] = fold_ids
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir()
    manifest = frame[["source_row", "source_rows", "message_id", "group_id", "label", "split", "cv_fold"]]
    manifest.to_csv(out / "splits.csv", index=False)
    write_json(out / "template_overlap.json", template_overlap(frame))
    dataset_sha = hashlib.sha256(Path(config.data).read_bytes()).hexdigest()
    publish_text = dataset_sha == PUBLIC_DATA_SHA256 or config.export_error_text
    specs = experiments(config.seed, config.mode == "quick")
    attack_seeds = [config.seed + 10000 + i for i in range(config.repeats)]
    metadata = {
        "protocol": "deduplicated-grouped-v2", "evaluation_revision": 2, "status": "training",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config), "python": sys.version, "platform": platform.platform(),
        "versions": {p: importlib.metadata.version(p) for p in
                     ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib", "pytest"]},
        "dataset": {"source": DATA_SOURCE if dataset_sha == PUBLIC_DATA_SHA256 else "user-supplied local CSV",
                    "path": config.data, "sha256": dataset_sha, "error_text_exported": publish_text},
        "git": _revision(root), "source_sha256": _source_hashes(root), "data_audit": audit,
        "labels": {"ham": 0, "spam": 1},
        "split_method": f"Stratified unique {config.grouping} groups, 2-stage split (seed, seed+1)",
        "split_counts": {name: {"total": len(part), "ham": int((part.label == 0).sum()),
                                "spam": int((part.label == 1).sum())}
                         for name, part in frame.groupby("split")},
        "split_manifest_sha256": hashlib.sha256((out / "splits.csv").read_bytes()).hexdigest(),
        "cv": {"method": "StratifiedGroupKFold", "folds": config.cv_folds, "shared_folds": True,
               "scoring": "f1 (spam=1)", "selection": "highest mean CV F1; ties by model_id alphabetically",
               "note": "Best searched CV score is not an independent quality estimate"},
        "attack_seeds": attack_seeds,
        "threshold_ties": {"default": "native pipeline.predict, including estimator-specific ties",
                           "validation": "score >= threshold predicts spam"},
        "search_spaces": {s.model_id: s.grid for s in specs}, "models": {},
    }
    write_json(out / "metadata.json", metadata)
    write_json(out / "config.json", asdict(config))
    cv_rows, validation_rows, fitted = [], [], {}
    cache = out / "_cv_cache"
    try:
        for spec in specs:
            print(f"CV {spec.model_id}: {len(list(ParameterGrid(spec.grid)))} candidates × 5 folds", flush=True)
            spec.pipeline.memory = joblib.Memory(cache, verbose=0)
            search = GridSearchCV(spec.pipeline, spec.grid, scoring="f1", cv=folds, n_jobs=config.jobs,
                                  refit=True, error_score="raise", return_train_score=False)
            search.fit(train.sms, train.label)
            model = search.best_estimator_
            model.memory = None
            scores = spam_scores(model, validation.sms)
            choice = select_threshold(validation.label, scores, config.max_fpr)
            bundle = SpamModel(model, choice["threshold"], spec.model_id)
            fitted[spec.model_id] = bundle
            path = out / "models" / f"{spec.model_id}.joblib"
            joblib.dump(bundle, path)
            probe = ["", "   ", "!!! £€$", "FREE offer! Call NOW", "Сәлем, hello 🙂"]
            reloaded = joblib.load(path)
            np.testing.assert_array_equal(bundle.predict(probe), reloaded.predict(probe))
            np.testing.assert_array_equal(bundle.score(probe), reloaded.score(probe))
            metadata["models"][spec.model_id] = {
                **_row(spec), "best_params": search.best_params_, "best_cv_f1": float(search.best_score_),
                "pipeline_params": model.get_params(deep=True), "score_kind": score_kind(model),
                "default_threshold": default_threshold(model), "threshold_selection": choice,
                "artifact": str(path.relative_to(out)),
                "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "refit_after_threshold": False,
            }
            for mode, threshold in [("default", default_threshold(model)), ("validation", bundle.threshold)]:
                pred = model.predict(validation.sms) if mode == "default" else predictions(scores, threshold)
                validation_rows.append({**_row(spec), "threshold_mode": mode, "threshold": threshold,
                                        **metrics(validation.label, scores, threshold, y_pred=pred)})
            res = search.cv_results_
            for j, params in enumerate(res["params"]):
                cv_rows.append({**_row(spec), "params": json.dumps(params, sort_keys=True),
                                "mean_cv_f1": float(res["mean_test_score"][j]),
                                "std_cv_f1": float(res["std_test_score"][j]),
                                "rank": int(res["rank_test_score"][j]),
                                **{f"fold_{i}_f1": float(res[f"split{i}_test_score"][j]) for i in range(5)}})
            write_json(out / "metadata.json", metadata)
    finally:
        if cache.exists():
            shutil.rmtree(cache)
    pd.DataFrame(cv_rows).to_csv(out / "cv_results.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(out / "validation_metrics.csv", index=False)
    # Freeze model choice and every threshold BEFORE any access to test scores.
    selected_model = min(fitted, key=lambda n: (-metadata["models"][n]["best_cv_f1"], n))
    metadata.update({"selected_model": selected_model, "status": "frozen_before_test"})
    write_json(out / "selection.json", {"selected_model": selected_model,
                                       "rule": metadata["cv"]["selection"],
                                       "thresholds": {n: b.threshold for n, b in fitted.items()}})
    write_json(out / "metadata.json", metadata)
    print(f"Frozen CV choice: {selected_model}. Starting final test evaluation.", flush=True)
    # A no-text reference for class imbalance, excluded from model/threshold selection.
    baseline_rows = [{"split": name, **majority_baseline(train.label, part.label)}
                     for name, part in [("validation", validation), ("test", test)]]
    pd.DataFrame(baseline_rows).to_csv(out / "baseline_metrics.csv", index=False)
    metadata["baseline"] = {"strategy": "DummyClassifier(strategy='prior')", "fit_split": "train",
                            "purpose": "class-imbalance reference; excluded from model selection"}
    clean_rows, errors, clean_scores, clean_predictions = [], [], {}, {}
    ytest = test.label.to_numpy()
    for spec in specs:
        bundle = fitted[spec.model_id]
        scores = spam_scores(bundle.pipeline, test.sms)
        clean_scores[spec.model_id] = scores
        for mode, threshold in [("default", default_threshold(bundle.pipeline)), ("validation", bundle.threshold)]:
            pred = bundle.pipeline.predict(test.sms) if mode == "default" else predictions(scores, threshold)
            clean_predictions[spec.model_id, mode] = pred
            clean_rows.append({**_row(spec), "threshold_mode": mode, "threshold": threshold,
                               **metrics(ytest, scores, threshold, y_pred=pred)})
            for position in np.flatnonzero(pred != ytest):
                sample = test.iloc[position]
                errors.append({**_row(spec), "threshold_mode": mode, "threshold": threshold,
                               "message_id": sample.message_id, "source_row": int(sample.source_row),
                               "text": sample.sms if publish_text else "[withheld: unrecognized dataset]",
                               "label": int(ytest[position]), "prediction": int(pred[position]),
                               "score": float(scores[position]), "score_kind": bundle.score_kind,
                               "error_type": "FP" if pred[position] else "FN"})
    pd.DataFrame(clean_rows).to_csv(out / "clean_metrics.csv", index=False)
    pd.DataFrame(errors, columns=[*_row(specs[0]), "threshold_mode", "threshold", "message_id", "source_row",
                                 "text", "label", "prediction", "score", "score_kind", "error_type"]
                 ).to_csv(out / "errors.csv", index=False)
    runs, attack_manifest = [], []
    spam_positions = np.flatnonzero(ytest == 1)
    for attack_type in config.attacks:
        for intensity in config.intensities:
            print(f"Attack {attack_type} p={intensity}: {config.repeats} seeds, all models", flush=True)
            for seed in attack_seeds:
                attacked = attack_messages(test.sms, ytest, seed, intensity, attack_type)
                attack_manifest.append({"attack_type": attack_type, "intensity": intensity, "seed": seed,
                                        "dataset_sha256": hashlib.sha256(json.dumps(attacked, ensure_ascii=False).encode()).hexdigest(),
                                        "changed_spam": sum(attacked[i] != test.sms.iloc[i] for i in spam_positions),
                                        "changed_ham": sum(attacked[i] != test.sms.iloc[i] for i in np.flatnonzero(ytest == 0))})
                for spec in specs:
                    bundle = fitted[spec.model_id]
                    # Ham is unchanged. Reuse its clean scores; all changed raw spam
                    # still passes through the ENTIRE saved preprocessing pipeline.
                    scores = clean_scores[spec.model_id].copy()
                    native_pred = clean_predictions[spec.model_id, "default"].copy()
                    if intensity != 0:
                        raw_spam = [attacked[i] for i in spam_positions]
                        scores[spam_positions] = spam_scores(bundle.pipeline, raw_spam)
                        native_pred[spam_positions] = bundle.pipeline.predict(raw_spam)
                    for mode, threshold in [("default", default_threshold(bundle.pipeline)), ("validation", bundle.threshold)]:
                        pred = native_pred if mode == "default" else predictions(scores, threshold)
                        runs.append({**_row(spec), "attack_type": attack_type, "intensity": intensity,
                                     "seed": seed, "threshold_mode": mode, "threshold": threshold,
                                     **metrics(ytest, scores, threshold, y_pred=pred),
                                     **attack_success_rate(ytest, clean_predictions[spec.model_id, mode], pred)})
            # Preserve completed draws even if a long run is interrupted.
            pd.DataFrame(runs).to_csv(out / "robustness_runs.csv", index=False, na_rep="undefined")
    run_frame = pd.DataFrame(runs)
    pd.DataFrame(attack_manifest).to_csv(out / "attack_manifest.csv", index=False)
    summary = run_frame.groupby(["model", "feature_type", "preprocessing", "threshold_mode", "attack_type", "intensity"])[
        ["precision", "recall", "f1", "roc_auc", "average_precision", "asr"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(out / "robustness_summary.csv", na_rep="undefined")
    plot_robustness(run_frame, out / "figures")
    metadata.update({"status": "complete", "elapsed_seconds": time.monotonic() - started,
                     "completed_utc": datetime.now(timezone.utc).isoformat(),
                     "robustness_rows": len(run_frame), "attack_datasets": len(attack_manifest)})
    write_json(out / "metadata.json", metadata)
    print(f"Complete: {out}; selected model: {selected_model}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument("--data", default="data/smshamspam.csv")
    parser.add_argument("--grouping", choices=["normalized", "template"], default="normalized",
                        help="Grouping unit shared by holdout splits and cross-validation")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-fpr", type=float, default=0.01)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--export-error-text", action="store_true",
                        help="Explicitly permit raw text in errors.csv for a non-bundled dataset")
    args = vars(parser.parse_args())
    args["output"] = args["output"] or f"results/{args['mode']}"
    if args["mode"] == "quick":
        args.update({"repeats": 2, "intensities": (0.0, 0.3)})
    run(Config(**args))


if __name__ == "__main__":
    main()
