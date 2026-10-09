"""Conditional percentile intervals with class-stratified whole-group resampling."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _rates(counts):
    tn, fp, fn, tp = np.asarray(counts).T
    divide = lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    return dict(precision=divide(tp, tp + fp), recall=divide(tp, tp + fn),
                f1=divide(2 * tp, 2 * tp + fp + fn), fpr=divide(fp, tn + fp))


def group_intervals(y, prediction, groups, *, draws=5000, confidence=0.95, seed=42):
    """Resample groups within each class; never refit or select a threshold.

    Intervals condition on the fitted model, grouping, split and class composition.
    Percentile intervals can be discrete or degenerate for rare errors. They do
    not measure search/training uncertainty, coverage under drift or attack SD.
    """
    y, prediction, groups = map(np.asarray, (y, prediction, groups))
    if not (y.ndim == prediction.ndim == groups.ndim == 1 and
            0 < len(y) == len(prediction) == len(groups)):
        raise ValueError("Expected nonempty aligned one-dimensional arrays")
    if (not np.isin(y, [0, 1]).all() or set(y) != {0, 1} or
            not np.isin(prediction, [0, 1]).all() or pd.isna(groups).any()):
        raise ValueError("Expected both binary classes, binary predictions and nonmissing groups")
    if not isinstance(draws, int) or isinstance(draws, bool) or draws < 100:
        raise ValueError("draws must be an integer of at least 100")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    frame = pd.DataFrame(dict(label=y, prediction=prediction, group=groups))
    counts = []
    for _, part in frame.groupby("group", sort=True):
        if part.label.nunique() != 1:
            raise ValueError("Each group must contain a single true class")
        truth, pred = part.label.to_numpy(), part.prediction.to_numpy()
        counts.append([truth[0], *((((truth == a) & (pred == b)).sum())
                                  for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)])])
    counts = np.asarray(counts)
    pools = [counts[counts[:, 0] == label, 1:] for label in [0, 1]]
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, 4), dtype=int)
    for i in range(draws):
        samples[i] = sum(pool[rng.integers(len(pool), size=len(pool))].sum(axis=0)
                         for pool in pools)
    point, sampled = _rates(counts[:, 1:].sum(axis=0)), _rates(samples)
    alpha = (1 - confidence) / 2
    return [dict(metric=name, estimate=float(value),
                 lower=float(np.quantile(sampled[name], alpha)),
                 upper=float(np.quantile(sampled[name], 1 - alpha)),
                 confidence=confidence, draws=draws, seed=seed,
                 n_messages=len(y), n_groups=len(counts))
            for name, value in point.items()]


def report(run, *, draws=5000, confidence=0.95, seed=42):
    """Reconstruct fixed predictions from the complete error ledger and audit totals."""
    run = Path(run)
    files = ["selection.json", "splits.csv", "errors.csv", "clean_metrics.csv"]
    selected = json.loads((run / files[0]).read_text())["selected_model"]
    splits, errors, metrics = [pd.read_csv(run / f) for f in files[1:]]
    test = splits[splits.split == "test"].copy().set_index("message_id")
    if test.empty or not test.index.is_unique or test.index.isna().any():
        raise ValueError("Expected unique, nonmissing test message IDs")
    records = []
    for mode in ["default", "validation"]:
        ledger = errors[(errors.model == selected) & (errors.threshold_mode == mode)]
        if (ledger.message_id.duplicated().any() or
                not ledger.message_id.isin(test.index).all()):
            raise ValueError("Duplicate or unknown error message")
        pred = test.label.copy()
        for row in ledger.itertuples():
            if row.label != test.at[row.message_id, "label"] or row.prediction != 1 - row.label:
                raise ValueError("Error ledger contains inconsistent labels or predictions")
            pred.at[row.message_id] = row.prediction
        saved = metrics[(metrics.model == selected) & (metrics.threshold_mode == mode)]
        if len(saved) != 1:
            raise ValueError("Expected exactly one clean metric row per decision mode")
        observed = [int(((test.label == a) & (pred == b)).sum())
                    for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]]
        if not np.array_equal(observed, saved.iloc[0][["tn", "fp", "fn", "tp"]].to_numpy()):
            raise ValueError("Error ledger disagrees with clean confusion counts")
        for name, value in _rates(observed).items():
            if not np.isclose(value, saved.iloc[0][name], rtol=0, atol=1e-14):
                raise ValueError("Clean metric disagrees with reconstructed predictions")
        records.extend(dict(model=selected, threshold_mode=mode, **row)
                       for row in group_intervals(test.label, pred, test.group_id,
                                                  draws=draws, confidence=confidence, seed=seed))
    metadata = dict(method="class-stratified whole-group percentile bootstrap",
                    scope="Fixed model, decision rule, split and observed class/group composition; no refitting. "
                          "Excludes model-selection, training and domain-shift uncertainty. "
                          "Rare errors can produce discrete or degenerate intervals.",
                    draws=draws, confidence=confidence, seed=seed,
                    source_sha256={f: hashlib.sha256((run / f).read_bytes()).hexdigest() for f in files})
    return pd.DataFrame(records), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows, meta = report(args.run, draws=args.draws, confidence=args.confidence, seed=args.seed)
    rows.to_csv(args.run / "confidence_intervals.csv", index=False)
    (args.run / "confidence_intervals.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(rows.to_string(index=False))


if __name__ == "__main__":
    main()
