import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spam_detector.uncertainty import group_intervals, report


def test_resamples_whole_groups_not_individual_messages():
    rows = group_intervals([0, 0, 1, 1], [0, 1, 1, 1], ["ham", "ham", "spam", "spam"], draws=100)
    f1 = next(row for row in rows if row["metric"] == "f1")
    assert f1["estimate"] == f1["lower"] == f1["upper"] == 0.8


def test_deterministic_non_degenerate_intervals():
    args = ([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 0, 1], list("abcdef"))
    rows = group_intervals(*args, draws=100)
    assert rows == group_intervals(*args, draws=100)
    assert all(0 <= r["lower"] <= r["upper"] <= 1 for r in rows)
    assert all(np.isfinite(r["estimate"]) for r in rows)
    assert rows[2]["lower"] < rows[2]["upper"]


@pytest.mark.parametrize("y,pred,groups", [([0, 1], [0], ["a", "b"]),
    ([0, 1], [0, 2], ["a", "b"]), ([0, 1], [0, 1], ["a", "a"]),
    ([0, 0], [0, 1], ["a", "b"]), ([0, 1], [0, 1], ["a", None])])
def test_rejects_invalid_inputs(y, pred, groups):
    with pytest.raises(ValueError):
        group_intervals(y, pred, groups, draws=100)


def test_saved_report_detects_missing_error(tmp_path):
    source = Path(__file__).resolve().parents[1] / "results/full"
    for name in ["selection.json", "splits.csv", "errors.csv", "clean_metrics.csv"]:
        shutil.copyfile(source / name, tmp_path / name)
    rows, meta = report(tmp_path, draws=100)
    assert len(rows) == 8
    assert rows.n_groups.unique().tolist() == [1032]
    assert len(meta["source_sha256"]) == 4
    errors = pd.read_csv(tmp_path / "errors.csv")
    selected = rows.model.iloc[0]
    errors.drop(errors[errors.model == selected].index[0]).to_csv(tmp_path / "errors.csv", index=False)
    with pytest.raises(ValueError, match="confusion counts"):
        report(tmp_path, draws=100)
