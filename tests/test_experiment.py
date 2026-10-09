import json

import joblib
import numpy as np
import pandas as pd
import pytest

from spam_detector.evaluation import metrics
from spam_detector.train import Config, run
from scripts.verify_results import verify


@pytest.mark.parametrize("grouping", ["normalized", "template"])
def test_quick_end_to_end_synthetic_only(tmp_path, grouping):
    # This verifies orchestration, not real-world model quality.
    data = tmp_path / "synthetic.csv"
    pd.DataFrame({"sms": [(f"meeting hello schedule {chr(97+i//26)}{chr(97+i%26)} number {i}" if i % 3
                           else f"FREE prize CASH offer {chr(97+i//26)}{chr(97+i%26)} reference {i}")
                          for i in range(120)],
                  "label": [int(i % 3 == 0) for i in range(120)]}).to_csv(data, index=False)
    out = tmp_path / "run"
    config = Config(mode="quick", data=str(data), output=str(out), repeats=1, intensities=(0, 0.3),
                    grouping=grouping)
    metadata = run(config)
    assert metadata["status"] == "complete"
    assert len(metadata["models"]) == 11
    assert metadata["dataset"]["error_text_exported"] is False
    chosen = json.loads((out / "selection.json").read_text())
    expected = min(metadata["models"], key=lambda n: (-metadata["models"][n]["best_cv_f1"], n))
    assert chosen["selected_model"] == expected
    splits = pd.read_csv(out / "splits.csv")
    assert splits.groupby("group_id").split.nunique().max() == 1
    assert splits[splits.split != "train"].cv_fold.isna().all()
    clean = pd.read_csv(out / "clean_metrics.csv")
    attacked = pd.read_csv(out / "robustness_runs.csv", float_precision="round_trip")
    assert len(clean) == 22 and len(attacked) == 132
    zero = attacked[attacked.intensity == 0].merge(clean, on=["model", "threshold_mode"], suffixes=("_attack", "_clean"))
    np.testing.assert_array_equal(zero.f1_attack, zero.f1_clean)
    for model, group in attacked.groupby("model"):
        bundle = joblib.load(out / "models" / f"{model}.joblib")
        assert set(group[group.threshold_mode == "validation"].threshold) == {bundle.threshold}
        assert metadata["models"][model]["refit_after_threshold"] is False
        native_row = clean[(clean.model == model) & (clean.threshold_mode == "default")].iloc[0]
        raw = pd.read_csv(data)
        held_out = raw.iloc[splits.loc[splits.split == "test", "source_row"].to_numpy()]
        native = metrics(held_out.label, bundle.score(held_out.sms), native_row.threshold,
                         y_pred=bundle.pipeline.predict(held_out.sms))
        assert native["fp"] == native_row.fp and native["fn"] == native_row.fn
    assert pd.read_csv(out / "attack_manifest.csv").changed_ham.eq(0).all()
    verification = verify(out, data, all_attacks=True)
    assert verification["attack_metric_rows_recomputed"] == 132
    assert verification["baseline_metric_rows_recomputed"] == 2
    assert verification["validation_metric_rows_recomputed"] == 22
    with pytest.raises(FileExistsError, match="overwrite"):
        run(config)


@pytest.mark.parametrize("options", [{"attacks": ()}, {"intensities": ()},
                                     {"attacks": ("leet", "leet")}, {"intensities": (0.1, 0.1)}])
def test_invalid_attack_grid_fails_before_data_access(tmp_path, options):
    out = tmp_path / "invalid"
    with pytest.raises(ValueError, match="required|duplicates"):
        run(Config(data="missing.csv", output=str(out), **options))
    assert not out.exists()
