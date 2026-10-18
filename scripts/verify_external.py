"""Recompute external scores from the saved pipeline and check all published rows."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from spam_detector.data import sha256_text
from spam_detector.external_data import COHORTS, PROTOCOL, file_hash, load_cohorts, without_placeholders
from spam_detector.external_eval import check_memberships, load_frozen_model, summarize


def verify(run, model_run):
    run = Path(run)
    metadata = json.loads((run / "metadata.json").read_text())
    protocol = json.loads(PROTOCOL.read_text())
    for name, digest in metadata["source_sha256"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError(f"Evaluation source changed: {name}")
    for name, digest in metadata["output_sha256"].items():
        if file_hash(run / name) != digest:
            raise ValueError(f"Result bytes changed: {name}")
    if (metadata["protocol_sha256"] != file_hash(PROTOCOL)
            or metadata["model_sha256"] != protocol["model_sha256"]
            or metadata["model_metadata_sha256"] != protocol["model_metadata_sha256"]
            or metadata["model_id"] != protocol["model_id"]
            or metadata["validation_threshold"] != protocol["validation_threshold"]
            or metadata["no_fit"] is not True or metadata["no_threshold_selection"] is not True):
        raise ValueError("Evaluation metadata differs from the frozen protocol")
    cohorts, audit = load_cohorts()
    check_memberships(cohorts, audit)
    model = load_frozen_model(protocol, model_run)
    ledger = pd.read_csv(run / "predictions.csv", float_precision="round_trip")
    keys = ["cohort", "view", "source_row"]
    expected = {(cohort, view, int(i)) for cohort, frame in cohorts.items()
                for view in protocol["input_views"] for i in frame.source_row}
    if ledger.duplicated(keys).any() or set(ledger[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Incomplete or duplicate prediction keys")
    for cohort in COHORTS:
        frame = cohorts[cohort].set_index("source_row")
        for view in protocol["input_views"]:
            part = ledger[(ledger.cohort == cohort) & (ledger.view == view)]
            texts = frame.loc[part.source_row].text
            if view == "placeholders_removed":
                texts = texts.map(without_placeholders)
            if part.input_sha256.tolist() != texts.map(sha256_text).tolist():
                raise ValueError("Inference text hashes differ")
            # Bypass SpamModel.score: call the fitted feature transform and classifier directly.
            features = model.pipeline[:-1].transform(texts)
            scores = model.pipeline.named_steps["clf"].decision_function(features)
            np.testing.assert_allclose(scores, part.score, rtol=0, atol=1e-14)
            np.testing.assert_array_equal(model.pipeline.named_steps["clf"].predict(features), part.default_prediction)
            np.testing.assert_array_equal(scores >= protocol["validation_threshold"], part.validation_prediction)
            np.testing.assert_array_equal(np.diff(features.tocsr().indptr), part.recognized_features)
    metrics, categories = summarize(ledger, cohorts, protocol)
    for name, recomputed in [("metrics", metrics), ("categories", categories)]:
        saved = pd.read_csv(run / f"{name}.csv", float_precision="round_trip")
        # None-valued unavailable metrics serialize as empty CSV cells.
        recomputed = recomputed.infer_objects().fillna(np.nan)
        pd.testing.assert_frame_equal(saved, recomputed, check_dtype=False, check_exact=False, rtol=0, atol=1e-14)
    for row in metrics.itertuples(index=False):
        part = ledger[(ledger.cohort == row.cohort) & (ledger.view == row.view)]
        prediction = part[f"{row.threshold_mode}_prediction"]
        assert row.detected == int(prediction.eq(1).sum())
        assert row.missed == int(prediction.eq(0).sum())
        assert row.recall == row.detected / row.n_messages
        assert row.n_ham == 0
    return dict(status="passed", model_fitting=False, protocol_sha256=file_hash(PROTOCOL),
                prediction_rows=len(ledger), binary_decisions_checked=len(ledger)*2,
                summary_rows=len(metrics), category_rows=len(categories),
                all_scores_and_decisions_recomputed=True, memberships_reconstructed=True,
                intervals_recomputed=True, output_hashes=metadata["output_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model-run", type=Path, default=ROOT / "results/full")
    args = parser.parse_args()
    report = verify(args.run, args.model_run)
    (args.run / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
