import json
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pytest
from sklearn.base import clone

from spam_detector.evaluation import SpamModel, default_threshold, spam_scores
from spam_detector.modeling import experiments, make_pipeline
from spam_detector.preprocessing import TextCleaner, clean_text, legacy_clean

TEXTS = ["hello meet for lunch", "see you at home", "call me tomorrow", "work dinner today",
         "WIN FREE £100 cash", "CLAIM prize NOW!", "FREE offer call cash", "win £500 reward"]
LABELS = [0, 0, 0, 0, 1, 1, 1, 1]
PROBES = ["", "   ", "!!! £$€", "Сәлем әлем 🙂", "FREE offer call NOW!", "see you at home"]


@pytest.mark.parametrize("spec", experiments(quick=True), ids=lambda s: s.model_id)
def test_round_trip_raw_inputs_including_new_python_process(tmp_path, spec):
    pipeline = spec.pipeline.fit(TEXTS, LABELS)
    bundle = SpamModel(pipeline, default_threshold(pipeline), spec.model_id)
    scores = bundle.score(PROBES)
    assert np.isfinite(scores).all()
    assert bundle.predict(PROBES).shape == (len(PROBES),)
    assert bundle.predict(PROBES[0]).shape == (1,)
    artifact = tmp_path / "model.joblib"
    joblib.dump(bundle, artifact)
    restored = joblib.load(artifact)
    np.testing.assert_array_equal(bundle.predict(PROBES), restored.predict(PROBES))
    np.testing.assert_array_equal(scores, restored.score(PROBES))
    script = ("import json,joblib,sys; m=joblib.load(sys.argv[1]); x=json.loads(sys.argv[2]); "
              "print(json.dumps({'pred':m.predict(x).tolist(),'scores':m.score(x).tolist(),"
              "'raw_pipeline':m.pipeline.predict(x).tolist()}))")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    process = subprocess.run([sys.executable, "-c", script, str(artifact), json.dumps(PROBES)],
                             cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    actual = json.loads(process.stdout)
    assert actual["pred"] == bundle.predict(PROBES).tolist()
    assert actual["scores"] == scores.tolist()
    assert actual["raw_pipeline"] == pipeline.predict(PROBES).tolist()


def test_legacy_exact_equivalence_and_symbol_ablations():
    raw = "  FREE: £50! Café — Сәлем\n"
    assert legacy_clean(raw) == "  free 50 caf  \n"
    assert clean_text(raw, "gentle") == "free: £50! café — сәлем"
    assert clean_text(raw, "no_punctuation") == "free £50 café  сәлем"
    assert clean_text(raw, "no_currency") == "free: 50! café — сәлем"
    assert TextCleaner().fit_transform([raw]) == [legacy_clean(raw)]


def test_vocabulary_and_idf_only_use_fold_fit_data():
    pipeline = make_pipeline()
    # Two independent folds: sentinel must not enter the other fold's vocabulary.
    one = clone(pipeline).fit(TEXTS[:2] + TEXTS[4:6], [0, 0, 1, 1])
    two = clone(pipeline).fit(["uniquefoldtoken ham", "uniquefoldtoken prize"], [0, 1])
    assert "uniquefoldtoken" not in one.named_steps["features"].vocabulary_
    assert "uniquefoldtoken" in two.named_steps["features"].vocabulary_
    before = one.named_steps["features"].idf_.copy()
    spam_scores(one, ["testonlysentinel uniquefoldtoken"])
    assert "testonlysentinel" not in one.named_steps["features"].vocabulary_
    np.testing.assert_array_equal(before, one.named_steps["features"].idf_)


@pytest.mark.parametrize("invalid", [None, 12, [None], [12], [["nested"]]])
def test_non_string_inputs_fail(invalid):
    pipeline = make_pipeline().fit(TEXTS, LABELS)
    with pytest.raises(TypeError, match="SMS|str"):
        pipeline.predict(invalid)


def test_empty_batch_is_explicit_error():
    with pytest.raises(ValueError, match="empty batch"):
        TextCleaner().transform([])


def test_one_shot_training_iterable_is_not_consumed_twice():
    expected = make_pipeline().fit(TEXTS, LABELS)
    actual = make_pipeline().fit(iter(TEXTS), LABELS)
    np.testing.assert_array_equal(expected.predict(PROBES), actual.predict(PROBES))
    np.testing.assert_array_equal(expected.decision_function(PROBES), actual.decision_function(PROBES))
