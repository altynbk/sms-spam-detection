import hashlib
from importlib.metadata import version
import json

import joblib
import numpy as np
import pytest

from spam_detector.demo import inspect_message, load_verified_model
from spam_detector.evaluation import SpamModel
from spam_detector.modeling import make_pipeline

TEXTS = ["hello meet for lunch", "see you at home", "call me tomorrow", "work dinner today",
         "WIN FREE cash", "CLAIM prize NOW", "FREE offer call cash", "win reward"]
LABELS = [0, 0, 0, 0, 1, 1, 1, 1]


def model_run(path):
    model = SpamModel(make_pipeline(preprocessing="gentle").fit(TEXTS, LABELS), -.1, "linear_svc_word_gentle")
    (path / "models").mkdir()
    artifact = path / "models/model.joblib"
    joblib.dump(model, artifact)
    metadata = dict(status="complete", selected_model=model.model_id, config=dict(mode="quick", grouping="normalized"),
                    versions={p: version(p) for p in ["numpy", "scipy", "scikit-learn", "joblib"]},
                    models={model.model_id: dict(artifact="models/model.joblib",
                            artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                            threshold_selection=dict(threshold=model.threshold))})
    (path / "metadata.json").write_text(json.dumps(metadata))
    return model, metadata


@pytest.mark.parametrize("features", ["word", "char", "word_char"])
def test_exact_contributions_reconstruct_saved_margin(features):
    model = SpamModel(make_pipeline(feature_type=features, preprocessing="gentle").fit(TEXTS, LABELS), -.1, "test")
    for text in ["FREE prize cash", "hello lunch tomorrow", "qzqxqzqx", "!!!", "Сәлем әлем"]:
        result = inspect_message(model, text)
        assert result["prediction"] == model.predict(text)[0]
        assert np.isclose(sum(r["contribution"] for r in result["contributions"]) + result["intercept"], model.score(text)[0], atol=1e-10)
    assert inspect_message(model, "!!!")["recognized_features"] == 0


@pytest.mark.parametrize("text", ["", "   ", "x" * 5001, None])
def test_demo_rejects_unusable_input(text, tmp_path):
    model, _ = model_run(tmp_path)
    with pytest.raises(ValueError):
        inspect_message(model, text)


def test_integrity_is_checked_before_unpickling(tmp_path, monkeypatch):
    expected, _ = model_run(tmp_path)
    actual = load_verified_model(tmp_path)
    np.testing.assert_array_equal(actual.predict(TEXTS), expected.predict(TEXTS))
    (tmp_path / "models/model.joblib").write_bytes(b"tampered")
    monkeypatch.setattr(joblib, "load", lambda _: pytest.fail("Unpickled before checking integrity"))
    with pytest.raises(ValueError, match="SHA-256"):
        load_verified_model(tmp_path)


def test_rejects_version_and_threshold_mismatch(tmp_path):
    _, meta = model_run(tmp_path)
    original = meta["versions"]["scikit-learn"]
    meta["versions"]["scikit-learn"] = "0.0"
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="requires scikit-learn"):
        load_verified_model(tmp_path)
    meta["versions"]["scikit-learn"] = original
    meta["models"][meta["selected_model"]]["threshold_selection"]["threshold"] = 2.0
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="threshold"):
        load_verified_model(tmp_path)
