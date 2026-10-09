"""Read-only inference helpers for the local demonstration."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import LinearSVC

from .evaluation import SpamModel

MAX_MESSAGE_CHARS = 5000
# Separators can add one character at each of the n - 1 boundaries.
MAX_ATTACK_INPUT_CHARS = (MAX_MESSAGE_CHARS + 1) // 2


def artifact_identity(run):
    """Inspect a trusted local run; hashes detect changes, not malicious authors."""
    run = Path(run).resolve()
    metadata_bytes = (run / "metadata.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    if metadata.get("status") != "complete":
        raise ValueError("The experiment has not completed")
    selected = metadata["selected_model"]
    info = metadata["models"][selected]
    artifact = (run / info["artifact"]).resolve()
    if not artifact.is_relative_to(run / "models"):
        raise ValueError("The model artifact must be inside this run's models directory")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if digest != info["artifact_sha256"]:
        raise ValueError("The model artifact does not match its recorded SHA-256")
    for package in ["numpy", "scipy", "scikit-learn", "joblib"]:
        if version(package) != metadata["versions"][package]:
            raise ValueError(f"Model requires {package}=={metadata['versions'][package]}")
    return artifact, metadata, hashlib.sha256(metadata_bytes).hexdigest(), digest


def load_verified_model(run):
    """Load only the CV-selected artifact after integrity and version checks."""
    artifact, meta, _, _ = artifact_identity(run)
    model = joblib.load(artifact)
    if not isinstance(model, SpamModel) or model.model_id != meta["selected_model"]:
        raise ValueError("Saved model identity differs from the experiment metadata")
    threshold = meta["models"][model.model_id]["threshold_selection"]["threshold"]
    if not np.isfinite(model.threshold) or model.threshold != threshold:
        raise ValueError("Saved threshold differs from the experiment metadata")
    return model


def inspect_message(model, text):
    """Return the saved decision plus exact linear contributions when available.

    Text is never cached or written. Out-of-vocabulary input has no lexical
    evidence even though the underlying classifier still returns an intercept.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a nonblank English SMS")
    if len(text) > MAX_MESSAGE_CHARS:
        raise ValueError(f"Use a message with at most {MAX_MESSAGE_CHARS:,} characters")
    features = model.pipeline[:-1].transform([text]).tocsr()
    score = float(model.score(text)[0])
    result = dict(prediction=int(model.predict(text)[0]), score=score,
                  threshold=float(model.threshold), score_kind=model.score_kind,
                  recognized_features=int(features.nnz), contributions=None)
    clf = model.pipeline.named_steps["clf"]
    if isinstance(clf, LinearSVC):
        names = model.pipeline.named_steps["features"].get_feature_names_out()
        values = features.data * clf.coef_[0, features.indices]
        intercept = float(clf.intercept_[0])
        if not np.isclose(intercept + values.sum(), score, atol=1e-10, rtol=0):
            raise ValueError("Feature contributions do not reconstruct the model score")
        result.update(intercept=intercept, contribution_sum=float(values.sum()),
                      contributions=sorted([dict(feature=str(names[index]), contribution=float(value))
                                            for index, value in zip(features.indices, values)],
                                           key=lambda r: (-abs(r["contribution"]), r["feature"])))
    return result
