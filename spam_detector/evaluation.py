"""Continuous scores, validation-only thresholds and paired attack metrics."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


def score_kind(model):
    return "probability" if hasattr(model, "predict_proba") else "decision_function"


def spam_scores(model, texts):
    if score_kind(model) == "probability":
        pos = list(model.classes_).index(1)
        return np.asarray(model.predict_proba(texts)[:, pos])
    if list(model.classes_) != [0, 1]:
        raise ValueError("Expected fitted classes [0, 1]")
    return np.asarray(model.decision_function(texts))


def default_threshold(model):
    return 0.5 if score_kind(model) == "probability" else 0.0


def predictions(scores, threshold):
    # Explicit tie policy shared by clean, attacked and deployed predictions.
    return (np.asarray(scores) >= threshold).astype(int)


def _arrays(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores) or not len(y):
        raise ValueError("Expected nonempty, aligned one-dimensional labels and scores")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Expected binary labels and finite continuous scores")
    return y, scores


def metrics(y, scores, threshold, *, y_pred=None):
    """Use native predictions for default mode, or an explicit tuned threshold."""
    y, scores = _arrays(y, scores)
    pred = predictions(scores, threshold) if y_pred is None else np.asarray(y_pred)
    if pred.shape != y.shape or not np.isin(pred, [0, 1]).all():
        raise ValueError("Predictions must be aligned binary labels")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, scores)) if len(set(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, scores)) if np.any(y == 1) else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fpr": float(fp / (fp + tn)) if fp + tn else None,
        "n_ham": int(fp + tn), "n_spam": int(fn + tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def select_threshold(y, scores, max_fpr=0.01):
    """Max recall subject to empirical FPR; ties: fewer FP, then higher threshold."""
    y, scores = _arrays(y, scores)
    if not np.isfinite(max_fpr) or not 0 <= max_fpr <= 1:
        raise ValueError("max_fpr must be finite and between 0 and 1")
    if set(y) != {0, 1}:
        raise ValueError("Validation must contain both ham and spam")
    # Include a finite reject-all threshold. Evaluate tied scores as one block.
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_y = scores[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    tp = np.r_[0, np.cumsum(sorted_y == 1)[ends]]
    fp = np.r_[0, np.cumsum(sorted_y == 0)[ends]]
    thresholds = np.r_[np.nextafter(scores.max(), np.inf), sorted_scores[ends]]
    n_ham = int((y == 0).sum())
    feasible = np.flatnonzero(fp / n_ham <= max_fpr)
    best = max(feasible, key=lambda i: (tp[i], -fp[i], thresholds[i]))
    threshold = float(thresholds[best])
    return {
        "threshold": threshold, "max_fpr": float(max_fpr), "n_ham": n_ham,
        "fpr_resolution": 1 / n_ham, "allowed_false_positives": int(np.floor(max_fpr * n_ham)),
        "rule": "maximize recall; ties minimize FP, then maximize threshold; score >= threshold",
        "validation_metrics": metrics(y, scores, threshold),
    }


def attack_success_rate(y, clean_pred, attacked_pred):
    y, clean_pred, attacked_pred = map(np.asarray, (y, clean_pred, attacked_pred))
    if not (y.ndim == clean_pred.ndim == attacked_pred.ndim == 1 and
            len(y) == len(clean_pred) == len(attacked_pred)):
        raise ValueError("ASR requires aligned one-dimensional arrays")
    if not all(np.isin(v, [0, 1]).all() for v in (y, clean_pred, attacked_pred)):
        raise ValueError("ASR requires binary labels and predictions")
    eligible = (y == 1) & (clean_pred == 1)
    denominator = int(eligible.sum())
    numerator = int((eligible & (attacked_pred == 0)).sum())
    return {"asr": numerator / denominator if denominator else None,
            "asr_numerator": numerator, "asr_denominator": denominator}


@dataclass
class SpamModel:
    """Persist the fitted raw-text pipeline and validation threshold together."""

    pipeline: object
    threshold: float
    model_id: str

    def predict(self, texts):
        return predictions(self.score(texts), self.threshold)

    def score(self, texts):
        return spam_scores(self.pipeline, texts)

    @property
    def score_kind(self):
        return score_kind(self.pipeline)
