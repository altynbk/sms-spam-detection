"""Stateless, importable preprocessing used at both fit and prediction time."""

import re
import unicodedata

from sklearn.base import BaseEstimator, TransformerMixin


def raw_batch(texts):
    """A string is one SMS; otherwise require a nonempty iterable of strings."""
    if isinstance(texts, str):
        return [texts]
    try:
        values = list(texts)
    except TypeError as exc:
        raise TypeError("Expected a raw SMS string or an iterable of strings") from exc
    if not values:
        raise ValueError("An empty batch is not supported; an empty SMS string is valid")
    for i, value in enumerate(values):
        if not isinstance(value, str):
            raise TypeError(f"SMS at position {i} must be str, got {type(value).__name__}")
    return values


def normalized_key(text):
    """Only case and whitespace; deliberately retain punctuation and Unicode."""
    return " ".join(text.lower().split())


def legacy_clean(text):
    """Exactly the original notebook's lowercase + ASCII regex cleanup."""
    return re.sub(r"[^a-z0-9\s]", "", text.lower())


def clean_text(text, mode):
    if mode == "legacy":
        return legacy_clean(text)
    text = normalized_key(text)
    if mode == "gentle":
        return text
    if mode == "no_punctuation":
        return "".join(c for c in text if not unicodedata.category(c).startswith("P"))
    if mode == "no_currency":
        return "".join(c for c in text if unicodedata.category(c) != "Sc")
    raise ValueError(f"Unknown preprocessing mode: {mode}")


class TextCleaner(TransformerMixin, BaseEstimator):
    def __init__(self, mode="legacy"):
        self.mode = mode

    def fit(self, X, y=None):
        raw_batch(X)
        clean_text("", self.mode)
        return self

    def transform(self, X):
        return [clean_text(text, self.mode) for text in raw_batch(X)]

    def fit_transform(self, X, y=None, **fit_params):
        # Materialize one-shot iterables once; TransformerMixin otherwise calls
        # fit and transform on the same already-consumed generator.
        values = raw_batch(X)
        return self.fit(values, y).transform(values)
