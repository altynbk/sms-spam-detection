"""Small predefined searches; all vectorizers fit inside each training fold."""

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC, SVC

from .preprocessing import TextCleaner


@dataclass
class Experiment:
    model_id: str
    classifier: str
    feature_type: str
    preprocessing: str
    pipeline: Pipeline
    grid: list


def make_pipeline(classifier="linear_svc", feature_type="word", preprocessing="legacy", seed=42):
    classifiers = {
        "nb": MultinomialNB(alpha=0.1),
        "lr": LogisticRegression(max_iter=2000, random_state=seed),
        "svc": SVC(kernel="linear", random_state=seed),
        "rf": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
        "linear_svc": LinearSVC(C=1, dual="auto", max_iter=10000, random_state=seed),
    }
    if classifier not in classifiers:
        raise ValueError(f"Unknown classifier: {classifier}")
    word = TfidfVectorizer(lowercase=False, max_features=1000, ngram_range=(1, 2))
    char = TfidfVectorizer(lowercase=False, max_features=1000, analyzer="char_wb", ngram_range=(3, 5))
    if feature_type == "word":
        features = word
    elif feature_type == "char":
        features = char
    elif feature_type == "word_char":
        features = Pipeline([("union", FeatureUnion([("word", word), ("char", char)])),
                             ("normalize", Normalizer())])
    else:
        raise ValueError(f"Unknown features: {feature_type}")
    return Pipeline([("clean", TextCleaner(preprocessing)), ("features", features), ("clf", classifiers[classifier])])


def experiments(seed=42, quick=False):
    # Keep the four original estimators. Add controlled LinearSVC comparisons.
    definitions = [(clf, "word", "legacy") for clf in ("nb", "lr", "svc", "rf")]
    definitions += [("linear_svc", "word", "legacy"), ("linear_svc", "char", "legacy")]
    definitions += [("linear_svc", f, "gentle") for f in ("word", "char", "word_char")]
    definitions += [("linear_svc", "char", p) for p in ("no_punctuation", "no_currency")]
    result = []
    for classifier, feature_type, preprocessing in definitions:
        grid = []
        for budget in ([1000] if quick else [1000, 5000]):
            if feature_type == "word_char":
                params = {"features__union__word__max_features": [budget // 2],
                          "features__union__char__max_features": [budget // 2]}
            else:
                params = {"features__max_features": [budget]}
            if classifier == "nb":
                params["clf__alpha"] = [0.1] if quick else [0.1, 1.0]
            elif classifier == "rf":
                params.update({"clf__max_depth": [None] if quick else [None, 20],
                               "clf__min_samples_leaf": [1] if quick else [1, 2]})
            else:
                params.update({"clf__C": [1.0] if quick else [0.5, 1.0],
                               "clf__class_weight": [None] if quick else [None, "balanced"]})
            grid.append(params)
        model_id = f"{classifier}_{feature_type}_{preprocessing}"
        result.append(Experiment(model_id, classifier, feature_type, preprocessing,
                                 make_pipeline(classifier, feature_type, preprocessing, seed), grid))
    return result
