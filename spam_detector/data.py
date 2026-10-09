"""Validate first, deduplicate raw strings, then split normalized groups."""

import hashlib
import json
import re
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .preprocessing import normalized_key

LABEL_MAP = {"ham": 0, "spam": 1, "0": 0, "1": 1}


def template_key(text):
    """A conservative, label-independent template key used only for grouping.

    Mask URLs and digit sequences in messages longer than 30 normalized
    characters. Short texts retain their original normalized key to avoid
    grouping unrelated replies such as dates or authentication codes.
    This heuristic does not identify every campaign or semantic paraphrase.
    """
    key = normalized_key(text)
    if len(key) > 30:
        key = re.sub(r"https?://\S+|www\.\S+", "<url>", key)
        key = re.sub(r"\d+", "#", key)
    return key


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _label(value):
    if isinstance(value, str) and value in LABEL_MAP:
        return LABEL_MAP[value]
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and value in (0, 1):
        return int(value)
    raise ValueError(f"Invalid label {value!r}; expected ham/spam or integer 0/1")


def prepare_data(frame, grouping="normalized"):
    if grouping not in {"normalized", "template"}:
        raise ValueError("grouping must be normalized or template")
    if not {"sms", "label"}.issubset(frame.columns):
        raise ValueError("Dataset must contain sms and label columns")
    if frame.empty:
        raise ValueError("Dataset is empty")
    df = frame[["sms", "label"]].copy().reset_index(drop=True)
    for i, text in enumerate(df.sms):
        if not isinstance(text, str):
            raise ValueError(f"sms at source row {i} is missing or not a string")
        if not text.strip():
            raise ValueError(f"sms at source row {i} is empty/whitespace-only in training data")
    df["label"] = df.label.map(_label)
    if set(df.label) != {0, 1}:
        raise ValueError("Dataset must contain both ham=0 and spam=1")
    df["source_row"] = np.arange(len(df))
    df["message_id"] = df.sms.map(sha256_text)
    df["group_id"] = df.sms.map(template_key if grouping == "template" else normalized_key).map(sha256_text)
    for column, description in [("message_id", "exact text"), ("group_id", f"{grouping} text")]:
        counts = df.groupby(column).label.nunique()
        conflicts = counts[counts > 1].index
        if len(conflicts):
            rows = df.loc[df[column].isin(conflicts), ["source_row", "label", column]]
            raise ValueError(f"Conflicting labels for {description}: {len(conflicts)} groups; "
                             f"source rows (zero-based): {rows.head(20).to_dict('records')}")
    source_rows = df.groupby("message_id", sort=False).source_row.agg(list)
    clean = df.drop_duplicates("sms").copy().reset_index(drop=True)
    clean["source_rows"] = clean.message_id.map(source_rows).map(json.dumps)
    group_sizes = clean.groupby("group_id").size()
    normalized_sizes = clean.sms.map(normalized_key).value_counts()
    audit = {
        "input_rows": len(df), "exact_duplicates_removed": len(df) - len(clean),
        "deduplicated_rows": len(clean), "normalized_groups": len(normalized_sizes),
        "normalized_groups_with_multiple_raw_texts": int((normalized_sizes > 1).sum()),
        "additional_normalized_duplicates": int((normalized_sizes - 1).sum()),
        "grouping": grouping, "split_groups": len(group_sizes),
        "split_groups_with_multiple_raw_texts": int((group_sizes > 1).sum()),
        "conflicting_exact_groups": 0, "conflicting_normalized_groups": 0,
        "class_counts": {str(k): int(v) for k, v in clean.label.value_counts().items()},
    }
    return clean, audit


def load_data(path, grouping="normalized"):
    # Preserve literal strings like "NA" and numeric-looking messages as text.
    return prepare_data(pd.read_csv(path, dtype={"sms": str, "label": str}, keep_default_na=False), grouping)


def template_overlap(frame):
    """Report template overlap without exporting SMS text or altering splits."""
    annotated = frame.assign(template=frame.sms.map(template_key).map(sha256_text))
    crossing = annotated.groupby("template").filter(lambda group: group.split.nunique() > 1)
    train_keys = set(annotated.loc[annotated.split == "train", "template"])
    test = annotated[annotated.split == "test"]
    return {
        "rule": "Lowercase and whitespace; mask URLs and digit runs only when normalized length > 30",
        "crossing_templates": int(crossing.template.nunique()),
        "crossing_messages": len(crossing),
        "test_messages_with_train_template": int(test.template.isin(train_keys).sum()),
        "groups": [{"template_sha256": key,
                    "source_rows": [int(x) for x in group.source_row],
                    "splits": group.split.tolist()}
                   for key, group in crossing.groupby("template", sort=True)],
    }


def assert_disjoint(frame):
    for column in ("message_id", "group_id"):
        if frame.groupby(column).split.nunique().max() != 1:
            raise ValueError(f"Leakage: {column} occurs in multiple splits")
    if set(frame.split) != {"train", "validation", "test"}:
        raise ValueError("Expected three nonempty splits")
    for split, part in frame.groupby("split"):
        if set(part.label) != {0, 1}:
            raise ValueError(f"{split} must contain both classes; use more data")


def split_data(frame, seed=42, validation_size=0.2, test_size=0.2):
    if not (0 < validation_size < 1 and 0 < test_size < 1 and validation_size + test_size < 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than 1")
    groups = frame.drop_duplicates("group_id")[["group_id", "label"]]
    # Stratify unique homogeneous groups; never select a split using model scores.
    rest, test = train_test_split(groups, test_size=test_size, stratify=groups.label, random_state=seed)
    train, validation = train_test_split(
        rest, test_size=validation_size / (1 - test_size), stratify=rest.label, random_state=seed + 1)
    membership = {gid: name for name, part in [("train", train), ("validation", validation), ("test", test)]
                  for gid in part.group_id}
    result = frame.copy()
    result["split"] = result.group_id.map(membership)
    assert_disjoint(result)
    return result


def make_cv_folds(train, seed=42, n_splits=5):
    per_class = train.drop_duplicates("group_id").label.value_counts()
    if len(per_class) != 2 or per_class.min() < n_splits:
        raise ValueError(f"Need at least {n_splits} normalized groups per class inside train")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(train.sms, train.label, train.group_id))
    fold_ids = np.full(len(train), -1, dtype=int)
    for i, (fit, held_out) in enumerate(folds):
        if set(train.iloc[fit].group_id) & set(train.iloc[held_out].group_id):
            raise ValueError("Normalized group overlap inside CV")
        if set(train.iloc[fit].label) != {0, 1} or set(train.iloc[held_out].label) != {0, 1}:
            raise ValueError("Each CV fold must contain both classes")
        fold_ids[held_out] = i
    return folds, fold_ids
