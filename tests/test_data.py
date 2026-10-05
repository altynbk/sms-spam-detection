from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spam_detector.data import assert_disjoint, load_data, make_cv_folds, prepare_data, split_data
from spam_detector.preprocessing import normalized_key

DATA = Path(__file__).resolve().parents[1] / "data" / "smshamspam.csv"


def test_bundled_data_group_split_and_shared_folds():
    clean, audit = load_data(DATA)
    assert audit["input_rows"] == 5574
    assert audit["deduplicated_rows"] == 5171
    assert audit["normalized_groups"] == 5159
    assert audit["normalized_groups_with_multiple_raw_texts"] == 12
    split = split_data(clean)
    assert_disjoint(split)
    pd.testing.assert_frame_equal(split, split_data(clean))
    assert not split.sms.duplicated().any()
    assert split.groupby("group_id").split.nunique().max() == 1
    train = split[split.split == "train"]
    folds, ids = make_cv_folds(train)
    assert sorted(np.concatenate([v for _, v in folds])) == list(range(len(train)))
    assert set(ids) == set(range(5))
    assert train.assign(fold=ids).groupby("group_id").fold.nunique().max() == 1
    for fit, held in folds:
        assert not set(train.iloc[fit].group_id) & set(train.iloc[held].group_id)
    # Stratification is approximate by rows, because the unit is the group.
    assert max(abs(part.label.mean() - clean.label.mean()) for _, part in split.groupby("split")) < 0.01


@pytest.mark.parametrize("texts,kind", [(["same", "same"], "exact"), ([" Pay   NOW! ", "pay now!"], "normalized")])
def test_conflicting_labels_fail_loudly(texts, kind):
    with pytest.raises(ValueError, match=f"Conflicting labels for {kind} text.*source rows"):
        prepare_data(pd.DataFrame({"sms": texts, "label": [0, 1]}))


@pytest.mark.parametrize("value", [None, np.nan, 12, "", "   "])
def test_invalid_training_sms(value):
    with pytest.raises(ValueError, match="sms at source row 0"):
        prepare_data(pd.DataFrame({"sms": [value, "OK"], "label": [0, 1]}))


@pytest.mark.parametrize("label", [None, np.nan, 2, -1, "SPAM", True, 0.5])
def test_invalid_labels(label):
    with pytest.raises(ValueError, match="Invalid label"):
        prepare_data(pd.DataFrame({"sms": ["hello", "free"], "label": [label, 1]}))


def test_raw_text_preserved_and_labels_mapped():
    raw = "  Pay £50!!!\n"
    df, audit = prepare_data(pd.DataFrame({"sms": [raw, raw, "hello"], "label": ["spam", "spam", "ham"]}))
    assert df.sms.iloc[0] == raw
    assert df.label.tolist() == [1, 0]
    assert df.source_rows.iloc[0] == "[0, 1]"
    assert audit["exact_duplicates_removed"] == 1
    assert normalized_key("Hello!") != normalized_key("Hello")
    assert normalized_key("а") != normalized_key("a")  # Cyrillic is deliberately distinct.


def test_csv_literal_na_is_a_text(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("sms,label\nNA,ham\n123,spam\n", encoding="utf-8")
    df, _ = load_data(path)
    assert df.sms.tolist() == ["NA", "123"]
