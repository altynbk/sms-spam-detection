import csv

import pandas as pd
import pytest

from spam_detector.adaptation_data import (
    annotate, campaign_key, clean_groups, exclude_reference_overlap, read_sms_history,
)
from spam_detector.adaptation_protocol import partition


def table(texts, labels):
    return annotate(pd.DataFrame({"text": texts, "label": labels,
                                  "source_row": range(len(texts))}), "fixture")


def test_campaign_key_masks_entities_and_compatibility_characters():
    assert campaign_key("PAY 42 at https://a.example!") == campaign_key("ＰＡＹ 73 at www.b.example")
    assert campaign_key("PAY 42 at <URL>") == campaign_key("PAY 73 at https://b.example")
    assert campaign_key("meet for dinner") != campaign_key("claim your prize")


def test_conflicts_are_excluded_as_whole_groups_and_duplicates_are_removed():
    frame = table(["Pay 42 now", "Pay 71 now", "See you soon", "SEE  YOU SOON"], [0, 1, 0, 0])
    result, audit = clean_groups(frame)
    assert result.text.tolist() == ["See you soon"]
    assert audit["conflicting_rows"] == 2
    assert audit["duplicates_removed"] == 1


def test_sms_history_excludes_redacted_ambiguous_and_malformed_records(tmp_path):
    path = tmp_path / "history.csv"
    records = [["date-received", "date-read", "sender", "category", "text"]]
    for category, text in [("spam", "Claim prize"), ("gov", "Weather warning"),
                           ("gov", "<REDACTED>"), ("ads", "New offer"),
                           ("notifs", "<REDACTED>"), ("OTP", "<REDACTED>")]:
        records.append(["2025-02-01 11:10:00", "", "sender", category, text])
    records.append(["2025-02-01 11:10:00", "", "sender", "ads", "notifs", "<REDACTED>"])
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(records)
    result, audit = read_sms_history(path)
    assert result.label.tolist() == [1, 0]
    assert result.source_row.tolist() == [0, 1]
    assert audit["malformed_source_rows"] == [6]
    assert audit["excluded_categories_or_redactions"] == 4
    assert "sender" not in result


def test_overlap_audit_removes_exact_templates_and_near_copies():
    reference = "Your delivery has arrived at the local collection centre today. Please collect it before closing time."
    frame = table([reference, reference + " Thanks.", "The family picnic is on Sunday by the lake."], [1, 1, 0])
    result, audit = exclude_reference_overlap(frame, [reference])
    assert result.source_row.tolist() == [2]
    assert audit["campaign_matches"] == 1
    assert audit["near_matches"] == 2
    assert audit["total_excluded"] == 2


@pytest.mark.parametrize("references", [[], [""], [None]])
def test_overlap_requires_valid_references(references):
    with pytest.raises(ValueError, match="reference"):
        exclude_reference_overlap(table(["hello"], [0]), references)


def test_invalid_timestamp_is_rejected(tmp_path):
    path = tmp_path / "history.csv"
    path.write_text("date-received,date-read,sender,category,text\ninvalid,,sender,spam,hello\n")
    with pytest.raises(ValueError):
        read_sms_history(path)


def test_partition_is_deterministic_and_keeps_groups_together():
    frame = pd.DataFrame({"group_id": [f"g{i}" for i in range(50)] * 2,
                          "label": [i % 2 for i in range(50)] * 2})
    first = partition(frame, test_fraction=0.2)
    pd.testing.assert_frame_equal(first, partition(frame, test_fraction=0.2))
    assert first.groupby("group_id").split.nunique().max() == 1
    assert first.groupby("split").label.nunique().eq(2).all()
    assert first.split.value_counts().to_dict() == {"train": 60, "test": 20, "validation": 20}


def test_positive_only_development_source_has_no_test_split():
    frame = pd.DataFrame({"group_id": [f"g{i}" for i in range(20)], "label": [1] * 20})
    result = partition(frame, test_fraction=0)
    assert result.split.value_counts().to_dict() == {"train": 16, "validation": 4}
