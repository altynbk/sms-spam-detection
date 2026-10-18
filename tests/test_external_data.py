import hashlib

import pandas as pd
import pytest

from spam_detector.external_data import check_source, explicit_year, prepare_cohorts, without_placeholders


@pytest.mark.parametrize("value,expected", [
    ("03/05/2023 14:31", 2023), ("17 February 2024 19:03", 2024),
    ("2023-10-06T18:00:00", 2023), ("Today 11:59 AM", None),
    ("03:08:21 Tuesday", None), ("2023 or 2024", None), ("120234", None),
])
def test_year_never_guesses_missing_or_ambiguous_years(value, expected):
    assert explicit_year(value) == expected


def test_cohorts_exclude_reference_overlap_and_deduplicate_after_date_filter():
    repeated = "A different message with a payment request and long text"
    matching = "Your delivery order 123456 has been delayed. Visit https://a.example/x"
    reference = ["meeting tomorrow", matching]
    records = [
        (repeated, "English", "2021", "banking"),
        (repeated.upper(), "English", "2024", "banking"),
        ("  " + repeated + "  ", "English", "2023", "banking"),
        ("MEETING   TOMORROW", "English", "2024", "others"),
        (matching.replace("123456", "999").replace("a.example/x", "b.example/y"), "English", "2024", "delivery"),
        ("UNKNOWN DATE", " English ", "Today", "others"),
        ("Translated English must not enter", "Spanish", "2024", "others"),
        (" ", "English", "2024", "others"),
    ]
    frame = pd.DataFrame(records, columns=["text", "language", "time", "scam_type"])
    cohorts, audit = prepare_cohorts(frame, reference)
    recent = cohorts["recent_2023_2024"]
    assert recent.text.tolist() == [repeated.upper()]
    assert recent.source_rows.tolist() == ["[1, 2]"]
    assert cohorts["all_english"].text.tolist() == [repeated, "UNKNOWN DATE"]
    assert audit["cohorts"]["recent_2023_2024"]["overlap_rows_excluded"] == 2
    assert audit["cohorts"]["recent_2023_2024"]["duplicate_rows_removed"] == 1
    assert audit["non_english_rows"] == audit["blank_english_rows"] == 1


def test_removes_only_declared_placeholder_tokens():
    text = "Hi <NAME>, use <URL> or <PHONE_NUMBER>. <ADV> stays."
    assert without_placeholders(text) == "Hi  , use   or  . <ADV> stays."


def test_source_check_rejects_truncated_or_changed_data(tmp_path):
    p = tmp_path / "source.csv"
    p.write_bytes(b"expected")
    spec = dict(bytes=8, sha256=hashlib.sha256(b"expected").hexdigest())
    check_source(p, spec)
    p.write_bytes(b"modified")
    with pytest.raises(ValueError, match="snapshot"):
        check_source(p, spec)
    p.write_bytes(b"short")
    with pytest.raises(ValueError, match="snapshot"):
        check_source(p, spec)


def test_rejects_missing_schema_and_nontext_fields():
    with pytest.raises(ValueError, match="table"):
        prepare_cohorts(pd.DataFrame({"text": ["x"]}), ["reference"])
    bad = pd.DataFrame([dict(text=42, language="English", time="2024", scam_type="spam")])
    with pytest.raises(ValueError, match="strings"):
        prepare_cohorts(bad, ["reference"])
