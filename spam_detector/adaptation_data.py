"""Source-audited adaptation splits, prepared without loading a classifier."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.request import urlopen

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from .external_data import ROOT, file_hash, without_placeholders

DIRECTORY = ROOT / "data/external/adaptation"


def campaign_key(text):
    """Conservative grouping only; never a fitted classifier feature."""
    text = unicodedata.normalize("NFKC", text).casefold()
    # Align publisher placeholders with live entities before punctuation removal.
    text = re.sub(r"<(?:url|email_address|phone_number)>", " entity ", text)
    text = re.sub(r"https?://\S+|www\.\S+|\b[\w.+-]+@[\w.-]+\.[a-z]+", " entity ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\d+", " number ", text)
    text = " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in text).split())
    return text


def annotate(frame, source):
    if not {"text", "label", "source_row"}.issubset(frame):
        raise ValueError("Expected text, label and source_row")
    if frame.empty or not frame.text.map(lambda t: isinstance(t, str) and bool(t.strip())).all():
        raise ValueError("Expected nonempty messages")
    if not frame.label.isin([0, 1]).all():
        raise ValueError("Expected binary labels")
    result = frame.copy()
    result["source"] = source
    result["text_sha256"] = result.text.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    result["group_id"] = result.text.map(campaign_key).map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    return result


def clean_groups(frame):
    """Drop conflicting groups, then normalized duplicate texts; publish counts."""
    conflicts = frame.groupby("group_id").label.nunique()
    conflicting = set(conflicts[conflicts > 1].index)
    clean = frame[~frame.group_id.isin(conflicting)].copy()
    keys = clean.text.map(lambda t: " ".join(unicodedata.normalize("NFKC", t).casefold().split()))
    clean = clean.loc[~keys.duplicated()].copy()
    return clean, {"input_rows": len(frame), "conflicting_groups": len(conflicting),
                   "conflicting_rows": int(frame.group_id.isin(conflicting).sum()),
                   "duplicates_removed": len(frame) - int(frame.group_id.isin(conflicting).sum()) - len(clean),
                   "retained_rows": len(clean)}


def read_sms_history(path):
    """Reject ambiguous rows explicitly instead of silently guessing CSV fields."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    expected = ["date-received", "date-read", "sender", "category", "text"]
    if rows[0] != expected:
        raise ValueError("Unexpected SMS history schema")
    valid, excluded = [], []
    for index, row in enumerate(rows[1:]):
        if len(row) != len(expected):
            excluded.append(index)
            continue
        date, _, _, category, text = row
        valid.append({"source_row": index, "received": date, "category": category, "text": text})
    frame = pd.DataFrame(valid)
    eligible = frame.category.isin(["spam", "gov"]) & frame.text.str.strip().ne("")
    eligible &= ~frame.text.str.contains("<REDACTED>", regex=False)
    retained = frame[eligible].copy()
    retained["label"] = retained.category.map({"spam": 1, "gov": 0})
    pd.to_datetime(retained.received, format="%Y-%m-%d %H:%M:%S", errors="raise")
    return annotate(retained, "sms_history"), {
        "raw_rows": len(rows) - 1, "malformed_source_rows": excluded,
        "excluded_categories_or_redactions": int((~eligible).sum()),
        "eligible_rows": len(retained),
        "scope": "All languages; unsolicited spam versus government PSAs; one Philippine recipient",
    }


def fetch(specification):
    """Download only declared source files and verify before using any bytes."""
    path = DIRECTORY / specification["filename"]
    if path.exists():
        if file_hash(path) != specification["sha256"]:
            raise ValueError(f"Source hash mismatch: {path.name}")
        return path
    with urlopen(specification["url"], timeout=60) as response:
        content = response.read(specification["bytes"] + 1)
    if len(content) != specification["bytes"] or hashlib.sha256(content).hexdigest() != specification["sha256"]:
        raise ValueError("Downloaded bytes differ from the frozen source")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
    return path


def exclude_reference_overlap(frame, reference_texts, similarity=0.85):
    """Conservative source decontamination; no learned spam scores or labels."""
    if not 0 < similarity <= 1:
        raise ValueError("Similarity must be in (0, 1]")
    references = list(reference_texts)
    if not references or not all(isinstance(t, str) and t.strip() for t in references):
        raise ValueError("Expected nonempty reference texts")
    if frame.empty:
        raise ValueError("Expected a nonempty candidate cohort")
    reference_keys = {campaign_key(t) for t in references}
    keys = frame.text.map(campaign_key)
    exact = keys.isin(reference_keys)
    # Stateless fixed character hashing avoids fitting a vocabulary to the test.
    vectorizer = HashingVectorizer(analyzer="char", ngram_range=(5, 5),
                                  n_features=2**18, alternate_sign=False, norm="l2")
    reference_matrix = vectorizer.transform(sorted(reference_keys))
    maxima = []
    for start in range(0, len(frame), 64):
        matrix = vectorizer.transform(keys.iloc[start:start + 64])
        maximum = (matrix @ reference_matrix.T).max(axis=1).toarray().ravel()
        maxima.extend(float(x) for x in maximum)
    annotated = frame.assign(reference_max_similarity=maxima)
    near = annotated.reference_max_similarity.ge(similarity)
    retained = annotated.loc[~(exact | near)].copy()
    return retained, {"reference_rows": len(references), "reference_keys": len(reference_keys),
                      "campaign_matches": int(exact.sum()), "near_matches": int(near.sum()),
                      "total_excluded": int((exact | near).sum()), "remaining": len(retained),
                      "similarity_threshold": similarity}


def prepare_history():
    specification = json.loads((DIRECTORY / "sms_history_source.json").read_text())
    path = DIRECTORY / specification["filename"]
    if file_hash(path) != specification["sha256"]:
        raise ValueError("SMS history does not match the pinned source")
    reference_spec = json.loads((DIRECTORY / "reference_sources.json").read_text())
    for relative, expected in reference_spec.items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError(f"Reference hash mismatch: {relative}")
    historical = pd.read_csv(ROOT / "data/smshamspam.csv", dtype=str, keep_default_na=False)
    imc = pd.read_csv(ROOT / "data/external/imc25/final_dataset_output.csv", dtype=str, keep_default_na=False)
    imc = imc[imc.language.str.strip().str.lower().eq("english")]
    imc_texts = [without_placeholders(t) for t in imc.text]
    references = list(historical.sms) + [t for t in imc_texts if t.strip()]
    frame, parsing = read_sms_history(path)
    frame, cleaning = clean_groups(frame)
    frame, overlap = exclude_reference_overlap(frame, references)
    audit = {"source_sha256": specification["sha256"], "parsing": parsing, "cleaning": cleaning,
             "empty_imc_texts_after_placeholder_removal": sum(not t.strip() for t in imc_texts),
             "overlap": overlap, "class_counts": frame.label.value_counts().sort_index().to_dict(),
             "class_groups": frame.groupby("label").group_id.nunique().to_dict(),
             "received_range": [frame.received.min(), frame.received.max()],
             "code_sha256": file_hash(__file__)}
    return frame, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an unused output directory")
    if args.download:
        fetch(json.loads((DIRECTORY / "sms_history_source.json").read_text()))
    frame, audit = prepare_history()
    args.output.mkdir(parents=True, exist_ok=True)
    frame.drop(columns="text").to_csv(args.output / "membership.csv", index=False)
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
