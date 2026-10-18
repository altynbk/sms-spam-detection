"""Prepare fixed external cohorts without loading or consulting a classifier."""

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen

import pandas as pd

from .data import sha256_text, template_key
from .preprocessing import normalized_key

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/imc25/source.json"
DATA = ROOT / "data/external/imc25/final_dataset_output.csv"
REFERENCE = ROOT / "data/smshamspam.csv"
PROTOCOL = ROOT / "data/external/imc25/protocol.json"
COHORTS = ("recent_2023_2024", "all_english")
PLACEHOLDERS = re.compile(
    r"<(?:URL|PHONE_NUMBER|NAMED_ENTITY|DATE_TIME|EMAIL_ADDRESS|CREDIT_CARD|"
    r"IP_ADDRESS|LOCATION|NAME|US_BANK_NUMBER|US_DRIVER_LICENSE|US_ITIN|"
    r"US_PASSPORT|US_SSN|MEDICAL_LICENSE|CRYPTO)>"
)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_source(path, specification):
    if Path(path).stat().st_size != specification["bytes"] or file_hash(path) != specification["sha256"]:
        raise ValueError("External dataset bytes differ from the pinned upstream snapshot")


def download_source():
    """Download data only; never visit links embedded inside message text."""
    source = json.loads(SOURCE.read_text())
    if DATA.exists():
        check_source(DATA, source)
        return DATA
    with urlopen(source["url"], timeout=120) as response:
        payload = response.read(source["bytes"] + 1)
    if len(payload) != source["bytes"] or hashlib.sha256(payload).hexdigest() != source["sha256"]:
        raise ValueError("Download does not match the expected size and SHA-256")
    DATA.parent.mkdir(parents=True, exist_ok=True)
    # An existing file is never replaced by a concurrent download.
    with DATA.open("xb") as output:
        output.write(payload)
    return DATA


def explicit_year(value):
    """Use a unique explicit four-digit year, never guess from day/month or 'today'."""
    years = set(re.findall(r"(?<!\d)(20\d{2})(?!\d)", value))
    return int(next(iter(years))) if len(years) == 1 else None


def without_placeholders(text):
    """Secondary paired view: remove only the declared publisher placeholder tokens."""
    return PLACEHOLDERS.sub(" ", text)


def prepare_cohorts(frame, reference_texts):
    required = {"text", "language", "time", "scam_type"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Expected a nonempty external table with text, language, time and scam_type")
    if not all(isinstance(v, str) for v in frame[list(required)].to_numpy().ravel()):
        raise ValueError("External text and metadata fields must be strings")
    reference = list(reference_texts)
    if not reference or not all(isinstance(s, str) and s.strip() for s in reference):
        raise ValueError("Expected nonblank reference SMS strings")
    raw_reference = set(reference)
    normalized_reference = {normalized_key(s) for s in reference}
    template_reference = {template_key(s) for s in reference}
    df = frame.reset_index(drop=True).copy()
    df["source_row"] = df.index
    df["year"] = df.time.map(explicit_year).astype("Int64")
    english = df.language.str.strip().str.lower().eq("english")
    nonblank = df.text.str.strip().ne("")
    base = df[english & nonblank].copy()
    base["normalized"] = base.text.map(normalized_key)
    base["template"] = base.text.map(template_key)
    audit = dict(source_rows=len(df), non_english_rows=int((~english).sum()),
                 english_rows=int(english.sum()), blank_english_rows=int((english & ~nonblank).sum()),
                 reference_rows=len(reference), cohorts={})
    prepared = {}
    for name in COHORTS:
        part = base[base.year.isin([2023, 2024])].copy() if name == COHORTS[0] else base.copy()
        exact = part.text.isin(raw_reference)
        normalized = part.normalized.isin(normalized_reference)
        template = part.template.isin(template_reference)
        overlap = normalized | template
        retained = part[~overlap]
        rows = []
        # Stable first-source-row representative; original published text is unchanged.
        for key, group in retained.groupby("normalized", sort=False):
            row = group.iloc[0]
            categories = sorted(set(group.scam_type.str.strip()) - {""})
            rows.append(dict(source_row=int(row.source_row), text=row.text,
                             text_sha256=sha256_text(row.text), normalized_sha256=sha256_text(key),
                             template_sha256=sha256_text(row.template),
                             source_rows=json.dumps(group.source_row.tolist()),
                             years=json.dumps(sorted(int(y) for y in group.year.dropna().unique())),
                             category=categories[0] if len(categories) == 1 else "mixed" if categories else "unknown"))
        prepared[name] = pd.DataFrame(rows, columns=["source_row", "text", "text_sha256",
            "normalized_sha256", "template_sha256", "source_rows", "years", "category"])
        audit["cohorts"][name] = dict(candidate_rows=len(part), exact_reference_matches=int(exact.sum()),
            normalized_reference_matches=int(normalized.sum()), template_reference_matches=int(template.sum()),
            overlap_rows_excluded=int(overlap.sum()), duplicate_rows_removed=len(retained) - len(rows),
            messages=len(rows), template_groups=int(prepared[name].template_sha256.nunique()),
            categories=prepared[name].category.value_counts().sort_index().to_dict(),
            candidate_years={str(k): int(v) for k, v in part.year.value_counts(dropna=False).sort_index().items()})
    return prepared, audit


def load_cohorts():
    source = json.loads(SOURCE.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    check_source(DATA, source)
    if file_hash(REFERENCE) != protocol["reference_sha256"]:
        raise ValueError("Reference corpus differs from the frozen protocol")
    frame = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    if len(frame) != source["rows"] or frame.columns.tolist() != source["columns"]:
        raise ValueError("External schema or row count differs from the source specification")
    reference = pd.read_csv(REFERENCE, dtype=str, keep_default_na=False)
    cohorts, audit = prepare_cohorts(frame, reference.sms)
    audit.update(source_sha256=source["sha256"], reference_sha256=protocol["reference_sha256"],
                 protocol_sha256=file_hash(PROTOCOL))
    return cohorts, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch the pinned source if absent")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an unused output directory")
    if args.download:
        download_source()
    cohorts, audit = load_cohorts()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in cohorts.items():
        frame.drop(columns="text").to_csv(args.output / f"{name}_membership.csv", index=False)
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
