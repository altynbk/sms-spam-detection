"""Prepare an untouched group holdout for the ordinary-SMS expansion study."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split

from .adaptation_data import annotate, campaign_key, clean_groups, exclude_reference_overlap
from .adaptation_protocol import load_frozen as load_previous, partition
from .external_data import ROOT, file_hash, without_placeholders

DIRECTORY = ROOT / "data/external/ham_expansion"
MEMBERSHIP = ROOT / "results/ham_expansion/protocol/membership.csv"


def fresh_partition(frame, seed):
    """Use integer group counts to avoid nested-fraction rounding artifacts."""
    groups = frame.drop_duplicates("group_id")[["group_id", "label"]]
    rest, test = train_test_split(groups, test_size=int(np.ceil(0.3 * len(groups))),
                                  stratify=groups.label, random_state=seed)
    train, validation = train_test_split(rest, test_size=int(np.ceil(0.2 * len(groups))),
                                        stratify=rest.label, random_state=seed + 1)
    mapping = {g: name for name, part in [("train", train), ("validation", validation), ("test", test)]
               for g in part.group_id}
    return frame.assign(split=frame.group_id.map(mapping))


def near_groups(frame, similarity=0.85):
    """Connected components keep indirect near-copy chains in one partition."""
    keys = sorted(set(frame.text.map(campaign_key)))
    if not keys or not 0 < similarity <= 1:
        raise ValueError("Expected messages and a similarity in (0, 1]")
    vectorizer = HashingVectorizer(analyzer="char", ngram_range=(5, 5),
                                  n_features=2**18, alternate_sign=False, norm="l2")
    matrix = vectorizer.transform(keys)
    rows, columns = [], []
    for start in range(0, len(keys), 64):
        product = (matrix[start:start + 64] @ matrix.T).tocoo()
        keep = product.data >= similarity
        rows.extend((product.row[keep] + start).tolist())
        columns.extend(product.col[keep].tolist())
    graph = coo_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(keys), len(keys))).tocsr()
    _, components = connected_components(graph, directed=False)
    representatives = {}
    for key, component in zip(keys, components):
        representatives.setdefault(component, hashlib.sha256(key.encode()).hexdigest())
    mapping = {key: representatives[c] for key, c in zip(keys, components)}
    result = frame.assign(group_id=frame.text.map(campaign_key).map(mapping))
    conflicting = result.groupby("group_id").label.nunique().loc[lambda s: s > 1].index
    removed = result.group_id.isin(conflicting)
    return result.loc[~removed].copy(), {
        "campaign_keys": len(keys), "near_components": len(set(components)),
        "conflicting_components": len(conflicting), "conflicting_rows": int(removed.sum()),
        "similarity": similarity,
    }


def fetch():
    spec = json.loads((DIRECTORY / "source.json").read_text())
    path = DIRECTORY / spec["filename"]
    if not path.exists():
        with urlopen(spec["url"], timeout=60) as response:
            content = response.read(spec["bytes"] + 1)
        if len(content) != spec["bytes"] or hashlib.sha256(content).hexdigest() != spec["sha256"]:
            raise ValueError("Unexpected source bytes")
        path.write_bytes(content)
    if file_hash(path) != spec["sha256"]:
        raise ValueError("Source hash mismatch")


def prepare():
    protocol = json.loads((DIRECTORY / "protocol.json").read_text())
    for relative, digest in protocol["input_sha256"].items():
        if file_hash(ROOT / relative) != digest:
            raise ValueError(f"Input hash mismatch: {relative}")
    previous, _ = load_previous()
    # Previous inspected tests cannot provide independent evidence again.
    # Keep financial test retired; use the dated SMS history as development now.
    history = partition(previous[previous.source.eq("sms_history")].drop(columns="split"),
                        test_fraction=0, seed=protocol["seed"])
    htrain = history[history.split.eq("train")]
    hval, history_overlap = exclude_reference_overlap(
        history[history.split.eq("validation")], htrain.text)
    development = pd.concat([previous[previous.split.isin(["train", "validation"])],
                             htrain, hval], ignore_index=True)

    raw = pd.read_csv(DIRECTORY / "DS_latest.csv", encoding="latin1", keep_default_na=False)
    if raw.columns.tolist() != ["SMSes", "Labels"] or set(raw.Labels) != {0, 1}:
        raise ValueError("Unexpected research corpus schema or labels")
    fresh = annotate(raw.rename(columns={"SMSes": "text", "Labels": "label"})
                     .reset_index(names="source_row"), "super_sms_2018_2023")
    fresh, cleaning = clean_groups(fresh)
    # Include every acquired message in the exclusion set, even earlier rejected rows.
    historical = pd.read_csv(ROOT / "data/smshamspam.csv", keep_default_na=False)
    financial = pd.read_csv(ROOT / "data/external/adaptation/financial.csv", keep_default_na=False)
    imc = pd.read_csv(ROOT / "data/external/imc25/final_dataset_output.csv", keep_default_na=False)
    from .adaptation_data import read_sms_history
    sms_history, _ = read_sms_history(ROOT / "data/external/adaptation/text-messages.csv")
    refs = (historical.sms.tolist() + financial.message.tolist() + sms_history.text.tolist()
            + [without_placeholders(t) for t in imc.text if isinstance(t, str) and without_placeholders(t).strip()])
    fresh, overlap = exclude_reference_overlap(fresh, refs)
    fresh, grouping = near_groups(fresh)
    fresh = fresh_partition(fresh, seed=protocol["seed"])
    combined = pd.concat([development, fresh], ignore_index=True)
    columns = ["source", "source_row", "text", "text_sha256", "group_id", "label", "split"]
    combined = combined[columns].sort_values(["source", "source_row"]).reset_index(drop=True)
    if combined.groupby("group_id").split.nunique().max() != 1:
        raise ValueError("Cross-split group overlap")
    if combined.groupby("group_id").label.nunique().max() != 1:
        raise ValueError("Conflicting labels")
    counts = combined.groupby(["source", "split", "label"]).size().rename("rows").reset_index()
    audit = {"raw_rows": len(raw), "raw_class_counts": raw.Labels.value_counts().sort_index().to_dict(),
             "cleaning": cleaning, "reference_overlap": overlap, "near_grouping": grouping,
             "retired_history_validation_overlap": history_overlap,
             "counts": counts.to_dict("records"),
             "groups": combined.groupby(["source", "split", "label"]).group_id.nunique()
                        .rename("groups").reset_index().to_dict("records"),
             "test_predictions_during_preparation": 0,
             "code_sha256": file_hash(__file__), "protocol_sha256": file_hash(DIRECTORY / "protocol.json")}
    return combined, audit


def manifest(frame):
    return frame.drop(columns="text").to_csv(index=False)


def load_frozen():
    frame, audit = prepare()
    if hashlib.sha256(manifest(frame).encode()).hexdigest() != file_hash(MEMBERSHIP):
        raise ValueError("Membership differs from the pretraining freeze")
    return frame, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an unused output directory")
    if args.download:
        fetch()
    frame, audit = prepare()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "membership.csv").write_text(manifest(frame))
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
