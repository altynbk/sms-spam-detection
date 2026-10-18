"""Freeze adaptation train/validation/test membership before candidate fitting."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .adaptation_data import (
    DIRECTORY, annotate, clean_groups, exclude_reference_overlap, prepare_history,
)
from .external_data import ROOT, file_hash, without_placeholders


def partition(frame, test_fraction, seed=20261018):
    groups = frame.drop_duplicates("group_id")[["group_id", "label"]]
    if test_fraction:
        rest, test = train_test_split(groups, test_size=test_fraction, random_state=seed,
                                      stratify=groups.label)
    else:
        rest, test = groups, groups.iloc[:0]
    train, validation = train_test_split(rest, test_size=0.2 / (1 - test_fraction),
                                        random_state=seed + 1, stratify=rest.label)
    mapping = {g: name for name, part in [("train", train), ("validation", validation), ("test", test)]
               for g in part.group_id}
    return frame.assign(split=frame.group_id.map(mapping))


def remove_overlap(frame, reference):
    if frame.empty:
        raise ValueError("Empty cohort after preparation")
    return exclude_reference_overlap(frame, reference, similarity=0.85)


def prepare():
    protocol = json.loads((DIRECTORY / "protocol.json").read_text())
    for relative, expected in protocol["input_sha256"].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError(f"Input differs from frozen protocol: {relative}")
    historical = pd.read_csv(ROOT / "data/smshamspam.csv", keep_default_na=False)
    membership = pd.read_csv(ROOT / "results/full/splits.csv", keep_default_na=False)
    original = membership.merge(historical.rename(columns={"sms": "text"}).reset_index(names="source_row"),
                                on=["source_row", "label"], validate="one_to_one")
    historical_all = list(historical.sms)
    historic = annotate(original[["text", "label", "source_row", "split"]], "historical")
    test_keys = set(historic.loc[historic.split.eq("test"), "group_id"])
    historic = historic[~historic.group_id.isin(test_keys)]
    val_keys = set(historic.loc[historic.split.eq("validation"), "group_id"])
    historic = historic[~(historic.split.eq("train") & historic.group_id.isin(val_keys))].copy()
    audit = {"historical_development_rows": len(historic)}

    financial = pd.read_csv(DIRECTORY / "financial.csv", dtype=str, keep_default_na=False)
    if financial.columns.tolist() != ["label", "message"] or set(financial.label) != {"ham", "scam"}:
        raise ValueError("Unexpected financial-message schema or labels")
    financial["source_row"] = financial.index
    # Source has no language column. This is a script filter, not language identification.
    latin = ~financial.message.str.contains("[\u0980-\u09ff]")
    audit["financial_raw_rows"] = len(financial)
    audit["financial_bengali_script_excluded"] = int((~latin).sum())
    financial = financial[latin].rename(columns={"message": "text"})
    financial["label"] = financial.label.map({"ham": 0, "scam": 1})
    financial, audit["financial_cleaning"] = clean_groups(annotate(financial, "financial"))
    financial, audit["financial_historical_overlap"] = remove_overlap(financial, historical_all)
    financial = partition(financial, test_fraction=0.2)
    # Reserve the entire acquired source against IMC reuse, including excluded holdout near-copies.
    financial_all = financial.text.tolist()
    ftrain = financial[financial.split.eq("train")]
    fval, audit["financial_validation_overlap"] = remove_overlap(
        financial[financial.split.eq("validation")], ftrain.text)
    ftest, audit["financial_test_overlap"] = remove_overlap(
        financial[financial.split.eq("test")], list(ftrain.text) + list(fval.text))

    imc = pd.read_csv(ROOT / "data/external/imc25/final_dataset_output.csv", dtype=str, keep_default_na=False)
    imc["source_row"] = imc.index
    imc = imc[imc.language.str.strip().str.lower().eq("english")].copy()
    imc["text"] = imc.text.map(without_placeholders)
    imc = imc[imc.text.str.strip().ne("")].assign(label=1)
    imc, audit["imc_cleaning"] = clean_groups(annotate(imc[["text", "label", "source_row"]], "imc25"))
    imc, audit["imc_reference_overlap"] = remove_overlap(imc, historical_all + financial_all)
    imc = partition(imc, test_fraction=0)
    itrain = imc[imc.split.eq("train")]
    ival, audit["imc_validation_overlap"] = remove_overlap(imc[imc.split.eq("validation")], itrain.text)

    development = pd.concat([historic, ftrain, fval, ftest, itrain, ival], ignore_index=True)
    history, audit["history_checkpoint"] = prepare_history()
    history, audit["history_financial_overlap"] = remove_overlap(history, financial_all)
    history["split"] = "test"
    combined = pd.concat([development, history], ignore_index=True)
    combined = combined[["source", "source_row", "text", "text_sha256", "group_id", "label", "split"]]
    if combined.groupby("group_id").split.nunique().max() != 1:
        raise ValueError("Cross-split campaign overlap")
    if combined.groupby("group_id").label.nunique().max() != 1:
        raise ValueError("Cross-source conflicting labels")
    for source in ["financial", "historical"]:
        for split in ["train", "validation"]:
            if set(combined.loc[combined.source.eq(source) & combined.split.eq(split), "label"]) != {0, 1}:
                raise ValueError(f"Missing class in {source}/{split}")
    for source in ["financial", "sms_history"]:
        if set(combined.loc[combined.source.eq(source) & combined.split.eq("test"), "label"]) != {0, 1}:
            raise ValueError(f"Missing test class in {source}")
    combined = combined.sort_values(["source", "source_row"]).reset_index(drop=True)
    audit["counts"] = combined.groupby(["source", "split", "label"]).size().rename("rows").reset_index().to_dict("records")
    audit["groups"] = combined.groupby(["source", "split", "label"]).group_id.nunique().rename("groups").reset_index().to_dict("records")
    audit["protocol_sha256"] = file_hash(DIRECTORY / "protocol.json")
    audit["code_sha256"] = file_hash(__file__)
    return combined, audit


def manifest(frame):
    return frame.drop(columns="text").to_csv(index=False)


def load_frozen():
    frame, audit = prepare()
    expected = ROOT / "results/adaptation/protocol/membership.csv"
    if hashlib.sha256(manifest(frame).encode()).hexdigest() != file_hash(expected):
        raise ValueError("Recomputed split differs from the pretraining membership")
    return frame, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an unused output directory")
    frame, audit = prepare()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "membership.csv").write_text(manifest(frame))
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit["counts"], indent=2))


if __name__ == "__main__":
    main()
