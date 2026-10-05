# Audit of the original repository

Inspected base commit: `8a47f22ab76061b07a9ba54cbb548588d826fa41` in
[altynbk/sms-spam-detection](https://github.com/altynbk/sms-spam-detection/tree/8a47f22ab76061b07a9ba54cbb548588d826fa41).
Cell numbers below are zero-based positions in the original notebook.

## Actual original workflow

1. Cell 3 loaded the CSV, lowercased each message, and removed `[^a-z0-9\s]`.
   The original text was retained separately in `text_raw`.
2. Cell 8 split cleaned texts 80/20 with seed 42 and label stratification, retaining
   duplicates. Word TF-IDF (1–2 grams, 1,000 features) and each classifier were
   inside a scikit-learn Pipeline. Classifiers: MultinomialNB, LogisticRegression,
   SVC(kernel="linear"), and RandomForestClassifier.
3. Each model was fitted on train and evaluated on test.
4. Cell 12 attacked already-cleaned spam at probability 0.3 over 20 seeds.
   Every model received the same attacked set; ham was unchanged.
5. Cell 15 ran a separate exact-deduplication experiment, retaining its metrics
   but not the newly fitted model objects.
6. Cell 17 saved the original `models` dictionary, trained with duplicates.
   The external regular-expression preprocessing was missing from those artifacts.

## Confirmed findings

| Finding | Evidence |
|---|---|
| Main split preceded deduplication | 5,574 input rows; 403 exact duplicates; 5,171 unique raw texts. Recreating the old split found 119 distinct raw texts on both sides and 135 test rows also present in train. |
| Additional normalized matches were not grouped | 5,159 normalized groups; 12 groups contain two distinct raw texts each after exact deduplication. |
| No defensive data validation | The bundled CSV itself has no missing/blank messages or contradictory labels. The issue was missing checks, not observed label corruption. |
| Saved pipeline did not implement the raw-SMS contract | Cleanup was outside the fitted pipeline; inference on original strings could differ from training preprocessing. |
| Wrong experiment's models were saved | Saved objects came from the experiment with duplicates, not the deduplicated sensitivity check. |
| No CV, validation split, or threshold selection | Original experiments used a single 80/20 split and default classifier decisions. |
| Limited robustness reporting | One attack and one nonzero intensity; no ASR, individual run table, AP, or explicit FPR reporting. |
| Uppercase limitation in the attack | Mapping keys were lowercase. The old experiment already lowercased inputs, so this was not an independent error in its saved numbers; it becomes a bug when attacking raw SMS. |
| Global warning suppression | Cell 1 called `warnings.filterwarnings("ignore")`. |
| Overstated reproducibility | No tested lock, metadata or saved membership manifest. WordCloud had no random_state. |

## Findings that did not hold up

- **There was no test-to-train vocabulary/IDF leakage:** TF-IDF was already fitted
  inside the pipeline on train only. The external deterministic cleanup did not
  learn anything from test. The fix concerns complete serialization and protocol.
- Exact deduplication already existed as a separate sensitivity experiment. It
  now belongs to the required main preparation step.
- No parameter or threshold search on test, adversarial training, or test-selected
  defense was found in the original code. However, its model-ranking claims were
  based on that test set. The revised model choice uses train-CV only.
- SVC's decision_function was already used as a continuous score for ROC-AUC,
  without calling it a probability.
- Shared attack sets and unchanged ham were already implemented correctly.

## Historical results and interpretation

`results/historical/` and `figures/historical/` retain the original saved outputs.
They were moved without changing their content; the historical ML experiment was
not retrained during this work. Original clean F1: NB 0.9032, LR 0.9124,
SVC 0.9404, RF 0.9291. Original separate deduplicated F1:
0.8797 / 0.8462 / 0.9127 / 0.9136.

Do not attribute differences from the revised experiment entirely to duplicate
leakage. Train/test membership and sizes, grouping, parameter search, thresholds,
software versions, and the attack-before-preprocessing order also changed.
Controlled comparisons are made within the new shared protocol.
The original DOCX/PDF in `paper/` describe the historical experiment and remain unchanged.
