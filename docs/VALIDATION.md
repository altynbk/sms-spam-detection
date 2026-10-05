# Executed validation and review history

## Reference review: latest checks

Reviewed on 5 October 2026, starting from commit
`bb8e308d6365984d4a07b74e0a9814651122f77e`. See
[REFERENCE_COMPARISON.md](REFERENCE_COMPARISON.md) for the primary sources and
the resulting corrections. The Python environment and dependency locks are
unchanged from the implementation review below.

- **70 tests passed in 12.43 seconds**, with warnings treated as pytest errors.
  New coverage includes a train-only no-text baseline, rejection of incomplete
  scenario grids and altered CV/summary aggregates, and running the expanded
  verifier through the synthetic end-to-end experiment.
- The real **full experiment completed in 75.08 seconds**: 84 candidates,
  420 CV fits, 11 train-only refits and 360 shared attack datasets.
- The eight existing tables (`splits`, `cv_results`, `validation_metrics`,
  `clean_metrics`, `robustness_runs`, `robustness_summary`, `attack_manifest`,
  `errors`) are **byte-identical** to the previous full run. All best parameters,
  CV scores, thresholds and the frozen selected model also match exactly.
- The additional baseline predicts ham for every test message: accuracy
  904/1034 = 0.874275, recall/F1 = 0, FP = 0, FN = 130. Its majority and prior
  come only from the training labels; it does not participate in model selection.
- All **nine notebook code cells** executed in a fresh kernel with local models.
  They also executed in a separate source-only copy made from Git's tracked and
  unignored files, with **no joblib files present**. That copy displayed the saved
  report and the expected instructions for enabling optional inference.
- Notebook format validation passed. Local Jupyter emitted its TCP transport
  warning, as in the earlier review; neither notebook run had a cell error.
- All three regenerated figures were visually inspected: native/default and
  validation-threshold panels, model colors, labels and legend are present.
- The expanded verifier passed in a separate process with `--all-attacks`:
  **all 7,920 attack rows**, 360 attack dataset hashes, 22 clean rows, 22 validation
  rows, 2 baseline rows, 383 error rows and 396 summary rows were recomputed.
  It also checked the 84 CV rows against their fold aggregates and selection,
  full scenario key sets, saved artifacts, source hashes and split membership.
  See [verification.json](../results/full/verification.json) for the machine-readable record.

The full run was moved aside before regeneration, so these comparisons use the
actual previous files. The repeat run checks reproducibility on the same fixed
split; it is not an independent generalization estimate. No test-driven model,
parameter, threshold or defense changes were introduced.

## Earlier implementation review

Local review on 5 October 2026. Base commit:
`8a47f22ab76061b07a9ba54cbb548588d826fa41`.
The checks below were completed locally before committing the implementation.
Python 3.12.14, macOS arm64; exact packages are recorded in the lock files and
experiment metadata.

## Review findings and fixes

- **Native default decisions:** the initial implementation reported default
  results using `score >= 0.5`. Random Forest's native tie-breaking chooses ham
  when both class probabilities are 0.5. This affected one clean-test prediction.
  Default-mode evaluation now uses `pipeline.predict`, matching CV and native
  inference; tuned decisions still use the explicitly defined `>=` rule.
- **Generator inputs:** TransformerMixin called fit and transform on the same
  iterable; fit consumed a one-shot generator. TextCleaner now materializes it
  once in fit_transform. A regression test verifies fitting equivalence.
- **Malformed attack grids:** empty/duplicate attacks or intensities now fail
  before data access or model training.
- **Reproducible verification:** the former scratch verification was replaced
  by `scripts/verify_results.py`, an included tool with explicit checks and no fitting.
- **Language:** README, audit, validation report and notebook narrative/output
  messages are in English. Unicode mappings and multilingual edge-case inputs
  remain as intentional test data; the public dataset is not translated.
- **Scope:** this update covers code, results and English project documentation.
  The original manuscript remains unchanged and describes the historical experiment.

## Commands actually run

Commands below use `python` as shorthand for the dedicated environment's executable.
Earlier result directories were retained outside the deliverable as comparison evidence;
new full/quick directories were generated without changing seed, grids or selection rules.

```bash
python -m pytest -q
python -m spam_detector.train --mode full --jobs 2 --output results/full
python -m spam_detector.train --mode quick --jobs 2 --output results/quick
python scripts/verify_results.py --run results/full --all-attacks
```

- **61 passed in 11.34 seconds**, with Python warnings treated as pytest errors.
  These include fresh-process serialization checks for all 11 model variants and
  the new generator/default-tie/configuration regressions.
- Full: 11 variants, 84 parameter combinations, 420 CV fits and 11 train-only refits;
  360 shared attacked datasets and 7,920 result rows.
- Quick: real bundled CSV, 11 variants, 5 folds, one candidate per variant,
  2 intensities × 2 seeds × 3 scenarios, 264 result rows.
- All nine English notebook code cells executed through nbclient in a fresh kernel;
  nbformat validation passed and there were no cell error outputs.
- The local Jupyter kernel emitted a transport warning about unencrypted TCP.
  It was not suppressed and did not cause a notebook execution failure.
- No Python warnings/errors were found in the full/quick training logs.

## Independent saved-result verification

The included verifier ran in a fresh Python process. The current, expanded report is
[verification.json](../results/full/verification.json). The earlier check covered:

- All 11 artifacts and source/data/manifest hashes matched metadata.
- Split membership and the five shared CV folds were reconstructed from configuration.
- Every tuned threshold was recomputed from validation and matched exactly.
- All 22 clean result rows were recomputed, including confusion matrices and ranking metrics.
- Every one of the 360 raw attacked datasets reproduced its saved SHA-256 and changed-message counts.
- **Every one of the 7,920 attacked result rows was independently recomputed** by
  passing the entire raw attacked batch through each loaded pipeline, bypassing
  the training runner's unchanged-ham score cache.
- FP counts remained constant under spam-only attacks; ASR numerators/denominators,
  zero-intensity identity and fixed thresholds were verified.

The verification did not train models or choose parameters using test results.
After the code corrections, the selected model, all 11 best parameter dictionaries,
all best CV scores and all validation thresholds matched the previous run exactly.
Tuned clean metrics also matched. The corrected RF native-default confusion matrix
is `[[902, 2], [24, 106]]`, rather than the initial explicit-0.5 table's
`[[901, 3], [24, 106]]`. This is a decision-rule correction, not a model improvement.

## Readiness judgment

The implementation and English analysis are suitable for review as a reproducible
educational/portfolio project. Keep Jupyter for narrative and the Python package
for reusable logic; a rewrite is not justified by the findings.

Claims about new languages, current SMS, on-device latency or real-world robustness
need separate evidence. The existing historical manuscript is outside the scope of
this code and documentation update.
