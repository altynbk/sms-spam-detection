# Executed validation and second review

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

The included verifier ran in a fresh Python process. Its report is
[verification.json](../results/full/verification.json):

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
