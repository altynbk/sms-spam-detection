# Validation

Reviewed on **9 October 2026**, using Python 3.12.14 on macOS arm64 and the
committed dependency locks. This report describes the current state; earlier
implementation history remains in Git and the [original audit](AUDIT.md).

## Executed checks

| Check | Result |
|---|---|
| Regression, integration and application suite | **96 tests passed**; warnings are errors |
| Installed dependencies | `python -m pip check` passed |
| Notebook | Format valid; all **11 code cells** executed in a fresh kernel without cell errors |
| Default full experiment | All **7,920 attack rows**, 360 attack datasets and 11 model artifacts verified |
| Template full experiment | All **7,920 attack rows**, 360 attack datasets and 11 model artifacts verified |
| Conditional intervals | Both runs' 5,000-draw estimates and source hashes reproduced |
| Manuscript | Current metrics present in the six-page PDF and four editable Word tables |
| Repository | Local links checked; published source and result files retained |

Independent verification also reconstructs both protocols' split/CV membership,
validation thresholds, 22 clean and 22 validation rows, two baseline rows,
84 search-result aggregates and 396 attack-summary rows per protocol. All 383
default and 407 template error records agree with model predictions. Dataset,
recorded training-source and artifact hashes match. No model is fitted or
selected using test outcomes during verification.

Machine-readable records:
[default verification](../results/full/verification.json),
[template verification](../results/template/verification.json).
The verifier checks saved CV scores and their aggregates; it does not repeat the
420 training fits in a full search.

## Application and manuscript

Inference tests check serialization, artifact changes, version/threshold mismatch
and exact linear contributions for word, character and combined features.
Application tests cover blank/OOV input, deterministic edits, zero-intensity
identity, long input with inserted separators, results controls and unavailable
models. The playground's 2,500-character source limit keeps every derived message
within the 5,000-character inference limit.

The browser flow was also exercised against the saved default model during the
demo's validation. Predictions and reconstructed scores agreed with local
inference. Application tests use synthetic fixtures to check behavior, not to
estimate spam-detection accuracy.

The current manuscript describes both full protocols and their limitations.
Its six rendered pages were inspected when the manuscript was synchronized.
The present review rechecked the PDF's result text and Word table structure;
neither file was changed.

## Reproduce checks locally

With the environment active:

```bash
python -m pytest -q
python -m pip check

# Requires the original matching local model artifacts.
python scripts/verify_results.py --run results/full --all-attacks
python scripts/verify_results.py --run results/template --all-attacks

# Uses only saved prediction/error tables; no model artifacts needed.
python -m spam_detector.uncertainty --run results/full
python -m spam_detector.uncertainty --run results/template
```

After a fresh clone, model binaries are absent. Follow the
[reproduction commands](../README.md#reproduce-the-experiments) to train into
`results/local/`, then verify those new directories. Never train over a completed
published run.

## GitHub checks

[Tests and reproducibility](https://github.com/altynbk/sms-spam-detection/actions/workflows/tests.yml)
runs on pull requests, pushes to `main` and manual dispatch. Four Python 3.12 jobs
cover Ubuntu/macOS and normalized/template grouping. Each checks dependencies,
runs core tests, trains a real-data quick experiment in a temporary directory,
independently verifies all 264 attack rows and computes conditional intervals.
A fifth job installs the optional app lock and tests the interactive flows.

All five checks are required before merging to `main`. Core-only jobs skip the
optional application module; the dedicated app job exercises it. Hosted run
status and logs are available at the workflow link above.

## Scope

Recomputation establishes consistency of the recorded experiment. The intervals
condition on fitted models and observed groups; attack-seed variation does not
measure training uncertainty. The template study is exploratory. None of these
checks establishes accuracy on new languages, current SMS or independent sources.
See the [model card](MODEL_CARD.md) and [research priorities](ROADMAP.md).
