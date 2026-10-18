# Validation

Updated on **18 October 2026**, using Python 3.12.14 on macOS arm64 and the
committed dependency locks. This report describes the current state; earlier
implementation history remains in Git and the [original audit](AUDIT.md).

## Executed checks

| Check | Result |
|---|---|
| Regression, integration and application suite | **134 tests passed**; warnings are errors |
| Installed dependencies | `python -m pip check` passed |
| Notebook | Format valid; all **13 code cells** executed in a fresh kernel without cell errors |
| Default full experiment (9 October) | All **7,920 attack rows**, 360 attack datasets and 11 model artifacts verified |
| Template full experiment (9 October) | All **7,920 attack rows**, 360 attack datasets and 11 model artifacts verified |
| Historical conditional intervals (9 October) | Both runs' 5,000-draw estimates and source hashes reproduced |
| External reported SMS | **31,806 prediction rows / 63,612 decisions**, eight metric rows, 72 category rows and all conditional intervals recomputed |
| Modern adaptation | All six candidates' **18,204 validation scores**, **1,664 test scores** on 832 messages, 18 validation and four test metric rows independently checked; thresholds and paired intervals reproduced |
| Manuscript | Historical, external and adaptation metrics present in the eight-page PDF, six editable Word tables and two figures; every rendered page inspected |
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

## External evaluation

The [external verification](../results/external/imc25/evaluation/verification.json)
reconstructs both predeclared cohorts from the pinned raw snapshot and checks
the unchanged model against the artifact hash fixed before external prediction.
It recomputes scores directly from the feature transform and classifier, checks
input hashes and both decisions, and reproduces the saved summaries and intervals.
All OOV messages remain included; no fitting or threshold selection occurs.

Regression tests exercise date eligibility, overlap and duplicate handling,
whole-group resampling, rejection of substituted artifacts and deliberate ledger
corruption, and the absence of misleading two-class metrics on positive-only data.
Synthetic fixtures require neither a network download nor the original model.

## Adaptation evaluation

The [adaptation verification](../results/adaptation/evaluation/verification.json)
checks the six saved pipelines against their artifact hashes and independently
recomputes their validation decision scores. Exhaustive threshold search agrees
with the selected operating points and validation-only candidate choice. The
financial and independent SMS holdouts contain 78 and 754 messages respectively.
Both original and selected pipelines' test scores and confusion counts agree
with the published records. A separate implementation resamples message rows
through whole class-stratified groups, reproducing all paired intervals with
5,000 draws per source and the failed replacement criterion.

Tests cover source parsing, overlap filtering, class/source/campaign weighting,
separate source FPR constraints, rejection of changed artifacts and the absence
of test data from fitting and selection. The inspected IMC reports are development
data for this follow-up. The original default model remains unchanged; no threshold
was retuned after the new test outcomes were inspected.

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

The current manuscript describes both full protocols, the external reported-SMS
evaluation and the adaptation follow-up. It reports the increased false-positive
rate on independent government notifications and the failed replacement criterion.
The Word source and PDF were rebuilt in place on 18 October. All eight rendered
pages were inspected; the external and adaptation tables match the saved metric
rows. The builder checks external artifact identity, result hashes and verification
records before incorporating results. The notebook includes a comparison chart
showing both spam recall and false positives with the cohort denominators.

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

# Requires the pinned external CSV and original matching default model.
python scripts/verify_external.py --run results/external/imc25/evaluation

# Requires the frozen source CSVs and matching local adaptation artifacts.
# See MODERN_ADAPTATION.md to reproduce a new run first.
python scripts/verify_adaptation.py --run results/local/modern-adaptation
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
measure training uncertainty. The template study is exploratory. IMC recall is
conditional on a positive-only, pseudonymized convenience sample. The adaptation
study measures F1/FPR on two new holdouts, but the financial sample mixes channels
without receipt dates, and independent SMS ham consists only of 55 government
notifications from one recipient. These results do not establish acceptable FPR
on a representative modern inbox or accuracy for a particular new language.
See the [model card](MODEL_CARD.md) and [research priorities](ROADMAP.md).
