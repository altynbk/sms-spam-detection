# Validation

Updated on **19 October 2026**, using Python 3.12.14 on macOS arm64 and the
committed dependency locks. This report describes the current state; earlier
implementation history remains in Git and the [original audit](AUDIT.md).

## Executed checks

The local application workflow, test suite, notebook and manuscript were checked
on 19 October. Research verification rows retain their earlier results; the
documentation review used saved outputs without retraining or rerunning studies.

| Check | Result |
|---|---|
| Regression, integration and application suite | **149 tests passed**; warnings are errors |
| Installed dependencies | `python -m pip check` passed |
| Notebook | Format valid; all **14 code cells** executed in a fresh kernel without cell errors |
| Default full experiment (9 October) | All **7,920 attack rows**, 360 attack datasets and 11 model artifacts verified |
| Template full experiment (9 October) | All **7,920 attack rows**, 360 attack datasets and 11 model artifacts verified |
| Historical conditional intervals (9 October) | Both runs' 5,000-draw estimates and source hashes reproduced |
| External reported SMS | **31,806 prediction rows / 63,612 decisions**, eight metric rows, 72 category rows and all conditional intervals recomputed |
| Modern adaptation | All six candidates' **18,204 validation scores**, **1,664 test scores** on 832 messages, 18 validation and four test metric rows independently checked; thresholds and paired intervals reproduced |
| Ordinary-SMS expansion | **59,101 validation scores**, **33,148 test scores** on 8,287 messages, seven exhaustive threshold choices, 35 validation/four test metric rows and three paired 5,000-draw comparisons independently checked |
| Manuscript | Historical, external and both adaptation studies present in the nine-page PDF, seven editable Word tables and two figures; every rendered page inspected |
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

The current manuscript describes both historical protocols, external reported-SMS
evaluation and both adaptation studies. It retains the first adaptation's higher
false-positive rate on PSAs and the latest study's uncertain FPR decrease and
failed 1% criterion. Word and PDF were rebuilt in place on 19 October; all nine
rendered pages were inspected. Seven editable tables match the reported metrics.
The builder verifies result hashes and verification records before incorporating
the follow-up results. The notebook's final study section compares all four
operating points and paired intervals from saved reports without model inference.

### Documentation consistency — 19 October 2026

The README, model card, notebook and manuscript were compared with the saved
historical, external and adaptation outputs. They identify the same frozen
default model and distinguish native classifier decisions from the validation
threshold. Both failed adaptation criteria and source/date limitations remain
explicit. The SMS-history holdout is described as independent for the first
adaptation only; the later study reused it for development. Detailed follow-up
methods remain in their supporting reports.

The notebook's inference example now uses the same verified loader as the README
and application. All 14 code cells executed in a fresh kernel with unchanged
outputs; the missing-model branch was also exercised. The 149-test suite passed.
The final text edit changed no code cells or stored outputs. The published
uncertainty and follow-up sections explicitly identify their fixed report paths.

The manuscript retained nine pages, seven editable tables and two figures.
Every table cell matched the previous version except the clarified `Native F1`
heading; all nine rendered pages were visually inspected. Local Markdown links
in the reviewed documents were checked. No training data, model, threshold or
published experiment output changed.

### Local workflow acceptance — 19 October 2026

The [README launch instructions](../README.md#interactive-demo) were checked in
the existing Python 3.12 environment. Activation worked, the offline dependency
dry run found all pinned application packages installed, and `pip check` passed.
This was not a fresh-environment installation test.

| Scenario | Verification and result |
|---|---|
| Default launch | `python -m streamlit run app.py` opened at `http://127.0.0.1:8501` with the saved default model. |
| Message checking | In the browser, the everyday and promotion examples matched direct model predictions (Ham and Spam). The displayed score and feature contributions agreed with inference; blank and out-of-vocabulary input produced guidance. |
| Existing interactive views | The browser displayed original/edited predictions, saved metrics and attack curves. AppTest additionally checked all three edit types and both decision rules against the committed result tables. |
| Missing model | The selected artifact was temporarily moved out of its load path. After reloading, both prediction buttons were disabled, setup guidance appeared, and saved results remained available. |
| Model restored | The original file was restored with an unchanged SHA-256; reloading enabled predictions again. |
| Run selection | AppTest loaded both `results/full` and `results/template` through `SMS_SPAM_RUN` and matched their saved-model predictions and metrics. A nonexistent directory produced the setup error without a traceback or file creation. |
| Browser CSV download | Not verified: the browser permission policy denied downloading from localhost. This does not establish an application defect. |

All 13 focused inference/application tests and the complete 149-test suite passed.
No reproducible application failure was found in the checked scenarios, so no
implementation change was needed. No study was retrained, no threshold changed,
and published data and results remained untouched. The illustrative messages
check application behavior, not detection accuracy on new evaluation data.

## Ordinary-SMS expansion verification

The [verification record](../results/ham_expansion/evaluation/verification.json)
checks all six fitted candidates and the threshold-only control. It reconstructs
source membership, hash-matched decision scores and exhaustive validation
threshold choices. A separate row-resampling implementation reproduces all
three 5,000-draw paired component-bootstrap comparisons and the failed replacement
gate. Regression tests cover near-copy chains, whole-component conflict exclusion,
class/source weighting, absence of test access during selection, and recall
safeguards. No test outcome was used to change a candidate or threshold.

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

# Requires the frozen new source and matching ham-expansion artifacts.
python scripts/verify_ham_expansion.py --run results/local/ham-expansion
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
The later ham-expansion holdout is larger but shares its author corpus with new
development data. It has no individual receipt dates and is not an unseen-provider
test. Its 14 / 1,178 false positives do not meet the 1% criterion; the FPR decrease
versus the previous model is statistically inconclusive.
See the [model card](MODEL_CARD.md) and [completion plan](ROADMAP.md).
