# Reducing false positives with a broader ordinary-SMS corpus

## Source and scope

The follow-up uses the authors' **2018–2023 temporal subset** of the
[Super SMS Dataset](https://github.com/smspamresearch/spstudy/tree/d83c99c6c3ae8c498b373e66b3a8129356a87b63/Data),
described by Salman, Ikram and Kaafar in
[IEEE Access (2024)](https://doi.org/10.1109/ACCESS.2024.3364671).
Its 29,403 rows comprise 4,028 ham and 25,375 spam messages before screening.
The paper describes 2,325 ham contributed by consenting volunteers during
October 2021–June 2023 in the combined corpus. The public CSV does not identify
those volunteers' rows, receipt dates, senders or component sources. The temporal
subset therefore must not be described as 4,028 individually dated modern SMS,
or as a representative sample of 2025–2026 traffic.

The authors provide the corpus for research and request citation, but the
repository contains no explicit standard data license. Raw messages remain local
and ignored; this project publishes hashes, row membership and aggregate results.
See [source.json](../data/external/ham_expansion/source.json) for exact bytes,
encoding, attribution and limitations. No generated ham or relabeled old UCI
republication is counted as new evidence.

## Protocol frozen before fitting

1. Remove conflicting campaign labels and normalized duplicate text. Remove
   exact campaign matches and character-5-gram hashed cosine similarity >= 0.85
   against **all previously acquired** historical, financial, IMC and eligible
   SMS-history messages, including earlier excluded rows.
2. Join new messages into connected components at the same similarity cutoff,
   including indirect chains. Exclude components with conflicting labels. Split
   components, stratified by class, into approximately 50% train, 20% validation
   and 30% untouched test (seed 20261019; integer group counts rounded upward
   for validation and test). All members of a component stay together.
3. Preserve existing historical, financial and IMC development partitions. The
   previously inspected financial test is retired. The previously inspected
   SMS-history test becomes development, split 80/20 with near-copy validation
   exclusions. It is **not new independent test evidence** in this study.
4. Fit six word/character TF-IDF LinearSVC candidates, C in {0.5, 1} and ham
   weight multipliers in {1, 2, 4}. Start with equal class, source-within-class and
   group-within-source/class mass; apply the ham multiplier and normalize weights
   to mean one. Vocabulary and IDF are fitted on training text only.
5. Select thresholds and the candidate using validation alone: maximize macro
   spam recall subject to empirical FPR <= 1% for **each** source with ham.
   Ties prefer lower mean FPR, then higher threshold within a model; remaining
   model ties follow C and multiplier order. Do not refit on validation.
6. Freeze selection, then score the new test once. Compare the original model,
   the previous adapted model, a **threshold-only control** recalibrating that
   previous model on the same validation set, and the selected refitted model.
   This separates the effects of adding data from simply raising a threshold.

The new test is independent of fitting and selection, after text-overlap
screening. It shares its author corpus with the new development data; it is
**not an unseen-provider or chronological evaluation**. Near-copy screening
cannot guarantee removal of every semantic paraphrase. Author labeling rules
may differ from a recipient's definition of unwanted advertising.

Report confusion counts, FPR, spam recall, precision, F1, ROC-AUC and AP, with
5,000 paired class-stratified whole-component bootstrap draws. Intervals are
conditional on these fitted models and sampled components. Promotion requires
test FPR <= 1% and <= the original baseline, a strictly negative upper 95% bound
on FPR change versus the previous adaptation, and a recall-change lower 95%
bound >= -3 percentage points versus both original and previous models.
No outcome permits claiming production safety from one corpus.

Exact rules and frozen inputs are in
[protocol.json](../data/external/ham_expansion/protocol.json); group membership
and preparation counts are in
[the protocol directory](../results/ham_expansion/protocol/).

## Reproduction

First acquire the preceding study's inputs as described in
[MODERN_ADAPTATION.md](MODERN_ADAPTATION.md), then run in the locked environment:

```bash
python -m spam_detector.ham_data --download --output results/local/ham-protocol
python -m spam_detector.ham_expansion fit --output results/local/ham-expansion-repeat
python -m spam_detector.ham_expansion evaluate --output results/local/ham-expansion-repeat
python scripts/verify_ham_expansion.py --run results/local/ham-expansion-repeat
```

The command verifies source hashes and prepares partitions without loading a
classifier. Frozen source text and model binaries are never committed.
Fitting requires the hash-matched original and previous adapted artifacts at the
paths recorded in the protocol. A changed artifact requires a separately declared
comparison; do not rewrite the frozen hashes to make an unmatched run pass.

## Observed results

Preparation removes 99 normalized duplicates, 1,790 overlaps with earlier sources
and 115 rows in two conflicting near-copy components. The new corpus retains
**3,933 ham and 23,466 spam**. Membership was merged in PR #14 before training.

| New source partition | Ham | Spam | Ham groups | Spam groups |
|---|---:|---:|---:|---:|
| Train | 1,968 | 11,860 | 1,922 | 9,952 |
| Validation | 787 | 4,497 | 769 | 3,982 |
| Test | 1,178 | 7,109 | 1,154 | 5,972 |

Including preceding development sources, training uses 29,593 rows and validation
8,443. Validation selects **`ham_word_char_c1_w1`**, C=1, ham multiplier=1 and
threshold **0.48587418949350525**. Its macro source recall is 88.77%; each ham
source has validation FPR below 1%. Increasing the extra ham weight did not win
selection. The previous model's threshold-only control requires threshold
2.087479952703472 under the same per-source constraints. Selection and validation
evidence were committed before final scoring in PR #15.

| Model on the same untouched test | FP / 1,178 ham | FPR | FN / 7,109 spam | Spam recall | Spam F1 |
|---|---:|---:|---:|---:|---:|
| Original historical model | 15 | 1.27% | 3,561 | 49.91% | 0.6649 |
| Previous adapted model | 17 | 1.44% | 1,901 | 73.26% | 0.8445 |
| Previous model, threshold only | 0 | 0.00% | 6,762 | 4.88% | 0.0931 |
| New expanded-data model | **14** | **1.19%** | **340** | **95.22%** | **0.9745** |

Against the previous adapted model, the new candidate corrects ten ham errors but
introduces seven new ones. Its FPR difference is **−0.25 percentage points**,
with a paired conditional 95% interval of **−1.02 to +0.51 points**. This does
**not establish a reduction**. The candidate's FPR interval is 0.59–1.94%; the
observed 1.19% exceeds the predeclared 1% limit. The replacement gate fails.
The historical demo stays unchanged; the newly trained artifact is experimental.

Spam recall improves by **21.96 points** versus the preceding adaptation
(paired interval +20.64 to +23.30 points); F1 improves by 0.1300 (interval
+0.1211 to +0.1390). Candidate ROC-AUC is 0.9946 and AP 0.9991. Its spam precision
is 99.79%, but precision and F1 depend on this deliberately spam-heavy corpus
and do not predict precision in a typical inbox. The threshold-only control
shows why lowering false positives by suppressing almost all detections is not
an adequate solution. The comparison does not isolate the effect of ham alone:
the new fit also adds spam and previously inspected PSA development messages.

Post-test inspection found remaining errors among delivery, registration and
event messages, plus advertising labeled ham by the authors. Some text is
transliterated or has encoding artifacts despite the authors' English filtering.
Recipient consent and sender identity are unavailable. Labels and test membership
were kept unchanged; no threshold or model was tuned after this inspection.

The [independent verification](../results/ham_expansion/evaluation/verification.json)
reconstructs 59,101 validation scores, 33,148 test scores, seven exhaustive
threshold choices, 35 validation and four test metric rows, and all three paired
5,000-draw comparisons. Full [metrics](../results/ham_expansion/evaluation/test_metrics.csv),
[intervals](../results/ham_expansion/evaluation/test_intervals.json) and
[gate criteria](../results/ham_expansion/evaluation/evaluation_metadata.json)
remain frozen. Future changes need a new untouched evaluation source with more
dated ordinary SMS, especially service and transaction messages.
