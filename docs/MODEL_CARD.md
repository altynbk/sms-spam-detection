# Model card: SMS spam classifier

## Intended use

An educational, reproducible experiment on historical English SMS. Suitable for
studying sparse text classification and the effect of controlled obfuscations,
and for a clearly labeled interactive demonstration. It has not been validated as
an automatic production blocker, a phishing/URL safety service, an email filter,
or a Russian/Kazakh classifier. A ham prediction does not establish that a message,
link or sender is safe.

## Model and decision rule

The frozen train-CV choice is `linear_svc_word_gentle`: lowercase/whitespace cleanup,
word unigram/bigram TF-IDF and LinearSVC. The saved `SpamModel` includes the complete
pipeline and the validation-selected threshold. Its input is the original SMS,
without caller-side preprocessing. Labels are **ham = 0, spam = 1**.
This is the default model in `results/full`. Both adaptation candidates described
below failed their replacement criteria and remain experimental.

At the validation threshold, score >= **-0.08620419778818611** predicts spam.
`score()` returns a signed decision-function value, **not a probability or a
percentage confidence**. The demo displays the label and, in optional
details, the margin and threshold. Turning that number into a percentage using
clipping, an arbitrary sigmoid or multiplication by 100 is not calibrated confidence.
Probability estimates would require a separately validated calibration protocol;
see [scikit-learn's calibration guide](https://scikit-learn.org/stable/modules/calibration.html).

The API accepts a single string or a nonempty iterable of strings. Empty,
whitespace-only, punctuation-only and Unicode strings are valid inference inputs,
but receiving a finite score does not establish predictive usefulness. An
interactive demo asks for nonblank text and withholds the displayed classification
when no vocabulary features were recognized. This is a UI guard, not a calibrated
abstention rule. The library rejects non-string inputs and empty batches.

## Data and evaluation

The bundled [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
contains 5,574 rows. Exact deduplication leaves 5,171 messages and 5,159 normalized
groups. Grouping retains punctuation and Unicode while ignoring case/whitespace.
This does not identify every related campaign or template. An additional
`--grouping template` experiment masks URLs and digit sequences for long messages;
its zero overlap is specific to that heuristic.

| Split | Ham | Spam | Role |
|---|---:|---:|---|
| Train | 2,710 | 392 | Five shared grouped CV folds and train-only refitting |
| Validation | 904 | 131 | Recall maximization under empirical FPR <= 1% |
| Test | 904 | 130 | Final clean and synthetic-attack reporting |

The model was selected from 11 variants using highest mean train-CV spam F1.
Test outcomes did not choose the variant or threshold. There was no refitting after
threshold selection. The default result uses a single grouped holdout. The additional template
experiment is exploratory; neither establishes temporal/source transfer by itself.

## Measured performance

For the frozen selected model at its validation threshold:

| Metric | Clean test |
|---|---:|
| Spam precision | 0.9683 |
| Spam recall | 0.9385 |
| Spam F1 | 0.9531 |
| Accuracy | 0.9884 |
| False positives / ham | 4 / 904 |
| False negatives / spam | 8 / 130 |
| FPR | 0.44% |

The conditional 95% group-bootstrap interval for this F1 is 0.9249–0.9769
(5,000 class-stratified whole-group resamples, seed 42). See
[confidence_intervals.csv](../results/full/confidence_intervals.csv) for both
decision rules. This conditions on the fitted model and observed split; it does
not include model-selection uncertainty, refitting or domain shift.

Native default decisions give 3 FP and 10 FN. A no-text training-majority reference
has 87.43% accuracy and zero spam recall/F1. Accuracy alone is insufficient here.
Some alternative models exceed 1% FPR on test despite satisfying the constraint
on validation; the constraint is empirical, not a future false-positive guarantee.

Random attacks modify raw spam only. At intensity 0.3, the selected model's mean
recall is 0.7696 for leet, 0.6035 for separators and 0.8473 for Unicode lookalikes.
These are nonadaptive scenarios on the same 130 spam messages, not an adversarial
security certification. Attack-seed SD is not uncertainty over new training sets.
Human readability, semantic preservation and changes to ham were not evaluated.

## Template sensitivity

The default split contains 13 crossing templates under a URL/digit masking rule,
including six test messages with a training counterpart. A separate full search
with template groups eliminates those crossings and selects the char/no-punctuation
LinearSVC. On its different test set, native predictions give F1 0.9531 (2 FP, 10 FN);
validation tuning gives F1 0.9288 (11 FP, 8 FN, FPR 1.22%). This does not replace the
default model or establish that one representation is universally best. The
analysis was introduced after inspecting the default run and is not an untouched
confirmatory evaluation. See `results/template/` and each run's template audit.

## External reported SMS

The same frozen artifact and validation threshold were evaluated on the
author-released IMC 2025 smishing dataset. The primary cohort contains 372
distinct English texts with an explicit 2023–2024 year, predominantly 2023.
There are no exact, normalized or template-key matches against the full
historical corpus. These checks cannot exclude related campaigns or paraphrases.

The model detects **176 / 372** reports: **47.31% recall**, 196 misses and a
conditional 95% whole-template-bootstrap interval of **42.32–52.28%**. Native
decisions detect 145 / 372 (38.98%). Removing fixed anonymization placeholders
changes validation-threshold recall to 47.04%. The supplementary English cohort
of 15,531 texts, including unknown/older years, gives 48.23% recall.

The dataset is a pseudonymized convenience sample of positive reports, with
publisher-supplied annotations and no legitimate-message class. External F1,
precision, accuracy and false-positive rate are unavailable. The difference
from the historical test is not a causal estimate of temporal degradation.
OOV inputs remain in the research denominator, unlike the demo's UI abstention.
The [external report](EXTERNAL_EVALUATION.md) records selection, source licensing,
all decision rules, verification and the protocol frozen before prediction.

## Modern-message adaptation candidate

The first follow-up added the inspected IMC reports and a financial-message
corpus published in 2025 to development data. Validation selected
`adapted_word_char_c1`, threshold 0.64907808395485, before scoring two new holdouts.
On the 78-message financial holdout, F1 rose from 0.6400 to 0.8605 and FP fell
from 5 to 3 of 32 ham. On the separate 754-message SMS history, recall rose from
42.63% to 58.66%, but FP rose from 11 to 19 of 55 ham (20.00% to 34.55% FPR).
The paired conditional 95% interval for the FPR increase is +1.79 to +28.07
percentage points. The replacement criterion **failed**.

The financial messages mix SMS, chats and email without receipt dates. The SMS
history spans 2022–2026, mixes languages, comes from one recipient and retains
only government PSAs as ham. It was independent of development for this study;
the later expansion explicitly reused it for development. The
[complete follow-up](MODERN_ADAPTATION.md) retains source audits, selection rules,
all metrics and verification records.

## Broader ordinary-SMS candidate

The second follow-up added development data from the authors' 2018–2023 Super SMS
subset and reserved a new group holdout of **1,178 ham and 7,109 spam**. Overlaps
with earlier acquired sources were removed. This holdout shares its author corpus
with the added training data; it is not an unseen-provider or chronological test.

Validation selected `ham_word_char_c1_w1`, threshold **0.48587418949350525**.
Against the previous adaptation on this holdout, FP changed **17 → 14**
(FPR 1.44% → 1.19%) and recall **73.26% → 95.22%**. The original default model
gave 15 FP and 49.91% recall. Adjusting only the previous model's threshold gave
zero FP but just 4.88% recall.

The paired conditional 95% interval for the candidate's FPR change versus the
previous adaptation is **−1.02 to +0.51 percentage points**, including zero.
Its 1.19% test FPR also exceeds the 1% limit. The replacement criterion **failed**;
no further tuning used this inspected test. Author labels, missing per-message
dates and consent metadata, and the spam-heavy mix limit generalization.
The source permits research access without a standard data license; raw messages
are not redistributed. See the [complete study](HAM_EXPANSION.md) for the source
audit, development counts, all four operating points and reproduction commands.

## Artifacts and reproducibility

The authoritative settings and exact versions are in
[metadata.json](../results/full/metadata.json); decisions are frozen in
[selection.json](../results/full/selection.json), and verification scope is recorded
in [verification.json](../results/full/verification.json). Dataset, training source
and saved models have SHA-256 records. The runtime is pinned in `requirements.txt`.

Git intentionally excludes joblib models. Reproduce into an unused output directory,
then load that run's selected artifact as shown in the README. Load only trusted
artifacts. The local demo checks the artifact hash, package versions, model identity
and recorded threshold before loading. It never trains on startup or saves input
messages. Exact LinearSVC feature contributions plus the intercept reconstruct
the score; these are associations, not causal explanations. Any hosted deployment
would require its own access, privacy and operational review.

All historical and follow-up results remain frozen. The external samples have
been inspected and cannot support another independent claim after tuning on them.
The completed studies do not establish reliable filtering of a modern inbox.
