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

## Independent reported SMS

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

A separate follow-up refits six pipelines using historical development data,
previously inspected IMC reports and a licensed 2025 financial-message corpus
containing both scam and ordinary messages. The final 78-message financial
holdout and a new 754-message SMS source were reserved before fitting. The
source, group splits, near-copy exclusions and selection rule were merged before
training; IMC is development data for this follow-up, not an independent test.

Validation selects `adapted_word_char_c1` with threshold 0.64907808395485. On the
financial holdout, F1 rises from 0.6400 to 0.8605, FP falls from 5 to 3 of 32 ham,
and FN from 22 to 9 of 46 spam. On the independent SMS source, recall rises from
42.63% to 58.66% and F1 from 0.5913 to 0.7270, but FP rises from 11 to 19 of 55
ham (FPR 20.00% to 34.55%). ROC-AUC falls from 0.6979 to 0.6466. The paired
conditional 95% interval for the FPR increase is +1.79 to +28.07 percentage points.

The predeclared replacement gate **fails**. The candidate is a research artifact,
not the demo default. Ordinary financial messages are mixed-channel and lack
receipt dates; the independent SMS source is mixed-language, one recipient and
has government PSAs as its only retained ham class. A higher spam F1 does not
establish reliable general filtering. See the [complete follow-up](MODERN_ADAPTATION.md).

## Broader ordinary-SMS candidate

The [ham-expansion study](HAM_EXPANSION.md) adds 1,968 training ham and 11,860
training spam from the authors' 2018–2023 Super SMS subset. Another 787 ham and
4,497 spam form validation; 1,178 ham and 7,109 spam are held out in near-copy
connected components before fitting. Text overlaps with all earlier acquired
sources are removed. The previous SMS-history test is explicitly retired into development.
The new test is independent of fitting/selection but shares its author corpus
with new training data. Receipt dates, senders and component provenance are
unavailable per row; this is not an evaluation on representative 2025–2026 SMS.

Validation selects `ham_word_char_c1_w1`, threshold **0.48587418949350525**,
from six C/ham-weight combinations. On the new test, FP changes **17 → 14**
versus the previous adaptation (FPR 1.44% → 1.19%); spam recall changes
**73.26% → 95.22%**, FN 1,901 → 340, and F1 0.8445 → 0.9745. The original
historical model gives 15 FP and 49.91% recall on this same test. A validation-only
threshold adjustment to the previous model gives zero FP but just 4.88% recall.

The new FPR difference versus the previous model has a paired conditional 95%
interval of **−1.02 to +0.51 percentage points**, including zero. The 1% test FPR
criterion also fails. Thus the candidate **does not replace the historical demo**.
The artifact is available locally at
`results/local/ham-expansion/models/ham_word_char_c1_w1.joblib`; reproducible
training, selection, scores and verification are recorded in
`results/ham_expansion/evaluation/`. Raw source messages and binaries are ignored.
No further tuning used this now-inspected test.

Author labels, occasional transliteration/encoding artifacts, unknown recipient
consent and the spam-heavy test mix limit generalization. The source offers data
for research without a standard data license; raw messages are not redistributed.

## Artifacts, reproducibility and release considerations

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

For broader use, obtain independent modern ham and new spam with reliable dates,
define the false-positive cost and target languages, and reserve a new final
holdout. The reported external samples have now been inspected. IMC may support further
development, but a new untouched test is required for another independent claim.
Keep the historical and follow-up results frozen.

## Template sensitivity

The default split contains 13 crossing templates under a URL/digit masking rule,
including six test messages with a training counterpart. A separate full search
with template groups eliminates those crossings and selects the char/no-punctuation
LinearSVC. On its different test set, native predictions give F1 0.9531 (2 FP, 10 FN);
validation tuning gives F1 0.9288 (11 FP, 8 FN, FPR 1.22%). This does not replace the
default model or establish that one representation is universally best. The
analysis was introduced after inspecting the default run and is not an untouched
confirmatory evaluation. See `results/template/` and each run's template audit.
