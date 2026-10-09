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
percentage confidence**. A future UI should display the label and, in optional
details, the margin and threshold. Turning that number into a percentage using
clipping, an arbitrary sigmoid or multiplication by 100 is not calibrated confidence.
Probability estimates would require a separately validated calibration protocol;
see [scikit-learn's calibration guide](https://scikit-learn.org/stable/modules/calibration.html).

The API accepts a single string or a nonempty iterable of strings. Empty,
whitespace-only, punctuation-only and Unicode strings are valid inference inputs,
but receiving a finite score does not establish predictive usefulness. An
interactive demo should ask for nonblank text and explain when no vocabulary
features were recognized. The library rejects non-string inputs and empty batches.

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
experiment is exploratory; neither establishes temporal/source transfer.

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

## Artifacts, reproducibility and release considerations

The authoritative settings and exact versions are in
[metadata.json](../results/full/metadata.json); decisions are frozen in
[selection.json](../results/full/selection.json), and verification scope is recorded
in [verification.json](../results/full/verification.json). Dataset, training source
and saved models have SHA-256 records. The runtime is pinned in `requirements.txt`.

Git intentionally excludes joblib models. Reproduce into a new output directory,
then load that run's selected artifact as shown in the README. Load only trusted
artifacts. A deployed demo should carry a known artifact, matching package versions
and the recorded threshold, and should never train on startup or silently replace
the model. Prediction input should not be stored as a side effect of using the demo.

For broader use, collect a separate representative evaluation set, define the
false-positive cost and target languages, and test source/time transfer. Keep the
current results frozen when evaluating any new model or defense.

## Template sensitivity

The default split contains 13 crossing templates under a URL/digit masking rule,
including six test messages with a training counterpart. A separate full search
with template groups eliminates those crossings and selects the char/no-punctuation
LinearSVC. On its different test set, native predictions give F1 0.9531 (2 FP, 10 FN);
validation tuning gives F1 0.9288 (11 FP, 8 FN, FPR 1.22%). This does not replace the
default model or establish that one representation is universally best. The
analysis was introduced after inspecting the default run and is not an untouched
confirmatory evaluation. See `results/template/` and each run's template audit.
