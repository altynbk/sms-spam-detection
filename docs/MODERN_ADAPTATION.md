# Adaptation to newer messages

## Data checkpoint before training

The existing historical model and its published results remain frozen. The
previously inspected IMC 2025 corpus is now development material for this follow-up;
it must not be presented as an unseen test of an adapted model.

A new external source is **Spam of the Times**, by Scott Lee Chua:
[author repository](https://github.com/scottleechua/data/tree/ad50c348c53f738645f78ee8baa8be5082a1de91/spam-and-marketing-sms).
The pinned 20 August 2026 snapshot contains 8,895 SMS records from one Philippine
recipient, collected since September 2022. Receipt timestamps are in UTC+8.
The author releases it under CC BY 4.0 and describes manual checks after initial
sender/phrase categorization. The source specification and license are in
[`data/external/adaptation`](../data/external/adaptation).

Only `spam` (unsolicited messages from unknown senders) and `gov` (government
public service announcements) enter this evaluation. We map them to spam=1 and
ham=0. We exclude advertising from known businesses because its consent status
is ambiguous. Private notifications and genuine OTPs are redacted by the author;
redaction markers cannot serve as negative examples. Sender and read-time fields
are never model inputs and are not republished.

This is a **mixed-language, single-recipient spam-versus-PSA challenge**, not a
representative English inbox or a general estimate of contemporary false-positive
rates. The publication date does not make every message a 2026 message. All dates
are retained in the membership ledger for separate descriptive reporting.

## Preparation fixed without classifier predictions

1. Validate the pinned file hash and the five-column CSV header. Explicitly
   exclude malformed source rows 1461 and 1977 (zero-based data-row indices): both
   have two category fields and a redacted body. Never guess their labels.
2. Exclude other categories, blank bodies and `<REDACTED>` text. There are 976
   eligible rows.
3. Compute a label-independent campaign key: Unicode NFKC/casefold, mask URLs,
   email addresses and digit runs, remove publisher placeholders and punctuation,
   then collapse whitespace. Exclude entire conflicting-label groups. There are
   no conflicts in this snapshot.
4. Keep the first normalized-text occurrence. Removing 191 repeats leaves 785
   messages: 55 PSAs and 730 spam messages. Distinct messages sharing a campaign
   key remain together for subsequent uncertainty calculations.
5. Remove matches to **all** historical SMS rows and all nonblank English IMC
   texts after removing the publisher's privacy placeholders. Also remove near
   copies at cosine similarity >=0.85 using fixed character 5-gram hashing,
   262,144 bins, nonnegative counts and L2 normalization. This screen uses text
   only; it neither fits a spam classifier nor examines prediction outcomes.
6. The reference screen excludes 31 messages, leaving **754 messages: 55 ham
   (54 groups) and 699 spam (643 groups)**. Receipt dates span 30 September 2022
   to 31 July 2026. The committed audit and membership contain hashes and source
   row indices, not SMS bodies.

Exact/template matching and the additional similarity screen reduce reuse of
known examples; they cannot certify semantic or campaign independence. Hash
collisions can conservatively exclude extra examples. This screened checkpoint
must also be checked against any newly acquired training/validation source
**before fitting or selecting candidates**. Publish that final membership before
scoring. No examples from this source may train the candidate or select its
threshold.

Reproduce the checkpoint with the existing pinned environment and IMC source:

```bash
python -m spam_detector.adaptation_data --download --output results/local/sms-history-protocol
```

The command checks source hashes and refuses to replace a nonempty output folder.
The reference outputs are in
[`results/adaptation/source_audit`](../results/adaptation/source_audit).
No model predictions have been produced for this source at this checkpoint.

## Additional ordinary-message data

The [Financial scams detection dataset, version 2](https://data.mendeley.com/datasets/znsk27yk3h/2)
by Al Rafi Ahmed and Gazi Faizul Islam was published on 22 October 2025 under
CC BY 4.0 (DOI: 10.17632/znsk27yk3h.2). Its description reports consented,
manually labeled real scam and ordinary messages in English and Bangla, gathered
from SMS, chats and email. Receipt dates and per-message channels are not promised.
It is a candidate **auxiliary development source**, not evidence of SMS-only
performance or a timestamped modern inbox. The acquired 47,442-byte CSV has 523 rows and two columns, `label` and
`message`. The source hash is pinned. We exclude 101 rows containing Bengali
script; this is a script filter, not a validated language detector. Three
normalized duplicates and seven historical overlaps are removed, leaving 412
rows before splitting. Claims about authenticity and annotation come from the
publisher; we have not independently audited collection or label accuracy.

## Frozen training and evaluation protocol

The complete membership is committed in
[`results/adaptation/protocol`](../results/adaptation/protocol) before training.
The [machine-readable protocol](../data/external/adaptation/protocol.json)
records source hashes, seed, six candidates, weights, selection rules and the
promotion criterion.

| Source | Train ham/spam | Validation ham/spam | New test ham/spam |
|---|---:|---:|---:|
| Historical benchmark | 2,704 / 366 | 900 / 126 | — |
| Financial messages | 99 / 147 | 33 / 48 | 32 / 46 |
| Previously inspected IMC reports | 0 / 11,856 | 0 / 1,927 | — |
| Independent SMS history | — | — | 55 / 699 |

Historical train/validation membership is preserved, except that any campaign
key present in the old test is excluded from development and a key shared by
old train/validation is reserved for validation. The historical test remains
unused for candidate selection or fitting.

Financial messages use stratified group splits of 60/20/20; IMC uses 80/20.
Seed is 20261018. Remove publisher privacy placeholders from IMC text to avoid
learning those source markers. Exclude IMC overlaps with the entire historical
and acquired financial corpora. After group splitting, exclude validation
near-copies of training messages and financial test near-copies of financial
training/validation using the same fixed 0.85 similarity rule. This removes two
financial validation rows, five test rows and 1,063 IMC validation rows.
The independent SMS history has no additional overlaps with the financial source.
The final two test sets have not been scored during preparation.

LinearSVC cannot incrementally fine-tune its weights. Adaptation therefore
**refits** six pipelines on the expanded training split: word 1–2 grams (5,000
features), character `char_wb` 3–5 grams (10,000), and their L2-normalized union,
each with C in {0.5, 1.0}, gentle preprocessing and fixed seed. Vectorizers fit
on training only. Class totals have equal weight; each source within a class
and each campaign within that source/class have equal total weight. This stops
the large positive-only IMC source from dominating the objective.

For each candidate, validation selects a threshold maximizing the mean spam
recall across the three validation sources, subject to empirical FPR <=1%
**separately** on historical and financial ham. Ties prefer lower mean FPR,
then a higher threshold. Candidate ties prefer word, character, then union,
and smaller C. The 33 financial validation ham messages permit zero false
positives under this empirical constraint; they do not establish a population
FPR guarantee. There is no refit after validation.

Score only the selected candidate and the frozen historical baseline on the two
new test sets. The financial holdout is unseen but shares its source with
training; the SMS history is a new source with a narrower negative class.
Report them separately. Use paired, class-stratified campaign-group bootstrap
intervals (5,000 draws, seed 20261018); uncertainty is conditional on these fixed
models and sampled sources. Never re-tune after looking at either test result.

The candidate remains research-only unless the 95% lower bound on SMS-history
F1 improvement is positive and its empirical FPR is both no worse than baseline
and <=1% on **each** new test source. Even passing this limited gate does not
establish representative production safety.

Reproduce the final membership after downloading the two licensed sources:

```bash
# Save the author's Financial scams detection dataset.csv as:
# data/external/adaptation/financial.csv
python -m spam_detector.adaptation_protocol --output results/local/adaptation-protocol
```

Raw messages and fitted model binaries stay outside Git. Training and evaluation
were performed only after the protocol was merged in PR #11.

## Measured outcome

The frozen protocol selected `adapted_word_char_c1`, using word and character
TF-IDF with C=1 and threshold **0.64907808395485**. It was fitted on 15,172 rows;
3,034 separate validation rows selected the candidate and threshold. Validation
macro recall was 0.8131; false positives were 0/33 financial ham and 3/900
historical ham. These small-sample validation constraints did not transfer.

| New evaluation set | Model | Spam recall | Spam F1 | False positives / ham | Misses / spam |
|---|---|---:|---:|---:|---:|
| Financial holdout | Original | 52.17% | 0.6400 | 5 / 32 | 22 / 46 |
| Financial holdout | Adapted | 80.43% | 0.8605 | 3 / 32 | 9 / 46 |
| Independent SMS history | Original | 42.63% | 0.5913 | 11 / 55 | 401 / 699 |
| Independent SMS history | Adapted | 58.66% | 0.7270 | 19 / 55 | 289 / 699 |

The adapted model finds 112 additional spam messages in the independent source,
but falsely flags eight additional PSAs. Its FPR there is **34.55%**, versus
20.00% for the original. Paired conditional 95% group-bootstrap intervals for
the change are **+0.0998 to +0.1721 F1**, **+11.99 to +20.12 percentage points
recall**, and **+1.79 to +28.07 percentage points FPR**. A higher spam F1 on this
spam-heavy sample does not establish safer filtering. Ranking quality also
falls on this source: ROC-AUC 0.6979 to 0.6466 and average precision 0.9667 to
0.9580. The apparent improvement is specific to the selected operating point.

On the smaller financial holdout, the paired F1 change interval is +0.1102 to
+0.3457. The FPR change interval includes zero (-18.75 to +6.25 percentage points);
32 ham examples cannot support a precise false-positive claim. This is an
unseen same-source, mixed-channel holdout, not a second independent SMS source.

**The preregistered promotion criterion failed.** The adapted artifact remains
an experimental candidate; the default model and demo are unchanged. Neither
model is suitable for automatic blocking on the measured modern samples. No
candidate, weight or threshold was changed after these test results were seen.
Further development needs more diverse real ordinary SMS, especially public
warnings and legitimate service messages, and a new untouched final test.
The two reported test sets are now inspected evidence, not reusable selection
sets with an independent-test claim.

Reproduce the frozen follow-up in an unused local directory:

```bash
python -m spam_detector.adaptation fit --output results/local/modern-adaptation
python -m spam_detector.adaptation evaluate --output results/local/modern-adaptation
python scripts/verify_adaptation.py --run results/local/modern-adaptation
```

The baseline artifact must exist at its recorded path and hash. Evaluation checks
all six training artifact hashes, runtime versions, source code, validation
outputs and the saved threshold before scoring the selected candidate. A run
with existing test predictions cannot be silently overwritten. The verifier
reconstructs all candidate validation scores and thresholds, both test-model
scores, confusion matrices, metrics and paired bootstrap intervals without
fitting a model. Published evidence is in
[`results/adaptation/evaluation`](../results/adaptation/evaluation); binaries
remain in the ignored local run.
