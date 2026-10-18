# External SMS source and fixed evaluation protocol

Source: Sharad Agarwal, Antonis Papasavva, Guillermo Suarez-Tangil and Marie
Vasek, *Fishing for Smishing: Understanding SMS Phishing Infrastructure and
Strategies by Mining Public User Reports*, ACM IMC 2025.
[Paper](https://doi.org/10.1145/3730567.3764431) ·
[Author artifact](https://github.com/reportsmishing/Smishing-Dataset-IMC25)

The upstream snapshot is pinned to commit
`a6175560b57387199871e51fbef6bc523d2516b4` (12 September 2025). Its
`final_dataset_output.csv` contains 33,869 rows. The collection covers public
reports from five sources during 2017–2024; publication in 2025 does not mean
the messages were received in 2025. The released text is pseudonymized and its
extraction/annotation involved language models. Some timestamps are incomplete.
This is a collection of reported smishing/spam, not a sample of ordinary inboxes.

License: **CC BY 4.0**, reproduced in [LICENSE.txt](LICENSE.txt). The source
metadata, exact download URL, byte count and SHA-256 are in
[source.json](source.json). Derived membership and prediction tables in this
repository are attributed to these authors under the same license. We do not
republish message text, sender metadata or translations. The source CSV is
downloaded locally and ignored by Git. No embedded message URLs are visited.

## Selection before scoring

[protocol.json](protocol.json) fixes the cohorts, model artifact, threshold,
input views, metrics and bootstrap parameters before any external predictions.
The preparation module does not load a classifier.

1. Keep nonblank original texts labeled English by the publisher. Do not use the
   translated-text column or infer English from an ASCII-only test.
2. **Primary cohort:** require exactly one distinct explicit four-digit year in
   the timestamp field, equal to 2023 or 2024. Day/month order is irrelevant to
   this year-only rule. Do not infer a year from "today", a time, or a two-digit
   year. A reported year is not independently verified ground truth.
3. **Supplementary cohort:** all eligible English texts, including older and
   undated records. It is not described as a uniformly recent sample.
4. Compare against the entire original SMS Spam Collection, including training,
   validation and test. Exclude normalized or existing template-key matches.
   These heuristics cannot rule out paraphrases, shared campaigns or matches
   obscured by upstream pseudonymization.
5. Deduplicate lowercase/whitespace-normalized texts within each cohort after
   eligibility filtering, retaining the first original source-row representative.
   Keep original text for inference. Record source rows and hashes without text.

The primary cohort has **372 distinct texts** from 488 eligible rows (480 rows
record 2023; eight record 2024). The supplementary cohort has **15,531 texts**
from 22,078 English rows. Neither cohort has a reference match under the declared
rules. Counts and exclusions are in the
[audit](../../../results/external/imc25/cohorts/audit.json). These cohorts overlap
and are not independent replications of each other.

## Frozen scoring plan

Use only the existing `linear_svc_word_gentle` artifact from `results/full`.
Primary prediction: its frozen validation threshold, **−0.08620419778818611**.
Report native default decisions alongside it, with no fitting, threshold
selection, model comparison or tuning on external outcomes.

The primary view uses the published pseudonymized text unchanged. A paired
sensitivity view removes only the declared publisher placeholder tokens (such
as `<URL>` and `<PHONE_NUMBER>`). Both views use exactly the same cohort. This
checks dependence on artificial tokens; it cannot recover the original SMS or
measure performance before anonymization.

Report detected/missed counts, recall and miss rate. **Do not report precision,
F1, accuracy, FPR, ROC-AUC or average precision:** without legitimate messages
they would be unavailable or misleading as classifier-quality estimates. Do
not combine historical ham with new positives to manufacture a modern F1.

Compute conditional 95% percentile intervals by resampling entire template
groups (5,000 draws, seed 42), retaining all members of each sampled group.
Intervals condition on the fitted model and this convenience sample; they do
not cover training, annotation, sampling or anonymization uncertainty. Category
breakdowns are descriptive. Related campaigns can span heuristic groups.

## Reproduce preparation

From the project root, with the pinned environment active:

```bash
python -m spam_detector.external_data --download --output results/local/imc25-cohorts
```

The download is verified against the pinned hash. Preparation writes membership
tables and an audit only, refuses to overwrite an existing run, and never trains
or predicts. The checked-in memberships were produced by this same module.

## Other sources considered

- [SMS Spam Collection mirrors](https://huggingface.co/datasets/codesignal/sms-spam-collection):
  recent upload dates do not make the original historical corpus independent.
- [Balanced LLM-generated SMS, 2025](https://data.mendeley.com/datasets/vmg875v4xs/1):
  generated using an older dataset; unsuitable as evidence on independently
  observed modern messages.
- [IIIT-D SMS release](https://india-data.org/dataset-details/171770c1-3e73-4c3f-b757-7357f1f22cd0):
  a recent portal release alone does not establish recent collection dates.

An independently collected, licensed modern **ham and spam** corpus remains
necessary to measure false positives and deployment tradeoffs.
