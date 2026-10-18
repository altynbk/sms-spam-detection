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
```

The command verifies source hashes and prepares partitions without loading a
classifier. Frozen source text and model binaries are never committed.
