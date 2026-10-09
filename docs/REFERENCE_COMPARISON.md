# Comparison with established implementations and primary references

Reviewed on 5 October 2026. This is a methodology and implementation comparison,
not a reproduction of external benchmarks. No SpamDam models were downloaded,
trained or evaluated here. External accuracy/F1 values are not a leaderboard for
this repository: datasets, duplicates, splits, class proportions, thresholds and
attack definitions differ.

## References and decisions

| Reference | Relevant practice | Assessment of this project |
|---|---|---|
| [scikit-learn: sparse text classification](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html) | TF-IDF with classical classifiers, including LinearSVC, Logistic Regression, Naive Bayes and Random Forest; the example uses 20 Newsgroups. | The model family is a reasonable baseline for sparse text. Retain the existing comparisons. This does not prove SMS performance, and BERT is not required merely to make the project valid. |
| [scikit-learn: text pipeline and search](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_text_feature_extraction.html) | A vectorizer and classifier form one pipeline passed to parameter search. | Our cleanup, TF-IDF and classifier are in one pipeline. Shared grouped folds fit their own vocabulary and IDF. The independently held-out test is distinct from train-CV and threshold validation. |
| [scikit-learn: decision thresholds](https://scikit-learn.org/stable/modules/classification_threshold.html) | Fitting a classifier and choosing its action threshold are separate steps; a prefit model needs new validation data for threshold selection. | Our separate validation split and lack of refitting after threshold selection follow this principle. The custom selector implements the explicit recall/FPR constraint and deterministic ties. A validation FPR limit is not a guarantee about test or future traffic. |
| [scikit-learn: DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) | A no-feature reference can predict the training majority class and return the training class prior. | Added this missing reference, outside the 11-model search. Always predicting ham gets 87.43% test accuracy here, but zero spam recall/F1. Accuracy alone would be misleading. |
| [SMS Spam Collection, UCI](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) | The corpus combines messages from several sources and accompanies Almeida et al.'s 2011 work. It is not chronologically sorted. | Keep the corpus for a historical benchmark. Lowercase/whitespace grouping controls a specific form of duplication; it does not establish sender, campaign or temporal independence. Row order must not be treated as a timestamp. |
| [SpamDam: paper](https://arxiv.org/html/2404.09481v1) and [author implementation](https://github.com/ChaseSecurity/SpamDam) | A broader SMS framework using newer collected spam, multilingual BERT, source-transfer comparisons, adversarial examples and poisoning experiments. | Our random leet/separator/lookalike tests cover a narrower threat model. They demonstrate sensitivity to these perturbations, not resistance to an adaptive adversary or modern production spam. A future external-data evaluation would address a larger evidence gap than adding another classifier to the same test set. |

The public SpamDam detector instructions name multilingual BERT; its analyzer
contains deletion, lookalike, invisible-character and reordering attacks. Our
Unicode experiment is related in subject, but the character mappings, sampling
and attack budget are not identical. Consequently, the respective attack-success
rates are not directly comparable. These observations come from the authors'
[detector directory](https://github.com/ChaseSecurity/SpamDam/tree/main/SMS_Spam_Detectors)
and [attack directory](https://github.com/ChaseSecurity/SpamDam/tree/main/SSD_Analyzer/adv_resistance/adversarial_examples).

## Findings acted on

1. **Fresh-clone notebook failure:** the previous introduction tried to train into
   the populated `results/full` directory, and the last cell unconditionally loaded
   an ignored model file. The report now works with the committed tables/figures;
   optional inference explains how to create `results/local/full` and switch `RUN`.
   The README inference path now agrees with that command.
2. **Missing class-imbalance reference:** `baseline_metrics.csv` reports a
   `DummyClassifier(strategy="prior")` fitted from training labels only. It is
   separate from CV, thresholds, frozen model selection and attack comparisons.
3. **Incomplete verifier coverage:** equal row counts did not prove that all
   configured attack seeds/scenarios were present. Verification now checks complete
   key sets, validation results, frozen thresholds, CV fold-score aggregates and
   selected parameters, the no-text baseline, error rows and robustness summaries.
   It recomputes predictions from saved models; it does not rerun CV training.
4. **Threshold context hidden in plots:** only tuned results were visible, and two
   models shared a color. Figures now show both native and tuned decisions with
   distinct, consistent colors. The notebook includes clean FPR beside the
   controlled word/character/combined comparison.

## What the evidence supports

The existing small Python package plus Jupyter report is appropriate for an
educational or portfolio experiment. A structural rewrite is not justified by
this review. Core checks cover data isolation, raw-text inference, native/tuned
decision semantics and deterministic attacks. The executed checks are recorded in
[VALIDATION.md](VALIDATION.md).

The selected word/gentle LinearSVC has test F1 **0.9531**, 4 false positives and
8 false negatives at its frozen validation threshold. That is a result for this
specific split, not evidence that it beats other published systems. Character
features are often more resilient in these attack scenarios, but their tuned
clean FPR is **2.10%**, versus **0.44%** for the selected word model. Do not describe
this as a free robustness improvement or select a replacement after seeing test.

This review changes diagnostics, verification and presentation. It preserves the
11 variants, parameter grids, split/CV seeds, selection criterion and thresholds.
Re-execution checks reproducibility on the same data; it is not a new independent
evaluation. A stronger deployment claim would require new, representative ham and
spam, source/time separation and a specified attack model. Such a study is outside
this bounded correction.
