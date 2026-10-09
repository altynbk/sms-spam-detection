"""Rebuild the existing paper from the committed experiment tables.

Requires requirements-paper.txt. Run build_paper_figures.py first.
Export the resulting DOCX to PDF with a compatible office renderer.
"""

import csv
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/SMS_Spam_Detection_Paper.docx"


def read_csv(protocol, name):
    with (ROOT / "results" / protocol / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(protocol, name):
    return json.loads((ROOT / "results" / protocol / name).read_text())


def main():
    meta = read_json("full", "metadata.json")
    template = read_json("template", "metadata.json")
    clean = read_csv("full", "clean_metrics.csv")
    selected = meta["selected_model"]
    # The discussion is specific to these experiments; fail instead of silently
    # attaching old interpretations to a newly trained model or changed split.
    assert selected == "linear_svc_word_gentle"
    assert template["selected_model"] == "linear_svc_char_no_punctuation"
    assert meta["split_counts"]["test"] == {"ham": 904, "spam": 130, "total": 1034}
    tuned = next(r for r in clean if r["model"] == selected and r["threshold_mode"] == "validation")
    assert (int(tuned["fp"]), int(tuned["fn"])) == (4, 8)

    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin = section.bottom_margin = Inches(0.65)
    section.left_margin = section.right_margin = Inches(0.75)
    section.footer_distance = Inches(0.3)
    for name in ["Normal", "Title", "Subtitle", "Heading 1", "Heading 2", "Caption"]:
        style = doc.styles[name]
        style.font.name = "Times New Roman"
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_after = Pt(6)
        for fonts in style.element.xpath(".//w:rFonts"):
            for attr in list(fonts.attrib):
                if "theme" in attr.lower():
                    del fonts.attrib[attr]
        for border in style.element.xpath(".//w:pBdr"):
            border.getparent().remove(border)
    normal = doc.styles["Normal"]
    normal.font.size = Pt(11)
    normal.paragraph_format.line_spacing = 1.08
    normal.paragraph_format.widow_control = True
    doc.styles["Title"].font.size = Pt(21)
    doc.styles["Title"].font.bold = True
    doc.styles["Title"].paragraph_format.space_after = Pt(10)
    for name, size in [("Heading 1", 14), ("Heading 2", 12)]:
        doc.styles[name].font.size = Pt(size)
        doc.styles[name].font.bold = True
        doc.styles[name].paragraph_format.space_before = Pt(10)
        doc.styles[name].paragraph_format.keep_with_next = True
    doc.styles["Caption"].font.size = Pt(9.5)
    doc.styles["Caption"].font.italic = False
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    doc.core_properties.title = "SMS Spam Detection with Grouped Evaluation and Character Obfuscation"
    doc.core_properties.author = "Kabiyev Altynbek"
    doc.core_properties.subject = "Reproducible evaluation of sparse SMS classifiers"
    doc.core_properties.comments = ""
    doc.core_properties.last_modified_by = "Kabiyev Altynbek"

    def p(text, style=None):
        return doc.add_paragraph(text, style)

    def h(text):
        doc.add_heading(text, level=1)

    def page():
        doc.add_page_break()

    def table(headers, rows, widths, caption):
        cap = p(caption, "Caption")
        cap.paragraph_format.keep_with_next = True
        t = doc.add_table(rows=1, cols=len(headers))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        for col, width in zip(t.columns, widths):
            col.width = Inches(width)
        for cells, values, header in [(t.rows[0].cells, headers, True)] + [
                (t.add_row().cells, row, False) for row in rows]:
            for i, (cell, value, width) in enumerate(zip(cells, values, widths)):
                cell.width = Inches(width)
                cell.text = str(value)
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                tcpr = cell._tc.get_or_add_tcPr()
                borders = OxmlElement("w:tcBorders")
                for edge in ["top", "left", "bottom", "right"]:
                    el = OxmlElement(f"w:{edge}")
                    for key, val in [("val", "single"), ("sz", "4"), ("color", "D9D9D9")]:
                        el.set(qn(f"w:{key}"), val)
                    borders.append(el)
                tcpr.append(borders)
                margin = OxmlElement("w:tcMar")
                for edge, val in [("top", "65"), ("bottom", "65"), ("left", "80"), ("right", "80")]:
                    el = OxmlElement(f"w:{edge}")
                    el.set(qn("w:w"), val)
                    el.set(qn("w:type"), "dxa")
                    margin.append(el)
                tcpr.append(margin)
                if header:
                    shade = OxmlElement("w:shd")
                    shade.set(qn("w:fill"), "EAEAEA")
                    tcpr.append(shade)
                for para in cell.paragraphs:
                    para.alignment = WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
                    para.paragraph_format.space_after = Pt(0)
                    para.paragraph_format.line_spacing = 1.05
                    for run in para.runs:
                        run.font.size = Pt(9.5)
                        run.bold = header
        repeat = OxmlElement("w:tblHeader")
        t.rows[0]._tr.get_or_add_trPr().append(repeat)
        for row in t.rows:
            row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        p("").paragraph_format.space_after = Pt(0)

    p(doc.core_properties.title, "Title")
    p("Kabiyev Altynbek\nAstana IT University, Kazakhstan")
    h("Abstract")
    p("This study evaluates eleven sparse text classifiers on the historical English SMS Spam Collection, with explicit separation of model search, threshold selection and final testing. Exact deduplication leaves 5,171 messages. Case and whitespace groups are kept together in train, validation and test partitions and in five training cross-validation folds. The selected word TF-IDF LinearSVC achieves spam F1 0.9531, with four false positives and eight false negatives among 1,034 test messages. Its conditional 95% group-bootstrap F1 interval is 0.9249 to 0.9769. At obfuscation intensity 0.3, mean spam recall falls from 0.9385 to 0.6035 for separator insertion. Character features preserve more recall in these tests but can incur more false positives. An exploratory split using URL and digit templates selects a different model and exposes a worse tradeoff after threshold tuning. These results support a reproducible historical benchmark and a controlled sensitivity study; they do not establish performance on current SMS, new languages or adaptive attacks.")
    p("Keywords: SMS spam, TF-IDF, grouped evaluation, threshold selection, character obfuscation.")
    h("1 Introduction")
    p("SMS filtering must detect unwanted messages without suppressing legitimate communication. Class imbalance makes accuracy an incomplete measure: on this test set, always predicting ham achieves 87.43% accuracy but detects no spam. We therefore report spam precision, recall and F1 together with false-positive and false-negative counts. We also separate clean-text quality from sensitivity to character edits that can disrupt a vocabulary-based classifier.")
    p("The contribution is an auditable comparison of representations and decision rules. All candidates share training folds and attack realizations. Model choice is frozen before test scoring, and a separate validation set chooses the operating threshold. A second grouping rule examines residual template overlap. Neither test set is used to repair a threshold or to replace the selected model after viewing its errors.")
    h("2 Related work")
    p("The SMS Spam Collection accompanies Almeida, Hidalgo and Yamakami's benchmark research [1, 2]. Its mixed historical sources make it useful for reproducible experiments, while limiting claims about deployment today. Uddin et al. study transformer-based detection and explainability in ExplainableDetector [3]. Li et al.'s SpamDam addresses newer SMS collection, federated learning and adversarial evaluation [4]. COPS studies a compact on-device smishing and URL pipeline [5]. These works motivate broader data and deployment studies; their published metrics are not directly comparable to our deduplicated splits. We do not reproduce those systems or claim to outperform them.")

    page()
    h("3 Data and evaluation protocol")
    p("The bundled corpus contains 5,574 labeled messages [1]. Input checks reject missing or blank training text, unsupported labels and conflicting labels within a group. Removing 403 exact duplicate rows leaves 4,518 ham and 653 spam messages. The remaining original strings are preserved as model inputs. Lowercasing and collapsing whitespace define 5,159 split groups; punctuation and Unicode remain in these keys. Message and group hashes provide an auditable membership record.")
    table(["Partition", "Ham", "Spam", "Total", "Purpose"],
          [[s.title(), meta["split_counts"][s]["ham"], meta["split_counts"][s]["spam"], meta["split_counts"][s]["total"], role]
           for s, role in [("train", "Model search and fitting"), ("validation", "Threshold selection"), ("test", "Final reporting")]],
          [1.1, .7, .7, .7, 3.8], "Table 1. Default protocol partitions after exact deduplication.")
    p("Unique homogeneous groups are stratified by label into approximately 60% training, 20% validation and 20% test data, using seeds 42 and 43. Whole groups remain in one partition. Five shared StratifiedGroupKFold folds inside training use seed 42. Every cleaner, vocabulary and inverse-document-frequency transformation is fitted within its training fold. No validation or test text enters feature fitting. Grouping reduces a declared form of overlap; it does not prove sender, campaign or temporal independence.")
    h("4 Models and operating thresholds")
    p("The implementation uses scikit-learn pipelines [6]. Four original word-feature baselines are Multinomial Naive Bayes, Logistic Regression, linear-kernel SVC and Random Forest. Seven LinearSVC variants compare word and character representations under legacy cleanup, gentle cleanup, their word-plus-character combination, and character models with punctuation or currency symbols removed. Legacy cleanup lowercases text and retains only a-z, 0-9 and whitespace. Gentle cleanup lowercases and collapses whitespace without stripping symbols.")
    p("Word TF-IDF uses unigrams and bigrams; char_wb TF-IDF uses character 3- to 5-grams within word boundaries. Vocabulary budgets are 1,000 or 5,000 features; the combined representation divides the budget equally and normalizes the concatenation. Search uses alpha in {0.1, 1} for Naive Bayes; C in {0.5, 1} and class weights in {none, balanced} for linear models; and depth in {unlimited, 20} with minimum leaf size in {1, 2} for the 100-tree forest. There are 84 candidates and 420 fold fits.")
    p("Each variant retains its highest mean training-CV spam F1 configuration. The overall choice uses that same criterion, breaking model ties alphabetically. Best searched CV scores are selection scores, not independent estimates. Each chosen pipeline is refitted on training only. Its validation threshold maximizes recall subject to empirical FPR <= 1%, then prefers fewer false positives and a higher threshold. Decisions use score >= threshold. The model is not refitted afterward. Native default predictions are reported separately; the validation constraint does not guarantee future FPR [7].")

    page()
    h("5 Clean test results")
    names = {"nb_word_legacy": "NB word legacy", "lr_word_legacy": "LR word legacy", "svc_word_legacy": "SVC word legacy", "rf_word_legacy": "RF word legacy", "linear_svc_word_legacy": "LinearSVC word legacy", "linear_svc_char_legacy": "LinearSVC char legacy", "linear_svc_word_gentle": "LinearSVC word gentle*", "linear_svc_char_gentle": "LinearSVC char gentle", "linear_svc_word_char_gentle": "LinearSVC word + char", "linear_svc_char_no_punctuation": "LinearSVC char no punctuation", "linear_svc_char_no_currency": "LinearSVC char no currency"}
    rows = []
    for name in meta["models"]:
        default, val = [next(r for r in clean if r["model"] == name and r["threshold_mode"] == mode) for mode in ["default", "validation"]]
        rows.append([names[name], f'{meta["models"][name]["best_cv_f1"]:.4f}', f'{float(default["f1"]):.4f}', f'{float(val["f1"]):.4f}', f'{val["fp"]}/{val["fn"]}', f'{100*float(val["fpr"]):.2f}%'])
    table(["Variant", "CV F1", "Default F1", "Tuned F1", "FP/FN", "FPR"], rows,
          [2.65, .75, .95, .9, .85, .9], "Table 2. Default protocol. FP/FN and FPR use the validation threshold. *CV-selected model.")
    p("Training CV selects word/gentle LinearSVC (C = 1, balanced class weights, vocabulary limit 5,000). At threshold -0.08620419778818611, test precision is 0.9683 and recall is 0.9385. The confusion counts are TN = 900, FP = 4, FN = 8 and TP = 122. Accuracy is 0.9884, ROC-AUC is 0.9957 and average precision is 0.9817. Native default predictions yield three false positives and ten false negatives, so tuning exchanges one additional false positive for two fewer missed spam messages.")
    p("Several alternatives have higher test F1 under particular decision rules. They do not replace the CV choice. Tuning reduces test F1 for several variants, including the character/gentle model: its recall increases, but 19 of 904 legitimate messages become false positives (2.10% FPR). This illustrates the difference between the validation objective and a guarantee on unseen messages. All eleven models are retained to make these tradeoffs visible.")
    h("6 Conditional uncertainty")
    p("For the fixed selected model, we sample whole test groups with replacement separately within ham and spam, preserving all messages in each selected group. We use 5,000 draws, seed 42, and the 2.5th and 97.5th percentiles of the resulting metric distribution [8]. The default protocol contains 1,032 test groups. The tuned F1 interval is 0.9249 to 0.9769; recall is 0.8931 to 0.9769 and FPR is 0.11% to 0.88%. Predictions reconstructed from the complete error ledger must match the saved confusion counts before resampling.")
    p("These intervals condition on the fitted model, decision rule, split and observed class/group composition. They exclude training and model-search variability, unknown campaign dependence and domain shift. With few errors, percentile intervals can be discrete or degenerate. They are approximate descriptive uncertainty estimates, not evidence that the same coverage holds in a new SMS population.")

    page()
    h("7 Character obfuscation experiments")
    p("Three nonadaptive scenarios modify raw spam before the fitted preprocessing pipeline. Leet replaces eligible ASCII letters using a fixed substitution map. Separator insertion places a period, hyphen or underscore between adjacent ASCII letters. Unicode substitution replaces eligible Latin letters with specified Cyrillic lookalikes. Intensity is the edit probability per eligible character or boundary, rather than the fraction of all characters changed. We use intensities 0.0 to 0.5 in steps of 0.1 and twenty seeds, 10042 to 10061.")
    p("Every model receives the same attacked messages for each scenario, intensity and seed. Ham remains unchanged, and clean validation thresholds stay fixed. Attack success rate (ASR) is the fraction of initially correctly detected spam that becomes ham after editing; its denominator excludes spam already missed on clean input. Twenty seeds measure randomness of these edits on the same test messages, not uncertainty across new training or test sets.")
    doc.add_picture(str(ROOT / "paper/figures/robustness.png"), width=Inches(7))
    p("Figure 1. Default protocol, gentle cleanup and validation thresholds. Curves show mean spam recall; bands show one attack-seed standard deviation. All classifiers are LinearSVC. Different clean false-positive rates must be considered alongside recall.", "Caption")
    robustness = read_csv("full", "robustness_summary.csv")
    rows = []
    for model, short in [(selected, "Word"), ("linear_svc_char_gentle", "Character"), ("linear_svc_word_char_gentle", "Word + character")]:
        row = next(r for r in clean if r["model"] == model and r["threshold_mode"] == "validation")
        vals = [next(r for r in robustness if r["model"] == model and r["threshold_mode"] == "validation" and r["attack_type"] == attack and float(r["intensity"]) == .3) for attack in ["leet", "separators", "unicode"]]
        rows.append([short, f'{100*float(row["fpr"]):.2f}%', *[f'{float(v["recall_mean"]):.3f} / {float(v["asr_mean"]):.3f}' for v in vals]])
    table(["Features", "Clean FPR", "Leet R / ASR", "Separators R / ASR", "Unicode R / ASR"], rows,
          [1.2, .85, 1.55, 1.75, 1.65], "Table 3. Mean recall (R) and ASR at intensity 0.3, with gentle cleanup and tuned thresholds.")
    p("The selected word model is most sensitive to separator insertion at intensity 0.3: mean recall drops to 0.6035 and ASR reaches 0.3582. Character features retain more recall across these scenarios, but their tuned clean FPR is 2.10%, compared with 0.44% for words. The combined model offers an intermediate measured tradeoff. These observations do not isolate a universal representation advantage: configurations and operating points differ. Human readability, meaning preservation, adaptive optimization and modifications to ham were not evaluated.")

    page()
    h("8 Sensitivity to message templates")
    p("Case and whitespace grouping can separate messages that differ only in a phone number or URL. A diagnostic rule masks URLs and digit runs in normalized messages longer than thirty characters. Short messages retain their normalized key to reduce accidental merging of unrelated short replies. The rule is stateless, uses no model scores and changes grouping only; classifiers still receive original strings. Conflicting labels within a template group are rejected.")
    p("The default split contains thirteen templates crossing partition boundaries, covering thirty-one messages; six test messages have a training counterpart under this rule. A separate complete experiment uses the 5,137 template groups for both holdout splitting and training CV. It produces 3,095 training, 1,040 validation and 1,036 test messages, including 904 ham and 132 spam in test. No template crosses partitions under the declared rule. This does not identify every semantic paraphrase or campaign.")
    template_clean = read_csv("template", "clean_metrics.csv")
    template_ci = read_csv("template", "confidence_intervals.csv")
    rows = []
    for mode, label in [("default", "Native default"), ("validation", "Validation tuned")]:
        r = next(r for r in template_clean if r["model"] == template["selected_model"] and r["threshold_mode"] == mode)
        ci = next(r for r in template_ci if r["metric"] == "f1" and r["threshold_mode"] == mode)
        rows.append([label, f'{float(r["precision"]):.4f}', f'{float(r["recall"]):.4f}', f'{float(r["f1"]):.4f}', f'{r["fp"]}/{r["fn"]}', f'{float(ci["lower"]):.4f} to {float(ci["upper"]):.4f}'])
    table(["Decision rule", "Precision", "Recall", "F1", "FP/FN", "95% F1 interval"], rows,
          [1.3, .8, .75, .75, .7, 2.7], "Table 4. Template protocol, CV-selected character LinearSVC without punctuation.")
    p("Training CV selects the character model without punctuation in this experiment. Tuning raises recall slightly but increases false positives from two to eleven and lowers F1 from 0.9531 to 0.9288. Test FPR reaches 1.22% despite meeting the validation constraint. The conditional intervals use 1,028 template test groups. These findings retain the unfavorable outcome rather than choosing a more flattering threshold after evaluation.")
    p("The template analysis was introduced after inspecting the default experiment and is exploratory. The two protocols change the split, fitted models and selected configurations together. Their difference is not a causal estimate of leakage, and their F1 intervals are not a paired significance test. The default model remains the frozen choice for its original protocol.")
    h("9 Limitations and next evaluation")
    p("Both protocols use one historical English corpus. Row order is not a timestamp, and the available text labels do not establish independent senders or campaigns [1]. Related messages may remain after either grouping rule. Performance on modern scams, Russian or Kazakh text, code-switching, email and URLs as separate objects remains unmeasured. A ham prediction does not establish that a link or sender is safe.")
    p("A stronger generalization study should first obtain independently labeled, licensed modern ham and spam, freeze the current model, and report errors by source, time and language before considering retraining. A new final holdout is needed for testing defenses designed after these attacks. Deployment also requires measurements of latency, memory, privacy behavior and the cost of false positives; sparse features alone do not establish on-device suitability.")

    page()
    h("10 Reproducibility and conclusion")
    p("The repository supplies raw-data checks, split and CV manifests, all 84 search results, clean and validation metrics, attack seeds and hashes, complete error ledgers, interval reports and fitted-pipeline export. Dataset, training source and model artifacts have SHA-256 records. Both full protocols were independently checked against saved models, including all 7,920 attacked metric rows per run. The environment uses Python 3.12.14 and scikit-learn 1.9.1; exact dependencies and execution metadata are recorded with each experiment.")
    p("The full run uses python -m spam_detector.train --mode full. Adding --grouping template selects the sensitivity protocol. Independent recomputation uses scripts/verify_results.py with --all-attacks; uncertainty uses python -m spam_detector.uncertainty. Output directories must be unused for training, preventing accidental replacement of a completed run. The README provides the complete commands. Saved model files must come from a trusted run and are regenerated locally rather than committed as binary release artifacts.")
    p("This study finds high clean-test quality for a CV-selected word model alongside substantial sensitivity to separator edits. Character features improve recall under the tested edits at operating points that can also increase false positives. Template grouping and conditional uncertainty qualify the result without changing it into a deployment guarantee. The next substantive evidence gain would come from independent modern data, rather than another classifier selected on the same test set.")
    p("Code and experiment artifacts: https://github.com/altynbk/sms-spam-detection")
    h("References")
    refs = [
        "[1] T. Almeida and J. Hidalgo. SMS Spam Collection. UCI Machine Learning Repository, 2012. Dataset DOI: 10.24432/C5CC84. https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
        "[2] T. A. Almeida, J. M. G. Hidalgo and A. Yamakami. Contributions to the study of SMS spam filtering: new collection and results. ACM Symposium on Document Engineering, 2011. Publication record linked from [1].",
        "[3] M. A. Uddin et al. ExplainableDetector: Exploring Transformer-based Language Modeling Approach for SMS Spam Detection with Explainability Analysis. arXiv:2405.08026, 2024. https://arxiv.org/abs/2405.08026",
        "[4] Y. Li, R. Zhang, W. Rong and X. Mi. SpamDam: Towards Privacy-Preserving and Adversary-Resistant SMS Spam Detection. arXiv:2404.09481, 2024. https://arxiv.org/abs/2404.09481",
        "[5] Harichandana B S S et al. COPS: A Compact On-device Pipeline for real-time Smishing detection. IEEE CCNC, 2024. arXiv:2402.04173. https://arxiv.org/abs/2402.04173",
        "[6] F. Pedregosa et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011. https://jmlr.org/papers/v12/pedregosa11a.html",
        "[7] Scikit-learn developers. Tuning the decision threshold for class prediction. User Guide, accessed 9 October 2026. https://scikit-learn.org/stable/modules/classification_threshold.html",
        "[8] SciPy developers. scipy.stats.bootstrap. Reference documentation, accessed 9 October 2026. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html",
    ]
    for ref in refs:
        para = p(ref)
        para.paragraph_format.space_after = Pt(7)
        for run in para.runs:
            run.font.size = Pt(9.5)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
