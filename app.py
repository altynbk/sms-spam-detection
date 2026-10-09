"""Local SMS Spam Lab. Run with: streamlit run app.py."""

import json
import os
from pathlib import Path
import random

import pandas as pd
import streamlit as st

from spam_detector.attacks import obfuscate
from spam_detector.demo import (
    MAX_ATTACK_INPUT_CHARS,
    MAX_MESSAGE_CHARS,
    artifact_identity,
    inspect_message,
    load_verified_model,
)

ROOT = Path(__file__).resolve().parent
RUN = Path(os.environ.get("SMS_SPAM_RUN", str(ROOT / "results/full"))).resolve()
REPOSITORY = "https://github.com/altynbk/sms-spam-detection"

st.set_page_config(page_title="SMS Spam Lab", page_icon="✉", layout="wide")


@st.cache_resource(show_spinner=False)
def saved_model(run, metadata_hash, artifact_hash):
    return load_verified_model(run)


def decision(model, text, *, details=True):
    try:
        result = inspect_message(model, text)
    except ValueError as exc:
        st.warning(str(exc))
        return
    if not result["recognized_features"]:
        st.warning("No recognized vocabulary features. There is not enough lexical evidence to show a useful classification.")
        return
    label = "Spam" if result["prediction"] else "Ham"
    st.subheader(label)
    st.caption("Model prediction. A ham label does not establish that a sender or link is safe.")
    if details:
        with st.expander("Why this prediction?"):
            st.write("These are associations learned from historical SMS. They explain the model score, not the sender's intent.")
            a, b = st.columns(2)
            a.metric("Model score", f'{result["score"]:.4f}')
            b.metric("Spam threshold", f'{result["threshold"]:.4f}')
            st.caption("Spam is predicted when score ≥ threshold. The decision margin is not a probability or confidence percentage."
                       if result["score_kind"] == "decision_function" else
                       "Spam is predicted when score ≥ threshold. These model probabilities have not been calibrated.")
            if result["contributions"] is not None:
                frame = pd.DataFrame(result["contributions"])
                st.caption("Positive contributions increase the spam score; negative contributions decrease it. Showing the 12 largest absolute contributions.")
                st.bar_chart(frame.head(12), x="feature", y="contribution", horizontal=True,
                             color="#246c76", height=320, sort=False)
                st.write(f'All {len(frame)} feature contributions ({result["contribution_sum"]:.6f}) '
                         f'+ intercept ({result["intercept"]:.6f}) = score ({result["score"]:.6f}).')
                st.dataframe(frame, hide_index=True, column_config={"contribution": st.column_config.NumberColumn(format="%.6f")})
            else:
                st.info("Exact feature contributions are available for LinearSVC models.")


st.title("SMS Spam Lab")
st.write("Explore how a text classifier detects spam and how small edits change its decisions.")
st.caption("Historical English SMS · Research demonstration · Messages are processed in memory and are not saved by this app")

try:
    metadata = json.loads((RUN / "metadata.json").read_text())
    clean = pd.read_csv(RUN / "clean_metrics.csv")
    summary = pd.read_csv(RUN / "robustness_summary.csv")
    if metadata["status"] != "complete":
        raise ValueError("Incomplete experiment")
except (OSError, ValueError, KeyError):
    st.error("The experiment report is unavailable. Set SMS_SPAM_RUN to a completed local experiment directory.")
    st.stop()

model = None
try:
    _, _, meta_hash, model_hash = artifact_identity(RUN)
    model = saved_model(str(RUN), meta_hash, model_hash)
except (OSError, ValueError, KeyError) as exc:
    st.info("Prediction is unavailable until the saved model is prepared. You can still explore the recorded results.")
    with st.expander("Model setup"):
        st.write("Use a trusted local experiment with the matching environment. The README includes the training and launch commands.")
        st.text(str(exc))
        st.link_button("Setup instructions", REPOSITORY + "#interactive-demo")

selected = metadata["selected_model"]

check, playground, results, about = st.tabs(["Check a message", "Attack playground", "Results explorer", "About"])

with check:
    st.subheader("Check an English SMS")
    st.write("Paste a message or start with an illustrative example.")
    example_cols = st.columns(3)
    for col, label, value in [(example_cols[0], "Everyday example", "Hi, are we still meeting for lunch tomorrow?"),
                              (example_cols[1], "Promotion example", "WIN a FREE cash prize! Call now to claim your reward."),
                              (example_cols[2], "Clear", "")]:
        if col.button(label):
            st.session_state["sms"] = value
    with st.form("classify"):
        message = st.text_area("Message", key="sms", max_chars=MAX_MESSAGE_CHARS, height=140,
                               placeholder="Type or paste an English message…")
        submitted = st.form_submit_button("Check message", type="primary", disabled=model is None)
    if submitted:
        decision(model, message)
    st.caption("Examples are illustrative, not additional evaluation data. Russian and Kazakh accuracy has not been measured.")

with playground:
    st.subheader("Small edits, different evidence")
    st.write("Compare the same message before and after a reproducible character edit. The model and threshold stay fixed.")
    attack_text = st.text_area("Original message", "WIN a FREE cash prize! Call now to claim your reward.",
                              max_chars=MAX_ATTACK_INPUT_CHARS, height=110,
                              help=f"Up to {MAX_ATTACK_INPUT_CHARS:,} characters, leaving room for inserted separators.")
    a, b, c = st.columns(3)
    attack = a.selectbox("Edit type", ["leet", "separators", "unicode"])
    intensity = b.slider("Edit probability", 0.0, 0.5, 0.3, 0.1)
    seed = c.number_input("Random seed", 0, 1000000, 42, step=1)
    st.caption("Probability applies only to eligible characters or boundaries. No edits are made at probability zero.")
    if st.button("Compare predictions", type="primary", disabled=model is None):
        if not attack_text.strip():
            st.warning("Enter a nonblank English SMS")
        else:
            changed = obfuscate(attack_text, random.Random(int(seed)), intensity, attack)
            left, right = st.columns(2)
            with left:
                st.write("**Original**")
                st.code(attack_text, language=None, wrap_lines=True)
                decision(model, attack_text, details=False)
            with right:
                st.write("**Edited**")
                st.code(changed, language=None, wrap_lines=True)
                decision(model, changed, details=False)
    st.caption("A single example has no verified label here and is not an attack success rate. Aggregate research metrics are in Results explorer.")

with results:
    st.subheader("Recorded test results")
    st.caption(f'Run: {RUN.name} · {metadata["config"]["mode"]} experiment · '
               f'{metadata["config"].get("grouping", "normalized")} grouping · CV choice: {selected}')
    st.write("These values come from the saved experiment. Interactive messages do not change them.")
    mode = st.radio("Decision rule", ["validation", "default"], horizontal=True,
                    format_func=lambda x: "Validation threshold" if x == "validation" else "Native default")
    rows = clean[clean.threshold_mode == mode]
    chosen = rows[rows.model == selected].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Selected model F1", f"{chosen.f1:.4f}")
    b.metric("Spam recall", f"{chosen.recall:.2%}")
    c.metric("False positives", f"{int(chosen.fp)} / {int(chosen.n_ham)}")
    d.metric("False negatives", f"{int(chosen.fn)} / {int(chosen.n_spam)}")
    st.caption("The model was chosen using training cross-validation. Test outcomes do not choose the model or threshold.")
    st.dataframe(rows[["model", "precision", "recall", "f1", "fp", "fn", "fpr"]], hide_index=True,
                 column_config={k: st.column_config.NumberColumn(format="%.4f") for k in ["precision", "recall", "f1", "fpr"]})
    if (RUN / "confidence_intervals.csv").exists():
        with st.expander("Conditional 95% intervals for the selected model"):
            st.dataframe(pd.read_csv(RUN / "confidence_intervals.csv").query("threshold_mode == @mode")[["metric", "estimate", "lower", "upper"]], hide_index=True)
            st.caption("Whole groups resampled within each class. Fixed model and split; excludes training/search uncertainty and domain shift.")
    scenario = st.selectbox("Recorded attack scenario", ["leet", "separators", "unicode"])
    defaults = [name for name in [selected, "linear_svc_char_gentle", "linear_svc_word_char_gentle"] if name in set(rows.model)]
    models = st.multiselect("Compare models", rows.model.tolist(), default=list(dict.fromkeys(defaults)))
    chart_rows = summary[(summary.threshold_mode == mode) & (summary.attack_type == scenario) & summary.model.isin(models)]
    if models:
        st.line_chart(chart_rows.pivot(index="intensity", columns="model", values="recall_mean"),
                      x_label="Edit probability", y_label="Mean spam recall", height=300)
        st.dataframe(rows[rows.model.isin(models)][["model", "fpr", "fp", "fn"]], hide_index=True)
        with st.expander("Attack means, seed variability and success rates"):
            st.dataframe(chart_rows[["model", "intensity", "recall_mean", "recall_std", "asr_mean", "asr_std"]], hide_index=True)
    else:
        st.info("Choose at least one model to draw a comparison.")
    st.caption("Attacks modify spam only, so clean FPR stays fixed. Seed variability is not uncertainty across independent datasets.")
    st.download_button("Download clean metrics", clean.to_csv(index=False), "clean_metrics.csv", "text/csv")

with about:
    st.subheader("What this experiment establishes")
    st.write("The study compares sparse text classifiers on the historical English SMS Spam Collection. Exact duplicates are removed before group-aware splits. Training CV selects the model, validation selects the threshold, and test data measure clean and synthetic-attack performance.")
    st.write("The template sensitivity study and group-bootstrap intervals qualify the results. Neither establishes accuracy on current SMS, new languages or adaptive attacks. A ham prediction is not a security verdict.")
    st.write("The app uses the saved pipeline without training on startup. Message input stays in session memory; the app does not save it to files, cache it across users or send it to an external prediction service. A hosted server would necessarily receive submitted text.")
    st.link_button("Read the paper", REPOSITORY + "/blob/main/paper/SMS_Spam_Detection_Paper.pdf")
    st.link_button("Model card and limitations", REPOSITORY + "/blob/main/docs/MODEL_CARD.md")
    st.link_button("Reproduce the experiment", REPOSITORY)
