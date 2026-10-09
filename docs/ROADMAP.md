# Development roadmap and comparable demos

Updated on 9 October 2026. The local SMS Spam Lab, exact linear explanations,
template sensitivity and conditional metric intervals are implemented. Public
hosting has not been provisioned. Project code, documentation and UI remain in English.

## What comparable projects implement

The repositories below were inspected for product ideas, including their UI source.
Their models and claimed metrics were not reproduced or independently audited.
They are useful implementation examples, not accuracy baselines for this project.

| Project | Observed implementation | Useful idea for this repository |
|---|---|---|
| [ang-1107/spam-detection](https://github.com/ang-1107/spam-detection) | Streamlit text area, a Classify button, blank-input handling and cached artifact loading; a CLI is also documented. | A small Python demo can reuse an existing inference function. Keep our single saved raw-text pipeline and threshold. |
| [raulradulescu/sms-spam-detection](https://github.com/raulradulescu/sms-spam-detection) | Streamlit examples, extracted URL/phone/text features, rule-based reason strings and a prediction breakdown. | Show examples and interpretable evidence. Distinguish keyword rules from actual contributions to our linear model. |
| [MubashirShafique/SMS-Email-Spam-Classifier-End-to-End-Project-](https://github.com/MubashirShafique/SMS-Email-Spam-Classifier-End-to-End-Project-) | A Streamlit client calls a FastAPI endpoint; the repository also contains a Flutter client. | An API becomes useful when several clients need the same model. A separate API and mobile app are optional later stages. |

The existing [SpamDam comparison](REFERENCE_COMPARISON.md) remains the research
reference for broader data and adversarial evaluation. An attractive demo and a
reliable evaluation solve different problems: a site improves accessibility of
the work, while new data determines how broadly its results apply.

## Implemented local demo

The **Streamlit application** reuses the existing package and saved model. It runs
in one Python process without a database; Jupyter remains the research report.
See the [launch instructions](../README.md#interactive-demo). The following flows
are implemented and covered by application or inference tests.

| Page | User experience | Completion criterion |
|---|---|---|
| Check a message | Paste an English SMS or choose a clearly labeled example; see Spam/Ham and optional model details. | UI and `SpamModel.predict` agree on fixed test inputs; blank input gets a useful prompt; no invented probability percentage. |
| Attack playground | Pick leet, separators or lookalikes, change intensity/seed, and compare original and perturbed text and decisions side by side. | Zero intensity is identical; a fixed seed is repeatable; the page never changes the model or threshold. It calls these predictions a demonstration, not aggregate ASR without known labels. |
| Results explorer | Inspect clean FP/FN, precision/recall/F1 and attack curves for both decision settings; compare word, char and combined features. | Values come from saved CSVs and identify the run/model; clean FPR is visible beside robustness results. |
| About the model | Short data/protocol explanation, limitations, GitHub and notebook links. | Clearly identifies the historical English scope and links to the model card. |

The attack playground is the strongest differentiator: it exposes an experiment
already implemented and tested in this repository. The model is fixed; interactive
examples are not used to optimize the existing holdout scores.

The optional dependency lock is `requirements-app.txt`. The loader verifies a
trusted selected-model artifact against its recorded hash, package versions,
identity and threshold. Messages stay in memory; blank and out-of-vocabulary
inputs do not receive an unqualified displayed decision.

Streamlit Community Cloud is one possible hosting route. Hugging Face Spaces is
another: its current SDK choices are Gradio, Docker and static HTML; Streamlit
uses the Docker route because its built-in SDK is deprecated.
See the [Spaces overview](https://huggingface.co/docs/hub/spaces-overview) and
[Streamlit deployment notice](https://huggingface.co/docs/hub/en/spaces-sdks-streamlit).
Choose hosting and artifact delivery before a public release; neither is provisioned
by this roadmap. A static GitHub Pages report alone would not run this Python model.

## Priorities after the first demo

| Priority | Extension | Why it helps | How to evaluate it |
|---|---|---|---|
| 1 | Independent modern-data benchmark | Addresses the largest gap in claims about real SMS. Include realistic ham as well as scams/spam. | Record licensing/provenance; separate sources, time and related templates; freeze the current model and report errors before considering retraining. |
| 2 | Repeated development splits | Conditional group-bootstrap intervals are implemented, but do not include refitting or search variability. | Evaluate a predefined split protocol on development data and preserve a new independent final holdout. |
| 2 | Calibrated probabilities or a review band | Makes confidence displays or "needs review" actions defensible. | Fit calibration/review rules on development data and evaluate calibration, coverage and FP/FN on a new holdout. Never turn the current SVM margin directly into a percentage. |
| 2 | Russian/Kazakh extension | Could make the project more locally useful. | Build independently labeled data in each language and code-switching examples; report per-language results. Changing UI language or translating the old corpus alone is insufficient evidence. |
| 3 | Stronger attacks and defenses | Explore invisible characters, spacing, paraphrases or adversarial augmentation. | Define perturbation budgets and meaning/readability checks; keep attack families/seeds used for development separate from final evaluation. |
| 3 | FastAPI and a custom web/mobile client | Useful if the demo becomes a multi-client service. | Reuse the same model artifact and contract; test latency, request limits and parity with local inference. Add authentication/storage only for a defined user need. |

The local lab and exact linear explanations complete the interactive release.
Independent modern data is the next priority for stronger ML claims.
Transformer comparisons, browser extensions and mobile apps can follow a concrete
need; adding them does not by itself improve the validity of the evaluation.

## Engineering foundation added now

GitHub Actions runs the pinned Python 3.12 environment on Linux and macOS, checks
dependencies, executes the tests, runs the real-data quick experiment and verifies
all of its attack rows. Its output is temporary and never replaces `results/full`.
The [model card](MODEL_CARD.md) records intended use, score semantics, measured
performance and artifact handling and serves as the inference contract for the demo.
