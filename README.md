# SMS Spam Detection

[![Tests and reproducibility](https://github.com/altynbk/sms-spam-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/altynbk/sms-spam-detection/actions/workflows/tests.yml)

A reproducible educational study of classical classifiers on the historical English
[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection),
with a local application for checking messages and exploring character-level attacks.

Eleven variants share grouped train/validation/test splits. Training
cross-validation selects the model; validation sets its decision threshold;
held-out data measure clean and attacked performance. Labels are **ham = 0, spam = 1**.

## Results at a glance

The selected word TF-IDF LinearSVC at its frozen validation threshold:

| Measure | Historical test |
|---|---:|
| Spam precision / recall / F1 | 0.9683 / 0.9385 / **0.9531** |
| False positives / legitimate messages | **4 / 904** |
| False negatives / spam messages | **8 / 130** |
| Conditional 95% group-bootstrap F1 interval | 0.9249–0.9769 |
| Mean spam recall under separators, probability 0.3 | **0.6035** |

These results do not establish performance on a modern inbox. An
[external evaluation](docs/EXTERNAL_EVALUATION.md) detects 176 of 372 dated
English smishing reports (**47.31% recall**); that source contains no legitimate
messages, so it cannot measure false-positive rates. Two adaptation studies failed
their replacement criteria, and the original model remains the demo default.
Russian/Kazakh and production traffic remain untested. See the
[model card](docs/MODEL_CARD.md) for intended use and limitations.

## Install

Run from the repository root with **Python 3.12**. If `.venv` already exists,
skip its creation and activate it.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Interactive demo

The committed reports can be explored immediately. Model binaries are excluded
from Git, so predictions require a matching local model; the app never trains on startup.

```bash
python -m pip install -r requirements-app.txt
python -m streamlit run app.py
```

Open [SMS Spam Lab](http://127.0.0.1:8501). Its four tabs cover message checking,
an attack playground, saved results and evaluation scope. The default run is
`results/full`.

To enable predictions when model files are missing, reproduce the full run below,
then launch against that run:

```bash
SMS_SPAM_RUN=results/local/full python -m streamlit run app.py
```

Load only trusted local artifacts. The app checks the recorded hash, versions,
model identity and threshold. Decision scores are not confidence percentages.
Messages stay in session memory without application logging or an external
prediction service; the server binds to localhost and disables usage telemetry.

## Reproduce the experiments

New runs belong under ignored `results/local/`. Training refuses to overwrite a
nonempty directory; choose an unused run name when repeating these commands.

```bash
# Smoke check; not the published full study.
python -m spam_detector.train --mode quick --jobs 2 --output results/local/quick
python scripts/verify_results.py --run results/local/quick --all-attacks

# Full study: 84 candidates, 420 cross-validation fits.
python -m spam_detector.train --mode full --jobs 2 --output results/local/full
python scripts/verify_results.py --run results/local/full --all-attacks
python -m spam_detector.uncertainty --run results/local/full
```

Use `--jobs 1` for less parallelism. Published tables remain in
[results/full](results/full). The [protocol](docs/PROTOCOL.md) records settings,
all eleven variants, attacks and the separate template sensitivity experiment.
Run `python -m pytest -q` for tests; see [validation](docs/VALIDATION.md) for
verification commands and GitHub checks.

<details>
<summary>Notebook, Python inference and paper commands</summary>

The notebook reads committed results without model binaries:

```bash
python -m pip install -r requirements-notebook.txt
python -m jupyterlab ML_Spam_Detector.ipynb
```

After reproducing the full run, load its selected model from the repository root.
The saved pipeline accepts original SMS strings and includes preprocessing:

```python
from spam_detector.demo import load_verified_model

model = load_verified_model("results/local/full")
print(model.predict(["FREE entry! Call NOW to claim £100", "Are we meeting for lunch?"]))
```

Rebuild the manuscript from the committed experimental tables:

```bash
python -m pip install -r requirements-paper.txt
python scripts/build_paper_figures.py
python scripts/build_external_figure.py
python scripts/build_paper.py
```

Export the DOCX to `paper/SMS_Spam_Detection_Paper.pdf` with Word or LibreOffice
and inspect every page.

</details>

## Documentation

- [Paper (PDF)](paper/SMS_Spam_Detection_Paper.pdf) · [Editable manuscript](paper/SMS_Spam_Detection_Paper.docx)
- [Executed notebook](ML_Spam_Detector.ipynb) — analysis, figures and error examples.
- [Model card](docs/MODEL_CARD.md) · [Experiment protocol](docs/PROTOCOL.md) · [Validation](docs/VALIDATION.md)
- Additional studies: [external evaluation](docs/EXTERNAL_EVALUATION.md), [modern-message adaptation](docs/MODERN_ADAPTATION.md), [ordinary-SMS expansion](docs/HAM_EXPANSION.md). Their results and limitations are retained separately.
- [Original audit](docs/AUDIT.md) · [Reference comparison](docs/REFERENCE_COMPARISON.md)
- [Contributing](CONTRIBUTING.md) · [Completion plan](docs/ROADMAP.md)

## License and data

Project code: [MIT](LICENSE). Historical dataset:
[CC BY 4.0, UCI DOI 10.24432/C5CC84](https://doi.org/10.24432/C5CC84).
The [protocol](docs/PROTOCOL.md#data-preparation-and-evaluation-protocol) records
the bundled CSV's provenance and hash. Additional datasets have separate terms,
documented in their study reports.
