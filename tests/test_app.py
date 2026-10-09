from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest
from spam_detector.evaluation import metrics
from spam_detector.demo import MAX_ATTACK_INPUT_CHARS, MAX_MESSAGE_CHARS
from test_demo import LABELS, TEXTS, model_run

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def demo(tmp_path, monkeypatch):
    # Synthetic fixture verifies the app contract, not model performance.
    model, _ = model_run(tmp_path)
    rows, attacks = [], []
    for mode, threshold in [("default", 0), ("validation", model.threshold)]:
        result = metrics(LABELS, model.score(TEXTS), threshold)
        rows.append(dict(model=model.model_id, threshold_mode=mode, **result))
        for attack in ["leet", "separators", "unicode"]:
            for intensity in [0, .3]:
                attacks.append(dict(model=model.model_id, threshold_mode=mode, attack_type=attack,
                                    intensity=intensity, recall_mean=result["recall"], recall_std=0,
                                    asr_mean=0, asr_std=0))
    pd.DataFrame(rows).to_csv(tmp_path / "clean_metrics.csv", index=False)
    pd.DataFrame(attacks).to_csv(tmp_path / "robustness_summary.csv", index=False)
    monkeypatch.setenv("SMS_SPAM_RUN", str(tmp_path))
    return tmp_path, model


def test_message_flow_matches_saved_model_and_handles_blank_and_oov(demo):
    _, model = demo
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    assert not app.exception
    app.button[3].click().run()
    assert any("nonblank" in w.value for w in app.warning)
    app.text_area(key="sms").input("FREE prize cash")
    app.button[3].click().run()
    expected = "Spam" if model.predict("FREE prize cash")[0] else "Ham"
    assert expected in [h.value for h in app.subheader]
    assert not app.exception
    app.text_area(key="sms").input("!!!")
    app.button[3].click().run()
    assert any("No recognized" in w.value for w in app.warning)
    assert not {"Spam", "Ham"} & {h.value for h in app.subheader}


def test_attack_identity_repeatability_and_results_controls(demo):
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    app.slider[0].set_value(0).run()
    app.button[4].click().run()
    assert app.code[0].value == app.code[1].value
    app.slider[0].set_value(.3).run()
    app.button[4].click().run()
    edited = app.code[1].value
    app.button[4].click().run()
    assert app.code[1].value == edited
    app.radio[0].set_value("default").run()
    app.multiselect[0].set_value([]).run()
    assert any("Choose at least" in info.value for info in app.info)
    assert not app.exception


def test_missing_artifact_keeps_results_available(demo):
    path, _ = demo
    (path / "models/model.joblib").unlink()
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    assert not app.exception
    assert app.button[3].disabled and app.button[4].disabled
    assert len(app.dataframe) > 0
    assert any("Prediction is unavailable" in info.value for info in app.info)


def test_long_attack_input_leaves_room_for_inserted_separators(demo):
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    assert app.text_area[0].proto.max_chars == MAX_MESSAGE_CHARS
    assert app.text_area[1].proto.max_chars == MAX_ATTACK_INPUT_CHARS
    original = ("FREE prize " * MAX_ATTACK_INPUT_CHARS)[:MAX_ATTACK_INPUT_CHARS]
    app.text_area[1].input(original)
    app.selectbox[0].set_value("separators")
    app.slider[0].set_value(.5)
    app.button[4].click().run()
    assert not app.exception
    assert len(app.code[1].value) > len(original)
    assert not any("at most" in w.value for w in app.warning)
    assert sum(h.value in {"Spam", "Ham"} for h in app.subheader) == 2
