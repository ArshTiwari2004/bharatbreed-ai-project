"""
test_ui.py
---------------------------------
Checks for the front end:
  - ui_helpers (quality check, upload saving, confirmation logging)
  - the Streamlit app itself, run headlessly with streamlit's AppTest,
    including a simulated "upload a photo -> confirm -> synced" flow.

Requires a trained model first:  python run_pipeline.py
Run:  python tests/test_ui.py      (or: python -m pytest tests/test_ui.py -v)
"""

import io
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ROOT = os.path.abspath(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import config  # noqa: E402
import ui_helpers as ui  # noqa: E402


def _sample_image_path():
    cls = sorted(d for d in os.listdir(config.RAW_DATA_DIR) if os.path.isdir(os.path.join(config.RAW_DATA_DIR, d)))[0]
    folder = os.path.join(config.RAW_DATA_DIR, cls)
    return os.path.join(folder, sorted(os.listdir(folder))[0])


def _jpeg_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype("uint8")).save(buf, format="JPEG")
    return buf.getvalue()


def test_quality_flags_dark_and_small():
    dark_small = _jpeg_bytes(np.full((60, 60, 3), 10))
    rep = ui.quality_report(dark_small)
    assert not rep["ok"]
    text = " ".join(rep["warnings"]).lower()
    assert "dark" in text and "small" in text


def test_quality_flags_blur():
    flat = _jpeg_bytes(np.full((300, 300, 3), 128))     # no edges at all -> very blurry
    assert any("blur" in w.lower() for w in ui.quality_report(flat)["warnings"])


def test_quality_rejects_non_image():
    for fn in (ui.quality_report, ui.save_upload):
        try:
            fn(b"this is not an image")
        except ValueError:
            continue
        raise AssertionError(f"{fn.__name__} accepted a non-image")


def test_save_upload_roundtrip():
    data = open(_sample_image_path(), "rb").read()
    path, digest = ui.save_upload(data)
    assert os.path.exists(path) and len(digest) == 12
    assert Image.open(path).mode == "RGB"


def test_decide_source():
    auto = {"status": "auto_accepted", "breed": "Gir", "confidence": 0.9}
    review = {"status": "needs_review", "shortlist": [{"breed": "Gir", "confidence": 0.4}, {"breed": "Sahiwal", "confidence": 0.3}]}
    assert ui.decide_source(auto, "Gir") == "auto_accepted"
    assert ui.decide_source(auto, "Sahiwal") == "worker_overrode"
    assert ui.decide_source(review, "Gir") == "worker_confirmed"
    assert ui.decide_source(review, "Sahiwal") == "worker_overrode"
    assert ui.decide_source(review, "Kankrej", manual=True) == "manual_entry"


def test_confirm_prediction_requires_animal_id_and_breed():
    res = {"status": "auto_accepted", "breed": "Gir", "confidence": 0.9}
    for kwargs in ({"final_label": "Gir", "animal_id": " "}, {"final_label": "", "animal_id": "A1"}):
        try:
            ui.confirm_prediction("x.jpg", res, {"Gir": 0.9}, worker_id="W", **kwargs)
        except ValueError:
            continue
        raise AssertionError("missing input was accepted")


def test_confirm_prediction_writes_both_records():
    before_fb, before_bpa = len(ui.read_feedback_df()), len(ui.read_bpa_df())
    res = {"status": "auto_accepted", "breed": "Gir", "confidence": 0.9}
    out = ui.confirm_prediction("x.jpg", res, {"Gir": 0.9}, "Gir", "TEST-UI-1", "TEST_WORKER", village="Test")
    assert out["source"] == "auto_accepted" and out["bpa_response"]["synced"] is True
    assert len(ui.read_feedback_df()) == before_fb + 1
    assert len(ui.read_bpa_df()) == before_bpa + 1


def test_app_loads_without_errors():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["Identify breed", "Records", "About"]


def test_app_upload_and_confirm_flow():
    """Simulates an upload (AppTest cannot drive file_uploader, so it is replaced by a stub) and a click on Confirm."""
    from streamlit.testing.v1 import AppTest
    code = f'''
import streamlit as st
class _Up:
    def __init__(self, b): self._b = b
    def getvalue(self): return self._b
st.file_uploader = lambda *a, **k: _Up(open({_sample_image_path()!r}, "rb").read())
__file__ = {os.path.join(ROOT, "app.py")!r}
exec(compile(open(__file__).read(), __file__, "exec"))
'''
    at = AppTest.from_string(code, default_timeout=180).run()
    assert not at.exception, at.exception
    at.text_input(key="animal_id").set_value("TEST-APP-ANIMAL")
    at.run()
    labels = [b.label for b in at.button]
    assert "Confirm and sync to BPA" in labels, labels
    before = len(ui.read_bpa_df())
    at.button[labels.index("Confirm and sync to BPA")].click().run()
    assert not at.exception, at.exception
    assert len(ui.read_bpa_df()) == before + 1
    assert any("Saved" in s.value for s in at.success)


def _app_code(extra_patch=""):
    return f'''
import sys
sys.path.insert(0, {os.path.join(ROOT, "src")!r})
import streamlit as st
class _Up:
    def __init__(self, b): self._b = b
    def getvalue(self): return self._b
st.file_uploader = lambda *a, **k: _Up(open({_sample_image_path()!r}, "rb").read())
{extra_patch}
__file__ = {os.path.join(ROOT, "app.py")!r}
exec(compile(open(__file__).read(), __file__, "exec"))
'''


def test_app_auto_accept_branch():
    from streamlit.testing.v1 import AppTest
    import inference
    original_predict = inference.BreedClassifier.predict      # the stub below patches the class, so restore it afterwards
    patch = (
        "import inference\n"
        "inference.BreedClassifier.predict = lambda self, p, threshold=None, top_k=None: "
        "{'status': 'auto_accepted', 'breed': 'Gir', 'confidence': 0.93, 'message': 'x'}\n"
    )
    try:
        at = AppTest.from_string(_app_code(patch), default_timeout=180).run()
        assert not at.exception, at.exception
        assert any("auto-accepted" in s.value for s in at.success), [s.value for s in at.success]
        at.text_input(key="animal_id").set_value("TEST-AUTO-1")
        at.run()
        at.button[[b.label for b in at.button].index("Confirm and sync to BPA")].click().run()
        assert not at.exception, at.exception
        assert any("auto_accepted" in s.value for s in at.success)
    finally:
        inference.BreedClassifier.predict = original_predict


def test_app_manual_entry_branch():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string(_app_code(), default_timeout=180).run()   # real model -> shortlist (low confidence)
    assert not at.exception, at.exception
    radio = [r for r in at.radio if r.label == "Likely breeds"][0]
    radio.set_value(radio.options[-1]).run()                           # "None of these - enter manually"
    assert not at.exception, at.exception
    at.text_input(key="animal_id").set_value("TEST-MANUAL-1")
    [t for t in at.text_input if t.label == "...or type the breed name"][0].set_value("Kankrej")
    at.run()
    at.button[[b.label for b in at.button].index("Confirm and sync to BPA")].click().run()
    assert not at.exception, at.exception
    assert any("manual_entry" in s.value for s in at.success), [s.value for s in at.success]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} UI tests passed")
    sys.exit(0 if passed == len(tests) else 1)
