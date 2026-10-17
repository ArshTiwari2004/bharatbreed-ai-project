"""
app.py  -  BharatBreed AI front end (Streamlit)
---------------------------------
Run:
    streamlit run app.py

A Field Level Worker can upload a photo or use the phone camera, see the
predicted breed with its confidence, confirm / choose another breed / enter one
manually, and sync the verified record to BPA (mocked). Every decision is logged
for retraining. It uses exactly the same code as the command-line pipeline.
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import pandas as pd
import streamlit as st

import config
import ui_helpers as ui

st.set_page_config(page_title="BharatBreed AI", page_icon="\U0001F404", layout="centered")


# ---------------------------------------------------------------- helpers
def model_ready():
    return os.path.exists(config.TFLITE_MODEL_PATH) and os.path.exists(config.LABEL_MAP_PATH)


@st.cache_resource(show_spinner="Loading model...")
def load_classifier():
    from inference import BreedClassifier   # imported lazily: TensorFlow is slow to import
    return BreedClassifier()


def pct(x):
    return f"{x:.0%}"


# ---------------------------------------------------------------- header + sidebar
st.title("\U0001F404 BharatBreed AI")
st.caption("AI-assisted breed identification for Indian cattle and buffaloes - "
           "the model suggests, the field worker decides.")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Auto-accept confidence threshold", 0.30, 0.99, float(config.CONFIDENCE_THRESHOLD), 0.01,
                          help="At or above this confidence the breed is auto-accepted; below it a shortlist is shown.")
    top_k = st.slider("Shortlist size (top-k)", 2, 5, int(config.TOP_K))
    worker_id = st.text_input("Field worker ID", value="FLW_DEMO", key="worker_id")
    st.divider()
    st.caption("Records are saved to the local `logs/` folder. The BPA sync is a mocked endpoint "
               "(see `src/bpa_sync_stub.py`).")

# ---------------------------------------------------------------- model availability
if not model_ready():
    st.warning("No trained model was found yet. Run the pipeline once to generate a demo dataset, "
               "train the model and export it for mobile (about 1-3 minutes on a laptop).")
    if st.button("Prepare demo model now"):
        with st.spinner("Running run_pipeline.py ... please wait"):
            proc = subprocess.run([sys.executable, "run_pipeline.py"], cwd=ROOT, capture_output=True, text=True)
        if proc.returncode == 0 and model_ready():
            st.success("Model is ready.")
            st.rerun()
        else:
            st.error("The pipeline failed. Last lines of output:")
            st.code((proc.stdout + "\n" + proc.stderr)[-3000:])
    st.stop()

clf = load_classifier()
labels = [clf.labels[i] for i in sorted(clf.labels)]

tab_id, tab_rec, tab_about = st.tabs(["Identify breed", "Records", "About"])

# ---------------------------------------------------------------- tab 1: identify
with tab_id:
    c1, c2 = st.columns(2)
    animal_id = c1.text_input("Animal ID / ear-tag no.", placeholder="e.g. UP-2024-00981234", key="animal_id")
    village = c2.text_input("Village (optional)", key="village")

    mode = st.radio("Photo source", ["Upload a photo", "Use the camera"], horizontal=True)
    if mode == "Upload a photo":
        shot = st.file_uploader("Photo of the animal (JPG / PNG)", type=["jpg", "jpeg", "png"])
    else:
        shot = st.camera_input("Take a photo of the animal")

    if shot is None:
        st.info("Add a clear, side-on photo in good light, with the whole animal in the frame.")
    else:
        data = shot.getvalue()
        try:
            quality = ui.quality_report(data)
            path, digest = ui.save_upload(data)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        st.image(path, caption="Photo to analyse")
        for w in quality["warnings"]:
            st.warning(w)

        # ---- prediction (decision layer = same code as the CLI)
        result = clf.predict(path, threshold=threshold, top_k=top_k)
        probs = clf.predict_proba(path)
        prob_map = {clf.labels[i]: float(p) for i, p in enumerate(probs)}

        manual = False
        if result["status"] == "auto_accepted":
            st.success(f"**{result['breed'].replace('_', ' ')}** - {pct(result['confidence'])} confidence "
                       f"(auto-accepted, threshold {pct(threshold)})")
            final_label = result["breed"]
        else:
            st.warning(f"Not sure - confidence is below {pct(threshold)}. Please choose the correct breed.")
            options = [f"{s['breed'].replace('_', ' ')}  -  {pct(s['confidence'])}" for s in result["shortlist"]]
            options.append("None of these - enter manually")
            choice = st.radio("Likely breeds", options)
            if choice == options[-1]:
                manual = True
                picked = st.selectbox("Pick from all known breeds", ["(type a different breed below)"] + labels)
                typed = st.text_input("...or type the breed name")
                final_label = typed.strip() if typed.strip() else ("" if picked.startswith("(") else picked)
            else:
                final_label = result["shortlist"][options.index(choice)]["breed"]
            st.caption("Tip: a better photo (closer, brighter, side-on) often raises confidence - you can retake it.")

        with st.expander("Confidence for every breed"):
            series = pd.Series(prob_map).sort_values(ascending=False)
            st.bar_chart(series)

        # ---- confirm + sync
        done = st.session_state.setdefault("confirmed", {})
        key = f"{digest}:{(animal_id or '').strip()}"
        if key in done:
            st.success(f"Already saved for this animal: **{done[key]['final']}** (source: {done[key]['source']}).")
        elif st.button("Confirm and sync to BPA", type="primary"):
            try:
                out = ui.confirm_prediction(path, result, prob_map, final_label, animal_id, worker_id,
                                            village=village, manual=manual)
            except ValueError as e:
                st.error(str(e))
            else:
                done[key] = {"final": final_label, "source": out["source"]}
                st.success(f"Saved **{final_label.replace('_', ' ')}** for animal `{animal_id.strip()}` "
                           f"(source: {out['source']}). BPA response: {out['bpa_response']['status']}, "
                           f"synced = {out['bpa_response']['synced']}.")
                st.balloons()

# ---------------------------------------------------------------- tab 2: records
with tab_rec:
    summary = ui.feedback_summary()
    m1, m2 = st.columns(2)
    m1.metric("Confirmed records", summary["total"])
    rate = summary["agreement_rate"]
    m2.metric("Worker agreed with model", "-" if rate is None else pct(rate),
              help="Share of records where the final label equals the model's top-1 guess.")

    st.subheader("Feedback log (used for retraining)")
    fb = ui.read_feedback_df()
    if fb.empty:
        st.caption("No records yet. Confirm a prediction in the first tab.")
    else:
        st.dataframe(fb.iloc[::-1], hide_index=True)
        st.download_button("Download feedback as CSV", fb.to_csv(index=False).encode("utf-8"),
                           file_name="bharatbreed_feedback.csv", mime="text/csv")

    st.subheader("Records sent to BPA (mock)")
    bpa = ui.read_bpa_df()
    if bpa.empty:
        st.caption("Nothing synced yet.")
    else:
        st.dataframe(bpa.iloc[::-1], hide_index=True)

# ---------------------------------------------------------------- tab 3: about
with tab_about:
    st.subheader("How it works")
    st.markdown(
        "1. **Capture** - photo from the camera or gallery, with a quick quality check.\n"
        "2. **Classify** - a transfer-learning CNN (EfficientNetB0 / ResNet50) runs as a TensorFlow Lite model.\n"
        "3. **Decide** - confident predictions are auto-accepted; otherwise a top-k shortlist is shown.\n"
        "4. **Confirm** - the worker confirms or corrects; the final label goes to BPA (mocked).\n"
        "5. **Learn** - every decision is logged and can be used to retrain the model."
    )
    st.subheader("Model in use")
    info = {"Classes": ", ".join(l.replace("_", " ") for l in labels),
            "TFLite model": f"{os.path.getsize(config.TFLITE_MODEL_PATH) / 1024:,.0f} KB"}
    if os.path.exists(config.TRAINING_HISTORY_PATH):
        with open(config.TRAINING_HISTORY_PATH) as f:
            hist = json.load(f)
        info["Backbone"] = hist.get("backbone", "-")
        info["Test accuracy (last training run)"] = f"{hist.get('test_accuracy', 0):.1%}"
    for k, v in info.items():
        st.write(f"**{k}:** {v}")
    st.info("If the model was trained on the built-in **placeholder dataset** (the default demo), its predictions "
            "only demonstrate the workflow. For real use, put real breed photos in `data/raw/<breed>/` and re-train.")
