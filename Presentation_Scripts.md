# BharatBreed AI — Talking Scripts

Three scripts: a 5-minute explanation, a 2-minute version, and a live-demo walkthrough to run a mentor through the actual code/terminal.

---

## A. The 5-minute explanation

**[0:00–0:45] The problem**
"India has the largest livestock population in the world, and every animal's record in the government's Bharat Pashudhan App has one field that's still entirely manual: breed. A field worker looks at a cow or buffalo and types in a breed name from memory. India has over 50 recognised cattle breeds and a dozen-plus buffalo breeds — many of them visually close, and made harder by widespread cross-breeding. That one manual field feeds into genetic-improvement programmes, disease-control targeting, insurance, and subsidy schemes. So errors there don't just sit in a database — they quietly propagate into real policy and financial decisions."

**[0:45–1:30] Why this isn't already solved**
"This isn't a new idea — there's real academic work training CNNs to classify cattle breeds from images, and this is literally an active government problem statement, SIH25004. But almost all of that work stops at 'we trained a model and it got some accuracy on a test set.' It doesn't answer the actual deployment questions: does it run on the ₹8,000 Android phone a field worker already has? What happens when the model isn't sure? And how does the prediction actually get back into the government database it's supposed to fix? We built the system around those three questions, not just the classifier."

**[1:30–3:00] What we actually built**
"Our pipeline has four stages. First, image capture and pre-processing — resize, denoise, and augmentation to simulate the messy lighting and angles of real field photography, not studio conditions. Second, a CNN classifier — we use transfer learning on EfficientNetB0, which gives strong accuracy for a comparatively small, India-specific dataset without needing to train a huge network from scratch. Third — and this is the part most projects skip — a confidence-aware decision layer. If the model is confident, above a threshold, we auto-accept the breed label. If it's not confident, instead of forcing a guess, we show the field worker a ranked shortlist of the top-3 most likely breeds and let them confirm or correct it. A human stays in the loop exactly where the model is weakest, not everywhere. Fourth, that confirmed label gets synced — through a defined API contract — into the Bharat Pashudhan App's data pipeline, so the correction actually lands. And every single confirmation or correction gets logged, so the model can be periodically retrained on real, field-verified Indian data — which means it should keep improving the more it's actually used."

**[3:00–4:00] Deployment-mindedness**
"Because the target device is a mid-range Android phone with unreliable connectivity, we export the trained model to TensorFlow Lite — both a full-precision version and an INT8-quantized version, which is roughly 3 to 4 times smaller with a small accuracy trade-off. Inference runs entirely on-device, so a field worker isn't blocked by a dead network at the exact moment they're standing in front of an animal. The only thing that needs connectivity is the final sync step, and that can queue and retry."

**[4:00–4:45] Honesty about scope, and what's real vs mocked**
"To be direct about scope: we don't have live access to the actual Bharat Pashudhan App, so our integration is a mocked API with the exact request and response shape a real integration would use — swapping it for a real authenticated call is a drop-in replacement, not a redesign. And the dataset in this repository is a small placeholder set we generated ourselves, specifically so the entire pipeline — training, TFLite export, inference, feedback logging — can be demonstrated end-to-end, live, right now, without bundling someone else's licensed dataset. With a real Indian-breed dataset and normal internet access for pretrained weights, the exact same code trains a real classifier."

**[4:45–5:00] Close**
"So in one sentence: this project doesn't just answer 'can a CNN tell cattle breeds apart' — several people have already shown that it can. It answers 'what does it take to actually get that working, reliably, in a field worker's hand, feeding back into India's national livestock database' — and that surrounding system is what we built."

---

## B. The 2-minute version

"India's national livestock database has one manual, error-prone field: breed. Field workers hand-type it from memory across 50-plus cattle breeds and a dozen-plus buffalo breeds, and that error rate quietly corrupts genetic-improvement, disease-control, and subsidy decisions downstream. This is a real government problem statement — SIH25004.

Existing academic work shows CNNs *can* classify cattle breeds — but almost all of it stops at a trained model and an accuracy number on a curated dataset. It never answers: does it run on a field worker's actual phone? What happens when the model isn't sure? How does the prediction reach the database it's meant to fix?

We built exactly that missing system. A transfer-learning CNN — EfficientNetB0 — classifies the breed from a photo. If it's confident, the label auto-accepts. If it's not, instead of guessing, it shows the field worker a top-3 shortlist to confirm or correct — a human stays in the loop exactly where the model is weakest. The trained model is exported to TensorFlow Lite, quantized down to a few megabytes, so it runs on-device on a mid-range Android phone with no connectivity dependency at capture time. Every confirmed or corrected label is logged and feeds a periodic retraining cycle, so the model gets better on real Indian field conditions the more it's actually used. And confirmed labels sync through a defined API contract into the Bharat Pashudhan App's data pipeline — mocked for now, since we don't have government system access at this scope, but built as a drop-in-ready contract.

In short: the classifier is the easy 30% of this problem. We built the harder 70% — the confidence handling, the mobile deployment, the feedback loop, and the integration path — because that's what actually makes this usable in a real field, not just a research notebook."

---

## C. Live demo walkthrough (talking a mentor through the actual project)

Use this when you have a terminal open and want to run the mentor through the real thing, not just slides.

**Step 0 — Before you start**
Have a terminal open in the project root with `python run_pipeline.py` ready to run, and this README open in a second window/tab. Say:

> "I'll run the whole pipeline live — data, training, mobile export, prediction, and the feedback loop — so you can see it's not just slides."

**Step 1 — Show the project structure (30 sec)**
Open `README.md`'s "Project structure" section, or run `tree src/` / `ls src/`. Say:

> "Each stage of the block diagram in the synopsis maps to exactly one file: `data_prep.py` for pre-processing, `model.py` and `train.py` for the CNN, `convert_tflite.py` for mobile export, `inference.py` for the confidence decision layer, `feedback_loop.py` and `bpa_sync_stub.py` for the loop back into the database. Nothing here is decorative — every box in the architecture diagram is real code."

**Step 2 — Run the pipeline (let it run, narrate while it trains)**
```bash
python run_pipeline.py
```
While training runs (~1 minute), say:

> "This is training a real transfer-learning model right now — EfficientNetB0 backbone, frozen first, then fine-tuned. In this environment, I'm using a small placeholder dataset I generated myself, so we can watch the whole pipeline run live without needing a multi-gigabyte dataset download. With a real breed dataset and normal internet access, this exact code trains a genuine classifier — the only thing that changes is what's inside `data/raw/`."

**Step 3 — Point out the TFLite export (when Step 3/6 prints)**
> "This is the deployment step — converting the trained model into TensorFlow Lite, and creating a quantized version that's about a third of the size. This is the file that would actually ship inside the Android app."

**Step 4 — Point out the confidence decision (when Step 4/6 prints)**
Show the printed JSON result. Say:

> "This is the core design decision of the project. The model doesn't just output one breed name — it outputs a status. If confidence clears the threshold, it auto-accepts. If not, like you can see here, it returns a ranked shortlist instead of forcing a wrong single answer, so a human confirms it."

**Step 5 — Point out the feedback log and BPA sync (Steps 5–6)**
Open `logs/feedback_log.jsonl` and `logs/bpa_sync_mock.jsonl` in a text editor. Say:

> "Every confirmed prediction gets logged here — this is what future retraining reads from. And this second file is the mocked sync into the Bharat Pashudhan App — same shape a real integration call would use, just written locally since we don't have government API access at this scope."

**Step 6 — Run the sanity tests (optional, shows engineering discipline)**
```bash
python -m pytest tests/ -v
```
> "And these are sanity checks — that the dataset, label map, TFLite model, decision layer, and both logs all actually work end-to-end. This is what we run before every demo so we're not debugging live in front of you."

**Step 7 — Close**
> "So that's the full loop: capture, classify, decide, sync, and feed back into retraining — running end-to-end, live, from one command."
