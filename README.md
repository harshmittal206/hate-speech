# Hate Speech & Offensive Language Detector (Improved)

A Streamlit application that classifies English social-media text into:
- Hate Speech
- Offensive Language
- No Hate or Offensive Language

## Improvements in this version

- Replaced the previous bag-of-words / Decision Tree approach with balanced Logistic Regression.
- Uses combined word TF-IDF (unigrams + bigrams) and character TF-IDF features.
- Keeps negation words such as `not`, `no`, and `never`; the old preprocessing removed these and could reverse meaning.
- Normalizes URLs, mentions, hashtags and whitespace without deleting profanity or negation.
- Shows held-out evaluation metrics and per-class precision, recall and F1.
- Displays the top predicted category and all three model-estimated probabilities.
- Uses project-relative paths, so it can be launched from another working directory.

## Files

- `app.py` — Streamlit user interface.
- `train_model.py` — data preprocessing, model training and evaluation.
- `labeled_data.csv` — Davidson et al. labeled tweet dataset.
- `model.pkl`, `vectorizer.pkl` — trained model and feature transformer.
- `metrics.json` — held-out test metrics from the latest training run.
- `Hate Speech.ipynb` — original notebook retained for reference.

## Training results

The latest run used a stratified 80/20 train/test split (random state 42):

| Metric | Result |
|---|---:|
| Accuracy | 88.9% |
| Weighted F1 | 0.891 |
| Macro F1 | 0.735 |
| Hate Speech F1 | 0.41 |
| Hate Speech recall | 0.43 |

**Important:** Overall accuracy is not the whole story. Hate Speech is the minority class and its F1/recall remain substantially lower than the other classes. The model is improved compared with the previous Decision Tree setup, but it is not reliable enough for automatic moderation or consequential decisions. Evaluate on newer, diverse, independently collected data before deployment.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Retrain the model

```bash
python train_model.py
```

Run the command from this folder, or run it by absolute path; the script locates `labeled_data.csv` relative to itself. Retraining overwrites `model.pkl`, `vectorizer.pkl` and `metrics.json`.

## Notes about probabilities

The percentages in the app are the classifier's estimated class probabilities. They are not guaranteed accuracy scores, and a desired 30–40% score should not be forced by changing inputs or manipulating the display. The probabilities depend on the text and the fitted model. If well-calibrated probabilities are required, calibration should be evaluated separately on a validation set.

## Limitations and responsible use

- The source dataset consists of English-language tweets and reflects the time, platform and annotation choices under which it was collected.
- The model may misclassify reclaimed language, sarcasm, quotations, dialects, counterspeech and context-dependent statements.
- The three labels are not a complete taxonomy of threats or violent rhetoric.
- Do not use the model as the sole basis for moderation, reporting, employment, educational or safety decisions. Human review is necessary.
