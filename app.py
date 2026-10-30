import json
import pickle
import re
import html
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"

LABEL_STYLE = {
    "Hate Speech": {"emoji": "🚫", "css": "result-hate"},
    "Offensive Language": {"emoji": "⚠️", "css": "result-offensive"},
    "No Hate or Offensive Language": {"emoji": "✅", "css": "result-clean"},
}
LABEL_ORDER = [
    "Hate Speech",
    "Offensive Language",
    "No Hate or Offensive Language",
]

st.set_page_config(
    page_title="Hate Speech & Offensive Language Detector",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
.block-container {max-width: 1150px; padding-top: 2rem; padding-bottom: 2rem;}
.big-title {font-size: 2rem; font-weight: 750; margin-bottom: .25rem;}
.subtitle {color: #9aa0ad; font-size: 1rem; margin-bottom: 1.25rem;}
.result-card {padding: 1.25rem 1.5rem; border-radius: 14px; margin: 1rem 0;}
.result-hate {background: linear-gradient(135deg,#3a1010,#5c1414); border:1px solid #ff4d4d;}
.result-offensive {background: linear-gradient(135deg,#3a2c10,#5c4514); border:1px solid #ffb84d;}
.result-clean {background: linear-gradient(135deg,#0f2b1d,#14432a); border:1px solid #2ecc71;}
.result-title {font-size:1.35rem;font-weight:700;}
.token {display:inline-block;padding:4px 10px;border-radius:20px;margin:3px;
background:rgba(128,128,128,.16);border:1px solid rgba(128,128,128,.3);font-size:.85rem;}
.small-note {font-size:.85rem;color:#9aa0ad;}
</style>
""", unsafe_allow_html=True)

def clean_data(text):
    """Same normalization used by the training vectorizer; keeps negation."""
    text = html.unescape(str(text)).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"\brt\b", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()

@st.cache_resource
def load_artifacts():
    missing = [p.name for p in (MODEL_PATH, VECTORIZER_PATH) if not p.exists()]
    if missing:
        st.error("Missing model files: " + ", ".join(missing) +
                 ". Run `python train_model.py` in this project folder first.")
        st.stop()
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    with VECTORIZER_PATH.open("rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_artifacts()

st.markdown('<div class="big-title">🛡️ Hate Speech & Offensive Language Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Three-class text classification using balanced Logistic Regression and word + character TF-IDF features.</div>', unsafe_allow_html=True)
st.info("Research/demo tool only. The training data is English-language social-media text and may contain bias. Do not use this prediction alone to make moderation, disciplinary, or safety decisions.")

examples = {
    "— pick an example —": "",
    "Neutral": "The meeting starts at ten and the report is ready.",
    "Offensive language": "That was a stupid and disgusting thing to say.",
    "Hate speech / violent rhetoric": "People from that community should be attacked and driven away.",
    "Ambiguous / context-dependent": "I am furious and someone needs to teach him a lesson.",
}
choice = st.selectbox("Try an example", list(examples))
message = st.text_area(
    "Text to analyze",
    value=examples[choice],
    height=130,
    placeholder="Type or paste a sentence or social-media comment..."
)

col_a, col_b = st.columns([1, 1])
with col_a:
    analyze = st.button("🔍 Analyze Text", use_container_width=True, type="primary")
with col_b:
    clear = st.button("🗑️ Clear", use_container_width=True)

if clear:
    st.session_state.pop("last_result", None)
    st.rerun()

def predict(text):
    features = vectorizer.transform([text])
    predicted = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    probability_map = dict(zip(model.classes_, probabilities))
    return predicted, probability_map

def make_chart(probability_map):
    labels = [label for label in LABEL_ORDER if label in probability_map]
    values = [probability_map[label] * 100 for label in labels]
    color_map = {
        "Hate Speech": "#ff4d4d",
        "Offensive Language": "#ffb84d",
        "No Hate or Offensive Language": "#2ecc71",
    }
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=[color_map[x] for x in labels],
        text=[f"{v:.1f}%" for v in values],
        textposition="auto",
    ))
    fig.update_layout(
        xaxis_title="Model-estimated probability (%)",
        xaxis=dict(range=[0, 100]),
        yaxis=dict(autorange="reversed"),
        height=300,
        margin=dict(l=10, r=20, t=15, b=35),
    )
    return fig

if analyze:
    if not message.strip():
        st.warning("Please enter some text to analyze.")
    else:
        predicted, probability_map = predict(message.strip())
        st.session_state.last_result = {
            "text": message.strip(),
            "prediction": predicted,
            "probabilities": probability_map,
        }

if "last_result" in st.session_state:
    result = st.session_state.last_result
    predicted = result["prediction"]
    probability_map = result["probabilities"]
    style = LABEL_STYLE[predicted]
    top_probability = probability_map[predicted] * 100

    st.markdown(
        f'<div class="result-card {style["css"]}">'
        f'<div class="result-title">{style["emoji"]} {predicted}</div>'
        f'<div>Model-estimated probability: <b>{top_probability:.1f}%</b></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(make_chart(probability_map), use_container_width=True)
    with right:
        st.subheader("Text statistics")
        st.write(f"**Word count:** {len(result['text'].split())}")
        st.write(f"**Character count:** {len(result['text'])}")
        st.subheader("Tokens in the input")
        tokens = re.findall(r"\b[\w']+\b", clean_data(result["text"]))
        if tokens:
            st.markdown("".join(f'<span class="token">{html.escape(t)}</span>' for t in tokens[:20]), unsafe_allow_html=True)
        st.caption("Tokens are shown for inspection; the model also uses word and character n-gram features.")

    st.caption("Probabilities are model estimates, not a measure of factual certainty or a guarantee that the text belongs to a category.")

if METRICS_PATH.exists():
    with METRICS_PATH.open(encoding="utf-8") as f:
        metrics = json.load(f)
    with st.expander("Model evaluation (held-out test set)"):
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        m2.metric("Weighted F1", f"{metrics['weighted_f1']:.3f}")
        m3.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
        st.write("Per-class precision, recall and F1:")
        report_rows = []
        for label in LABEL_ORDER:
            row = metrics["class_report"].get(label, {})
            report_rows.append({
                "Class": label,
                "Precision": round(row.get("precision", 0), 3),
                "Recall": round(row.get("recall", 0), 3),
                "F1-score": round(row.get("f1-score", 0), 3),
                "Support": int(row.get("support", 0)),
            })
        st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

st.divider()
st.caption("Model: balanced Logistic Regression · Features: word TF-IDF (unigrams/bigrams) + character TF-IDF · Dataset: Davidson et al. labeled tweet dataset. For educational and research use.")
