"""
=============================================================
  CyberShield — Upgraded Streamlit App
  Requires .pkl files from train_model.py

  New in this version:
    - SBERT all-mpnet-base-v2 (768-dim, higher accuracy)
    - Implicit / sarcasm detection flag
    - Low-confidence borderline warning
    - Multi-model consensus with agreement %
    - Radar chart across all 6 classes
    - Severity index (0-10, original formula)
    - SHAP explainability
    - Session analytics dashboard

  Run: streamlit run app.py
=============================================================
"""

import streamlit as st
import joblib
import re
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from collections import Counter

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="CyberShield — Cyberbullying Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .main { background-color: #0e1117; }

    .verdict-safe {
        background: linear-gradient(135deg, #0d2b1a, #0f3d24);
        border: 1px solid #22c55e; border-radius: 12px;
        padding: 20px 28px; color: #22c55e;
        font-size: 1.4rem; font-weight: 700;
        text-align: center; letter-spacing: 0.05em;
    }
    .verdict-danger {
        background: linear-gradient(135deg, #2b0d0d, #3d0f0f);
        border: 1px solid #ef4444; border-radius: 12px;
        padding: 20px 28px; color: #ef4444;
        font-size: 1.4rem; font-weight: 700;
        text-align: center; letter-spacing: 0.05em;
    }
    .verdict-borderline {
        background: linear-gradient(135deg, #1a1500, #2b2000);
        border: 1px solid #eab308; border-radius: 12px;
        padding: 20px 28px; color: #eab308;
        font-size: 1.2rem; font-weight: 700;
        text-align: center; letter-spacing: 0.04em;
    }
    .flag-box {
        background: #1a1400;
        border: 1px solid #ca8a04;
        border-radius: 10px;
        padding: 12px 16px;
        color: #fbbf24;
        font-size: 0.9rem;
        margin: 10px 0;
    }
    .model-card {
        background: #1a1d26; border: 1px solid #2d3148;
        border-radius: 10px; padding: 14px 18px; margin: 6px 0;
        font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
    }
    .section-title {
        font-size: 1rem; font-weight: 600; color: #94a3b8;
        text-transform: uppercase; letter-spacing: 0.1em;
        margin: 20px 0 10px 0;
        border-bottom: 1px solid #2d3148; padding-bottom: 6px;
    }
    .insight-box {
        background: #13151f; border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0; padding: 12px 16px;
        font-size: 0.9rem; color: #c4c9e2; margin: 10px 0;
    }
    .stat-card {
        background: #1a1d26; border: 1px solid #2d3148;
        border-radius: 10px; padding: 16px; text-align: center;
    }
    .stat-number { font-size: 1.8rem; font-weight: 700; color: #6366f1; }
    .stat-label  { font-size: 0.78rem; color: #64748b;
                   text-transform: uppercase; letter-spacing: 0.08em; }
    .source-badge {
        display: inline-block; padding: 2px 10px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; margin: 2px;
    }
    /* Animated verdict banners */
    .verdict-safe, .verdict-danger, .verdict-borderline {
        animation: fadeSlideIn 0.4s ease;
    }
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(-8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    /* Hover effect on model cards */
    .model-card:hover {
        border-color: #6366f1;
        transition: border-color 0.2s ease;
    }
    /* Input area styling */
    .stTextArea textarea {
        background: #13151f !important;
        border: 1px solid #2d3148 !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.95rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
    }
    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button[kind="primary"]:hover {
        opacity: 0.85 !important;
    }
    /* Category pill badges */
    .category-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-top: 6px;
    }
    /* History item */
    .history-item {
        background: #13151f;
        border: 1px solid #2d3148;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.78rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# ── NLTK ─────────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet',   quiet=True)
    nltk.download('omw-1.4',   quiet=True)

download_nltk()

# ── Load SBERT (upgraded to mpnet) ──────────────────────────
@st.cache_resource
def load_sbert():
    # all-mpnet-base-v2: 768-dim, highest accuracy SBERT model on CPU
    return SentenceTransformer("all-mpnet-base-v2")

# ── Load Models ──────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    try:
        le  = joblib.load('label_encoder.pkl')
        xgb = joblib.load('model_xgboost.pkl')
        svm = joblib.load('model_svm.pkl')
        lr  = joblib.load('model_lr.pkl')
        return le, xgb, svm, lr
    except FileNotFoundError as e:
        st.error(f"Model file missing: {e}. Please run train_model.py first.")
        return None, None, None, None

@st.cache_resource
def load_berkeley_models():
    """Load Berkeley severity regressor and sarcasm classifier.
    Returns None gracefully if not yet trained."""
    try:
        sev_model    = joblib.load('severity_model.pkl')
        sev_scaler   = joblib.load('severity_scaler.pkl')
        sar_model    = joblib.load('sarcasm_model.pkl')
        return sev_model, sev_scaler, sar_model
    except FileNotFoundError:
        return None, None, None

# ── Preprocessing ────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

# Minimal cleaning — SBERT needs full sentence context.
# Stopword removal destroys phrases like "your age", "go back",
# "too old" which are critical classification signals.
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if len(t) > 1]
    return " ".join(tokens).strip()

# ── Implicit / Sarcasm Detection ─────────────────────────────
# Lightweight rule-based + pattern detector
# Flags text that may contain hidden or sarcastic toxicity
# even if the main classifier marks it safe

SARCASM_PATTERNS = [
    r'\bso\s+(smart|brave|strong|clever|great|talented)\b.*(like you|for you|someone like)',
    r'\b(oh wow|oh great|wow amazing|how wonderful|how nice)\b',
    r'\bsure(ly)?\b.*(they|those people|your kind)',
    r'\bjust\s+(stay|go back|belong)',
    r'\b(protect|save|defend)\b.*(our|this)\s+(neighborhood|country|community|culture)',
    r'\bthose\s+people\b',
    r'\byour\s+kind\b',
    r'\bpeople\s+like\s+you\b',
    r'\bgo\s+back\s+(to\s+)?(where|your)',
    r'\bnot\s+(racist|sexist|bigot)\s+but\b',
    r'\bi.m\s+not\s+(saying|implying).*(but|however)',
    r'\bwe\s+(don.t\s+)?(need|want)\s+(your|their|those)\b',
]

IMPLICIT_PATTERNS = [
    r'\bthese\s+people\b',
    r'\btaking\s+over\b',
    r'\binvading\b',
    r'\bour\s+(jobs|women|country|culture|values)\b',
    r'\b(they|them)\s+(don.t|never|always|all)\b',
    r'\bnaturally\s+(inferior|criminal|violent|lazy)\b',
    r'\b(crime|violence|poverty)\s+(rate|problem)\b.*(they|their|those)',
]

def detect_implicit_sarcasm(text: str) -> tuple:
    """
    Returns (is_flagged: bool, flag_type: str, matched_pattern: str)
    Operates on raw text (before cleaning) to preserve sarcasm markers.
    """
    text_lower = text.lower()
    for p in SARCASM_PATTERNS:
        if re.search(p, text_lower):
            return True, "Possible Sarcasm / Coded Language", p
    for p in IMPLICIT_PATTERNS:
        if re.search(p, text_lower):
            return True, "Possible Implicit Bias / Dog-Whistle", p
    return False, "", ""

# ── Severity Engine ──────────────────────────────────────────
SEVERITY_MAP = {
    'not_cyberbullying': 0,
    'age':               5,
    'gender':            6,
    'ethnicity':         8,
    'religion':          8,
    'other_cyberbullying': 4,
}
SEVERITY_LABELS = {
    0:  ("None",     "#22c55e"),
    1:  ("Low",      "#84cc16"),
    2:  ("Low",      "#84cc16"),
    3:  ("Moderate", "#eab308"),
    4:  ("Moderate", "#eab308"),
    5:  ("High",     "#f97316"),
    6:  ("High",     "#f97316"),
    7:  ("Critical", "#ef4444"),
    8:  ("Critical", "#ef4444"),
    9:  ("Critical", "#ef4444"),
    10: ("Critical", "#ef4444"),
}

def compute_severity(label: str, confidence: float,
                     is_flagged: bool = False,
                     embedding=None) -> tuple:
    """
    Returns (severity_score: int, source: str)
    Uses Berkeley-trained Ridge regressor when available,
    falls back to formula-based scoring otherwise.
    """
    if berkeley_loaded and embedding is not None:
        # Use Berkeley-trained severity regressor
        raw = float(sev_model.predict(embedding)[0])
        raw = np.clip(raw, 0, 10)
        # Boost if cyberbullying detected (not safe)
        if label != 'not_cyberbullying':
            raw = max(raw, SEVERITY_MAP.get(label, 3))
        flag_mod = 1 if is_flagged and label == 'not_cyberbullying' else 0
        return int(np.clip(round(raw + flag_mod), 0, 10)), "Berkeley ML Model"
    else:
        # Fallback formula
        base     = SEVERITY_MAP.get(label, 3)
        modifier = round((confidence - 0.5) * 4) if confidence > 0.5 else 0
        flag_mod = 1 if is_flagged and label == 'not_cyberbullying' else 0
        return min(10, max(0, base + modifier + flag_mod)), "Formula-based"

def detect_sarcasm_ml(embedding) -> tuple:
    """
    ML-based sarcasm/implicit detector using Berkeley-trained classifier.
    Returns (is_flagged: bool, confidence: float)
    """
    if berkeley_loaded and embedding is not None:
        prob = sar_model.predict_proba(embedding)[0]
        is_implicit = bool(sar_model.predict(embedding)[0])
        return is_implicit, float(prob[1])
    return False, 0.0

# ── Charts ───────────────────────────────────────────────────
def make_radar_chart(probs, class_names):
    labels = [c.replace('_', ' ').title() for c in class_names]
    values = list(probs) + [probs[0]]
    labels_closed = labels + [labels[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=labels_closed,
        fill='toself',
        fillcolor='rgba(99,102,241,0.15)',
        line=dict(color='#6366f1', width=2),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='#13151f',
            radialaxis=dict(visible=True, range=[0,1],
                            color='#475569', gridcolor='#2d3148',
                            tickfont=dict(size=9, color='#64748b')),
            angularaxis=dict(color='#94a3b8', gridcolor='#2d3148')
        ),
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=20), height=320
    )
    return fig

def make_session_pie(history):
    if not history: return None
    counts = Counter(h['consensus'] for h in history)
    fig = px.pie(names=list(counts.keys()), values=list(counts.values()),
                 color_discrete_sequence=px.colors.qualitative.Vivid, hole=0.45)
    fig.update_layout(paper_bgcolor='#0e1117', font=dict(color='#94a3b8'),
                      legend=dict(font=dict(size=10)),
                      margin=dict(l=10,r=10,t=10,b=10), height=250)
    return fig

def make_severity_trend(history):
    if len(history) < 2: return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=[h['severity'] for h in history],
        mode='lines+markers',
        line=dict(color='#6366f1', width=2),
        marker=dict(color='#818cf8', size=7),
        fill='tozeroy', fillcolor='rgba(99,102,241,0.1)'
    ))
    fig.update_layout(
        paper_bgcolor='#0e1117', plot_bgcolor='#13151f',
        font=dict(color='#94a3b8'),
        xaxis=dict(showgrid=False, title='Analysis #', color='#64748b'),
        yaxis=dict(showgrid=True, gridcolor='#2d3148',
                   range=[0,10], title='Severity', color='#64748b'),
        margin=dict(l=10,r=10,t=10,b=30), height=200
    )
    return fig

# ── Session State ─────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ── Load Resources ────────────────────────────────────────────
sbert = load_sbert()
le, xgb, svm, lr = load_all_models()
models_loaded = all([le, xgb, svm, lr])
CLASS_NAMES   = list(le.classes_) if models_loaded else []
MODEL_REGISTRY = {"XGBoost": xgb, "SVM": svm, "Logistic Regression": lr} \
                  if models_loaded else {}

# Berkeley models — optional, degrade gracefully if not present
sev_model, sev_scaler, sar_model = load_berkeley_models()
berkeley_loaded = all([sev_model, sev_scaler, sar_model])

# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ CyberShield")
    st.markdown(
        "<small style='color:#64748b'>Multi-dataset · Multi-model · "
        "Explainable AI · Severity Scoring</small>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown("### 📊 Session Analytics")
    history = st.session_state.history

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-number">'
                    f'{len(history)}</div><div class="stat-label">Analyzed'
                    f'</div></div>', unsafe_allow_html=True)
    with c2:
        flagged = sum(1 for h in history if h['consensus'] != 'not_cyberbullying')
        st.markdown(f'<div class="stat-card"><div class="stat-number">'
                    f'{flagged}</div><div class="stat-label">Flagged'
                    f'</div></div>', unsafe_allow_html=True)

    if history:
        avg_sev     = np.mean([h['severity'] for h in history])
        most_common = Counter(h['consensus'] for h in history).most_common(1)[0][0]
        implicit_ct = sum(1 for h in history if h.get('implicit_flagged'))

        st.markdown(f'<div class="stat-card" style="margin-top:8px">'
                    f'<div class="stat-number">{avg_sev:.1f}</div>'
                    f'<div class="stat-label">Avg Severity</div></div>',
                    unsafe_allow_html=True)

        if implicit_ct > 0:
            st.markdown(f'<div class="stat-card" style="margin-top:8px">'
                        f'<div class="stat-number" style="color:#eab308">'
                        f'{implicit_ct}</div>'
                        f'<div class="stat-label">Implicit Flags</div></div>',
                        unsafe_allow_html=True)

        st.markdown(f'<div class="insight-box" style="margin-top:10px">'
                    f'Most flagged: <b>{most_common.replace("_"," ").title()}'
                    f'</b></div>', unsafe_allow_html=True)

        pie = make_session_pie(history)
        if pie:
            st.plotly_chart(pie, use_container_width=True,
                            config={'displayModeBar': False})

        trend = make_severity_trend(history)
        if trend:
            st.markdown("<div class='section-title'>Severity Trend</div>",
                        unsafe_allow_html=True)
            st.plotly_chart(trend, use_container_width=True,
                            config={'displayModeBar': False})

        # ── Recent history log ───────────────────────────
        st.markdown("<div class='section-title'>Recent</div>",
                    unsafe_allow_html=True)
        for item in reversed(st.session_state.history[-4:]):
            dot = "🔴" if item['consensus'] != 'not_cyberbullying' else "🟢"
            label = item['consensus'].replace('_',' ').title()
            sev   = item['severity']
            st.markdown(
                f'<div class="history-item">{dot} {label} '
                f'<span style="float:right;color:#475569">S:{sev}</span><br>'
                f'<span style="color:#334155;font-size:0.72rem">'
                f'{item["text"]}</span></div>',
                unsafe_allow_html=True
            )

        if st.button("🗑 Clear Session", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown("<small style='color:#475569'>No analyses yet.</small>",
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <small style='color:#475569'>
    <b>Classifiers:</b> XGBoost · SVM · Logistic Regression<br>
    <b>Embeddings:</b> SBERT all-mpnet-base-v2 (768-dim)<br>
    <b>Dataset 1:</b> Twitter Cyberbullying (47k tweets)<br>
    <b>Dataset 2:</b> Berkeley Hate Speech (135k annotations)<br>
    <b>Augmentation:</b> Domain-adaptive formal English<br>
    <b>Balancing:</b> SMOTE<br>
    <b>Severity:</b> Berkeley-trained Ridge Regressor<br>
    <b>Sarcasm:</b> Berkeley-trained LR Classifier<br>
    <b>Explainability:</b> SHAP (XAI)
    </small>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  MAIN — Header
# ════════════════════════════════════════════════════════════
st.markdown("""
<div style='margin-bottom:4px'>
    <span style='font-size:2rem;font-weight:700;color:#e2e8f0'>🛡️ CyberShield</span>
    <span style='margin-left:12px;background:#1e1b4b;color:#818cf8;
                 font-size:0.75rem;font-weight:600;padding:3px 10px;
                 border-radius:20px;border:1px solid #3730a3;
                 vertical-align:middle'>v2.0 · Multi-Model</span>
</div>
<p style='color:#475569;margin-top:2px;font-size:0.9rem'>
    Semantic sentence embeddings &nbsp;·&nbsp; 3-model ensemble &nbsp;·&nbsp;
    Explainable AI &nbsp;·&nbsp; Implicit language detection &nbsp;·&nbsp;
    Domain-adaptive augmentation
</p>
""", unsafe_allow_html=True)

# Dataset info banner
st.markdown("""
<div style='background:#13151f;border:1px solid #2d3148;border-radius:10px;
            padding:10px 18px;margin-bottom:16px;font-size:0.85rem;color:#64748b;
            display:flex;align-items:center;flex-wrap:wrap;gap:8px'>
    <span style='color:#475569;font-weight:600'>Trained on:</span>
    <span class="source-badge" style="background:#1e3a5f;color:#60a5fa">
        🐦 Twitter Cyberbullying Dataset &nbsp;47k tweets
    </span>
    <span class="source-badge" style="background:#2d1a4a;color:#c084fc">
        ✍️ Domain-Adaptive Augmentation &nbsp;900 sentences
    </span>
    <span class="source-badge" style="background:#1a2535;color:#38bdf8">
        🏛️ Berkeley Hate Speech Dataset &nbsp;135k annotations
    </span>
    <span style='margin-left:auto;color:#334155'>
        SBERT all-mpnet-base-v2 (768-dim) &nbsp;·&nbsp; 3-model ensemble &nbsp;·&nbsp; SMOTE balanced
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Input ─────────────────────────────────────────────────────
input_text = st.text_area(
    "Enter a tweet or message to analyze:",
    height=120,
    placeholder="Paste any message here — the system detects explicit, "
                "implicit, and sarcastic cyberbullying..."
)
if input_text.strip():
    word_count = len(input_text.split())
    char_count = len(input_text)
    st.markdown(
        f"<small style='color:#334155'>{word_count} words · "
        f"{char_count} characters</small>",
        unsafe_allow_html=True
    )

col_btn1, col_btn2, _ = st.columns([1, 1, 4])
with col_btn1:
    analyze_clicked = st.button("🔍 Analyze", type="primary",
                                use_container_width=True)
with col_btn2:
    clear_clicked = st.button("✕ Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

# ════════════════════════════════════════════════════════════
#  ANALYSIS ENGINE
# ════════════════════════════════════════════════════════════
if analyze_clicked and input_text.strip() and models_loaded:

    with st.spinner("Encoding with SBERT all-mpnet-base-v2..."):
        cleaned   = clean_text(input_text)
        embedding = sbert.encode([cleaned])   # shape: (1, 768)

    # ── Implicit / Sarcasm Check ─────────────────────────────
    # Step 1: ML-based detector (Berkeley-trained)
    ml_flagged, ml_sarcasm_conf = detect_sarcasm_ml(embedding)
    # Step 2: Rule-based detector (regex patterns)
    rule_flagged, rule_flag_type, _ = detect_implicit_sarcasm(input_text)
    # Combine: flag if either detector fires
    is_flagged  = ml_flagged or rule_flagged
    flag_type   = rule_flag_type if rule_flagged else "Implicit / Coded Language (ML detected)"
    flag_source = "ML + Rule-based" if (ml_flagged and rule_flagged) else \
                  ("Berkeley ML Model" if ml_flagged else "Rule-based Pattern")

    # ── Multi-Model Predictions ──────────────────────────────
    predictions   = {}
    probabilities = {}

    for name, model in MODEL_REGISTRY.items():
        pred_idx          = model.predict(embedding)[0]
        predictions[name] = le.inverse_transform([pred_idx])[0]
        try:
            probs = model.predict_proba(embedding)[0]
        except Exception:
            probs = np.zeros(len(CLASS_NAMES))
            probs[pred_idx] = 1.0
        probabilities[name] = probs

    # ── Consensus (majority vote) ────────────────────────────
    vote_counts = Counter(predictions.values())
    consensus_label, consensus_votes = vote_counts.most_common(1)[0]
    agreement_pct = (consensus_votes / len(MODEL_REGISTRY)) * 100

    # ── Ensemble confidence ──────────────────────────────────
    avg_probs  = np.mean([probabilities[n] for n in MODEL_REGISTRY], axis=0)
    confidence = float(avg_probs.max())

    # ── Low-confidence borderline check ─────────────────────
    CONFIDENCE_THRESHOLD = 0.60
    is_borderline = confidence < CONFIDENCE_THRESHOLD

    # ── Severity ─────────────────────────────────────────────
    severity, sev_source = compute_severity(
        consensus_label, confidence, is_flagged, embedding
    )
    sev_label, sev_color = SEVERITY_LABELS[severity]

    # ── Save to session ──────────────────────────────────────
    st.session_state.history.append({
        'text':             input_text[:60] + ('…' if len(input_text)>60 else ''),
        'consensus':        consensus_label,
        'severity':         severity,
        'confidence':       confidence,
        'implicit_flagged': is_flagged,
    })

    # ════════════════════════════════════════════════════════
    #  RESULTS
    # ════════════════════════════════════════════════════════
    st.markdown("---")

    # ── Verdict Banner ───────────────────────────────────────
    if is_borderline:
        st.markdown(
            f'<div class="verdict-borderline">'
            f'⚠️ BORDERLINE CASE — Low Confidence ({confidence:.0%})<br>'
            f'<small style="font-weight:400;font-size:0.9rem">'
            f'Leaning toward: {consensus_label.replace("_"," ").upper()} '
            f'— Manual review recommended</small></div>',
            unsafe_allow_html=True
        )
    elif consensus_label == 'not_cyberbullying':
        st.markdown(
            '<div class="verdict-safe">✅ SAFE — Not Cyberbullying</div>',
            unsafe_allow_html=True
        )
    else:
        display = consensus_label.replace('_', ' ').upper()
        CATEGORY_COLORS = {
            'age':               ('#f97316', '#431407'),
            'ethnicity':         ('#ef4444', '#450a0a'),
            'gender':            ('#ec4899', '#500724'),
            'religion':          ('#a855f7', '#3b0764'),
            'other_cyberbullying': ('#eab308', '#422006'),
        }
        pill_color, pill_bg = CATEGORY_COLORS.get(consensus_label, ('#ef4444','#450a0a'))
        st.markdown(
            f'<div class="verdict-danger" style="border-color:{pill_color}">'
            f'🚨 CYBERBULLYING DETECTED<br>'
            f'<span class="category-pill" style="background:{pill_bg};'
            f'color:{pill_color};border:1px solid {pill_color}">'
            f'{display}</span></div>',
            unsafe_allow_html=True
        )

    # ── Implicit / Sarcasm Flag ──────────────────────────────
    if is_flagged:
        ml_conf_str = f" (ML confidence: {ml_sarcasm_conf:.0%})" if ml_flagged else ""
        st.markdown(f"""
        <div class="flag-box">
            ⚠️ <b>Implicit Language Flag: {flag_type}</b>{ml_conf_str}<br>
            <small style='color:#d97706'>Detected by: {flag_source}</small><br>
            <small>This message contains patterns associated with coded, implicit,
            or sarcastic language. Even if classified as safe, it may carry
            harmful intent through indirect phrasing. Treat with caution.</small>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column layout ────────────────────────────────────
    left_col, right_col = st.columns([1.1, 1])

    with left_col:

        # ── Model Breakdown ──────────────────────────────────
        st.markdown("<div class='section-title'>Model Breakdown</div>",
                    unsafe_allow_html=True)
        for name, pred in predictions.items():
            icon  = "✅" if pred == 'not_cyberbullying' else "🚨"
            conf  = float(probabilities[name].max())
            match = "✓ Agrees" if pred == consensus_label else "✗ Differs"
            color = "#22c55e" if pred == consensus_label else "#f97316"
            st.markdown(f"""
            <div class="model-card">
                {icon} <b style='color:#e2e8f0'>{name}</b>
                <span style='float:right;color:{color};font-size:0.8rem'>
                    {match}</span><br>
                <span style='color:#94a3b8'>
                    → {pred.replace('_',' ').title()}</span>
                <span style='float:right;color:#64748b'>{conf:.1%}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="insight-box" style='margin-top:12px'>
            🗳 <b>{int(agreement_pct)}% model agreement</b>
            ({consensus_votes}/{len(MODEL_REGISTRY)} models concur)
        </div>""", unsafe_allow_html=True)

        # ── Severity Index ───────────────────────────────────
        st.markdown("<div class='section-title'>Severity Index</div>",
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;margin-bottom:4px'>
            <span style='color:#e2e8f0;font-weight:600;font-size:1.2rem'>
                {severity}/10</span>
            <span style='color:{sev_color};font-weight:700'>{sev_label}</span>
        </div>
        <div style='background:#1a1d26;border-radius:8px;
                    height:14px;overflow:hidden'>
            <div style='width:{severity*10}%;height:100%;
                        background:{sev_color};border-radius:8px'></div>
        </div>
        <small style='color:#475569'>
            Source: {sev_source}
            {'· implicit language penalty applied' if is_flagged else ''}
        </small>""", unsafe_allow_html=True)

        # ── Preprocessed text ────────────────────────────────
        st.markdown("<div class='section-title'>Preprocessed Input</div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<code style='color:#94a3b8;font-size:0.8rem'>{cleaned}</code>",
            unsafe_allow_html=True
        )

    with right_col:

        # ── Radar Chart ──────────────────────────────────────
        st.markdown(
            "<div class='section-title'>Confidence Across All Classes</div>",
            unsafe_allow_html=True
        )
        radar = make_radar_chart(avg_probs, CLASS_NAMES)
        st.plotly_chart(radar, use_container_width=True,
                        config={'displayModeBar': False})

        # ── Top 3 probabilities ──────────────────────────────
        sorted_idx = np.argsort(avg_probs)[::-1][:3]
        st.markdown("<div class='section-title'>Top Predictions</div>",
                    unsafe_allow_html=True)
        for idx in sorted_idx:
            cname = CLASS_NAMES[idx].replace('_', ' ').title()
            prob  = avg_probs[idx]
            color = "#6366f1" if CLASS_NAMES[idx] == consensus_label else "#334155"
            st.markdown(f"""
            <div style='margin:6px 0'>
                <div style='display:flex;justify-content:space-between;
                            font-size:0.85rem;color:#94a3b8'>
                    <span>{cname}</span>
                    <span style='color:#e2e8f0'>{prob:.1%}</span>
                </div>
                <div style='background:#1a1d26;border-radius:6px;
                            height:8px;margin-top:3px;overflow:hidden'>
                    <div style='width:{int(prob*100)}%;height:100%;
                                background:{color};border-radius:6px'></div>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── SHAP Explainability ───────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div class='section-title'>🧠 Explainable AI — Why This Verdict?</div>",
        unsafe_allow_html=True
    )

    with st.expander("View Token-Level SHAP Explanation", expanded=True):
        try:
            import shap

            explainer      = shap.TreeExplainer(xgb)
            shap_vals      = explainer.shap_values(embedding)
            pred_class_idx = int(np.argmax(avg_probs))

            if isinstance(shap_vals, list):
                class_shap = shap_vals[pred_class_idx][0]
            else:
                class_shap = shap_vals[0]

            tokens      = cleaned.split()
            n_tokens    = min(len(tokens), 15)
            dim_per_tok = max(1, len(class_shap) // max(1, len(tokens)))
            token_shap  = []

            for i in range(n_tokens):
                start = i * dim_per_tok
                end   = start + dim_per_tok
                token_shap.append(
                    (tokens[i], float(np.mean(class_shap[start:end])))
                )

            token_shap.sort(key=lambda x: abs(x[1]), reverse=True)
            top_tokens = token_shap[:10]
            words  = [t[0] for t in top_tokens]
            values = [t[1] for t in top_tokens]
            colors = ['#ef4444' if v > 0 else '#22c55e' for v in values]

            fig_shap = go.Figure(go.Bar(
                x=values, y=words, orientation='h',
                marker_color=colors, marker_line_width=0,
            ))
            fig_shap.update_layout(
                paper_bgcolor='#0e1117', plot_bgcolor='#13151f',
                font=dict(color='#94a3b8', size=12),
                xaxis=dict(title='SHAP Contribution', color='#64748b',
                           gridcolor='#2d3148', zeroline=True,
                           zerolinecolor='#475569'),
                yaxis=dict(color='#e2e8f0'),
                margin=dict(l=10,r=10,t=10,b=30), height=320,
            )
            st.plotly_chart(fig_shap, use_container_width=True,
                            config={'displayModeBar': False})

            st.markdown(f"""
            <div class='insight-box'>
                🔴 <b>Red</b> = tokens pushing <i>toward</i>
                {consensus_label.replace('_',' ').title()}<br>
                🟢 <b>Green</b> = tokens pushing <i>away</i> from that class<br>
                <small style='color:#475569'>
                    SHAP computed on XGBoost over 768-dim SBERT embeddings
                    (all-mpnet-base-v2)
                </small>
            </div>""", unsafe_allow_html=True)

        except ImportError:
            st.info("Run `pip install shap` to enable explainability.")
        except Exception as e:
            st.warning(f"SHAP unavailable: {e}")

elif analyze_clicked and not input_text.strip():
    st.warning("Please enter some text to analyze.")
elif analyze_clicked and not models_loaded:
    st.error("Models not loaded. Run train_model.py first.")