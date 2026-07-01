import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Crop Recommendation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Root & background ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0e1a0e !important;
    color: #d4e8d4;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stAppViewContainer"] > .main {
    background-color: #0e1a0e;
}
[data-testid="stHeader"], [data-testid="stToolbar"] {
    background: transparent !important;
}
section[data-testid="stSidebar"] { display: none; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0e1a0e; }
::-webkit-scrollbar-thumb { background: #2d5c2d; border-radius: 3px; }

/* ── Hero title ── */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 900;
    text-align: center;
    color: #e8f5e8;
    letter-spacing: -0.5px;
    margin: 0.2rem 0 0.3rem;
    line-height: 1.15;
}
.hero-title span {
    color: #5cb85c;
    font-style: italic;
}
.hero-sub {
    text-align: center;
    color: #7a9e7a;
    font-size: 1.08rem;
    font-weight: 300;
    margin-bottom: 1.8rem;
    letter-spacing: 0.02em;
}

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #5cb85c;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(92,184,92,0.2);
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(92,184,92,0.15);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* ── Number inputs ── */
[data-testid="stNumberInput"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(92,184,92,0.2) !important;
    border-radius: 8px !important;
    color: #e8f5e8 !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stNumberInput"] input {
    color: #e8f5e8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 1.08rem !important;
    background: transparent !important;
}
[data-testid="stNumberInput"] label {
    color: #9ab89a !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stNumberInput"] button {
    color: #5cb85c !important;
    background: rgba(92,184,92,0.08) !important;
    border-radius: 4px !important;
    font-size: 1.1rem !important;
}
[data-testid="stNumberInput"] button:hover {
    background: rgba(92,184,92,0.18) !important;
}

/* ── Range hint badges ── */
.range-badge {
    display: inline-block;
    background: rgba(92,184,92,0.1);
    border: 1px solid rgba(92,184,92,0.2);
    border-radius: 4px;
    padding: 1px 8px;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #7ab87a;
    margin-top: 2px;
}

/* ── Recommend button ── */
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #2d7a2d 0%, #4caf50 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2rem !important;
    letter-spacing: 0.03em;
    box-shadow: 0 4px 20px rgba(76,175,80,0.3) !important;
    transition: all 0.2s !important;
}
[data-testid="baseButton-primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(76,175,80,0.45) !important;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, rgba(45,122,45,0.18) 0%, rgba(14,26,14,0.9) 100%);
    border: 1px solid rgba(92,184,92,0.35);
    border-radius: 14px;
    padding: 1.6rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 1rem;
}
.result-crop-name {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #e8f5e8;
    line-height: 1;
}
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #5cb85c;
    margin-bottom: 4px;
}

/* ── Progress bars (top 5) ── */
.crop-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 0.65rem;
}
.crop-rank {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #5c7a5c;
    width: 16px;
    text-align: center;
    flex-shrink: 0;
}
.crop-emoji { font-size: 1rem; flex-shrink: 0; }
.crop-name {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    font-weight: 500;
    color: #c8dfc8;
    width: 110px;
    flex-shrink: 0;
}
.bar-bg {
    flex: 1;
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #2d7a2d, #5cb85c);
}
.crop-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #9ab89a;
    width: 48px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Parameter guide table ── */
[data-testid="stTable"] table {
    background: transparent !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.92rem !important;
}
[data-testid="stTable"] th {
    background: rgba(92,184,92,0.1) !important;
    color: #5cb85c !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid rgba(92,184,92,0.2) !important;
}
[data-testid="stTable"] td {
    color: #b8d4b8 !important;
    border-bottom: 1px solid rgba(255,255,255,0.05) !important;
}
[data-testid="stTable"] tr:hover td {
    background: rgba(92,184,92,0.05) !important;
}

/* ── Tip box ── */
.tip-box {
    background: rgba(92,184,92,0.07);
    border-left: 3px solid #5cb85c;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.92rem;
    color: #9ab89a;
    margin-top: 1rem;
    line-height: 1.5;
}
.tip-box strong { color: #5cb85c; }

/* ── Divider ── */
hr { border-color: rgba(92,184,92,0.1) !important; }

/* ── Alert / error ── */
[data-testid="stAlert"] {
    background: rgba(255,80,80,0.08) !important;
    border: 1px solid rgba(255,80,80,0.25) !important;
    border-radius: 8px !important;
    color: #ffaaaa !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #5cb85c !important; }
</style>
""", unsafe_allow_html=True)

# ── JS: clear default 0 on focus, restore if left empty ──────────────────────
st.markdown("""
<script>
(function pollInputs() {
    const inputs = document.querySelectorAll('[data-testid="stNumberInput"] input');
    inputs.forEach(inp => {
        if (inp.dataset.clearBound) return;
        inp.dataset.clearBound = "1";
        inp.addEventListener('focus', function() {
            if (this.value === '0' || this.value === '0.00' || this.value === '0.0') {
                this.value = '';
            }
        });
        inp.addEventListener('blur', function() {
            if (this.value === '') {
                this.value = '0';
                this.dispatchEvent(new Event('input', { bubbles: true }));
            }
        });
    });
    setTimeout(pollInputs, 800);
})();
</script>
""", unsafe_allow_html=True)


# ── Model loader ──────────────────────────────────────────────────────────────
# CRITICAL: this must match the exact column order model.py trained on.
# Mismatching this order silently produces wrong / low-confidence predictions.
FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def _find_model_dir():
    """Look for a 'models' folder next to app.py, then one level up, so the
    app works whether it lives at project root or inside a subfolder."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "..", "models"),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "stack_model_combined.pkl")):
            return c
    # Fall back to the sibling-folder convention even if not found yet,
    # so the resulting error message points at the expected location.
    return candidates[-1]


@st.cache_resource
def load_models():
    MODEL_DIR = _find_model_dir()
    with open(os.path.join(MODEL_DIR, "stack_model_combined.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoder_combined.pkl"), "rb") as f:
        encoder = pickle.load(f)
    return model, encoder


CROP_EMOJIS = {
    "rice": "🌾", "wheat": "🌾", "maize": "🌽", "coffee": "☕",
    "jute": "🌿", "cotton": "🌸", "coconut": "🥥", "papaya": "🍈",
    "orange": "🍊", "apple": "🍎", "muskmelon": "🍈", "watermelon": "🍉",
    "grapes": "🍇", "mango": "🥭", "banana": "🍌", "pomegranate": "🍎",
    "lentil": "🫘", "blackgram": "🫘", "mungbean": "🫘", "mothbeans": "🫘",
    "pigeonpeas": "🫘", "kidneybeans": "🫘", "chickpea": "🫘",
}
def get_emoji(name): return CROP_EMOJIS.get((name or "").lower(), "🌱")


# ── SHAP helpers (fully inlined — no external file dependency) ────────────────
SHAP_FEATURE_META = [
    ("N",           "Nitrogen",    "kg/ha", "🌿"),
    ("P",           "Phosphorus",  "kg/ha", "🧪"),
    ("K",           "Potassium",   "kg/ha", "⚗️"),
    ("temperature", "Temperature", "°C",    "🌡️"),
    ("humidity",    "Humidity",    "%",     "💧"),
    ("pH",          "Soil pH",     "",      "🔬"),
    ("rainfall",    "Rainfall",    "mm",    "🌧️"),
]

@st.cache_resource
def get_shap_explainer(_model):
    """Build TreeExplainer from the RF base learner — cached for the session."""
    import shap
    rf = _model.named_estimators_["rf"]
    return shap.TreeExplainer(rf)

def compute_shap_values(model, input_array, class_idx):
    """
    Returns shap_vals (shape: 7,) for the predicted class.
    Handles both shap output formats:
      - list of arrays → older shap  (n_classes items, each (n_samples, n_features))
      - np.ndarray     → newer shap  (n_samples, n_features, n_classes)
    """
    explainer = get_shap_explainer(model)
    sv = explainer.shap_values(input_array)          # input shape (1, 7)

    if isinstance(sv, list):
        vals     = np.array(sv[class_idx])[0]        # (7,)
        base_val = float(explainer.expected_value[class_idx])
    else:
        sv = np.array(sv)
        if sv.ndim == 3:                             # (1, 7, n_classes)
            vals     = sv[0, :, class_idx]
            base_val = (
                float(explainer.expected_value[class_idx])
                if hasattr(explainer.expected_value, "__len__")
                else float(explainer.expected_value)
            )
        else:                                        # (n_classes, 1, 7)
            vals     = sv[class_idx, 0, :]
            base_val = float(explainer.expected_value[class_idx])

    return vals, base_val

def render_shap_chart(shap_vals, input_array, top_crop):
    """Themed horizontal SHAP impact chart rendered via components.html."""
    import streamlit.components.v1 as components

    order   = np.argsort(np.abs(shap_vals))[::-1]   # most impactful first
    max_abs = max(float(np.abs(shap_vals).max()), 1e-9)

    rows_html = ""
    for idx in order:
        sv               = float(shap_vals[idx])
        raw_val          = float(input_array[0, idx])
        _, label, unit, emoji = SHAP_FEATURE_META[idx]
        bar_pct          = min(abs(sv) / max_abs * 100, 100)
        color            = "#4caf50" if sv >= 0 else "#e05252"
        unit_str         = f" {unit}" if unit else ""
        sign             = "+" if sv >= 0 else ""

        rows_html += f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:11px;">
          <div style="width:135px;flex-shrink:0;text-align:right;">
            <span style="font-size:1rem;">{emoji}</span>
            <span style="font-family:'DM Sans',sans-serif;font-size:0.9rem;
                         color:#c8dfc8;font-weight:500;"> {label}</span>
          </div>
          <div style="width:86px;flex-shrink:0;text-align:center;">
            <span style="font-family:'DM Mono',monospace;font-size:0.78rem;
                         color:#7ab87a;background:rgba(92,184,92,0.1);
                         border:1px solid rgba(92,184,92,0.2);
                         border-radius:4px;padding:2px 7px;">
              {raw_val:.2f}{unit_str}
            </span>
          </div>
          <div style="flex:1;background:rgba(255,255,255,0.05);
                      border-radius:4px;height:10px;overflow:hidden;">
            <div style="width:{bar_pct:.1f}%;height:100%;border-radius:4px;
                        background:{color};box-shadow:0 0 6px {color}55;"></div>
          </div>
          <div style="width:72px;flex-shrink:0;text-align:right;">
            <span style="font-family:'DM Mono',monospace;font-size:0.84rem;
                         color:{color};font-weight:600;">{sign}{sv:.4f}</span>
          </div>
        </div>
        """

    legend = """
    <div style="display:flex;gap:20px;margin-top:14px;padding-top:10px;
                border-top:1px solid rgba(92,184,92,0.12);">
      <div style="display:flex;align-items:center;gap:7px;">
        <div style="width:13px;height:13px;border-radius:3px;background:#4caf50;"></div>
        <span style="font-family:'DM Sans',sans-serif;font-size:0.8rem;color:#9ab89a;">
          Positive — favours this crop
        </span>
      </div>
      <div style="display:flex;align-items:center;gap:7px;">
        <div style="width:13px;height:13px;border-radius:3px;background:#e05252;"></div>
        <span style="font-family:'DM Sans',sans-serif;font-size:0.8rem;color:#9ab89a;">
          Negative — works against this crop
        </span>
      </div>
    </div>
    """

    note = f"""
    <p style="font-family:'DM Sans',sans-serif;font-size:0.84rem;color:#7a9e7a;
              margin:0 0 14px;line-height:1.6;">
      SHAP contributions toward predicting
      <strong style="color:#e8f5e8;">{top_crop.title()}</strong>
      — derived from the Random Forest base learner inside the stacking ensemble.
      Sorted by absolute impact (most influential at top).
    </p>
    """

    html = f"""
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700
                &family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
    <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(92,184,92,0.15);
                border-radius:12px;padding:1.4rem 1.6rem;">
      {note}
      {rows_html}
      {legend}
    </div>
    """
    components.html(html, height=len(SHAP_FEATURE_META) * 54 + 160)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">Smart <span>Crop</span> Recommendation</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Enter your soil and climate parameters to discover the ideal crop for your field.</p>', unsafe_allow_html=True)

# ── Main layout: left inputs | right guide ────────────────────────────────────
st.markdown("""
<div class="tip-box">
    📏 <strong>Enter precise values, not rounded ones.</strong> Type the exact
    reading you have — e.g. if your rainfall measurement is <strong>101.05 mm</strong>,
    enter <strong>101.05</strong>, not <strong>101</strong>. The model is sensitive to
    decimals, and rounding can shift the recommendation and its confidence.
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    # ── Soil Nutrients ──
    st.markdown('<div class="section-header">🌿 &nbsp;Soil Nutrients</div>', unsafe_allow_html=True)
    with st.container():
        n1, n2, n3 = st.columns(3)
        with n1:
            N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
            st.markdown('<span class="range-badge">0 – 200 kg/ha</span>', unsafe_allow_html=True)
        with n2:
            P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
            st.markdown('<span class="range-badge">0 – 200 kg/ha</span>', unsafe_allow_html=True)
        with n3:
            K = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
            st.markdown('<span class="range-badge">0 – 200 kg/ha</span>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ph_col, _ = st.columns([1, 2])
        with ph_col:
            pH = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=0.0, step=0.01, format="%.2f")
            st.markdown('<span class="range-badge">Acidic 0 – 14 → Alkaline</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Climate Conditions ──
    st.markdown('<div class="section-header">🌤 &nbsp;Climate Conditions</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=0.0, step=0.5, format="%.2f")
        st.markdown('<span class="range-badge">0 – 50 °C</span>', unsafe_allow_html=True)
    with c2:
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, format="%.2f")
        st.markdown('<span class="range-badge">0 – 100 %</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    rf_col, _ = st.columns([1, 2])
    with rf_col:
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0, step=5.0, format="%.2f")
        st.markdown('<span class="range-badge">0 – 500 mm</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    btn = st.button("🌾  Recommend Crop", type="primary", use_container_width=False)


@st.cache_data
def get_guide_df():
    return pd.DataFrame({
        "Parameter": ["N", "P", "K", "pH", "Temp", "Humidity", "Rainfall"],
        "Unit":      ["kg/ha", "kg/ha", "kg/ha", "—", "°C", "%", "mm"],
        "Typical Range": ["0–140", "5–145", "5–205", "3.5–9.9", "8–44", "14–100", "20–299"],
    }).set_index("Parameter")


with col_right:
    st.markdown('<div class="section-header">📋 &nbsp;Parameter Guide</div>', unsafe_allow_html=True)
    st.table(get_guide_df())

    st.markdown("""
    <div class="tip-box">
        <strong>💡 Tip:</strong> Values outside typical ranges are allowed — the model was trained on diverse real-world datasets and handles edge cases gracefully.
    </div>
    """, unsafe_allow_html=True)


# ── Prediction ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

if btn:
    try:
        model, encoder = load_models()
    except Exception as e:
        st.error(f"❌ Could not load model files: {e}")
        st.stop()

    # Feature order MUST match FEATURE_ORDER / training order:
    # N, P, K, temperature, humidity, ph, rainfall.
    # (Previously this array put pH before temperature/humidity, which
    # silently fed the model garbage — fixed here.)
    inputs = np.array([[N, P, K, temperature, humidity, pH, rainfall]])

    if all(v == 0 for v in [N, P, K, pH, temperature, humidity, rainfall]):
        st.warning(
            "⚠️ All fields are still at their default value of 0 — that's not a "
            "realistic soil/climate reading. Enter your actual measurements above "
            "for a meaningful recommendation."
        )

    with st.spinner("Analysing soil and climate data…"):
        pred_encoded = model.predict(inputs)
        pred_crop = encoder.inverse_transform(pred_encoded)[0]
        proba = model.predict_proba(inputs)[0]

    proba_df = (
        pd.DataFrame({"Crop": encoder.classes_, "Probability": proba * 100})
        .sort_values("Probability", ascending=False)
        .reset_index(drop=True)
    )
    top_conf = proba_df.iloc[0]["Probability"]
    top_crop = proba_df.iloc[0]["Crop"]

    # ── Result banner ──
    st.markdown(f"""
    <div class="result-card">
        <div>
            <div class="result-label">Recommended Crop</div>
            <div class="result-crop-name">{get_emoji(top_crop)} &nbsp;{top_crop.title()}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 &nbsp;Top 5 Crop Probabilities</div>', unsafe_allow_html=True)

    max_prob = proba_df.iloc[0]["Probability"]
    runner_up = proba_df.iloc[1]["Probability"] if len(proba_df) > 1 else 0
    if top_conf < 60:
        st.markdown(
            '<div class="tip-box">⚖️ These soil/climate values sit between several '
            'suitable crops — the top pick is the best match, but it\'s worth '
            'reviewing the runner-ups below too.</div>',
            unsafe_allow_html=True,
        )
    elif (max_prob - runner_up) < 10:
        st.markdown(
            '<div class="tip-box">⚖️ The top two crops are close — either could work well '
            "for these conditions.</div>",
            unsafe_allow_html=True,
        )

    top5 = proba_df.head(5)

    bar_rows = ""
    for i, row in top5.iterrows():
        rank = i + 1
        fill_pct = (row["Probability"] / max_prob) * 100
        bold = "font-weight:700;color:#e8f5e8;" if rank == 1 else ""
        bar_rows += f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.7rem;">
            <span style="font-family:'DM Mono',monospace;font-size:0.85rem;color:#5c7a5c;width:18px;text-align:center;flex-shrink:0;">{rank}</span>
            <span style="font-size:1.15rem;flex-shrink:0;">{get_emoji(row['Crop'])}</span>
            <span style="font-family:'DM Sans',sans-serif;font-size:1rem;font-weight:500;color:#c8dfc8;width:120px;flex-shrink:0;{bold}">{row['Crop'].title()}</span>
            <div style="flex:1;background:rgba(255,255,255,0.05);border-radius:4px;height:9px;overflow:hidden;">
                <div style="width:{fill_pct:.1f}%;height:100%;border-radius:4px;background:linear-gradient(90deg,#2d7a2d,#5cb85c);"></div>
            </div>
            <span style="font-family:'DM Mono',monospace;font-size:0.9rem;color:#9ab89a;width:56px;text-align:right;flex-shrink:0;">{row['Probability']:.2f}%</span>
        </div>
        """

    import streamlit.components.v1 as components
    components.html(f"""
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(92,184,92,0.15);border-radius:12px;padding:1.4rem 1.6rem;">
        {bar_rows}
    </div>
    """, height=len(top5) * 54 + 50)

    # ── SHAP Explainability ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔍 &nbsp;SHAP Feature Explainability</div>', unsafe_allow_html=True)

    try:
        import shap  # verify installed; raises ImportError if missing
        class_idx = list(encoder.classes_).index(top_crop)
        with st.spinner("Computing SHAP values…"):
            shap_vals, _ = compute_shap_values(model, inputs, class_idx)
        render_shap_chart(shap_vals, inputs, top_crop)
    except ImportError:
        st.warning("⚠️ `shap` is not installed. Run:  pip install shap  then restart the app.")
    except Exception as e:
        st.warning(f"⚠️ SHAP computation failed: {e}")


# ── Credits ─────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="border-top:1px solid rgba(92,184,92,0.15); padding-top:1.2rem; margin-top:1rem;
            text-align:center;">
    <div style="font-family:'DM Mono',monospace; font-size:0.72rem; letter-spacing:0.14em;
                text-transform:uppercase; color:#5cb85c; margin-bottom:6px;">
        Developed by
    </div>
    <div style="font-family:'DM Sans',sans-serif; font-size:0.9rem; color:#9ab89a;
                line-height:1.7;">
        Abhinav Manoj &nbsp;•&nbsp; Adithyan M C &nbsp;•&nbsp; Gagan Dev P V &nbsp;•&nbsp; Jyothis C R
    </div>
</div>
""", unsafe_allow_html=True)