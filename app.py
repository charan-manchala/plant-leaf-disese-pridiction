"""
app.py — Plant Leaf Disease Prediction — Main Streamlit App
Run: streamlit run app.py
"""

import os
import sys
import io
import json
import numpy as np
from PIL import Image
import streamlit as st

st.set_page_config(
    page_title="Plant Leaf Disease Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from data.disease_cure_db import DISEASE_CURE_DB, PLANT_TYPES
from utils.visualization import (overlay_gradcam,
                                  make_confidence_bar_chart,
                                  severity_color)

MODEL_PATH     = os.path.join(ROOT, "models",
                               "efficientnet_b0_best.pth")
CLASS_IDX_PATH = os.path.join(ROOT, "models",
                               "class_indices.json")
DEMO_MODE      = not (os.path.exists(MODEL_PATH) and
                      os.path.exists(CLASS_IDX_PATH))

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main .block-container{padding-top:1.5rem;max-width:1200px}
.app-header{background:linear-gradient(135deg,#1b4332,#2d6a4f);
  border-radius:12px;padding:1.8rem 2rem;margin-bottom:1.5rem;color:white}
.app-header h1{font-size:2rem;font-weight:700;margin:0 0 .3rem}
.app-header p{font-size:1rem;opacity:.85;margin:0}
.step-row{display:flex;gap:8px;align-items:center;
  margin-bottom:1.2rem;flex-wrap:wrap}
.step-pill{background:#d8f3dc;color:#1b4332;border-radius:999px;
  padding:4px 14px;font-size:.82rem;font-weight:600;
  border:1.5px solid #52b788}
.step-arrow{color:#64748b;font-size:.9rem}
.result-card{background:#fff;border:1px solid #e2e8f0;
  border-radius:12px;padding:1.4rem 1.6rem;margin-bottom:1rem;
  box-shadow:0 1px 4px rgba(0,0,0,.05)}
.result-card h3{font-size:.78rem;font-weight:600;
  text-transform:uppercase;letter-spacing:.08em;
  color:#64748b;margin:0 0 .5rem}
.result-value{font-size:1.3rem;font-weight:700;color:#1e293b}
.badge{display:inline-block;border-radius:999px;
  padding:5px 16px;font-size:.9rem;font-weight:600}
.badge-healthy{background:#d1fae5;color:#065f46}
.badge-diseased{background:#fee2e2;color:#991b1b}
.pct-bar-wrap{background:#f1f5f9;border-radius:999px;
  height:12px;overflow:hidden;margin-top:.4rem}
.pct-bar-fill{height:100%;border-radius:999px}
.cure-item{display:flex;gap:10px;align-items:flex-start;
  padding:8px 0;border-bottom:1px solid #e2e8f0;font-size:.95rem}
.cure-item:last-child{border-bottom:none}
.cure-num{background:#52b788;color:white;border-radius:50%;
  width:24px;height:24px;min-width:24px;display:flex;
  align-items:center;justify-content:center;
  font-size:.75rem;font-weight:700;margin-top:1px}
.sev-chip{display:inline-block;border-radius:6px;
  padding:3px 12px;font-size:.82rem;font-weight:600;
  margin-left:8px}
.demo-banner{background:#fffbeb;border:1px solid #fcd34d;
  border-radius:10px;padding:.9rem 1.2rem;
  font-size:.9rem;color:#92400e;margin-bottom:1rem}
.conf-ring{font-size:2.5rem;font-weight:800;color:#2d6a4f}
#MainMenu{visibility:hidden}footer{visibility:hidden}
header[data-testid="stHeader"]{display:none}
</style>
""", unsafe_allow_html=True)


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_predictor():
    if DEMO_MODE:
        return None
    try:
        from utils.predictor import PlantDiseasePredictor
        return PlantDiseasePredictor(MODEL_PATH, CLASS_IDX_PATH)
    except Exception as e:
        st.warning(f"Model load warning: {e}")
        return None


# ── Demo predictions ──────────────────────────────────────────────────────────
DEMO_PREDS = [
    {
        "plant_name": "Tomato",
        "plant_type": "Vegetable crop",
        "is_healthy": False,
        "disease_name": "Early Blight",
        "disease_key": "Tomato___Early_blight",
        "confidence": 87.4,
        "disease_percentage": 38.5,
        "top_k_predictions": [
            ("Tomato___Early_blight", 87.4),
            ("Tomato___Septoria_leaf_spot", 8.2),
            ("Tomato___healthy", 4.4),
        ],
        "cam_array": None,
    },
    {
        "plant_name": "Potato",
        "plant_type": "Root/Tuber crop",
        "is_healthy": False,
        "disease_name": "Late Blight",
        "disease_key": "Potato___Late_blight",
        "confidence": 91.1,
        "disease_percentage": 62.0,
        "top_k_predictions": [
            ("Potato___Late_blight", 91.1),
            ("Potato___Early_blight", 6.7),
            ("Potato___healthy", 2.2),
        ],
        "cam_array": None,
    },
    {
        "plant_name": "Apple",
        "plant_type": "Fruit tree",
        "is_healthy": True,
        "disease_name": "Healthy",
        "disease_key": "Apple___healthy",
        "confidence": 96.3,
        "disease_percentage": None,
        "top_k_predictions": [
            ("Apple___healthy", 96.3),
            ("Apple___Apple_scab", 2.4),
            ("Apple___Black_rot", 1.3),
        ],
        "cam_array": None,
    },
]


def mock_predict(image: Image.Image) -> dict:
    key    = hash(image.tobytes()) % len(DEMO_PREDS)
    result = DEMO_PREDS[key].copy()
    h, w   = 224, 224
    cam    = np.zeros((h, w), dtype=np.float32)
    if not result["is_healthy"]:
        rng = np.random.default_rng(42)
        for _ in range(5):
            cx, cy = rng.integers(40, 184, size=2)
            r      = rng.integers(20, 50)
            Y, X   = np.ogrid[:h, :w]
            mask   = ((X-cx)**2 + (Y-cy)**2) <= r**2
            cam[mask] = rng.uniform(0.5, 1.0)
        cam = np.clip(cam + rng.uniform(0, 0.1, cam.shape), 0, 1)
    result["cam_array"] = cam
    return result


# ── Header ────────────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="app-header">
        <h1>🌿 Plant Leaf Disease Predictor</h1>
        <p>Upload a leaf image for instant AI-powered disease
        detection, severity assessment, and cure recommendations.</p>
    </div>
    <div class="step-row">
        <span class="step-pill">① Upload Leaf</span>
        <span class="step-arrow">→</span>
        <span class="step-pill">② Detect Plant</span>
        <span class="step-arrow">→</span>
        <span class="step-pill">③ Assess Health</span>
        <span class="step-arrow">→</span>
        <span class="step-pill">④ Identify Disease</span>
        <span class="step-arrow">→</span>
        <span class="step-pill">⑤ Recommend Cure</span>
    </div>
    """, unsafe_allow_html=True)


# ── Result cards ──────────────────────────────────────────────────────────────
def render_result_cards(result: dict, cure_info: dict):
    is_healthy = result["is_healthy"]
    pct        = result.get("disease_percentage")
    conf       = result["confidence"]

    # Row 1
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="result-card">
            <h3>🌱 Plant Name</h3>
            <div class="result-value">{result['plant_name']}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="result-card">
            <h3>🏷️ Plant Type</h3>
            <div class="result-value">{result['plant_type']}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        badge_cls = "badge-healthy" if is_healthy else "badge-diseased"
        badge_txt = "✅ Healthy" if is_healthy else "⚠️ Diseased"
        pct_html  = ""
        if pct is not None:
            col = ("#e76f51" if pct > 50
                   else "#f4a261" if pct > 25 else "#e9c46a")
            pct_html = (
                f'<div style="margin-top:.6rem">'
                f'<span style="font-size:.85rem;color:#64748b">'
                f'Affected: <b style="color:{col}">{pct}%</b>'
                f'</span>'
                f'<div style="background:#f1f5f9;border-radius:999px;'
                f'height:12px;overflow:hidden;margin-top:.4rem">'
                f'<div style="height:100%;border-radius:999px;'
                f'width:{pct}%;background:{col}"></div>'
                f'</div></div>'
            )
        st.markdown(f"""
        <div class="result-card">
            <h3>🔍 Leaf Status</h3>
            <span class="badge {badge_cls}">{badge_txt}</span>
            {pct_html}
        </div>""", unsafe_allow_html=True)

    # Row 2 — Disease details
    if not is_healthy:
        d1, d2, d3 = st.columns(3)
        sev     = cure_info.get("severity", "Unknown")
        sev_col = severity_color(sev)
        with d1:
            st.markdown(f"""
            <div class="result-card">
                <h3>🦠 Disease</h3>
                <div class="result-value">{result['disease_name']}</div>
            </div>""", unsafe_allow_html=True)
        with d2:
            st.markdown(f"""
            <div class="result-card">
                <h3>⚡ Severity</h3>
                <div class="result-value">{sev}
                  <span class="sev-chip"
                    style="background:{sev_col}22;color:{sev_col}">
                    {sev}</span>
                </div>
            </div>""", unsafe_allow_html=True)
        with d3:
            st.markdown(f"""
            <div class="result-card">
                <h3>🎯 Confidence</h3>
                <div class="conf-ring">{conf}%</div>
            </div>""", unsafe_allow_html=True)

    # Symptoms & Causes
    if not is_healthy and cure_info:
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(f"""
            <div class="result-card">
                <h3>🔬 Symptoms</h3>
                <p style="margin:0;font-size:.95rem;line-height:1.6">
                {cure_info.get('symptoms','—')}</p>
            </div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""
            <div class="result-card">
                <h3>🧫 Cause</h3>
                <p style="margin:0;font-size:.95rem;line-height:1.6">
                {cure_info.get('causes','—')}</p>
            </div>""", unsafe_allow_html=True)

    # Cure steps
    cure_steps = cure_info.get("cure",
                               ["No specific treatment needed."])
    items_html = "".join(
        f'<div class="cure-item">'
        f'<div class="cure-num">{i}</div>'
        f'<div>{step}</div></div>'
        for i, step in enumerate(cure_steps, 1)
    )
    st.markdown(f"""
    <div class="result-card">
        <h3>💊 Suggested Cure & Treatment</h3>
        {items_html}
    </div>""", unsafe_allow_html=True)

    # Prevention
    prevention = cure_info.get("prevention", "")
    if prevention:
        st.markdown(f"""
        <div class="result-card"
          style="border-left:4px solid #52b788">
            <h3>🛡️ Prevention</h3>
            <p style="margin:0;font-size:.95rem;line-height:1.6">
            {prevention}</p>
        </div>""", unsafe_allow_html=True)


# ── Image analysis tab ────────────────────────────────────────────────────────
def render_image_analysis(original_img: Image.Image, result: dict):
    img_col, cam_col = st.columns(2)
    with img_col:
        st.markdown("**Original Leaf Image**")
        st.image(original_img, use_container_width=True)
    with cam_col:
        cam = result.get("cam_array")
        if cam is not None and not result["is_healthy"]:
            st.markdown("**Disease Region Heatmap** *(red = affected)*")
            overlay = overlay_gradcam(original_img, cam)
            st.image(overlay, use_container_width=True)
        else:
            st.markdown("**No disease region detected**")
            st.image(original_img, use_container_width=True)
            st.success("✅ Leaf appears healthy.")


# ── Confidence chart tab ──────────────────────────────────────────────────────
def render_confidence_chart(result: dict):
    top_k = result.get("top_k_predictions", [])
    if top_k:
        chart_bytes = make_confidence_bar_chart(top_k)
        st.image(chart_bytes, use_container_width=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Info")
        st.markdown("---")
        if DEMO_MODE:
            st.warning("**Demo Mode** — Train model to enable live inference")
        else:
            st.success("✅ Model loaded: EfficientNet-B0")
        st.markdown("---")
        st.markdown("### 📦 Datasets")
        st.markdown("""
        - **PlantVillage** — 54,306 images, 38 classes
        - **PlantDoc** — 2,569 real-world images
        - **Leaf Segmentation DS** — pixel-level masks
        """)
        st.markdown("---")
        st.markdown("### 🌿 Supported Plants")
        for p in ["Tomato","Potato","Corn","Apple",
                  "Grape","Pepper","Strawberry","Peach",
                  "Cherry","Orange","Squash","Soybean"]:
            st.markdown(f"- {p}")
        st.markdown("---")
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. Upload a clear, well-lit leaf photo
        2. Single leaf preferred
        3. Supports JPG, PNG, WEBP
        """)


# ── Text report ───────────────────────────────────────────────────────────────
def generate_report(result: dict, cure_info: dict) -> str:
    lines = [
        "=" * 50,
        "   PLANT LEAF DISEASE PREDICTION REPORT",
        "=" * 50,
        f"Plant Name   : {result['plant_name']}",
        f"Plant Type   : {result['plant_type']}",
        f"Leaf Status  : "
        f"{'Healthy' if result['is_healthy'] else 'Diseased'}",
    ]
    if result.get("disease_percentage") is not None:
        lines.append(
            f"Affected Area: {result['disease_percentage']}%")
    if not result["is_healthy"]:
        lines += [
            f"Disease      : {result['disease_name']}",
            f"Severity     : "
            f"{cure_info.get('severity', 'Unknown')}",
        ]
    lines.append(f"Confidence   : {result['confidence']}%")

    if not result["is_healthy"]:
        lines += [
            "", "-" * 50,
            "SYMPTOMS:",
            cure_info.get("symptoms", "—"),
            "", "CAUSE:",
            cure_info.get("causes", "—"),
            "", "TREATMENT:",
        ]
        for i, step in enumerate(
                cure_info.get("cure", []), 1):
            lines.append(f"  {i}. {step}")
        lines += ["", "PREVENTION:",
                  cure_info.get("prevention", "—")]

    lines += ["", "-" * 50, "TOP-3 PREDICTIONS:"]
    for cls, conf in result.get("top_k_predictions", []):
        name = cls.replace("___", ": ").replace("_", " ")
        lines.append(f"  • {name:<40} {conf:.1f}%")
    lines += ["", "=" * 50,
              "Plant Disease AI — EfficientNet-B0",
              "Trained on PlantVillage + PlantDoc",
              "=" * 50]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    render_header()
    render_sidebar()

    if DEMO_MODE:
        st.markdown("""
        <div class="demo-banner">
        ⚠️ <b>Demo Mode</b> — No trained model found.
        Train first: <code>python models/train_model.py
        --dataset_path "your/path" --plants Tomato Potato
        Corn_(maize) Apple Grape Pepper,_bell
        --samples_per_class 200 --epochs 10</code>
        </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "📤 Upload a leaf image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded is None:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;
          border:2px dashed #cbd5e1;border-radius:12px;
          background:#f8fafc;margin-top:1rem">
            <div style="font-size:3rem">🍃</div>
            <div style="font-size:1.1rem;color:#475569;
              margin-top:.5rem;font-weight:500">
                Upload a leaf image to begin analysis
            </div>
            <div style="font-size:.9rem;color:#94a3b8;
              margin-top:.3rem">
                Supports JPG, PNG, JPEG, WEBP
            </div>
        </div>""", unsafe_allow_html=True)
        return

    # Load image
    image_bytes  = uploaded.read()
    original_img = Image.open(
        io.BytesIO(image_bytes)).convert("RGB")

    st.markdown("---")

    # Predict
    with st.spinner("🔍 Analysing leaf..."):
        predictor = load_predictor()
        result    = (predictor.predict(original_img)
                     if predictor else mock_predict(original_img))

    # Cure lookup
    disease_key = result.get("disease_key", "")
    cure_info   = DISEASE_CURE_DB.get(
        disease_key, DISEASE_CURE_DB["default"])

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "📋 Diagnosis & Treatment",
        "🖼️ Visual Analysis",
        "📊 Prediction Confidence",
    ])
    with tab1:
        render_result_cards(result, cure_info)
    with tab2:
        render_image_analysis(original_img, result)
    with tab3:
        st.markdown("#### Model Prediction Breakdown")
        render_confidence_chart(result)

    # Download report
    st.markdown("---")
    report = generate_report(result, cure_info)
    st.download_button(
        label="📥 Download Diagnosis Report (.txt)",
        data=report,
        file_name=f"diagnosis_{result['plant_name'].lower()}.txt",
        mime="text/plain",
    )


if __name__ == "__main__":
    main()