import asyncio
import sys
import hashlib

if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ANEES — CPAK System",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .header-box {
        background: linear-gradient(135deg,
            #1a3a5c 0%, #2d6a9f 100%);
        padding: 28px 32px;
        border-radius: 14px;
        margin-bottom: 28px;
        text-align: center;
    }
    .header-box h1 {
        color: white; font-size: 28px;
        margin: 0 0 4px 0; font-weight: 900;
        letter-spacing: 12px;
    }
    .header-box h2 {
        color: #cce4ff; font-size: 15px;
        margin: 0 0 4px 0; font-weight: 400;
        letter-spacing: 1px;
    }
    .header-box p {
        color: #a8c8f0; font-size: 12px; margin: 0;
    }
    .login-box {
        max-width: 420px;
        margin: 80px auto;
        padding: 40px;
        background: white;
        border-radius: 16px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.10);
    }
    .login-title {
        text-align: center;
        color: #1a3a5c;
        font-size: 26px;
        font-weight: 900;
        letter-spacing: 10px;
        margin-bottom: 6px;
    }
    .login-sub {
        text-align: center;
        color: #666;
        font-size: 12px;
        margin-bottom: 24px;
    }
    .disclaimer {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 13px;
        color: #92400e;
        margin-top: 16px;
    }
    .borderline-warning {
        background: #fef9c3;
        border: 2px solid #eab308;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 14px;
        color: #713f12;
        margin-top: 12px;
        font-weight: 600;
    }
    div[data-testid="stButton"] > button {
        width: 100%; height: 52px;
        font-size: 17px; font-weight: 600;
        background: linear-gradient(135deg,
            #1a3a5c, #2d6a9f);
        color: white; border: none;
        border-radius: 10px;
    }
    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg,
            #2d6a9f, #1a3a5c);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ── Authentication ─────────────────────────────────────────────────────────────
USERS = {
    "dr.arshad": hashlib.sha256(
        "hukm2026".encode()).hexdigest(),
    "eva.ray": hashlib.sha256(
        "ukm2026".encode()).hexdigest(),
    "admin": hashlib.sha256(
        "cpakadmin".encode()).hexdigest(),
}

def check_password(username, password):
    hashed = hashlib.sha256(
        password.encode()).hexdigest()
    return USERS.get(username) == hashed

def login_page():
    st.markdown("""
    <div class="login-box">
        <div class="login-title">
            A · N · E · E · S
        </div>
        <div style='text-align:center;
                    color:#1a3a5c;
                    font-size:13px;
                    font-weight:500;
                    margin-bottom:4px'>
            Artificial Neural Engine for
            Enhanced Scanogram Analysis
        </div>
        <div style='text-align:center;
                    color:#666;
                    font-size:12px;
                    margin-bottom:8px'>
            (Coronal Plane Alignment of the
            Knee — CPAK)
        </div>
        <div class="login-sub">
            Hospital Universiti Kebangsaan Malaysia
            <br>
            Restricted Access —
            Authorised Personnel Only
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown(
            "<h3 style='text-align:center;"
            "color:#1a3a5c;margin-bottom:20px'>"
            "🔒 Sign In</h3>",
            unsafe_allow_html=True)

        username = st.text_input(
            "Username",
            placeholder="Enter username",
            key="login_user")
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter password",
            key="login_pass")

        if st.button("Sign In", type="primary"):
            if check_password(username, password):
                st.session_state['authenticated'] = True
                st.session_state['username']      = username
                st.rerun()
            else:
                st.error("❌ Invalid username "
                          "or password.")

        st.markdown(
            "<p style='text-align:center;"
            "color:#999;font-size:12px;"
            "margin-top:16px'>"
            "⚠️ This system contains sensitive "
            "patient data. Unauthorised access "
            "is prohibited.</p>",
            unsafe_allow_html=True)

# ── Check authentication ───────────────────────────────────────────────────────
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    login_page()
    st.stop()

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_H, IMG_W = 512, 128
MODEL_DIR    = Path(__file__).parent

ALIGN_MAP = {
    1:"Varus",   2:"Neutral", 3:"Valgus",
    4:"Varus",   5:"Neutral", 6:"Valgus",
    7:"Varus",   8:"Neutral", 9:"Valgus",
}
ALIGN_FULL = {
    "Varus":   "Varus — Knee bows outward (Bowleg)",
    "Neutral": "Neutral — Normal mechanical alignment",
    "Valgus":  "Valgus — Knee bows inward (Knock-knee)",
}
JLO_MAP = {
    1:"Apex Distal (JLO ≤ 179°)",
    2:"Apex Distal (JLO ≤ 179°)",
    3:"Apex Distal (JLO ≤ 179°)",
    4:"Neutral (JLO = 180°)",
    5:"Neutral (JLO = 180°)",
    6:"Neutral (JLO = 180°)",
    7:"Apex Proximal (JLO ≥ 181°)",
    8:"Apex Proximal (JLO ≥ 181°)",
    9:"Apex Proximal (JLO ≥ 181°)",
}
EMOJI_MAP = {
    "Varus":"🔴", "Neutral":"🟢", "Valgus":"🔵"}
BADGE_COLS = {
    1:(180,30,30),  2:(30,140,30),  3:(30,100,180),
    4:(150,20,20),  5:(20,120,20),  6:(20,80,160),
    7:(120,10,10),  8:(10,100,10),  9:(10,60,140),
}

# ── Load ensemble models ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ensemble models...")
def load_models():
    models = []
    for name in ['unet_v2_s1.keras',
                  'unet_v2_s2.keras',
                  'unet_v2_s3.keras']:
        path = MODEL_DIR / name
        if path.exists():
            m = keras.models.load_model(str(path))
            models.append(m)
        else:
            print(f"Not found: {path}")
    return models

# ── Core functions ─────────────────────────────────────────────────────────────
def preprocess(img_bgr, side):
    clahe  = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8,8))
    gray   = cv2.cvtColor(img_bgr,
                           cv2.COLOR_BGR2GRAY)
    ih, iw = gray.shape
    if side == "LEFT":
        crop, x_off = gray[:, iw//2:], iw//2
    else:
        crop, x_off = gray[:, :iw//2], 0
    ch, cw = crop.shape
    tr = IMG_H/IMG_W; ar = ch/cw
    if ar > tr:
        nw=int(ch/tr); pw=(nw-cw)//2
        crop=np.pad(crop,((0,0),(pw,nw-cw-pw)),
                    constant_values=0)
        px,py,sc=pw,0,IMG_H/ch
    else:
        nh=int(cw*tr); ph=(nh-ch)//2
        crop=np.pad(crop,((ph,nh-ch-ph),(0,0)),
                    constant_values=0)
        px,py,sc=0,ph,IMG_H/nh
    crop=cv2.resize(crop,(IMG_W,IMG_H),
                    interpolation=cv2.INTER_AREA)
    crop=clahe.apply(crop)
    return (crop.astype(np.float32)/255.0)\
               [...,np.newaxis], px, py, sc, x_off

def hm_peak(hm):
    idx=np.argmax(hm)
    y,x=np.unravel_index(idx,hm.shape)
    return float(x),float(y)

def to_orig(cx,cy,px,py,sc,x_off):
    return int(cx/sc-px+x_off), int(cy/sc-py)

def calc_ahka(fh,knee,ankle,side):
    fh   =np.array(fh,   dtype=float)
    knee =np.array(knee, dtype=float)
    ankle=np.array(ankle,dtype=float)
    leg  =np.linalg.norm(ankle-fh)
    if leg<1: return 0.0
    ax,ay=ankle-fh; kx,ky=knee-fh
    cross=ax*ky-ay*kx
    fk   =np.linalg.norm(knee-fh)
    if fk<1: return 0.0
    sin_a=np.clip((cross/leg)/fk,-1,1)
    angle=np.degrees(np.arcsin(sin_a))
    return -angle if side=="RIGHT" else angle

def get_cpak_full(ahka, jlo):
    h = 0 if ahka<-3 else (2 if ahka>3 else 1)
    v = 0 if jlo<=179 else (2 if jlo>=181 else 1)
    return [[1,2,3],[4,5,6],[7,8,9]][v][h]

def is_borderline(ahka):
    return (abs(ahka-(-3))<=2 or
            abs(ahka-3)   <=2)

def ensemble_predict(img_bgr, side, models, jlo):
    inp,px,py,sc,x_off = preprocess(img_bgr, side)
    hm_sum = None
    for model in models:
        pred   = model.predict(
            inp[np.newaxis], verbose=0)[0]
        hm_sum = pred if hm_sum is None \
                  else hm_sum + pred
    hm_avg   = hm_sum / len(models)
    fh_pt    = to_orig(*hm_peak(hm_avg[:,:,0]),
                        px,py,sc,x_off)
    knee_pt  = to_orig(*hm_peak(hm_avg[:,:,1]),
                        px,py,sc,x_off)
    ankle_pt = to_orig(*hm_peak(hm_avg[:,:,2]),
                        px,py,sc,x_off)
    ahka     = calc_ahka(
        fh_pt,knee_pt,ankle_pt,side)
    cpak     = get_cpak_full(ahka, jlo)
    border   = is_borderline(ahka)
    return fh_pt,knee_pt,ankle_pt,ahka,cpak,border

def draw_annotated(img_bgr,fh,knee,ankle,
                   ahka,cpak,borderline):
    vis = img_bgr.copy()
    h,w = vis.shape[:2]
    sc  = min(700/h, 1.5)
    vis = cv2.resize(vis,(int(w*sc),int(h*sc)))

    def sp(p):
        return (int(p[0]*sc),int(p[1]*sc))

    F,K,A = sp(fh),sp(knee),sp(ankle)

    axis_col=(0,200,255) if borderline \
              else (0,220,0)
    cv2.line(vis,F,A,axis_col,2,cv2.LINE_AA)
    cv2.line(vis,F,K,(0,180,255),2,cv2.LINE_AA)
    cv2.line(vis,K,A,(0,180,255),2,cv2.LINE_AA)

    for pt,col,lbl in [
        (F,(0,255,255),"FH"),
        (K,(255,0,200),"KNEE"),
        (A,(0,255,100),"ANKLE"),
    ]:
        cv2.circle(vis,pt,9,col,-1)
        cv2.circle(vis,pt,10,(0,0,0),1)
        cv2.putText(vis,lbl,(pt[0]+12,pt[1]+5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,col,2,cv2.LINE_AA)

    cv2.putText(
        vis,f"aHKA: {ahka:+.1f}\u00b0",
        (K[0]+15,K[1]-18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,(255,255,255),2,cv2.LINE_AA)

    txt=f" CPAK Type {cpak}  |  {ALIGN_MAP[cpak]} "
    (tw,th),_=cv2.getTextSize(
        txt,cv2.FONT_HERSHEY_SIMPLEX,0.75,2)
    cv2.rectangle(vis,(8,8),(tw+18,th+22),
                  BADGE_COLS.get(cpak,(80,80,80)),
                  -1)
    cv2.putText(vis,txt,(13,32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,(255,255,255),2,cv2.LINE_AA)

    if borderline:
        cv2.putText(vis,"BORDERLINE - VERIFY",
                    (8,th+45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,(0,200,255),2,cv2.LINE_AA)

    return cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)

# ── Header with logout ─────────────────────────────────────────────────────────
st.markdown('<div id="top"></div>',
             unsafe_allow_html=True)

header_col, logout_col = st.columns([5,1])
with header_col:
    st.markdown("""
    <div class="header-box">
        <h1>A · N · E · E · S</h1>
        <h2>Artificial Neural Engine for Enhanced
        Scanogram Analysis</h2>
        <p>(Coronal Plane Alignment of the Knee
        — CPAK)</p>
    </div>
    """, unsafe_allow_html=True)

with logout_col:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        f"👤 **{st.session_state['username']}**")
    if st.button("🔓 Logout"):
        st.session_state['authenticated'] = False
        st.session_state['username']      = ''
        st.rerun()

# Load models
models = load_models()
if not models:
    st.error("❌ No models found. Place "
             "unet_v2_s1.keras, unet_v2_s2.keras, "
             "unet_v2_s3.keras in the same folder "
             "as app.py")
    st.stop()

st.success(f"✅ {len(models)} ensemble "
            f"model(s) loaded.")

col_left, col_right = st.columns(2, gap="large")

# ── Left column ────────────────────────────────────────────────────────────────
with col_left:
    st.markdown("### 📤 Upload Image")

    uploaded = st.file_uploader(
        "Select plain X-ray scanogram (PNG/JPG)",
        type=["png","jpg","jpeg"],
        label_visibility="collapsed",
    )
    if uploaded:
        st.image(uploaded,
                 caption="Uploaded scanogram",
                 use_column_width=True)

    side_choice = st.radio(
        "🦵 Select leg to analyse:",
        ["Left Leg", "Right Leg"],
        horizontal=True,
    )

    st.markdown("**Joint Line Obliquity (JLO)**")
    st.caption(
        "Enter the JLO value from PACS measurement "
        "to classify all 9 CPAK types. "
        "Default 175 = Apex Distal (most common).")
    jlo_input = st.number_input(
        "JLO value (°)",
        min_value=160, max_value=200,
        value=175, step=1,
        help="JLO ≤ 179 = Apex Distal | "
             "JLO = 180 = Neutral | "
             "JLO ≥ 181 = Apex Proximal")

    if jlo_input <= 179:
        st.info(f"JLO = {jlo_input}° → "
                f"**Apex Distal** (Types 1, 2, 3)")
    elif jlo_input == 180:
        st.info(f"JLO = {jlo_input}° → "
                f"**Neutral** (Types 4, 5, 6)")
    else:
        st.info(f"JLO = {jlo_input}° → "
                f"**Apex Proximal** (Types 7, 8, 9)")

    analyze = st.button(
        "🔍  Analyse Knee Alignment",
        type="primary")

    st.markdown("""
    **Instructions:**
    1. Upload a plain X-ray full-leg scanogram
    2. Select the leg side
    3. Enter JLO value from PACS (optional)
    4. Click **Analyse Knee Alignment**
    5. Review results on the right
    """)

# ── Right column ───────────────────────────────────────────────────────────────
with col_right:
    st.markdown("### 📋 Analysis Results")

    if uploaded and analyze:

        # Scroll to top
        st.markdown("""
        <script>
        document.getElementById('top')
            .scrollIntoView({behavior:'smooth'});
        </script>
        """, unsafe_allow_html=True)

        with st.spinner(
            "Running ensemble inference — "
            "please wait..."):
            try:
                file_bytes = np.frombuffer(
                    uploaded.read(), dtype=np.uint8)
                img_bgr    = cv2.imdecode(
                    file_bytes, cv2.IMREAD_COLOR)
                side = "LEFT" \
                       if "Left" in side_choice \
                       else "RIGHT"

                (fh_pt, knee_pt, ankle_pt,
                 ahka, cpak, borderline) = \
                    ensemble_predict(
                        img_bgr, side,
                        models, jlo_input)

                align     = ALIGN_MAP[cpak]
                jlo_label = JLO_MAP[cpak]
                result_img= draw_annotated(
                    img_bgr,fh_pt,knee_pt,
                    ankle_pt,ahka,cpak,
                    borderline)

                st.image(
                    result_img,
                    caption="Annotated image with "
                            "detected landmarks "
                            "and mechanical axis",
                    use_column_width=True)

                m1,m2,m3 = st.columns(3)
                m1.metric("CPAK Type",
                           f"Type {cpak}")
                m2.metric("Alignment", align)
                m3.metric("aHKA Angle",
                           f"{ahka:+.2f}°")

                if borderline:
                    st.markdown("""
<div class="borderline-warning">
⚠️ <strong>BORDERLINE CASE:</strong>
Predicted aHKA is within ±2° of the ±3°
classification boundary. Alignment classification
may be uncertain.
<strong>Manual verification strongly
recommended.</strong>
</div>
""", unsafe_allow_html=True)

                with st.expander(
                    "📄 Detailed Report",
                    expanded=True):
                    st.markdown(f"""
**{EMOJI_MAP[align]} Diagnosis:**
{ALIGN_FULL[align]}

| Parameter | Value |
|-----------|-------|
| CPAK Type | **Type {cpak}** |
| Alignment (aHKA axis) | {align} |
| Joint Line (JLO axis) | {jlo_label} |
| aHKA Angle | {ahka:+.2f}° |
| JLO (entered) | {jlo_input}° |
| Borderline | {"⚠️ Yes — verify manually" if borderline else "✅ No"} |

**Detected Landmark Coordinates (pixels):**

| Landmark | x | y |
|----------|---|---|
| Femoral Head (FH) | {fh_pt[0]} | {fh_pt[1]} |
| Knee Joint Centre | {knee_pt[0]} | {knee_pt[1]} |
| Ankle Centre | {ankle_pt[0]} | {ankle_pt[1]} |

**Model:** Ensemble of {len(models)} U-Nets
(arithmetic mean of heatmaps)
                    """)

                st.markdown("""
<div class="disclaimer">
⚠️ <strong>Important Notice:</strong>
This system is a research prototype only
and has not been validated for clinical use.
All results must be verified by a qualified
medical professional before any clinical
decision is made.
</div>
""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    elif uploaded and not analyze:
        st.info("👆 Click **Analyse** to proceed.")

    else:
        st.info(
            "👈 Upload a scanogram to get started.")

        with st.expander(
            "📖 CPAK 9-Type Classification Matrix",
            expanded=True):
            st.markdown("""
| | **Varus** (aHKA<−3°) | **Neutral** (−3°≤aHKA≤3°) | **Valgus** (aHKA>3°) |
|--|--|--|--|
| **Apex Distal** (JLO≤179°) | Type I | Type II | Type III |
| **Neutral** (JLO=180°) | Type IV | Type V | Type VI |
| **Apex Proximal** (JLO≥181°) | Type VII | Type VIII | Type IX |

*aHKA is predicted by AI. JLO is entered manually.*
            """)

        with st.expander("📊 Model Performance"):
            st.markdown("""
| Metric | Value |
|--------|-------|
| Dataset | 495 patients (HUKM) |
| Architecture | Ensemble of 3 U-Nets |
| Regularisation | L2 + SpatialDropout2D + Early Stopping |
| Augmentation | Flip + Zoom + Brightness + Noise |
| Valgus Oversampling | 10× heavy augmentation |
| 3-Class Accuracy | **79.8%** |
| 9-Type Accuracy | **74.7%** |
| aHKA MAE | **4.23°** |
| Macro-F1 | **0.773** |
| Cohen's Kappa | **0.626** |
| SVM (C=100, RBF) | **93.9%** |
| Mathematical Ceiling | **86.9%** |
| Borderline accuracy | **73.4%** |
| Non-borderline accuracy | **91.4%** |
            """)
