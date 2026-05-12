import os
import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

# -------------------- SETTINGS --------------------
MODEL_PATH = "mnist_model.keras"
SAVE_DIR = "collected_mistakes"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_digit_model():
    model = load_model(MODEL_PATH)
    return model

model = load_digit_model()

# -------------------- PREPROCESSING --------------------
def preprocess_and_segment_rgba_image(rgba_image):

    # Convert RGBA to grayscale
    gray = cv2.cvtColor(rgba_image.astype("uint8"), cv2.COLOR_RGBA2GRAY)

    # Invert image (white digit on black background)
    gray = 255 - gray

    # Blur slightly
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Binary threshold
    _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Ignore tiny noise
        if w*h < 100:
            continue

        boxes.append((x, y, w, h))

    # Sort left to right
    boxes = sorted(boxes, key=lambda b: b[0])

    results = []

    for (x, y, w, h) in boxes:

        roi = thresh[y:y+h, x:x+w]

        # Create square image
        size = max(w, h) + 20

        square = np.zeros((size, size), dtype=np.uint8)

        x_offset = (size - w) // 2
        y_offset = (size - h) // 2

        square[y_offset:y_offset+h, x_offset:x_offset+w] = roi

        # Resize to MNIST size
        resized = cv2.resize(square, (28,28))

        # Normalize
        normalized = resized.astype("float32") / 255.0

        normalized = normalized.reshape(1,28,28,1)

        results.append((x,y,w,h,normalized,resized))

    return results, thresh

# -------------------- PREDICTION --------------------
def predict_sequence_and_render(image_rgba, model):
    segs, thresh = preprocess_and_segment_rgba_image(image_rgba)
    preview_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    predicted, confidences = [], []
    for (x,y,w,h,roi_norm,roi_resized) in segs:
        pred = model.predict(roi_norm, verbose=0)[0]
        digit = int(np.argmax(pred))
        conf = float(np.max(pred))
        predicted.append(str(digit))
        confidences.append(conf)
        cv2.rectangle(preview_rgb, (x,y), (x+w, y+h), (255, 120, 0), 2)
        cv2.putText(
            preview_rgb,
            f"{digit} ({conf*100:.1f}%)",
            (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 60, 120), 2, cv2.LINE_AA)
    return preview_rgb, "".join(predicted), confidences

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Digit Sequence Recognizer ✨", page_icon="✏️", layout="wide")

# -------------------- BEAUTIFUL CSS --------------------
st.markdown("""
<style>

/* ---------- MAIN APP ---------- */

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #111827,
        #1e293b
    );
    color: white;
    overflow-x: hidden;
}

/* ---------- HIDE STREAMLIT DEFAULT ---------- */

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ---------- TITLE ---------- */

.main-title {
    text-align: center;
    font-size: 3.5rem;
    font-weight: 800;
    margin-top: -20px;
    background: linear-gradient(
        to right,
        #38bdf8,
        #818cf8,
        #ec4899
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: glow 3s infinite ease-in-out;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 1.1rem;
    margin-bottom: 30px;
}

/* ---------- ANIMATION ---------- */

@keyframes glow {
    0%,100% {
        filter: drop-shadow(0px 0px 10px #38bdf8);
    }
    50% {
        filter: drop-shadow(0px 0px 25px #818cf8);
    }
}

/* ---------- GLASS CARD ---------- */

.glass-card {

    background: rgba(255,255,255,0.08);

    border-radius: 24px;

    padding: 28px;

    backdrop-filter: blur(14px);

    border: 1px solid rgba(255,255,255,0.1);

    box-shadow:
        0px 8px 32px rgba(0,0,0,0.35);

    transition: 0.4s ease;
}

.glass-card:hover {

    transform: translateY(-4px);

    box-shadow:
        0px 12px 40px rgba(56,189,248,0.2);
}

/* ---------- BUTTON ---------- */

.stButton > button {

    width: 100%;

    height: 52px;

    border-radius: 14px;

    border: none;

    font-size: 17px;

    font-weight: 700;

    color: white;

    background: linear-gradient(
        135deg,
        #06b6d4,
        #3b82f6
    );

    transition: 0.3s;
}

.stButton > button:hover {

    transform: scale(1.03);

    box-shadow:
        0px 0px 20px rgba(59,130,246,0.6);
}

/* ---------- PREDICTION BOX ---------- */

.prediction-box {

    background: linear-gradient(
        135deg,
        #8b5cf6,
        #ec4899
    );

    padding: 24px;

    border-radius: 20px;

    text-align: center;

    font-size: 2rem;

    font-weight: bold;

    margin-top: 25px;

    box-shadow:
        0px 0px 25px rgba(236,72,153,0.35);

    animation: pulse 2s infinite;
}

@keyframes pulse {

    0%,100% {
        transform: scale(1);
    }

    50% {
        transform: scale(1.02);
    }
}

/* ---------- CONFIDENCE LABEL ---------- */

.confidence-label {

    margin-top: 12px;

    font-size: 15px;

    color: #cbd5e1;
}

/* ---------- FOOTER ---------- */

.footer {

    text-align: center;

    margin-top: 50px;

    color: #94a3b8;

    font-size: 0.95rem;
}

/* ---------- SCROLLBAR ---------- */

::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: #111827;
}

::-webkit-scrollbar-thumb {
    background: #3b82f6;
    border-radius: 10px;
}

/* ---------- SIDEBAR ---------- */

section[data-testid="stSidebar"] {

    background: rgba(15,23,42,0.95);

    border-right:
        1px solid rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.title("⚙️ Controls")

    stroke_size = st.slider(
        "Pen Thickness",
        1,
        20,
        8
    )

    st.markdown("---")

    st.info("""
    ✨ Tips:
    
    • Write clearly
    
    • Leave spaces between digits
    
    • Avoid overlapping
    
    • Use center area
    """)

    st.markdown("---")

    st.success("✅ AI Model Loaded")

# -------------------- MAIN APP --------------------
st.markdown("<h1>🧠 Handwritten Digit Sequence Recognition</h1>", unsafe_allow_html=True)

st.markdown("<div class='glass-card'>Draw your digits below — e.g. <b>12345</b> — with clear gaps between each digit.</div>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("🖊️ Draw Here")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=stroke_size,
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=420,
        height=160,
        drawing_mode="freedraw",
        key="canvas",
    )

    predict_btn = st.button("🔍 Predict Sequence")
    clear_btn = st.button("🗑️ Clear Canvas")

    if clear_btn:
        st.rerun()
    save_btn = st.button("💾 Save Incorrect Sample")

with col2:
    if predict_btn:
        if canvas_result.image_data is None:
            st.warning("Please draw something first!")
        else:
            img_rgba = canvas_result.image_data.astype("uint8")
            with st.spinner("🤖 AI is analyzing handwriting..."):

                preview, seq, confs = predict_sequence_and_render(
                    img_rgba,
                    model
            )
            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                     caption="Detected Digits & Confidence Levels", use_column_width=True)
            if seq:
                st.markdown(
                    f"<div class='prediction-box'>Prediction: {seq}</div>",
                    unsafe_allow_html=True
               )
                st.subheader("📊 Confidence Scores")

                for i, conf in enumerate(confs):

                    st.progress(conf)

                    st.markdown(
                        f"<div class='confidence-label'>Digit <b>{seq[i]}</b> → {conf*100:.2f}% confidence</div>",
                        unsafe_allow_html=True
                )
            else:
                st.warning("No digits detected — try writing more clearly.")

    if save_btn:
        if canvas_result.image_data is None:
            st.warning("Draw something first!")
        else:
            img_rgba = canvas_result.image_data.astype("uint8")
            segs, _ = preprocess_and_segment_rgba_image(img_rgba)
            base_name = os.path.join(SAVE_DIR, f"sample_{len(os.listdir(SAVE_DIR))+1}")
            for idx, (_,_,_,_,_,roi_resized) in enumerate(segs):
                cv2.imwrite(f"{base_name}_{idx}.png", roi_resized)
            st.success(f"✅ Saved {len(segs)} incorrect samples to `{SAVE_DIR}`.")

st.markdown(
    "<div class='main-title'>🧠 AI Handwritten Digit Recognizer</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Deep Learning + TensorFlow + OpenCV + Streamlit</div>",
    unsafe_allow_html=True
)