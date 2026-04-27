import streamlit as st
import cv2
import numpy as np
import subprocess
import os
import re
import tempfile
from PIL import Image
from pathlib import Path
import base64

# ─── PAGE CONFIG ─────────────────────────────────────────
st.set_page_config(page_title="VectoLine", layout="wide")

# ─── CSS ────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --bg: #0e0e11;
    --bg2: #17171c;
    --bg3: #1e1e26;
    --accent: #7cfc8e;
    --text: #ffffff;
}

/* Background */
body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Remove spinner box */
.stSpinner,
[data-testid="stSpinnerContainer"],
[data-testid="stStatusWidget"] {
    background: transparent !important;
    border: none !important;
}

/* Hide default progress bar */
[data-testid="stProgress"] {
    display: none !important;
}

/* Slider */
[data-baseweb="slider"] > div > div > div:first-child {
    background: var(--accent) !important;
}
[data-baseweb="slider"] > div > div > div:nth-child(3) {
    background: #2a2a38 !important;
}

/* Slider knob */
[data-testid="stSlider"] [role="slider"] {
    background: var(--accent) !important;
    border: 2px solid var(--accent) !important;
}

/* Slider value text */
[data-testid="stSlider"] div {
    color: var(--accent) !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── FUNCTIONS ──────────────────────────────────────────
def image_to_line_art(img, mode='canny', threshold=128):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if mode == 'canny':
        edges = cv2.Canny(gray, 100, 200)
        return cv2.bitwise_not(edges)

    elif mode == 'adaptive':
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2)

    elif mode == 'xdog':
        g1 = cv2.GaussianBlur(gray.astype(float), (0,0), 1.4)
        g2 = cv2.GaussianBlur(gray.astype(float), (0,0), 2.0)
        xdog = np.where((g1-g2)>=0.01, 1.0, 1.0 + np.tanh(20*(g1-g2)))
        xdog = (xdog*255).astype(np.uint8)
        _, res = cv2.threshold(xdog, threshold, 255, cv2.THRESH_BINARY)
        return res

def line_art_to_svg(line_art, out_svg):
    with tempfile.NamedTemporaryFile(suffix='.pbm', delete=False) as f:
        pbm = f.name
    Image.fromarray(line_art).convert('1').save(pbm)
    subprocess.run(['potrace', pbm, '--svg', '-o', out_svg])

def simplify_svg(content):
    return content  # simplified placeholder

# ─── UI ─────────────────────────────────────────────────
st.title("⚡ VectoLine")

mode = st.selectbox("Mode", ["canny", "adaptive", "xdog"])
epsilon = st.slider("RDP EPSILON", 0.5, 5.0, 1.0, 0.5)
threshold = st.slider("THRESHOLD", 64, 200, 128, 8)

uploaded = st.file_uploader("Upload image", type=["png","jpg","jpeg"])

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Original", use_container_width=True)

    if st.button("VECTORIZE"):
        status = st.empty()

        with st.spinner(""):
            status.markdown("⚡ Vectorizing... **33%**")

            line_art = image_to_line_art(img, mode, threshold)

            status.markdown("Tracing SVG... **66%**")

            with tempfile.TemporaryDirectory() as tmp:
                raw_svg = os.path.join(tmp, "out.svg")
                line_art_to_svg(line_art, raw_svg)

                status.markdown("Simplifying... **90%**")

                content = open(raw_svg).read()
                content = simplify_svg(content)
                svg_bytes = content.encode()

            status.markdown("✅ Done! **100%**")

        st.image(line_art, caption="Line Art", use_container_width=True)

        b64 = base64.b64encode(svg_bytes).decode()
        st.markdown(f'<img src="data:image/svg+xml;base64,{b64}" style="width:100%;background:white"/>', unsafe_allow_html=True)

        st.download_button("Download SVG", svg_bytes, "output.svg")
