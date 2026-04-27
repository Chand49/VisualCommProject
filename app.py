import streamlit as st
import cv2
import numpy as np
import subprocess
import os
import re
import tempfile
import zipfile
import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path
import io
import base64

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VectoLine — Image Vectorizer",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Force sidebar open via JS (fixes hidden-toggle bug on Streamlit Cloud)
st.markdown("""
<script>
(function() {
    function openSidebar() {
        var btn = window.parent.document.querySelector('[data-testid="collapsedControl"]');
        if (btn) { btn.click(); }
    }
    setTimeout(openSidebar, 400);
})();
</script>
""", unsafe_allow_html=True)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ── Root & Background ── */
:root {
    --bg:        #0e0e11;
    --bg2:       #17171c;
    --bg3:       #1e1e26;
    --accent:    #7cfc8e;
    --accent2:   #4af0c8;
    --accent3:   #ffbb55;
    --muted:     #555566;
    --text:      #cccce0;
    --border:    #2a2a38;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: var(--bg2) !important;
    border-right: 1px solid var(--border);
}

/* ── Header Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0e0e11 0%, #17171c 50%, #1a1a24 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(124,252,142,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 200px;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(74,240,200,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    color: var(--accent);
    letter-spacing: -1px;
    margin: 0 0 0.3rem 0;
    line-height: 1;
}
.hero-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #c8c8e0;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0 0 1rem 0;
}
.pipeline-badge {
    display: inline-block;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.4rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent2);
    letter-spacing: 1px;
}
.pipeline-badge span { color: var(--muted); margin: 0 6px; }

/* ── Metric Cards ── */
.metric-row {
    display: flex;
    gap: 12px;
    margin: 1.5rem 0;
}
.metric-card {
    flex: 1;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.metric-lbl {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
}

/* ── Mode Cards in Sidebar ── */
.mode-card {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
    color: var(--text);
}
.mode-card strong { color: var(--accent); font-family: 'Space Mono', monospace; font-size: 0.78rem; }
.mode-card .tag {
    display: inline-block;
    background: rgba(124,252,142,0.1);
    border: 1px solid rgba(124,252,142,0.2);
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.68rem;
    color: var(--accent);
    margin-top: 4px;
}

/* ── Section Headers ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    border-left: 3px solid var(--accent);
    padding-left: 8px;
}

/* ── Image Panels ── */
.img-panel {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
}
.img-panel-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.8rem;
}

/* ── Streamlit Widget Overrides ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSlider"] > div {
    background-color: var(--bg3) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
label, .stSlider label, .stSelectbox label {
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px;
}
[data-testid="stFileUploader"] {
    background: var(--bg3) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
}
.stButton > button {
    background: var(--accent) !important;
    color: #0e0e11 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    letter-spacing: 1.5px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    text-transform: uppercase !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: #a0ffb0 !important;
    transform: translateY(-1px) !important;
}
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: var(--accent2) !important;
    border: 1px solid var(--accent2) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.2rem !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(74,240,200,0.08) !important;
}
.stAlert {
    background: var(--bg3) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
}
/* Progress bar — filled green, track dark, text always white */
[data-testid="stProgress"] > div {
    background-color: var(--bg3) !important;
    border-radius: 4px !important;
}
[data-testid="stProgress"] > div > div {
    background-color: var(--accent) !important;
}
[data-testid="stProgress"] p,
[data-testid="stProgress"] ~ p,
[data-testid="stProgress"] + p {
    color: var(--text) !important;
    background: transparent !important;
}
/* Slider — thumb green */
[data-testid="stSlider"] [role="slider"] {
    background-color: var(--accent) !important;
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,252,142,0.25) !important;
}
/* Slider filled (left) track — green */
[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div:first-child {
    background: var(--accent) !important;
}
/* Slider unfilled (right) track — dim */
[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div:nth-child(3) {
    background: #3a3a50 !important;
}
/* Slider current value label — green */
[data-testid="stSlider"] div[data-baseweb="slider"] > div:last-child div,
[data-testid="stSlider"] [data-baseweb="slider"] > div:last-child {
    color: var(--accent) !important;
}
/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    color: var(--muted) !important;
    letter-spacing: 1px;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}
/* Spinner / status box — keep dark background, no green */
[data-testid="stStatusWidget"],
[data-testid="stSpinner"],
div[data-testid="stAlert"][kind="info"],
.stSpinner > div,
[data-baseweb="notification"] {
    background-color: var(--bg2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
/* st.spinner wrapper that Streamlit wraps in a light/colored box */
.stSpinner,
[data-testid="stSpinnerContainer"],
[data-testid="stSpinnerContainer"] > div {
    background: var(--bg2) !important;
    background-color: var(--bg2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
[data-testid="stStatusWidget"] > div,
[data-testid="stStatusWidget"] svg {
    color: var(--accent) !important;
    fill: var(--accent) !important;
    background: transparent !important;
}
/* Progress bar text label — always white, above the bar, never inside it */
[data-testid="stProgress"] {
    display: flex !important;
    flex-direction: column-reverse !important;
}
[data-testid="stProgress"] p {
    color: var(--text) !important;
    background: transparent !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    margin-bottom: 4px !important;
}
[data-testid="stProgress"] > div {
    position: relative !important;
    z-index: 0 !important;
}
/* Progress text inside spinner steps */
[data-testid="stProgress"] + div,
[data-testid="stProgressText"] {
    color: var(--text) !important;
    background: transparent !important;
}
/* Expander */
[data-testid="stExpander"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
/* Divider */
hr { border-color: var(--border) !important; }

/* Hide default Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Hide header bar visually but keep it in layout so toggle stays accessible */
header[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
}
/* Keep only the deploy/share buttons hidden, show sidebar toggle */
header[data-testid="stHeader"] > div:first-child {
    visibility: hidden;
}
/* Sidebar collapse/expand arrow — always visible */
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    color: var(--accent) !important;
}
[data-testid="collapsedControl"] svg path {
    fill: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Pipeline Functions ──────────────────────────────────────────────────────

def image_to_line_art(img_array, mode='canny', threshold=128):
    """Convert a numpy BGR image to binary line-art."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    if mode == 'canny':
        otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(gray, otsu * 0.5, otsu)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        return cv2.bitwise_not(edges)
    elif mode == 'adaptive':
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif mode == 'xdog':
        g1 = cv2.GaussianBlur(gray.astype(float), (0, 0), 1.4)
        g2 = cv2.GaussianBlur(gray.astype(float), (0, 0), 1.4 * 1.6)
        xdog = np.where((g1 - g2) >= 0.01, 1.0, 1.0 + np.tanh(20.0 * (g1 - g2)))
        xdog = (xdog * 255).clip(0, 255).astype(np.uint8)
        _, result = cv2.threshold(xdog, threshold, 255, cv2.THRESH_BINARY)
        return result
    raise ValueError(f"Unknown mode: {mode}")


def line_art_to_svg(line_art, out_svg):
    """Trace binary bitmap → SVG via potrace."""
    with tempfile.NamedTemporaryFile(suffix='.pbm', delete=False) as f:
        pbm = f.name
    try:
        Image.fromarray(line_art).convert('1').save(pbm)
        r = subprocess.run(
            ['potrace', pbm, '--svg', '-o', out_svg,
             '--turdsize', '2', '--alphamax', '1.0', '--opttolerance', '0.2'],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
    finally:
        if os.path.exists(pbm):
            os.remove(pbm)


def _tokenize(d):
    return re.findall(
        r'[MmCcLlZzHhVvSsQqTtAa]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', d)


def _cubic(p0, p1, p2, p3, n=8):
    t = np.linspace(0, 1, n + 1)
    u = 1 - t
    x = u**3*p0[0] + 3*u**2*t*p1[0] + 3*u*t**2*p2[0] + t**3*p3[0]
    y = u**3*p0[1] + 3*u**2*t*p1[1] + 3*u*t**2*p2[1] + t**3*p3[1]
    return list(zip(x.tolist(), y.tolist()))


def _rdp(pts, eps):
    if len(pts) < 3:
        return pts
    arr  = np.array(pts)
    keep = np.zeros(len(arr), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(arr) - 1)]
    while stack:
        s, e = stack.pop()
        if e - s < 2:
            continue
        seg     = arr[e] - arr[s]
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-10:
            dists = np.linalg.norm(arr[s+1:e] - arr[s], axis=1)
        else:
            u       = seg / seg_len
            vecs    = arr[s+1:e] - arr[s]
            proj    = np.clip(vecs @ u, 0, seg_len)
            closest = arr[s] + np.outer(proj, u)
            dists   = np.linalg.norm(arr[s+1:e] - closest, axis=1)
        idx = int(np.argmax(dists))
        if dists[idx] > eps:
            mid = s + 1 + idx
            keep[mid] = True
            stack.append((s, mid))
            stack.append((mid, e))
    return arr[keep].tolist()


def _parse_d_to_subpaths(d):
    tokens   = _tokenize(d)
    if not tokens:
        return []
    tok_cmds = [(i, t) for i, t in enumerate(tokens) if re.match(r'^[A-Za-z]$', t)]
    blocks   = []
    for ci, (ti, cmd_char) in enumerate(tok_cmds):
        next_ti      = tok_cmds[ci+1][0] if ci+1 < len(tok_cmds) else len(tokens)
        nums_for_cmd = [float(t) for t in tokens[ti+1:next_ti]
                        if re.match(r'^[-+]?(?:\d+\.?\d*|\.\d+)', t)]
        blocks.append((cmd_char, nums_for_cmd))

    subpaths = []
    cur      = [0.0, 0.0]
    sub_pts  = []
    closed   = False

    for cmd_char, nums in blocks:
        if cmd_char == 'M':
            if sub_pts: subpaths.append((sub_pts, closed))
            cur = [nums[0], nums[1]]; sub_pts = [tuple(cur)]; closed = False
            j = 2
            while j + 1 <= len(nums) - 1:
                cur = [nums[j], nums[j+1]]; sub_pts.append(tuple(cur)); j += 2
        elif cmd_char == 'm':
            if sub_pts: subpaths.append((sub_pts, closed))
            cur = [cur[0]+nums[0], cur[1]+nums[1]]; sub_pts = [tuple(cur)]; closed = False
            j = 2
            while j + 1 <= len(nums) - 1:
                cur = [cur[0]+nums[j], cur[1]+nums[j+1]]; sub_pts.append(tuple(cur)); j += 2
        elif cmd_char == 'C':
            j = 0
            while j + 6 <= len(nums):
                p0 = tuple(cur); p1 = (nums[j], nums[j+1])
                p2 = (nums[j+2], nums[j+3]); p3 = (nums[j+4], nums[j+5])
                sub_pts.extend(_cubic(p0, p1, p2, p3, n=8)[1:]); cur = list(p3); j += 6
        elif cmd_char == 'c':
            j = 0
            while j + 6 <= len(nums):
                p0 = tuple(cur)
                p1 = (cur[0]+nums[j], cur[1]+nums[j+1])
                p2 = (cur[0]+nums[j+2], cur[1]+nums[j+3])
                p3 = (cur[0]+nums[j+4], cur[1]+nums[j+5])
                sub_pts.extend(_cubic(p0, p1, p2, p3, n=8)[1:]); cur = list(p3); j += 6
        elif cmd_char == 'L':
            j = 0
            while j + 2 <= len(nums):
                cur = [nums[j], nums[j+1]]; sub_pts.append(tuple(cur)); j += 2
        elif cmd_char == 'l':
            j = 0
            while j + 2 <= len(nums):
                cur = [cur[0]+nums[j], cur[1]+nums[j+1]]; sub_pts.append(tuple(cur)); j += 2
        elif cmd_char in ('Z', 'z'):
            closed = True

    if sub_pts:
        subpaths.append((sub_pts, closed))
    return subpaths


def _simplify_d(d, eps):
    subpaths = _parse_d_to_subpaths(d)
    if not subpaths:
        return d, 0, 0
    out_parts  = []
    pts_before = 0
    pts_after  = 0
    for pts, closed in subpaths:
        pts_before += len(pts)
        if len(pts) < 2: continue
        simplified  = _rdp(pts, eps)
        pts_after  += len(simplified)
        if len(simplified) < 2: continue
        s    = simplified
        part = f"M {s[0][0]:.2f},{s[0][1]:.2f} L " + \
               ' '.join(f'{x:.2f},{y:.2f}' for x, y in s[1:])
        if closed: part += ' Z'
        out_parts.append(part)
    return (' '.join(out_parts) if out_parts else d), pts_before, pts_after


def simplify_svg_content(content, epsilon=1.0):
    total_before = 0
    total_after  = 0
    def _replace(m):
        nonlocal total_before, total_after
        simplified, pb, pa = _simplify_d(m.group(1), epsilon)
        total_before += pb; total_after += pa
        return f'd="{simplified}"'
    result = re.sub(r'd="([^"]+)"', _replace, content)
    return result, total_before, total_after


def run_pipeline(img_array, mode='canny', threshold=128, epsilon=1.0):
    """Full pipeline: numpy BGR → line art → SVG bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        raw_svg   = os.path.join(tmp, 'raw.svg')
        final_svg = os.path.join(tmp, 'final.svg')

        line_art = image_to_line_art(img_array, mode=mode, threshold=threshold)
        line_art_to_svg(line_art, raw_svg)

        size_kb    = os.path.getsize(raw_svg) / 1024
        path_count = len(re.findall(r'<path', open(raw_svg).read()))

        content = open(raw_svg).read()
        simplified, before, after = simplify_svg_content(content, epsilon)
        open(final_svg, 'w').write(simplified)

        svg_bytes = open(final_svg, 'rb').read()
        reduction = (1 - after / before) * 100 if before > 0 else 0

    return svg_bytes, line_art, {
        'size_kb': size_kb,
        'paths': path_count,
        'pts_before': before,
        'pts_after': after,
        'reduction': reduction,
    }


# ─── Sidebar (informational only — settings moved inline) ────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 0.5rem 0;">
        <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
                    color:#555566;letter-spacing:3px;text-transform:uppercase;
                    margin-bottom:1rem;">⚙ Pipeline Settings</div>
        <div style="font-size:0.75rem;color:#aaaacc;line-height:1.6;">
            Use the <b style="color:#7cfc8e;">⚙ PIPELINE SETTINGS</b> expander
            in the main panel to configure mode, epsilon and threshold.
        </div>
    </div>
    <hr style="border-color:#2a2a38;">
    <div class="mode-card">
        <strong>CANNY</strong><br>
        Edge detection via Canny algorithm. Great for photos and complex scenes.<br>
        <span class="tag">photos</span> <span class="tag">detailed</span>
    </div>
    <div class="mode-card">
        <strong>ADAPTIVE</strong><br>
        Adaptive thresholding. Best for manuscripts, sketches, printed documents.<br>
        <span class="tag">documents</span> <span class="tag">sketches</span>
    </div>
    <div class="mode-card">
        <strong>XDOG</strong><br>
        Extended Difference of Gaussians. Stylized, artistic line art.<br>
        <span class="tag">artistic</span> <span class="tag">stylized</span>
    </div>
    <hr style="border-color:#2a2a38;">
    <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
                color:#555566;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.8rem;">
        ε — Epsilon Guide
    </div>
    <div style="font-size:0.75rem;color:#aaaacc;line-height:1.8;">
        <b style="color:#7cfc8e;">0.5</b> — Fine detail, large file<br>
        <b style="color:#7cfc8e;">1.0</b> — Balanced (default)<br>
        <b style="color:#7cfc8e;">3.0</b> — Clean geometric<br>
        <b style="color:#ffbb55;">5.0+</b> — Very coarse, tiny file
    </div>
    """, unsafe_allow_html=True)


# ─── Main Layout ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-banner">
    <div class="hero-title">VectoLine</div>
    <div class="hero-subtitle">Image Vectorization Pipeline</div>
    <div class="pipeline-badge">
        Input Image <span>→</span> Line Art <span>→</span> SVG Vector <span>→</span> Simplified SVG
    </div>
</div>
""", unsafe_allow_html=True)

# ── Inline Pipeline Settings (always accessible) ─────────────────────────────
with st.expander("⚙  PIPELINE SETTINGS", expanded=False):
    col_s1, col_s2, col_s3 = st.columns([2, 2, 2])
    with col_s1:
        mode = st.selectbox(
            "LINE ART MODE",
            options=["canny", "adaptive", "xdog"],
            index=0,
            key="mode_inline",
        )
    with col_s2:
        epsilon = st.slider(
            "RDP EPSILON (simplification)",
            min_value=0.5, max_value=5.0, value=1.0, step=0.5,
            key="epsilon_inline",
            help="Higher = fewer path points, smoother/simpler output"
        )
    with col_s3:
        threshold = st.slider(
            "THRESHOLD (xdog only)",
            min_value=64, max_value=200, value=128, step=8,
            key="threshold_inline",
        )
    st.markdown("""
    <div style="display:flex;gap:12px;margin-top:0.5rem;flex-wrap:wrap;">
        <div class="mode-card" style="flex:1;min-width:160px;">
            <strong>CANNY</strong><br>Edge detection. Great for photos.<br>
            <span class="tag">photos</span>
        </div>
        <div class="mode-card" style="flex:1;min-width:160px;">
            <strong>ADAPTIVE</strong><br>Best for documents &amp; sketches.<br>
            <span class="tag">documents</span>
        </div>
        <div class="mode-card" style="flex:1;min-width:160px;">
            <strong>XDOG</strong><br>Stylized, artistic line art.<br>
            <span class="tag">artistic</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Use inline values (sidebar is now informational only)
mode      = st.session_state.get("mode_inline", "canny")
epsilon   = st.session_state.get("epsilon_inline", 1.0)
threshold = st.session_state.get("threshold_inline", 128)

# Tabs
tab_single, tab_compare, tab_pdf = st.tabs(["  🖼  SINGLE IMAGE  ", "  🔬  MODE COMPARE  ", "  📄  PDF BATCH  "])


# ══════════════════════════════════════════════════════════
# TAB 1 — Single Image
# ══════════════════════════════════════════════════════════
with tab_single:
    st.markdown('<div class="section-label">Upload Image</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop a PNG, JPG, or BMP here",
        type=["png", "jpg", "jpeg", "bmp"],
        key="single_upload",
        label_visibility="collapsed",
    )

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w       = img_bgr.shape[:2]

        col_orig, col_la, col_svg = st.columns(3)
        with col_orig:
            st.markdown('<div class="img-panel-title">01 — ORIGINAL</div>', unsafe_allow_html=True)
            st.image(img_rgb, use_container_width=True)
            st.caption(f"{w} × {h} px  ·  {uploaded.size // 1024} KB")

        run_btn = st.button("⚡  VECTORIZE", use_container_width=False)

        if run_btn:
            with st.spinner("Running pipeline…"):
                prog = st.progress(0, text="Step 1/3 — Line art…")
                line_art = image_to_line_art(img_bgr, mode=mode, threshold=threshold)
                prog.progress(33, text="Step 2/3 — Tracing with potrace…")

                with tempfile.TemporaryDirectory() as tmp:
                    raw_svg   = os.path.join(tmp, 'raw.svg')
                    final_svg = os.path.join(tmp, 'final.svg')
                    line_art_to_svg(line_art, raw_svg)
                    size_kb    = os.path.getsize(raw_svg) / 1024
                    path_count = len(re.findall(r'<path', open(raw_svg).read()))
                    prog.progress(66, text="Step 3/3 — Simplifying paths (RDP)…")
                    content = open(raw_svg).read()
                    simplified, before, after = simplify_svg_content(content, epsilon)
                    open(final_svg, 'w').write(simplified)
                    svg_bytes = open(final_svg, 'rb').read()

                prog.progress(100, text="Done!")
                reduction = (1 - after / before) * 100 if before > 0 else 0

            # Show line art and SVG preview
            with col_la:
                st.markdown('<div class="img-panel-title">02 — LINE ART</div>', unsafe_allow_html=True)
                st.image(line_art, use_container_width=True, clamp=True)
                st.caption(f"Mode: {mode.upper()}")

            with col_svg:
                st.markdown('<div class="img-panel-title">03 — VECTOR SVG</div>', unsafe_allow_html=True)
                svg_b64 = base64.b64encode(svg_bytes).decode()
                st.markdown(
                    f'<img src="data:image/svg+xml;base64,{svg_b64}" '
                    f'style="width:100%;border-radius:8px;background:#fff;" />',
                    unsafe_allow_html=True,
                )
                st.caption(f"{size_kb:.1f} KB  ·  {path_count} paths")

            # Metrics
            reduction_color = "#7cfc8e" if reduction > 0 else "#ffbb55"
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="metric-val">{size_kb:.1f}<span style="font-size:0.9rem;color:#555566"> KB</span></div>
                    <div class="metric-lbl">SVG Size</div>
                </div>
                <div class="metric-card">
                    <div class="metric-val">{path_count}</div>
                    <div class="metric-lbl">SVG Paths</div>
                </div>
                <div class="metric-card">
                    <div class="metric-val">{before:,}</div>
                    <div class="metric-lbl">Points Before</div>
                </div>
                <div class="metric-card">
                    <div class="metric-val">{after:,}</div>
                    <div class="metric-lbl">Points After</div>
                </div>
                <div class="metric-card">
                    <div class="metric-val" style="color:{reduction_color};">{reduction:.1f}<span style="font-size:0.9rem;color:#555566">%</span></div>
                    <div class="metric-lbl">Reduction</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            stem = Path(uploaded.name).stem
            st.download_button(
                label="⬇  Download SVG",
                data=svg_bytes,
                file_name=f"{stem}_vector.svg",
                mime="image/svg+xml",
                use_container_width=False,
            )
    else:
        st.info("Upload an image above to get started.")


# ══════════════════════════════════════════════════════════
# TAB 2 — Mode Comparison
# ══════════════════════════════════════════════════════════
with tab_compare:
    st.markdown('<div class="section-label">Compare All 3 Line Art Modes Side-by-Side</div>', unsafe_allow_html=True)

    uploaded_cmp = st.file_uploader(
        "Drop image here",
        type=["png", "jpg", "jpeg", "bmp"],
        key="compare_upload",
        label_visibility="collapsed",
    )

    if uploaded_cmp:
        file_bytes = np.frombuffer(uploaded_cmp.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        run_cmp = st.button("🔬  COMPARE ALL MODES", use_container_width=False)
        if run_cmp:
            with st.spinner("Generating all 3 modes…"):
                c_orig, c_canny, c_adapt, c_xdog = st.columns(4)
                labels = {
                    c_orig:  ("ORIGINAL",  None,       "#aaaacc"),
                    c_canny: ("CANNY",     "canny",    "#7cfc8e"),
                    c_adapt: ("ADAPTIVE",  "adaptive", "#4af0c8"),
                    c_xdog:  ("XDOG",      "xdog",     "#ffbb55"),
                }
                for col, (title, m, color) in labels.items():
                    with col:
                        st.markdown(
                            f'<div style="font-family:Space Mono,monospace;font-size:0.68rem;'
                            f'color:{color};letter-spacing:2px;text-transform:uppercase;'
                            f'margin-bottom:0.5rem;">{title}</div>',
                            unsafe_allow_html=True)
                        if m is None:
                            st.image(img_rgb, use_container_width=True)
                        else:
                            la = image_to_line_art(img_bgr, mode=m, threshold=threshold)
                            st.image(la, use_container_width=True, clamp=True)
    else:
        st.info("Upload an image to compare the three line art modes.")


# ══════════════════════════════════════════════════════════
# TAB 3 — PDF Batch
# ══════════════════════════════════════════════════════════
with tab_pdf:
    st.markdown('<div class="section-label">Extract & Vectorize All Images from a PDF</div>', unsafe_allow_html=True)

    uploaded_pdf = st.file_uploader(
        "Drop a PDF here",
        type=["pdf"],
        key="pdf_upload",
        label_visibility="collapsed",
    )

    if uploaded_pdf:
        # Download options
        dl_col1, dl_col2 = st.columns([3, 1])
        with dl_col2:
            show_lineart = st.toggle("🎨 Show Line Art Preview", value=True)

        run_pdf = st.button("📄  PROCESS PDF", use_container_width=False)

        if run_pdf:
            with tempfile.TemporaryDirectory() as tmp:
                pdf_path = os.path.join(tmp, uploaded_pdf.name)
                open(pdf_path, 'wb').write(uploaded_pdf.read())

                doc   = fitz.open(pdf_path)
                found = 0
                svgs       = {}   # name → svg bytes
                line_arts  = {}   # name → PNG bytes of line art
                previews   = []   # list of (label, original_rgb, line_art_array, svg_b64)

                prog = st.progress(0, text=f"Scanning {len(doc)} pages…")

                for pn, page in enumerate(doc):
                    for idx, info in enumerate(page.get_images(full=True)):
                        found += 1
                        label = f"p{pn+1}_img{idx+1}"
                        try:
                            bi  = doc.extract_image(info[0])
                            arr = np.frombuffer(bi['image'], np.uint8)
                            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if img is None: continue

                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            la      = image_to_line_art(img, mode=mode, threshold=threshold)

                            # Save line art as PNG bytes
                            la_pil     = Image.fromarray(la)
                            la_buf     = io.BytesIO()
                            la_pil.save(la_buf, format='PNG')
                            line_arts[f"{label}_lineart.png"] = la_buf.getvalue()

                            # Trace to SVG
                            raw_svg   = os.path.join(tmp, f'{label}_raw.svg')
                            final_svg = os.path.join(tmp, f'{label}.svg')
                            line_art_to_svg(la, raw_svg)
                            content = open(raw_svg).read()
                            simp, _, _ = simplify_svg_content(content, epsilon)
                            open(final_svg, 'w').write(simp)
                            svg_bytes = open(final_svg, 'rb').read()
                            svgs[f"{label}.svg"] = svg_bytes

                            # Store preview data
                            svg_b64 = base64.b64encode(svg_bytes).decode()
                            previews.append((label, img_rgb, la, svg_b64))

                        except Exception as e:
                            st.warning(f"{label}: {e}")
                    prog.progress(int((pn + 1) / len(doc) * 100), text=f"Page {pn+1}/{len(doc)}")

                if svgs:
                    st.success(f"✅ Converted {len(svgs)} / {found} images")

                    # ── Download buttons ──────────────────────────────────
                    btn_col1, btn_col2 = st.columns(2)

                    with btn_col1:
                        svg_zip = io.BytesIO()
                        with zipfile.ZipFile(svg_zip, 'w') as zf:
                            for name, data in svgs.items():
                                zf.writestr(name, data)
                        svg_zip.seek(0)
                        st.download_button(
                            label=f"⬇  Download {len(svgs)} SVGs as ZIP",
                            data=svg_zip.getvalue(),
                            file_name=f"{Path(uploaded_pdf.name).stem}_svgs.zip",
                            mime="application/zip",
                            use_container_width=True,
                        )

                    with btn_col2:
                        la_zip = io.BytesIO()
                        with zipfile.ZipFile(la_zip, 'w') as zf:
                            for name, data in line_arts.items():
                                zf.writestr(name, data)
                        la_zip.seek(0)
                        st.download_button(
                            label=f"⬇  Download {len(line_arts)} Line Arts as ZIP",
                            data=la_zip.getvalue(),
                            file_name=f"{Path(uploaded_pdf.name).stem}_linearts.zip",
                            mime="application/zip",
                            use_container_width=True,
                        )

                    # ── Preview Grid ──────────────────────────────────────
                    if show_lineart and previews:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(
                            '<div class="section-label">Line Art Preview</div>',
                            unsafe_allow_html=True
                        )
                        # Show 3 columns: original | line art | SVG
                        st.markdown("""
                        <div style="display:grid;grid-template-columns:repeat(3,1fr);
                                    gap:6px;margin-bottom:0.5rem;text-align:center;">
                            <div style="font-family:'Space Mono',monospace;font-size:0.62rem;
                                        color:#555566;letter-spacing:2px;">ORIGINAL</div>
                            <div style="font-family:'Space Mono',monospace;font-size:0.62rem;
                                        color:#7cfc8e;letter-spacing:2px;">LINE ART</div>
                            <div style="font-family:'Space Mono',monospace;font-size:0.62rem;
                                        color:#4af0c8;letter-spacing:2px;">VECTOR SVG</div>
                        </div>
                        """, unsafe_allow_html=True)

                        for label, orig_rgb, la_arr, svg_b64 in previews:
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.image(orig_rgb, use_container_width=True, caption=label)
                            with c2:
                                st.image(la_arr, use_container_width=True, clamp=True,
                                         caption=f"{mode.upper()} line art")
                            with c3:
                                st.markdown(
                                    f'<img src="data:image/svg+xml;base64,{svg_b64}" '
                                    f'style="width:100%;border-radius:6px;background:#fff;" />',
                                    unsafe_allow_html=True
                                )
                                st.caption("Vector SVG")
                            st.markdown("<hr style='border-color:#2a2a38;margin:0.5rem 0;'>",
                                        unsafe_allow_html=True)
                else:
                    st.warning("No images found or extractable in this PDF.")
    else:
        st.info("Upload a PDF to batch-vectorize all embedded images.")
