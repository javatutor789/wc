"""
✨ Word Cloud Studio — a chic Streamlit word cloud generator.
Run:  streamlit run wordcloud_app.py
"""
import io
import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw
from wordcloud import STOPWORDS, WordCloud

# ------------------------------------------------------------------ page setup
st.set_page_config(page_title="Word Cloud Studio", page_icon="☁️", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;800&family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .hero {
        padding: 2.2rem 2rem; border-radius: 22px; margin-bottom: 1.5rem;
        background: linear-gradient(135deg, #1e1b4b 0%, #6d28d9 50%, #db2777 100%);
        color: white; box-shadow: 0 12px 40px rgba(109,40,217,.25);
    }
    .hero h1 { font-family: 'Playfair Display', serif; font-weight: 800;
               font-size: 2.8rem; margin: 0; letter-spacing: .5px; }
    .hero p { margin: .4rem 0 0; font-weight: 300; font-size: 1.1rem; opacity: .9; }
    .metric-card {
        background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.18);
        border-radius: 16px; padding: 1rem 1.2rem; text-align: center;
    }
    .metric-card .num { font-family: 'Playfair Display', serif; font-size: 1.9rem;
                        font-weight: 800; color: #a855f7; }
    .metric-card .lbl { font-size: .8rem; text-transform: uppercase;
                        letter-spacing: 1.5px; opacity: .65; }
    section[data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.2); }
    .stDownloadButton button {
        background: linear-gradient(90deg, #6d28d9, #db2777); color: white;
        border: none; border-radius: 12px; padding: .6rem 1.4rem; font-weight: 600;
    }
    .stDownloadButton button:hover { filter: brightness(1.1); color: white; }
    </style>
    <div class="hero">
        <h1>☁️ Word Cloud Studio</h1>
        <p>Upload a text file, tweak the style, and watch your words bloom.</p>
        <p> RIU
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ palettes
PALETTES = {
    "Amethyst Dusk": ["#6d28d9", "#a855f7", "#db2777", "#f472b6", "#4338ca"],
    "Ocean Breeze": ["#0f766e", "#14b8a6", "#0ea5e9", "#1d4ed8", "#38bdf8"],
    "Sunset Glow": ["#b91c1c", "#ea580c", "#f59e0b", "#db2777", "#7c2d12"],
    "Forest Sage": ["#14532d", "#16a34a", "#65a30d", "#84cc16", "#365314"],
    "Rose Gold": ["#9f1239", "#e11d48", "#fb7185", "#c2410c", "#be185d"],
    "Midnight Neon": ["#22d3ee", "#a78bfa", "#f472b6", "#34d399", "#facc15"],
    "Monochrome Ink": ["#111827", "#374151", "#6b7280", "#9ca3af", "#1f2937"],
    "Matplotlib · viridis": "viridis",
    "Matplotlib · plasma": "plasma",
    "Matplotlib · cool": "cool",
}


def get_cmap(name: str):
    spec = PALETTES[name]
    if isinstance(spec, str):
        return spec
    return LinearSegmentedColormap.from_list(name, spec)


# ------------------------------------------------------------------ shapes
def make_mask(shape: str, size: int = 900):
    """Return a numpy mask (255 = empty, 0 = where words go) or None for rectangle."""
    if shape == "Rectangle":
        return None
    img = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(img)
    c, r = size / 2, size * 0.47

    if shape == "Circle":
        d.ellipse([c - r, c - r, c + r, c + r], fill=0)
    elif shape == "Heart":
        t = np.linspace(0, 2 * np.pi, 500)
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        s = r / 16.5
        pts = [(c + xi * s, c - (yi + 2.5) * s) for xi, yi in zip(x, y)]
        d.polygon(pts, fill=0)
    elif shape == "Star":
        pts = []
        for i in range(10):
            ang = -np.pi / 2 + i * np.pi / 5
            rad = r if i % 2 == 0 else r * 0.42
            pts.append((c + rad * np.cos(ang), c + rad * np.sin(ang)))
        d.polygon(pts, fill=0)
    elif shape == "Diamond":
        d.polygon([(c, c - r), (c + r * 0.8, c), (c, c + r), (c - r * 0.8, c)], fill=0)
    elif shape == "Hexagon":
        pts = [(c + r * np.cos(np.pi / 6 + i * np.pi / 3),
                c + r * np.sin(np.pi / 6 + i * np.pi / 3)) for i in range(6)]
        d.polygon(pts, fill=0)
    elif shape == "Triangle":
        d.polygon([(c, c - r), (c + r, c + r * 0.8), (c - r, c + r * 0.8)], fill=0)
    return np.array(img)


SHAPES = ["Rectangle", "Circle", "Heart", "Star", "Diamond", "Hexagon", "Triangle"]

SAMPLE_TEXT = """
Creativity is intelligence having fun. Design is thinking made visual. Simplicity is the ultimate
sophistication. Imagine create inspire elegance style beauty balance harmony color texture light
shadow form space rhythm motion story craft detail vision courage curiosity wonder dream explore
build play learn grow share dream design create inspire imagine beauty elegance style craft vision
color color design design creativity creativity story story wonder light light inspire inspire
"""

# ------------------------------------------------------------------ upload
uploaded = st.file_uploader("📄 Upload a text file", type=["txt", "md", "csv", "log"])

text, source = SAMPLE_TEXT, "sample text"
if uploaded is not None:
    raw = uploaded.getvalue()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    source = uploaded.name
else:
    st.info("No file yet — showing a demo with sample text. Upload your own to begin ✨")

if not text.strip():
    st.warning("The file appears to be empty.")
    st.stop()

# ------------------------------------------------------------------ controls
with st.container(border=True):
    st.markdown("#### 🎨 Style Studio")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        max_words = st.slider("Number of words", 10, 500, 100, step=10)
    with k2:
        palette_name = st.selectbox("Color palette", list(PALETTES.keys()), index=0)
    with k3:
        shape = st.selectbox("Shape", SHAPES, index=1)
    with k4:
        bg_color = st.color_picker("Background color", "#FFFFFF")

    with st.expander("⚙️ Advanced options"):
        a1, a2, a3 = st.columns(3)
        with a1:
            min_len = st.slider("Minimum word length", 1, 10, 3)
            remove_stop = st.toggle("Remove common stop words", value=True)
            extra_stop = st.text_input("Extra words to exclude (comma-separated)", "")
        with a2:
            contour = st.slider("Outline thickness", 0, 10, 0)
            contour_color = st.color_picker("Outline color", "#6d28d9")
        with a3:
            prefer_h = st.slider("Horizontal word preference", 0.0, 1.0, 0.9, 0.05)
            seed = st.number_input("Random seed (layout)", 0, 9999, 42)

# ------------------------------------------------------------------ build cloud
stopwords = set(STOPWORDS) if remove_stop else set()
stopwords |= {w.strip().lower() for w in extra_stop.split(",") if w.strip()}

mask = make_mask(shape)
wc_kwargs = dict(
    background_color=bg_color,
    max_words=max_words,
    colormap=get_cmap(palette_name),
    stopwords=stopwords,
    min_word_length=min_len,
    contour_width=contour if mask is not None else 0,
    contour_color=contour_color,
    prefer_horizontal=prefer_h,
    random_state=int(seed),
    collocations=False,
    relative_scaling=0.5,
)
if mask is not None:
    wc_kwargs.update(mask=mask)
else:
    wc_kwargs.update(width=1400, height=800)

try:
    wc = WordCloud(**wc_kwargs).generate(text)
except ValueError:
    st.error("No words left to display — try lowering the minimum word length "
             "or disabling stop-word removal.")
    st.stop()

# ------------------------------------------------------------------ stats
tokens = re.findall(r"[A-Za-z\u00C0-\u024F']+", text.lower())
counts = Counter(t for t in tokens if t not in stopwords and len(t) >= min_len)

c1, c2, c3 = st.columns(3)
for col, num, lbl in [
    (c1, f"{len(tokens):,}", "Total words"),
    (c2, f"{len(counts):,}", "Unique words"),
    (c3, f"{len(wc.words_):,}", "Words shown"),
]:
    col.markdown(f'<div class="metric-card"><div class="num">{num}</div>'
                 f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.write("")

# ------------------------------------------------------------------ display
left, right = st.columns([3, 1.4], gap="large")

with left:
    st.subheader(f"Your cloud · {source}")
    image = wc.to_image()
    st.image(image, width="stretch")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    st.download_button("⬇️ Download PNG", buf.getvalue(), "wordcloud.png", "image/png")

with right:
    st.subheader("Top words")
    top = pd.DataFrame(counts.most_common(12), columns=["word", "count"])
    if not top.empty:
        st.bar_chart(top.set_index("word"), horizontal=True, color="#a855f7")

st.caption("Made with Streamlit · wordcloud · Pillow - Python")
