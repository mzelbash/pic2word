"""
Pic2Word Interactive Walkthrough
SEAS 8525: Computer Vision and Generative AI
Dr. Elbasheer, Week 6

Run locally:
    pip install -r requirements.txt
    streamlit run pic2word_app.py

How this app relates to the paper:
    Pic2Word (Saito et al., CVPR 2023) trains a small mapper that turns a CLIP image
    embedding into one pseudo-word token. This app uses the same frozen CLIP and inserts
    the image as a real token at the text encoder's input. Instead of trained mapper
    weights, it finds each image's pseudo-word by solving the mapper's training objective
    directly (about 150 gradient steps, a few seconds on CPU).

Image database:
    By default the app downloads the 25 Unsplash images listed in GALLERY.
    To use your own, put them in a folder named "samples" next to this file
    (optionally in subfolders named by category, e.g. samples/shoes/red_sneakers.jpg).
    The file name becomes the label.
"""

import base64
import io
import urllib.request
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"

st.set_page_config(
    page_title="Pic2Word Walkthrough · SEAS 8525",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS (house style: Lexend, navy titles, gold tags, light panels)
# ═════════════════════════════════════════════════════════════════════════════
NAVY = "#002147"
GOLD = "#FFC400"
BG = "#F3F6FB"
BORDER = "#D9E1EC"
INK = "#1E293B"
MUTED = "#64748B"

IMG, IMG_BG = "#1D4ED8", "#E3ECFF"        # image side = blue
TXT, TXT_BG = "#0E9F6E", "#DDF5EC"        # text side = green
PROJ, PROJ_BG = "#D97706", "#FFF2DB"      # projection = orange
SPACE, SPACE_BG = "#7C3AED", "#F1EAFE"    # shared space = purple
POS, POS_BG = "#0E9F6E", "#CFEFDF"        # matching pair
NEG, NEG_BG = "#E11D48", "#FDE2E8"        # non-matching pair

PAIR_COLORS = ["#1D4ED8", "#E11D48", "#D97706", "#0E9F6E",
               "#7C3AED", "#0891B2", "#BE185D", "#4B5563"]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, p, li, label, input, textarea, button, h1, h2, h3, h4,
.stMarkdown, [data-testid="stWidgetLabel"], [data-testid="stExpander"] summary p {
    font-family: 'Lexend', system-ui, sans-serif !important;
}
code, pre, .stCode code { font-family: 'JetBrains Mono', monospace !important; }

[data-testid="stAppViewContainer"] { background: #F3F6FB; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1320px; }
footer, [data-testid="stDecoration"] { display: none; }
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #D9E1EC; }
p, li { font-size: 16px; color: #1E293B; line-height: 1.6; }
[data-testid="stExpander"] { background: #FFFFFF; border-radius: 12px; border-color: #D9E1EC; }

/* section banner */
.hero { background: #002147; border-radius: 16px; padding: 18px 28px 16px; margin: 4px 0 16px;
        display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
.hero .badge { background: #FFC400; color: #002147; font-weight: 800; font-size: 18px;
        width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; }
.hero .t { color: #FFFFFF; font-weight: 800; font-size: 32px; letter-spacing: -0.3px; }
.hero .s { color: #C9D6EA; font-size: 17px; flex-basis: 100%; margin-left: 56px; margin-top: -6px; }

/* takeaway box at the end of each section */
.key { background: #FFFFFF; border: 1px solid #D9E1EC; border-top: 6px solid #FFC400;
       border-radius: 16px; padding: 22px 36px; margin: 22px auto 10px; text-align: center;
       box-shadow: 0 4px 14px rgba(0, 33, 71, 0.08); }
.key .x { font-size: 21px; font-weight: 500; color: #002147; line-height: 1.6; max-width: 980px; margin: 0 auto; }
.key .x b { font-weight: 800; }

/* top navigation: big, clear section buttons */
.st-key-topnav { background: #FFFFFF; border: 1px solid #D9E1EC; border-radius: 16px; padding: 10px 12px;
                 box-shadow: 0 2px 10px rgba(0, 33, 71, 0.06); margin-bottom: 4px; }
.st-key-topnav [data-testid="stButtonGroup"], .st-key-topnav [data-testid="stButtonGroup"] > div { width: 100%; }
.st-key-topnav [data-testid="stButtonGroup"] button { flex: 1 1 0; min-height: 54px; padding: 8px 14px;
                 border-radius: 12px !important; border: 1.5px solid #D9E1EC; background: #F3F6FB; }
.st-key-topnav [data-testid="stButtonGroup"] button p { font-size: 18px !important; font-weight: 700 !important; color: #002147; }
.st-key-topnav [data-testid="stButtonGroup"] button:hover { border-color: #002147; }
.st-key-topnav [data-testid="stBaseButton-segmented_controlActive"] { background: #002147 !important; border-color: #002147 !important; }
.st-key-topnav [data-testid="stBaseButton-segmented_controlActive"] p { color: #FFC400 !important; }
.st-key-topnav [role="radiogroup"] label p { font-size: 18px !important; font-weight: 700; color: #002147; }

/* "by the numbers" fact cards */
.facts { display: grid; grid-template-columns: 1.35fr 1fr 1fr; gap: 14px; margin: 6px 0 4px; }
@media (max-width: 1000px) { .facts { grid-template-columns: 1fr; } }
.fgroup { background: #FFFFFF; border: 1px solid #D9E1EC; border-radius: 16px; padding: 14px 16px; }
.fgroup .gh { font-weight: 800; font-size: 16px; margin-bottom: 2px; }
.fgroup .gs { font-size: 13px; color: #64748B; margin-bottom: 10px; }
.fact { padding: 10px 0; border-top: 1px solid #EEF2F7; }
.fact .n { font-size: 32px; font-weight: 800; line-height: 1.1; white-space: nowrap; }
.fact .k { font-weight: 700; font-size: 15.5px; margin-top: 2px; color: #002147; }
.fact .d { font-size: 13.5px; color: #475569; line-height: 1.45; margin-top: 2px; }

/* small stat chips */
.chips { display: flex; gap: 12px; flex-wrap: wrap; margin: 4px 0 12px; }
.chip { background: #FFFFFF; border: 1px solid #D9E1EC; border-radius: 14px; padding: 10px 18px; min-width: 150px; }
.chip b { display: block; font-size: 24px; font-weight: 800; color: #002147; }
.chip span { font-size: 13px; color: #64748B; }

/* panels and pipelines */
.panel { background: #FFFFFF; border: 1px solid #D9E1EC; border-radius: 16px; padding: 18px 20px; margin: 6px 0 10px; }
.ptitle { font-weight: 700; color: #002147; font-size: 17px; margin-bottom: 10px; display: flex; gap: 10px; align-items: center; }
.tag { display: inline-block; border-radius: 999px; padding: 2px 12px; font-size: 13px; font-weight: 700; }
.pipe { display: flex; align-items: center; gap: 6px; flex-wrap: nowrap; overflow-x: auto; padding-bottom: 2px; }
.pipe > * { flex-shrink: 0; }
.arrow { font-size: 22px; color: #94A3B8; font-weight: 700; }
.node { border-radius: 12px; padding: 8px 11px; text-align: center; font-weight: 700; font-size: 14px; line-height: 1.3; }
.node small { display: block; font-weight: 500; color: #64748B; font-size: 12px; margin-top: 2px; }
.n-img  { background: #E3ECFF; border: 2px solid #1D4ED8; color: #1D4ED8; }
.n-txt  { background: #DDF5EC; border: 2px solid #0E9F6E; color: #0B7A55; }
.n-proj { background: #FFF2DB; border: 2px solid #D97706; color: #B45309; }
.n-space{ background: #F1EAFE; border: 2px solid #7C3AED; color: #6D28D9; }
.n-plain{ background: #FFFFFF; border: 1px dashed #94A3B8; color: #1E293B; }
.thumb { border-radius: 10px; display: block; }

/* token sequences */
.seq { display: flex; gap: 4px; align-items: center; flex-wrap: wrap; justify-content: center; }
.seq img { width: 24px; height: 24px; border-radius: 4px; border: 1px solid #93B4F5; }
.tk { display: inline-block; border-radius: 8px; padding: 3px 7px; font-size: 13px; font-weight: 600;
      background: #FFFFFF; border: 1.5px solid #0E9F6E; color: #0B7A55; }
.tk.sp { background: #F1F5F9; border-color: #94A3B8; color: #475569; }
.tk.eot { background: #0E9F6E; border-color: #0E9F6E; color: #FFFFFF; }
.tk.cls { background: #1D4ED8; border-color: #1D4ED8; color: #FFFFFF; }

/* embedding strips */
.strip { display: flex; gap: 1px; }
.strip i { display: block; width: 8px; height: 30px; border-radius: 2px; }
.grid5 { display: grid; grid-template-columns: auto 34px auto 34px 1fr; gap: 12px 6px; align-items: center; }
.slabel { font-size: 12.5px; color: #64748B; margin-top: 4px; font-weight: 500; }

/* similarity matrix */
table.mx { border-collapse: separate; border-spacing: 6px; margin: 0 auto; }
table.mx td { width: 104px; height: 74px; text-align: center; border-radius: 10px; font-size: 19px;
              color: #1E293B; border: 1px solid rgba(0,0,0,0.06); }
table.mx td.d { font-weight: 800; }
table.mx td.win { box-shadow: inset 0 0 0 3px #002147; }
table.mx th { font-weight: 600; font-size: 14px; color: #1E293B; }
table.mx th.col span { display: block; border-radius: 10px; padding: 6px 8px; border: 2px solid; line-height: 1.25; }
table.mx th.row { text-align: left; }
table.mx th.row div { display: flex; align-items: center; gap: 8px; }
table.mx th.row img { width: 66px; height: 66px; object-fit: cover; border-radius: 10px; border: 3px solid; }

/* zero-shot step cards */
.steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
@media (max-width: 1000px) { .steps { grid-template-columns: 1fr 1fr; } }
.step { background: #FFFFFF; border: 1px solid #D9E1EC; border-radius: 16px; padding: 14px; }
.step h4 { margin: 0 0 12px 0; padding: 8px 12px; border-radius: 10px; font-size: 17px; font-weight: 700; }
.bar { display: grid; grid-template-columns: 1fr 62px; gap: 6px; align-items: center; margin: 5px 0; font-size: 14px; }
.bar .track { background: #F1F5F9; border-radius: 6px; height: 26px; position: relative; overflow: hidden; }
.bar .fill { height: 100%; border-radius: 6px; }
.bar .lab { position: absolute; left: 8px; top: 3px; font-weight: 600; color: #1E293B; white-space: nowrap; }
.bar .v { text-align: right; font-weight: 700; font-family: 'JetBrains Mono', monospace; font-size: 13.5px; }

/* sidebar gallery */
.gal { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
.gal div { font-size: 11px; color: #475569; text-align: center; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.gal img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 8px; }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SMALL HTML HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def html(s: str):
    """Render an HTML snippet. Lines are joined so Markdown never sees
    indentation or blank lines (which would break the HTML)."""
    one_line = " ".join(line.strip() for line in s.splitlines() if line.strip())
    st.markdown(f"<div>{one_line}</div>", unsafe_allow_html=True)


def esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("$", "&#36;"))


def hero(num, title, subtitle):
    html(f'<div class="hero"><div class="badge">{num}</div><div class="t">{title}</div>'
         f'<div class="s">{subtitle}</div></div>')


def key_idea(text):
    html(f'<div class="key"><div class="x">{text}</div></div>')


def chips(items):
    inner = "".join(f'<div class="chip"><b>{v}</b><span>{k}</span></div>' for v, k in items)
    html(f'<div class="chips">{inner}</div>')


def mix(c1, c2, t):
    t = float(np.clip(t, 0, 1))
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b))


def val_color(t, pos_color):
    """t in [-1, 1]: positive values tint toward pos_color, negative toward red."""
    return mix("#FFFFFF", pos_color, t) if t >= 0 else mix("#FFFFFF", NEG, -t)


def strip_html(vec, color, n=64):
    v = np.asarray(vec[:n], dtype=float)
    m = max(float(np.abs(v).max()), 1e-6)
    cells = "".join(f'<i style="background:{val_color(x / m, color)}"></i>' for x in v)
    return f'<div class="strip">{cells}</div>'


def article(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


def seg(label, options, key, default=None):
    """Segmented control with a radio fallback for older Streamlit versions."""
    if key not in st.session_state:
        st.session_state[key] = default or options[0]
    if hasattr(st, "segmented_control"):
        val = st.segmented_control(label, options, key=key)
    else:
        val = st.radio(label, options, key=key, horizontal=True)
    return val or options[0]


def code_panel(title, code):
    with st.expander(f"💻  Code · {title}", expanded=False):
        st.code(code, language="python")



def place_labels(anchors, sizes, points, bounds, iters=400):
    """Spread label boxes apart in pixel space.
    anchors: where each label wants to be; sizes: (w, h) per label;
    points: marker positions labels should not cover; bounds: (x0, y0, x1, y1) of the axes."""
    anchors = np.asarray(anchors, float)
    n = len(anchors)
    center = anchors.mean(0)
    d = anchors - center
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    rng = np.random.default_rng(0)
    pos = anchors + d * 30 + rng.normal(0, 3, size=anchors.shape)
    half = sizes / 2.0
    x0, y0, x1, y1 = bounds
    for it in range(iters + 300):
        final = it >= iters                      # last phase: only separate labels, nothing pulls them back
        moved = 0.0
        for i in range(n):                       # label vs label
            for j in range(i + 1, n):
                dx, dy = pos[j] - pos[i]
                ox = half[i, 0] + half[j, 0] + 6 - abs(dx)
                oy = half[i, 1] + half[j, 1] + 6 - abs(dy)
                if ox > 0 and oy > 0:
                    if ox < oy:
                        s = (np.sign(dx) or 1) * ox / 2
                        pos[i, 0] -= s; pos[j, 0] += s
                    else:
                        s = (np.sign(dy) or 1) * oy / 2
                        pos[i, 1] -= s; pos[j, 1] += s
                    moved += min(ox, oy)
        for i in range(n if not final else 0):   # label vs markers
            for p in points:
                dx, dy = pos[i] - p
                ox = half[i, 0] + 9 - abs(dx)
                oy = half[i, 1] + 9 - abs(dy)
                if ox > 0 and oy > 0:
                    if ox < oy:
                        pos[i, 0] += (np.sign(dx) or 1) * ox * 0.5
                    else:
                        pos[i, 1] += (np.sign(dy) or 1) * oy * 0.5
        if not final:
            pos += 0.02 * (anchors - pos)        # gentle pull back toward the true point
        pos[:, 0] = np.clip(pos[:, 0], x0 + half[:, 0] + 2, x1 - half[:, 0] - 2)
        pos[:, 1] = np.clip(pos[:, 1], y0 + half[:, 1] + 2, y1 - half[:, 1] - 2)
        if final and moved < 0.5:
            break
    return pos



P2W_CSS = """
.stButton > button[kind="primary"] { background: #002147; border: none; color: #FFFFFF; font-weight: 700;
    font-size: 16px; padding: 10px 26px; border-radius: 12px; }
.stButton > button[kind="primary"]:hover { background: #0B3A6E; color: #FFC400; }
.stButton > button[kind="primary"] p { color: #FFFFFF; font-size: 16px; font-weight: 700; }
.tk.pr { background: #F1F5F9; border-color: #94A3B8; color: #334155; }
.cards3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
@media (max-width: 1000px) { .cards3 { grid-template-columns: 1fr; } }
.rgrid { display: grid; gap: 10px; }
.rt { background: #FFFFFF; border: 1px solid #D9E1EC; border-radius: 12px; padding: 6px; text-align: center; }
.rt img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 8px; display: block; }
.rt.good { border: 2.5px solid #0E9F6E; box-shadow: 0 0 0 3px #DDF5EC; }
.rt .rl { font-size: 12.5px; font-weight: 700; color: #002147; margin-top: 4px; white-space: nowrap;
          overflow: hidden; text-overflow: ellipsis; }
.rt .rs { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #64748B; }
.bd { display: inline-block; font-size: 11px; font-weight: 700; border-radius: 999px; padding: 1px 7px; margin: 2px 1px 0; }
.bd.y { background: #DDF5EC; color: #0B7A55; }
.bd.n { background: #F1F5F9; color: #94A3B8; }
.wchip { display: inline-flex; align-items: center; gap: 4px; border: 2.5px solid #D97706; background: #FFF2DB;
         border-radius: 8px; padding: 2px 5px; font-weight: 800; color: #B45309; vertical-align: middle; }
.wchip img { width: 30px; height: 30px; border-radius: 5px; object-fit: cover; }
.mrow { display: grid; grid-template-columns: 220px 1fr; gap: 14px; align-items: center; padding: 12px 0;
        border-top: 1px solid #EEF2F7; }
.mname { font-weight: 800; font-size: 16px; }
.mname small { display: block; font-weight: 500; color: #64748B; font-size: 12.5px; line-height: 1.35; margin-top: 2px; }
.score { font-size: 13px; font-weight: 700; margin-top: 6px; }
.trow { display: grid; grid-template-columns: 96px 112px 1fr; gap: 8px; align-items: center; margin: 5px 0; }
.trow .tk, .trow .wchip { justify-self: start; }
.trow .src { font-size: 12px; color: #64748B; font-family: 'JetBrains Mono', monospace; }
.vpipe { display: flex; flex-direction: column; align-items: stretch; gap: 4px; }
.vpipe .down { text-align: center; color: #94A3B8; font-weight: 700; font-size: 18px; line-height: 1; }
.spaces { display: grid; grid-template-columns: 1fr 120px 1fr; gap: 10px; align-items: center; }
@media (max-width: 900px) { .spaces { grid-template-columns: 1fr; } }
.sp { background: #FFFFFF; border: 1px solid #D9E1EC; border-radius: 16px; padding: 14px 18px; }
.sp .gh { font-weight: 800; font-size: 17px; margin-bottom: 6px; }
.sp li { font-size: 14.5px; margin: 3px 0; }
.wordchips { display: flex; gap: 6px; flex-wrap: wrap; }
"""
st.markdown(f"<style>{P2W_CSS}</style>", unsafe_allow_html=True)


def facts_panel(facts, note):
    html(f'<div style="font-weight:800;font-size:20px;color:{NAVY};margin:14px 0 6px">{note[0]} '
         f'<span style="font-weight:500;font-size:15px;color:{MUTED}">· {note[1]}</span></div>')
    cols = ""
    for title, sub, color, items in facts:
        rows = "".join(f'<div class="fact"><div class="n" style="color:{color}">{n}</div>'
                       f'<div><div class="k">{k}</div><div class="d">{esc(d)}</div></div></div>'
                       for n, k, d in items)
        cols += (f'<div class="fgroup" style="border-top:5px solid {color}"><div class="gh" style="color:{color}">{title}</div>'
                 f'<div class="gs">{sub}</div>{rows}</div>')
    html(f'<div class="facts">{cols}</div>')


def run_button(label, key):
    return st.button(f"▶  {label}", key=key, type="primary")


# ═════════════════════════════════════════════════════════════════════════════
# IMAGES: the retrieval database
# ═════════════════════════════════════════════════════════════════════════════
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}

GALLERY = [
    ("white sneakers", "shoes", "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=300"),
    ("red high heels", "shoes", "https://images.unsplash.com/photo-1543163521-1bf539c55dd2?w=300"),
    ("brown leather boots", "shoes", "https://images.unsplash.com/photo-1520639888713-7851133b1ed0?w=300"),
    ("black running shoes", "shoes", "https://images.unsplash.com/photo-1491553895911-0055eca6402d?w=300"),
    ("blue canvas shoes", "shoes", "https://images.unsplash.com/photo-1525966222134-fcfa99b8ae77?w=300"),
    ("black leather bag", "bags", "https://images.unsplash.com/photo-1548036328-c9fa89d128fa?w=300"),
    ("red handbag", "bags", "https://images.unsplash.com/photo-1584917865442-de89df76afd3?w=300"),
    ("brown backpack", "bags", "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=300"),
    ("beige tote bag", "bags", "https://images.unsplash.com/photo-1600857544200-b2f666a9a2ec?w=300"),
    ("blue denim bag", "bags", "https://images.unsplash.com/photo-1590874103328-eac38a683ce7?w=300"),
    ("wooden chair", "furniture", "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=300"),
    ("red armchair", "furniture", "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=300"),
    ("white modern chair", "furniture", "https://images.unsplash.com/photo-1567538096630-e0c55bd6374c?w=300"),
    ("black office chair", "furniture", "https://images.unsplash.com/photo-1589384267710-7a170981ca78?w=300"),
    ("blue accent chair", "furniture", "https://images.unsplash.com/photo-1506439773649-6e0eb8cfb237?w=300"),
    ("red sports car", "cars", "https://images.unsplash.com/photo-1583121274602-3e2820c69888?w=300"),
    ("white SUV", "cars", "https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?w=300"),
    ("black sedan", "cars", "https://images.unsplash.com/photo-1502877338535-766e1452684a?w=300"),
    ("blue convertible", "cars", "https://images.unsplash.com/photo-1618843479313-40f8afb4b4d8?w=300"),
    ("yellow taxi", "cars", "https://images.unsplash.com/photo-1559416523-140ddc3d238c?w=300"),
    ("red roses", "flowers", "https://images.unsplash.com/photo-1490750967868-88df5691cc54?w=300"),
    ("yellow sunflowers", "flowers", "https://images.unsplash.com/photo-1597848212624-a19eb35e2651?w=300"),
    ("purple lavender", "flowers", "https://images.unsplash.com/photo-1498462440456-0dba182e775b?w=300"),
    ("white daisies", "flowers", "https://images.unsplash.com/photo-1467736518471-8d521f1dc1e8?w=300"),
    ("pink tulips", "flowers", "https://images.unsplash.com/photo-1585952285284-5e68f9f09fc0?w=300"),
]
NO_KIND = {"uploaded", "my images"}


@st.cache_data(show_spinner=False)
def fetch_bytes(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.read()
    except Exception:
        return None


@st.cache_data(show_spinner=False, max_entries=256)
def to_pil(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def clip_crop(img, size=224):
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = img.size
    l, t = (w - size) // 2, (h - size) // 2
    return img.crop((l, t, l + size, t + size))


def pil_uri(img, fmt="JPEG", quality=85):
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


@st.cache_data(show_spinner=False, max_entries=512)
def thumb_uri(b: bytes, size=160):
    return pil_uri(clip_crop(to_pil(b), size))


def build_pool(uploads):
    """Ordered dict: label -> {"bytes", "cat"}. Every image is both a possible query and a database item."""
    pool = {}

    def add(label, cat, data):
        name, k = label, 2
        while name in pool:
            name = f"{label} ({k})"
            k += 1
        pool[name] = {"bytes": data, "cat": cat}

    local = Path(__file__).parent / "samples"
    files = sorted(p for p in local.rglob("*") if p.suffix.lower() in IMG_EXT) if local.exists() else []
    if files:
        for p in files:
            cat = p.parent.name if p.parent != local else "my images"
            add(p.stem.replace("_", " "), cat, p.read_bytes())
    else:
        for lbl, cat, url in GALLERY:
            data = fetch_bytes(url)
            if data:
                add(lbl, cat, data)
    for f in uploads or []:
        add(Path(f.name).stem.replace("_", " "), "uploaded", f.getvalue())
    return pool


# ═════════════════════════════════════════════════════════════════════════════
# MODEL: frozen CLIP, plus "image -> pseudo-word"
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading CLIP ViT-B/32 (first run downloads about 600 MB)...")
def load_clip():
    model = CLIPModel.from_pretrained(MODEL_ID)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    model.requires_grad_(False)          # both encoders stay frozen
    return model, processor, device


def _unit(x):
    return x / x.norm(dim=-1, keepdim=True)


@st.cache_resource(show_spinner=False)
def star_id():
    """Token id of '*', the placeholder that marks where the image goes."""
    _, processor, _ = load_clip()
    ids = processor.tokenizer("*")["input_ids"]
    return int(ids[1])


@st.cache_data(show_spinner=False, max_entries=512)
def embed_image(b: bytes) -> np.ndarray:
    model, processor, device = load_clip()
    pv = processor(images=[to_pil(b)], return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        f = model.visual_projection(model.vision_model(pixel_values=pv).pooler_output)
    return _unit(f)[0].cpu().numpy()


def _tok(prompts):
    _, processor, device = load_clip()
    t = processor.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    return t["input_ids"].to(device), t["attention_mask"].to(device)


def _text_feats(ids, att, word=None):
    """CLIP's frozen text encoder. If `word` is given, every '*' token is swapped for that
    vector at the token-embedding layer, so the image enters the sentence as a word."""
    model, _, _ = load_clip()
    hook = None
    if word is not None:
        mask = (ids == star_id()).unsqueeze(-1)

        def swap(module, inputs, output):
            return torch.where(mask, word.to(output.dtype), output)

        hook = model.text_model.embeddings.token_embedding.register_forward_hook(swap)
    try:
        pooled = model.text_model(input_ids=ids, attention_mask=att).pooler_output
    finally:
        if hook is not None:
            hook.remove()
    return _unit(model.text_projection(pooled))


@st.cache_data(show_spinner=False, max_entries=2048)
def embed_text(t: str) -> np.ndarray:
    ids, att = _tok([t])
    with torch.no_grad():
        return _text_feats(ids, att)[0].cpu().numpy()


@st.cache_data(show_spinner=False, max_entries=64)
def image_to_word(b: bytes, steps: int = 150, lr: float = 0.02):
    """Find the pseudo-word w that makes "a photo of *" land on this image's embedding.
    This is the objective Pic2Word trains its mapper on, solved directly for one image."""
    model, processor, device = load_clip()
    target = torch.tensor(embed_image(b), device=device)
    ids, att = _tok(["a photo of *"])
    table = model.text_model.embeddings.token_embedding.weight
    init_id = int(processor.tokenizer("thing")["input_ids"][1])
    w = table[init_id].detach().clone().requires_grad_(True)
    opt = torch.optim.AdamW([w], lr=lr, weight_decay=0.0)
    hist, best, best_w = [], -2.0, None
    with torch.enable_grad():
        for _ in range(steps):
            sim = (_text_feats(ids, att, w)[0] * target).sum()
            s = float(sim.item())
            hist.append(s)
            if s > best:
                best, best_w = s, w.detach().clone()
            opt.zero_grad()
            (1 - sim).backward()
            opt.step()
    return best_w.cpu().numpy(), np.array(hist), best


def word_for(b):
    with st.spinner("Turning the image into a word (first time only, a few seconds)..."):
        return image_to_word(b)


@st.cache_data(show_spinner=False, max_entries=512)
def compose(b: bytes, prompt: str) -> np.ndarray:
    _, _, device = load_clip()
    w = image_to_word(b)[0]
    ids, att = _tok([prompt])
    with torch.no_grad():
        return _text_feats(ids, att, torch.tensor(w, device=device))[0].cpu().numpy()


@st.cache_data(show_spinner=False, max_entries=64)
def nearest_words(w_np: np.ndarray, k: int = 6):
    model, processor, _ = load_clip()
    table = model.text_model.embeddings.token_embedding.weight
    with torch.no_grad():
        w = torch.tensor(w_np, device=table.device)
        sims = (table @ w) / (table.norm(dim=-1) * w.norm() + 1e-8)
        top = torch.topk(sims, 60).indices.tolist()
    words = []
    for i in top:
        t = str(processor.tokenizer.convert_ids_to_tokens(int(i))).replace("</w>", "")
        if t.isalpha() and len(t) > 2 and t.lower() not in words:
            words.append(t.lower())
        if len(words) == k:
            break
    return words


@st.cache_resource(show_spinner=False)
def token_stats():
    model, _, _ = load_clip()
    table = model.text_model.embeddings.token_embedding.weight
    with torch.no_grad():
        return int(table.shape[0]), int(table.shape[1]), float(table.norm(dim=-1).mean().item())


def database():
    with st.spinner("Encoding the image database (first time only)..."):
        return np.stack([embed_image(POOL[l]["bytes"]) for l in LABELS])


def search(q, D, exclude, k):
    s = D @ q
    s = s.astype(float)
    s[exclude] = -np.inf
    return list(np.argsort(-s)[:k]), s


STOP = {"a", "an", "the", "of", "in", "on", "photo", "picture", "image", "and", "with", "but", "that", "is",
        "it", "its", "made", "style", "more", "less", "same", "one", "version"}


def change_words(text):
    return {w for w in "".join(c if c.isalnum() else " " for c in text.lower()).split() if w not in STOP and len(w) > 2}


def badges(query_lbl, res_lbl, words):
    """Two checks a composed search should pass: same kind of thing, and the requested change."""
    qc, rc = POOL[query_lbl]["cat"], POOL[res_lbl]["cat"]
    kind = None if (qc in NO_KIND or rc in NO_KIND) else (qc == rc)
    change = None if not words else bool(words & set(res_lbl.lower().split()))
    return kind, change


def badge_html(kind, change):
    out = ""
    if kind is not None:
        out += f'<span class="bd {"y" if kind else "n"}">kind {"✓" if kind else "✗"}</span>'
    if change is not None:
        out += f'<span class="bd {"y" if change else "n"}">change {"✓" if change else "✗"}</span>'
    return out


def results_html(query_lbl, idx, scores, words, cols=None):
    tiles = ""
    for i in idx:
        lbl = LABELS[i]
        kind, change = badges(query_lbl, lbl, words)
        good = kind is not False and change is not False and (kind or change)
        tiles += (f'<div class="rt{" good" if good else ""}"><img src="{thumb_uri(POOL[lbl]["bytes"], 150)}"/>'
                  f'<div class="rl">{esc(lbl)}</div><div class="rs">{scores[i]:.3f}</div>'
                  f'<div>{badge_html(kind, change)}</div></div>')
    n = cols or len(idx)
    return f'<div class="rgrid" style="grid-template-columns:repeat({n}, minmax(0, 150px))">{tiles}</div>'


def both_count(query_lbl, idx, words):
    marks = [badges(query_lbl, LABELS[i], words) for i in idx]
    if any(k is None or c is None for k, c in marks):
        return None
    return sum(1 for k, c in marks if k and c)


def wchip(b, size=30):
    return f'<span class="wchip"><img src="{thumb_uri(b, 80)}" style="width:{size}px;height:{size}px"/>*</span>'


def sentence_html(prompt, b):
    parts = []
    for piece in prompt.split():
        if piece == "*":
            parts.append(wchip(b))
        elif "*" in piece:
            parts.extend([wchip(b), f'<span class="tk pr">{esc(piece.replace("*", ""))}</span>'])
        else:
            parts.append(f'<span class="tk pr">{esc(piece)}</span>')
    return '<div class="seq" style="justify-content:flex-start">' + "".join(parts) + "</div>"


def p2w_prompt(mod):
    return TEMPLATE.replace("{}", mod.replace("*", "").strip())


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    html(f'<div style="font-weight:800;font-size:22px;color:{NAVY}">🔎 Pic2Word Walkthrough</div>'
         f'<div style="margin-top:4px"><span class="tag" style="background:{GOLD};color:{NAVY}">SEAS 8525</span>'
         f'<span style="color:{MUTED};font-size:14px;margin-left:8px">Week 6 · Dr. Elbasheer</span></div>')
    st.divider()
    st.markdown("**Image database**")
    st.caption("Any image here can be a query or a search result. Add your own below.")
    uploads = st.file_uploader("Add images", type=["png", "jpg", "jpeg", "webp"],
                               accept_multiple_files=True, label_visibility="collapsed")
    POOL = build_pool(uploads)
    if POOL:
        with st.expander(f"Show all {len(POOL)} images"):
            tiles = "".join(f'<div><img src="{thumb_uri(v["bytes"], 120)}"/>{esc(k)}</div>'
                            for k, v in POOL.items())
            html(f'<div class="gal">{tiles}</div>')
    else:
        st.warning("No images could be loaded. Upload a few images above.")
    st.divider()
    if torch.cuda.is_available():
        st.caption(f"⚡ GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.caption("Running on CPU. CLIP is small; turning a new image into a word takes a few seconds.")
    with st.expander("Run it locally"):
        st.code("pip install -r requirements.txt\nstreamlit run pic2word_app.py", language="bash")

LABELS = list(POOL)
TEMPLATE = "a photo of * , {}"
QUICK_MODS = ["red", "black", "blue", "white", "made of leather", "yellow"]


def need_images(k=2):
    if len(LABELS) < k:
        st.info(f"This section needs at least {k} images. Add some in the sidebar.")
        return True
    return False


def pick_image(key, label="Query image", default="white sneakers"):
    idx = LABELS.index(default) if default in LABELS else 0
    return st.selectbox(label, LABELS, index=idx, key=key)


def mod_input(key):
    if key not in st.session_state:
        st.session_state[key] = "red"
    mod = st.text_input("Change to make", key=key)
    cols = st.columns(len(QUICK_MODS))
    for k, m in enumerate(QUICK_MODS):
        with cols[k]:
            st.button(m, key=f"{key}_q{k}", on_click=lambda m=m: st.session_state.__setitem__(key, m))
    return mod.replace("*", "").strip() or "red"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 · BIG PICTURE
# ═════════════════════════════════════════════════════════════════════════════
def vboxes(x, color, bg, cy, n=5, w=24, h=18, g=3):
    top = cy - (n * h + (n - 1) * g) / 2
    return "".join(f'<rect x="{x}" y="{top + k * (h + g):.1f}" width="{w}" height="{h}" rx="3" '
                   f'fill="{bg}" stroke="{color}" stroke-width="1.8"/>' for k in range(n))


def arch_svg(q_uri, mod, results):
    AR = 'stroke="#334155" stroke-width="2.4" marker-end="url(#ah)"'
    cy = 175

    def arrow(x1, x2, y=cy):
        return f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" {AR}/>'

    m = mod if len(mod) <= 14 else mod[:13] + "…"
    res = ""
    spots = [(1020, cy - 104), (1126, cy - 104), (1020, cy + 4), (1126, cy + 4)]
    for k, (x, y) in enumerate(spots):
        if results and k < len(results):
            res += (f'<image href="{results[k]}" x="{x}" y="{y}" width="100" height="100" preserveAspectRatio="xMidYMid slice"/>'
                    f'<rect x="{x}" y="{y}" width="100" height="100" rx="6" fill="none" stroke="{NEG}" stroke-width="2"/>')
        else:
            res += (f'<rect x="{x}" y="{y}" width="100" height="100" rx="8" fill="#FFFFFF" stroke="#CBD5E1" '
                    f'stroke-width="2" stroke-dasharray="6 5"/>'
                    f'<text x="{x + 50}" y="{y + 60}" text-anchor="middle" font-size="30" fill="#CBD5E1" font-weight="800">?</text>')
    return f"""
    <div class="panel">
    <svg viewBox="0 48 1240 300" width="100%" style="font-family:Lexend, sans-serif">
      <defs><marker id="ah" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="#334155"/></marker></defs>

      <text x="85" y="{cy - 72}" text-anchor="middle" font-size="16" font-weight="700" fill="{IMG}">Query image</text>
      <image href="{q_uri}" x="25" y="{cy - 60}" width="120" height="120" preserveAspectRatio="xMidYMid slice"/>
      <rect x="25" y="{cy - 60}" width="120" height="120" fill="none" stroke="{IMG}" stroke-width="3" rx="6"/>
      {arrow(150, 172)}

      <polygon points="178,{cy - 80} 298,{cy - 47} 298,{cy + 47} 178,{cy + 80}" fill="{IMG_BG}" stroke="{IMG}" stroke-width="2.5"/>
      <text x="238" y="{cy - 10}" text-anchor="middle" font-size="15" font-weight="700" fill="{IMG}">CLIP image</text>
      <text x="238" y="{cy + 10}" text-anchor="middle" font-size="15" font-weight="700" fill="{IMG}">encoder</text>
      <text x="238" y="{cy + 34}" text-anchor="middle" font-size="13" font-weight="700" fill="{IMG}">❄ frozen</text>
      {arrow(302, 322)}

      {vboxes(328, IMG, IMG_BG, cy)}
      <text x="340" y="{cy + 74}" text-anchor="middle" font-size="12.5" fill="#475569">image</text>
      <text x="340" y="{cy + 90}" text-anchor="middle" font-size="12.5" fill="#475569">embedding</text>
      {arrow(356, 376)}

      <polygon points="382,{cy - 62} 482,{cy - 32} 482,{cy + 32} 382,{cy + 62}" fill="{PROJ_BG}" stroke="{PROJ}" stroke-width="2.5"/>
      <text x="430" y="{cy - 4}" text-anchor="middle" font-size="16" font-weight="800" fill="#B45309">Mapper</text>
      <text x="430" y="{cy + 18}" text-anchor="middle" font-size="12.5" fill="#B45309">🔥 trained</text>
      {arrow(486, 506)}

      <rect x="512" y="{cy - 32}" width="30" height="64" rx="5" fill="{PROJ_BG}" stroke="{PROJ}" stroke-width="2.5"/>
      <text x="527" y="{cy + 9}" text-anchor="middle" font-size="22" font-weight="800" fill="#B45309">*</text>
      <text x="527" y="{cy + 56}" text-anchor="middle" font-size="12.5" fill="#475569">pseudo-</text>
      <text x="527" y="{cy + 72}" text-anchor="middle" font-size="12.5" fill="#475569">word</text>
      {arrow(546, 566)}

      <text x="676" y="{cy - 70}" text-anchor="middle" font-size="14" font-weight="700" fill="#0B7A55">prompt</text>
      <rect x="572" y="{cy - 58}" width="208" height="116" rx="14" fill="#FFFFFF" stroke="{TXT}" stroke-width="2"/>
      <text x="676" y="{cy - 18}" text-anchor="middle" font-size="18" fill="#1E293B">a photo of</text>
      <rect x="598" y="{cy + 2}" width="44" height="44" rx="7" fill="{PROJ_BG}" stroke="{PROJ}" stroke-width="2.5"/>
      <image href="{q_uri}" x="602" y="{cy + 6}" width="36" height="36" preserveAspectRatio="xMidYMid slice"/>
      <text x="650" y="{cy + 31}" font-size="18" fill="#1E293B">, {esc(m)}</text>
      {arrow(784, 806)}

      <polygon points="812,{cy - 47} 932,{cy - 80} 932,{cy + 80} 812,{cy + 47}" fill="{TXT_BG}" stroke="{TXT}" stroke-width="2.5"/>
      <text x="872" y="{cy - 10}" text-anchor="middle" font-size="15" font-weight="700" fill="#0B7A55">CLIP text</text>
      <text x="872" y="{cy + 10}" text-anchor="middle" font-size="15" font-weight="700" fill="#0B7A55">encoder</text>
      <text x="872" y="{cy + 34}" text-anchor="middle" font-size="13" font-weight="700" fill="#0B7A55">❄ frozen</text>
      {arrow(936, 956)}

      {vboxes(962, SPACE, SPACE_BG, cy)}
      <text x="974" y="{cy + 74}" text-anchor="middle" font-size="12.5" fill="#475569">composed</text>
      <text x="974" y="{cy + 90}" text-anchor="middle" font-size="12.5" fill="#475569">query</text>
      {arrow(990, 1012)}

      <text x="1123" y="{cy - 114}" text-anchor="middle" font-size="15" font-weight="700" fill="{NEG}">Top matches</text>
      {res}
      <text x="1123" y="{cy + 128}" text-anchor="middle" font-size="12.5" fill="#475569">cosine search in the database</text>

      <text x="25" y="{cy + 130}" font-size="13.5" font-weight="700" fill="{IMG}">❄ frozen: never updated</text>
      <text x="230" y="{cy + 130}" font-size="13.5" font-weight="700" fill="#C2410C">🔥 trained: only the small mapper</text>
    </svg></div>"""


FACTS = [
    ("Training setup", "What the original authors used", PROJ, [
        ("~3M", "unlabeled images",
         "Images from Conceptual Captions. The captions are not used: each image supervises its own word."),
        ("0", "composed triplets",
         "No (image, change, target) examples are ever needed, which is what makes it zero-shot."),
        ("1", "small trained network", "Only the mapper learns. Both CLIP encoders are frozen."),
    ]),
    ("This app", "What runs here", IMG, [
        ("ViT-B/32", "CLIP backbone", "The paper uses the larger ViT-L/14. The idea is identical."),
        ("1 × 512", "one pseudo-word", "The image becomes a single 512-number token, the same size as any word."),
        ("150", "gradient steps per image",
         "No trained mapper weights here, so the app solves the mapper's objective directly for each image."),
    ]),
    ("Results", "Reported in the paper (Saito et al., CVPR 2023)", POS, [
        ("4", "zero-shot benchmarks",
         "CIRR, Fashion-IQ, ImageNet domain conversion, and COCO object composition."),
        ("1", "model for every task", "The same mapper is used everywhere; only the sentence changes."),
    ]),
]


def sec_big_picture():
    hero(1, "Pic2Word: an image as a word in a sentence",
         "Zero-shot composed image retrieval · Saito et al., Google 2023")
    if need_images():
        return
    c1, c2 = st.columns([1, 2])
    with c1:
        lbl = pick_image("s1_img")
    with c2:
        mod = mod_input("s1_mod")
    b = POOL[lbl]["bytes"]
    prompt = p2w_prompt(mod)

    sig = ("s1", lbl, prompt)
    if run_button("Run this example", "s1_run"):
        D = database()
        idx, _ = search(compose_with_spinner(b, prompt), D, LABELS.index(lbl), 4)
        st.session_state["s1_res"] = {"sig": sig, "val": [thumb_uri(POOL[LABELS[i]]["bytes"], 200) for i in idx]}
    r = st.session_state.get("s1_res")
    results = r["val"] if r and r["sig"] == sig else None

    html(arch_svg(thumb_uri(b, 240), mod, results))
    if not results:
        st.caption("Press ▶ to fill in the top matches from the database.")

    t = thumb_uri(b, 120)
    html(f"""
    <div class="cards3">
      <div class="step" style="border-top:5px solid {TXT}"><h4 style="background:{TXT_BG};color:#0B7A55">Text only</h4>
        <div style="text-align:center;margin:8px 0"><span class="tk">"{esc(mod)}"</span></div>
        <div class="slabel" style="text-align:center;font-size:14px">✗ forgets the exact item you had in mind</div></div>
      <div class="step" style="border-top:5px solid {IMG}"><h4 style="background:{IMG_BG};color:{IMG}">Image only</h4>
        <div style="text-align:center"><img class="thumb" src="{t}" width="64" style="display:inline-block"/></div>
        <div class="slabel" style="text-align:center;font-size:14px">✗ cannot say what to change</div></div>
      <div class="step" style="border-top:5px solid {PROJ}"><h4 style="background:{PROJ_BG};color:#B45309">Image + text (composed)</h4>
        <div style="text-align:center;display:flex;gap:8px;align-items:center;justify-content:center">
          <img class="thumb" src="{t}" width="64"/><b style="font-size:22px;color:#94A3B8">+</b><span class="tk">"{esc(mod)}"</span></div>
        <div class="slabel" style="text-align:center;font-size:14px">✓ this item, with this change</div></div>
    </div>""")

    facts_panel(FACTS, ("Pic2Word by the numbers", "from the original paper, Saito et al. 2023"))

    key_idea("Pic2Word turns the query image into <b>one pseudo-word</b>. Put it in a sentence like "
             f"<b style='color:#B45309'>\"a photo of * , {esc(mod)}\"</b> and CLIP's frozen text encoder reads it like any "
             f"other word. Only the small <b style='color:#C2410C'>mapper</b> is ever trained.")

    code_panel("the whole method in a few lines", '''\
e_img = clip.encode_image(query_image)            # frozen image encoder
w     = mapper(e_img)                             # trained: image -> one pseudo-word
q     = clip.encode_text("a photo of * , red", star=w)   # frozen text encoder reads w as a word
scores = database_image_embeddings @ q            # cosine similarity (all unit vectors)
top    = scores.argsort()[::-1][:4]''')


def compose_with_spinner(b, prompt):
    word_for(b)
    return compose(b, prompt)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 · WHERE A WORD LIVES
# ═════════════════════════════════════════════════════════════════════════════
def sec_word():
    hero(2, "Where does a word live?", "The pseudo-word enters CLIP's text encoder at the very first layer")
    if need_images(1):
        return
    c1, c2 = st.columns([1, 2])
    with c1:
        lbl = pick_image("s2_img", "Image for the * slot")
    with c2:
        prompt = st.text_input("Prompt (* marks where the image goes)", value="a photo of * , red", key="s2_prompt")
    if "*" not in prompt:
        st.warning("Put a * in the prompt to mark where the image goes.")
        return
    b = POOL[lbl]["bytes"]
    model, processor, _ = load_clip()
    w, _, _ = word_for(b)
    table = model.text_model.embeddings.token_embedding.weight
    vocab, dim, avg_norm = token_stats()
    ids = processor.tokenizer(prompt, truncation=True)["input_ids"]
    sid = star_id()

    rows = ""
    for k, t in enumerate(ids):
        if t == sid:
            tok, src, vec, color = wchip(b, 26), '<b style="color:#B45309">from the mapper</b>', w, PROJ
        else:
            name = "[SOT]" if k == 0 else "[EOT]" if k == len(ids) - 1 else (processor.tokenizer.decode([t]).strip() or "·")
            cls = "sp" if k == 0 else "eot" if k == len(ids) - 1 else ""
            tok = f'<span class="tk {cls}">{esc(name)}</span>'
            src = f"table row {t:,}"
            with torch.no_grad():
                vec = table[t].detach().float().cpu().numpy()
            color = TXT
        rows += f'<div class="trow">{tok}<span class="src">{src}</span>{strip_html(vec, color, 48)}</div>'

    html(f"""
    <div class="panel">
      <div class="ptitle"><span class="tag" style="background:{TXT_BG};color:#0B7A55">INPUT</span>
        Every word is looked up in a table of {vocab:,} rows. The * is not: its row comes from the image.</div>
      <div style="display:grid;grid-template-columns:1.7fr 40px 1fr;gap:10px;align-items:center">
        <div>{rows}<div class="slabel">first 48 of {dim} numbers per word · blue-green = positive, red = negative</div></div>
        <div class="arrow" style="text-align:center">→</div>
        <div class="vpipe">
          <div class="node n-plain">+ position embeddings</div><div class="down">↓</div>
          <div class="node n-txt">Text encoder × 12 ❄<small>causal self-attention</small></div><div class="down">↓</div>
          <div class="node n-txt"><span class="tk eot">[EOT]</span><small>sentence summary</small></div><div class="down">↓</div>
          <div class="node n-proj">Projection<small>{dim} → 512</small></div><div class="down">↓</div>
          <div class="node n-space">Composed query<small>512-d, unit length</small></div>
        </div>
      </div>
    </div>""")

    wn = float(np.linalg.norm(w))
    html(f"""
    <div class="spaces">
      <div class="sp" style="border-top:5px solid {PROJ}"><div class="gh" style="color:#B45309">Token space · the input</div>
        <ul><li>{vocab:,} words, each a row of {dim} numbers</li>
            <li><b>Pic2Word writes the image here</b>, as one new row</li>
            <li>rows are not normalized: a typical word has length {avg_norm:.2f}, this pseudo-word {wn:.2f}</li></ul></div>
      <div style="text-align:center;font-weight:700;color:#64748B">→<br>text<br>encoder<br>→</div>
      <div class="sp" style="border-top:5px solid {SPACE}"><div class="gh" style="color:{SPACE}">Embedding space · the output</div>
        <ul><li>512-number unit vectors</li>
            <li>whole sentences and whole images are compared here</li>
            <li>the image encoder outputs here too, so search is a dot product</li></ul></div>
    </div>""")

    key_idea("A word is just a <b>row in a lookup table</b>. Pic2Word leaves the table alone and slips in "
             f"<b style='color:#B45309'>a new row made from the image</b> wherever * appears. "
             "Everything after that is ordinary, frozen CLIP.")

    code_panel("a mapper, and how the word is inserted", '''\
import torch.nn as nn

class Mapper(nn.Module):
    """Small MLP: CLIP image embedding -> one token embedding (the paper uses 3 linear layers)."""
    def __init__(self, img_dim=512, hidden=512, token_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, token_dim),
        )
    def forward(self, e_img):
        return self.net(e_img)          # NOT normalized: it must look like a word embedding

# Insert w wherever the prompt has "*", at the token-embedding layer
star = tokenizer("*")["input_ids"][1]
def swap(module, inputs, out):
    return torch.where((ids == star).unsqueeze(-1), w, out)
hook = clip.text_model.embeddings.token_embedding.register_forward_hook(swap)
q = clip.text_projection(clip.text_model(input_ids=ids).pooler_output)
hook.remove()''')


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 · TRAINING THE MAPPER
# ═════════════════════════════════════════════════════════════════════════════
def loop_svg(q_uri):
    AR = 'stroke="#334155" stroke-width="2.4" marker-end="url(#ah)"'
    OR = f'stroke="{PROJ}" stroke-width="2.6" stroke-dasharray="7 5" marker-end="url(#ao)"'

    def strip(x, y, color, bg, w=110, h=30, n=5):
        cw = w / n
        return "".join(f'<rect x="{x + i * cw:.1f}" y="{y}" width="{cw:.1f}" height="{h}" fill="{bg}" stroke="{color}" stroke-width="2"/>'
                       for i in range(n))

    return f"""
    <div class="panel">
    <svg viewBox="0 70 1230 300" width="100%" style="font-family:Lexend, sans-serif">
      <defs><marker id="ah" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="#334155"/></marker>
      <marker id="ao" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="{PROJ}"/></marker></defs>

      <image href="{q_uri}" x="20" y="95" width="100" height="100" preserveAspectRatio="xMidYMid slice"/>
      <rect x="20" y="95" width="100" height="100" rx="6" fill="none" stroke="{IMG}" stroke-width="3"/>
      <text x="70" y="214" text-anchor="middle" font-size="12.5" fill="#475569">an unlabeled image</text>
      <line x1="124" y1="145" x2="146" y2="145" {AR}/>
      <polygon points="152,100 262,124 262,166 152,190" fill="{IMG_BG}" stroke="{IMG}" stroke-width="2.5"/>
      <text x="207" y="142" text-anchor="middle" font-size="14" font-weight="700" fill="{IMG}">image encoder</text>
      <text x="207" y="162" text-anchor="middle" font-size="12.5" font-weight="700" fill="{IMG}">❄ frozen</text>
      <line x1="266" y1="145" x2="290" y2="145" {AR}/>
      {strip(296, 130, IMG, IMG_BG)}
      <text x="351" y="120" text-anchor="middle" font-size="12.5" font-weight="700" fill="{IMG}">image embedding = target</text>
      <path d="M410,145 L1100,145 L1100,166" fill="none" stroke="{IMG}" stroke-width="2.4" marker-end="url(#ah)"/>

      <line x1="351" y1="162" x2="351" y2="262" {AR}/>
      <rect x="296" y="268" width="110" height="64" rx="12" fill="{PROJ_BG}" stroke="{PROJ}" stroke-width="2.5"/>
      <text x="351" y="296" text-anchor="middle" font-size="15" font-weight="800" fill="#B45309">Mapper</text>
      <text x="351" y="316" text-anchor="middle" font-size="12.5" fill="#B45309">🔥 trained</text>
      <line x1="410" y1="300" x2="432" y2="300" {AR}/>
      <rect x="438" y="272" width="26" height="56" rx="5" fill="{PROJ_BG}" stroke="{PROJ}" stroke-width="2.5"/>
      <text x="451" y="307" text-anchor="middle" font-size="20" font-weight="800" fill="#B45309">*</text>
      <line x1="468" y1="300" x2="490" y2="300" {AR}/>
      <rect x="496" y="268" width="170" height="64" rx="12" fill="#FFFFFF" stroke="{TXT}" stroke-width="2"/>
      <text x="556" y="306" text-anchor="middle" font-size="16" fill="#1E293B">a photo of</text>
      <rect x="614" y="284" width="32" height="32" rx="6" fill="{PROJ_BG}" stroke="{PROJ}" stroke-width="2.2"/>
      <text x="630" y="306" text-anchor="middle" font-size="18" font-weight="800" fill="#B45309">*</text>
      <line x1="670" y1="300" x2="692" y2="300" {AR}/>
      <polygon points="698,268 808,250 808,350 698,332" fill="{TXT_BG}" stroke="{TXT}" stroke-width="2.5"/>
      <text x="753" y="296" text-anchor="middle" font-size="14" font-weight="700" fill="#0B7A55">text encoder</text>
      <text x="753" y="316" text-anchor="middle" font-size="12.5" font-weight="700" fill="#0B7A55">❄ frozen</text>
      <line x1="812" y1="300" x2="834" y2="300" {AR}/>
      {strip(840, 285, TXT, TXT_BG)}
      <text x="895" y="350" text-anchor="middle" font-size="12.5" font-weight="700" fill="#0B7A55">sentence embedding</text>
      <path d="M950,300 L1100,300 L1100,276" fill="none" stroke="{TXT}" stroke-width="2.4" marker-end="url(#ah)"/>

      <rect x="1000" y="170" width="210" height="102" rx="14" fill="#FFFBEB" stroke="{GOLD}" stroke-width="2.5"/>
      <text x="1105" y="200" text-anchor="middle" font-size="16" font-weight="800" fill="{NAVY}">Loss</text>
      <text x="1105" y="222" text-anchor="middle" font-size="13" fill="#334155">make the two</text>
      <text x="1105" y="240" text-anchor="middle" font-size="13" fill="#334155">embeddings match</text>
      <text x="1105" y="260" text-anchor="middle" font-size="11.5" fill="#64748B">(contrastive over a batch)</text>

      <path d="M996,222 L400,222 L400,262" fill="none" {OR}/>
      <text x="700" y="212" text-anchor="middle" font-size="13.5" font-weight="700" fill="#C2410C">gradient: only the mapper learns</text>
    </svg></div>"""


def curve_svg(hist, ref, ref_label):
    W, H, L, R, T, B = 640, 250, 56, 18, 18, 40
    lo = min(float(hist.min()), ref) - 0.02
    hi = max(float(hist.max()), ref) + 0.02
    X = lambda i: L + i / max(len(hist) - 1, 1) * (W - L - R)
    Y = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    grid = ""
    for k in range(5):
        v = lo + k * (hi - lo) / 4
        grid += (f'<line x1="{L}" y1="{Y(v):.1f}" x2="{W - R}" y2="{Y(v):.1f}" stroke="#EEF2F7"/>'
                 f'<text x="{L - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" font-size="11" fill="#64748B">{v:.2f}</text>')
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(hist))
    return f"""
    <svg viewBox="0 0 {W} {H}" width="100%" style="font-family:Lexend, sans-serif;max-width:{W}px">
      {grid}
      <line x1="{L}" y1="{Y(ref):.1f}" x2="{W - R}" y2="{Y(ref):.1f}" stroke="{TXT}" stroke-width="2" stroke-dasharray="6 5"/>
      <text x="{W - R}" y="{Y(ref) - 6:.1f}" text-anchor="end" font-size="11.5" fill="#0B7A55" font-weight="700">{esc(ref_label)}</text>
      <polyline points="{pts}" fill="none" stroke="{PROJ}" stroke-width="3"/>
      <circle cx="{X(len(hist) - 1):.1f}" cy="{Y(hist[-1]):.1f}" r="5" fill="{PROJ}"/>
      <text x="{(L + W - R) / 2}" y="{H - 8}" text-anchor="middle" font-size="12" fill="#475569">gradient step</text>
      <text x="14" y="{(T + H - B) / 2}" text-anchor="middle" font-size="12" fill="#475569"
            transform="rotate(-90 14 {(T + H - B) / 2})">cosine to the image</text>
    </svg>"""


def sec_training():
    hero(3, "Teaching an image its word", "How the mapper is trained, with no captions and no triplets")
    if need_images(1):
        return
    lbl = pick_image("s3_img", "Image")
    b = POOL[lbl]["bytes"]
    html(loop_svg(thumb_uri(b, 200)))

    st.markdown(f"<div style='font-weight:700;color:{NAVY};font-size:17px;margin-top:6px'>"
                "Watch it happen for one image</div>", unsafe_allow_html=True)
    sig = ("s3", lbl)
    if run_button("Turn this image into a word", "s3_run"):
        st.session_state["s3_res"] = {"sig": sig, "val": word_for(b)}
    r = st.session_state.get("s3_res")
    if r and r["sig"] == sig:
        w, hist, best = r["val"]
        caption = f"a photo of {lbl}"
        ref = float(embed_text(caption) @ embed_image(b))
        words = nearest_words(w)
        left, right = st.columns([1.5, 1])
        with left:
            html(f'<div class="panel">{curve_svg(hist, ref, f"human caption: {caption}")}</div>')
        with right:
            chips([(f"{hist[0]:.2f}", 'start: "a photo of thing"'),
                   (f"{best:.2f}", f"after {len(hist)} steps"),
                   (f"{ref:.2f}", "a human-written caption")])
            wc = "".join(f'<span class="tk">{esc(x)}</span>' for x in words)
            html(f'<div class="panel"><div class="ptitle" style="font-size:15px">Closest real words to this pseudo-word</div>'
                 f'<div class="wordchips">{wc}</div><div class="slabel" style="margin-top:8px">'
                 f'Often only loosely related: the pseudo-word does not have to be a real word.</div></div>')
    else:
        st.caption("Press ▶ to run the optimization and see the similarity climb.")

    html(f"""<div class="panel" style="font-size:15px;border-left:5px solid {GOLD}">
      <b style="color:{NAVY}">About this app.</b> Pic2Word trains one mapper on millions of images, so at search time
      the word comes from a single forward pass. The trained weights are not available in a form this app can load,
      so here we solve <b>the same objective</b> for each image directly: 150 small gradient steps on the pseudo-word itself.
    </div>""")

    key_idea("The training signal is <b>the image itself</b>: a good pseudo-word makes "
             f"<b style='color:#B45309'>\"a photo of *\"</b> land on the image's own embedding. So Pic2Word needs only "
             f"<b>unlabeled images</b>: no captions, no composed triplets. Both CLIP encoders stay "
             f"<b style='color:{IMG}'>frozen</b>.")

    code_panel("training the mapper (Pic2Word) and what this app does instead", '''\
# Pic2Word: train one mapper on a big set of unlabeled images
for images in loader:
    e_img = clip.encode_image(images)                       # [B, d], frozen
    w = mapper(e_img)                                       # [B, token_dim], one word per image
    e_txt = encode_text_with_word("a photo of *", w)        # [B, d], frozen text encoder
    logits = e_txt @ e_img.T / tau                          # contrastive, like CLIP
    loss = cross_entropy(logits, arange(B))
    loss.backward(); opt.step()                             # only the mapper has gradients

# This app: optimize the word directly for ONE image (same objective, no mapper)
w = token_table["thing"].clone().requires_grad_()
opt = torch.optim.AdamW([w], lr=0.02)
for step in range(150):
    sim = encode_text_with_word("a photo of *", w) @ e_img
    (1 - sim).backward(); opt.step(); opt.zero_grad()''')


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 · COMPOSED RETRIEVAL
# ═════════════════════════════════════════════════════════════════════════════
def sec_retrieval():
    hero(4, "Composed image retrieval", "Search with an image and a change at the same time")
    if need_images(3):
        return
    c1, c2, c3 = st.columns([1, 2.2, 0.8])
    with c1:
        lbl = pick_image("s4_img")
    with c2:
        mod = mod_input("s4_mod")
    with c3:
        k = st.slider("Results", 3, 6, 4, key="s4_k")
    b = POOL[lbl]["bytes"]
    qi = LABELS.index(lbl)
    prompt = p2w_prompt(mod)
    words = change_words(mod)

    D = database()
    e_img = embed_image(b)
    e_txt = embed_text(mod)
    e_avg = e_img + e_txt
    e_avg = e_avg / np.linalg.norm(e_avg)
    e_p2w = compose_with_spinner(b, prompt)

    idx, s = search(e_p2w, D, qi, k)
    html(f"""
    <div class="panel">
      <div class="ptitle"><span class="tag" style="background:{PROJ_BG};color:#B45309">PIC2WORD</span>Your search, as one sentence</div>
      <div class="pipe" style="margin-bottom:14px">
        <img class="thumb" src="{thumb_uri(b, 120)}" width="74"/>
        <div class="arrow">→</div>
        <div class="node n-proj">image → word</div>
        <div class="arrow">→</div>
        <div class="node n-plain" style="min-width:250px">{sentence_html(prompt, b)}</div>
        <div class="arrow">→</div>
        <div class="node n-txt">CLIP text encoder ❄</div>
        <div class="arrow">→</div>
        <div class="node n-space">composed query</div>
      </div>
      {results_html(lbl, idx, s, words)}
    </div>""")

    methods = [
        ("Image only", "the query image's own embedding", IMG, e_img),
        ("Text only", f'just "{esc(mod)}"', TXT, e_txt),
        ("Average", "normalize(image + text): the simple baseline", SPACE, e_avg),
        ("Pic2Word", f'"{esc(prompt)}"', PROJ, e_p2w),
    ]
    rows = ""
    for name, sub, color, q in methods:
        ix, sc = search(q, D, qi, k)
        n_ok = both_count(lbl, ix, words)
        score = "" if n_ok is None else (f'<div class="score" style="color:{POS if n_ok else MUTED}">'
                                         f'{n_ok} of {k} have both ✓</div>')
        rows += (f'<div class="mrow"><div><div class="mname" style="color:{color}">{name}<small>{sub}</small></div>{score}</div>'
                 f'{results_html(lbl, ix, sc, words, cols=k)}</div>')
    html(f'<div class="panel"><div class="ptitle">Four ways to search the same database</div>{rows}'
         f'<div class="slabel" style="margin-top:6px"><b>kind ✓</b> = same kind of thing as the query · '
         f'<b>change ✓</b> = has the change you asked for · green frame = both</div></div>')

    with st.expander("🧪  Experiment: can we just slide between the image and the text?"):
        a = st.slider("Weight on the text  (0 = image only, 1 = text only)", 0.0, 1.0, 0.5, 0.05, key="s4_alpha")
        q = (1 - a) * e_img + a * e_txt
        q = q / np.linalg.norm(q)
        ix, sc = search(q, D, qi, k)
        ii = float(np.sort(np.delete(D @ e_img, qi))[-1])
        tt = float(np.sort(np.delete(D @ e_txt, qi))[-1])
        html(results_html(lbl, ix, sc, words))
        html(f'<div class="slabel" style="font-size:14px;margin-top:8px">The best image-to-image score here is '
             f'<b>{ii:.2f}</b>, the best text-to-image score only <b>{tt:.2f}</b>. Because of CLIP\'s modality gap '
             f'the image side dominates the average, so a single weight rarely gets both the item and the change.</div>')

    key_idea(f"<b style='color:{IMG}'>Image only</b> ignores the change and <b style='color:{TXT}'>text only</b> forgets "
             f"the item. <b style='color:{SPACE}'>Averaging</b> the two vectors mostly stays on the image side. "
             f"<b style='color:#B45309'>Pic2Word</b> lets the text encoder combine them the way it combines words.")

    code_panel("the four queries", f'''\
e_img = encode_image(query)                          # image only
e_txt = encode_text("{esc(mod)}")                    # text only
e_avg = normalize(e_img + e_txt)                     # average baseline
e_p2w = encode_text_with_word("{esc(prompt)}", w)    # Pic2Word

for q in (e_img, e_txt, e_avg, e_p2w):
    scores = database @ q                            # cosine similarity
    scores[query_index] = -inf                       # never return the query itself
    print(scores.argsort()[::-1][:{k}])''')


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 · ONE WORD, MANY SENTENCES
# ═════════════════════════════════════════════════════════════════════════════
def sec_sentences():
    hero(5, "One word, many sentences", "Once the image is a word, you can use it in any sentence")
    if need_images(3):
        return
    c1, c2 = st.columns([1, 2])
    with c1:
        lbl = pick_image("s5_img")
        html(f'<img class="thumb" src="{thumb_uri(POOL[lbl]["bytes"], 220)}" style="width:100%;max-width:200px"/>')
    with c2:
        txt = st.text_area("Sentences (one per line, * = your image)",
                           value="a photo of * , red\na photo of * , black\na photo of * , blue\n"
                                 "a photo of * , yellow\na photo of * , made of leather",
                           height=170, key="s5_prompts")
    prompts = [p.strip() for p in txt.splitlines() if p.strip()]
    bad = [p for p in prompts if "*" not in p]
    if bad:
        st.warning("These lines have no * and were skipped: " + "; ".join(bad))
    prompts = [p for p in prompts if "*" in p][:8]
    if not prompts:
        return
    b = POOL[lbl]["bytes"]
    qi = LABELS.index(lbl)
    D = database()
    word_for(b)
    rows = ""
    for p in prompts:
        ix, sc = search(compose(b, p), D, qi, 4)
        rows += (f'<div class="mrow" style="grid-template-columns:300px 1fr"><div>{sentence_html(p, b)}</div>'
                 f'{results_html(lbl, ix, sc, change_words(p.replace("*", " ")), cols=4)}</div>')
    html(f'<div class="panel"><div class="ptitle">The same pseudo-word, reused in every sentence</div>{rows}</div>')

    key_idea("The image was turned into a word <b>once</b>. Each sentence just puts that word in a new context, "
             "exactly like reusing a real word. The paper uses the same trick for domain changes, such as "
             f"<b style='color:#B45309'>\"a sketch of *\"</b> or <b style='color:#B45309'>\"an origami of *\"</b>.")

    code_panel("reusing one pseudo-word", '''\
w = mapper(encode_image(query))                      # computed once
for sentence in ["a photo of * , red", "a sketch of *", "* on a beach"]:
    q = encode_text_with_word(sentence, w)
    print(sentence, (database @ q).argsort()[::-1][:4])''')


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 · EMBEDDING SPACE
# ═════════════════════════════════════════════════════════════════════════════
def sec_space():
    hero(6, "Inside the shared space", "Where the query, the change, and the composed query land")
    if need_images(4):
        return
    c1, c2, c3 = st.columns([1, 1.6, 0.8])
    with c1:
        lbl = pick_image("s6_img")
    with c2:
        mod = mod_input("s6_mod")
    with c3:
        method = seg("Projection", ["PCA", "t-SNE"], key="s6_method")
    b = POOL[lbl]["bytes"]
    qi = LABELS.index(lbl)
    prompt = p2w_prompt(mod)
    D = database()
    e_img = D[qi]
    e_txt = embed_text(mod)
    e_avg = (e_img + e_txt) / np.linalg.norm(e_img + e_txt)
    e_p2w = compose_with_spinner(b, prompt)
    top, _ = search(e_p2w, D, qi, 3)

    n = len(D)
    X = np.vstack([D, e_txt, e_avg, e_p2w])
    if method == "PCA":
        XY = PCA(n_components=2, random_state=0).fit_transform(X)
    else:
        XY = TSNE(n_components=2, perplexity=max(2, min(10, len(X) - 1)), random_state=42,
                  init="pca", learning_rate="auto").fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFBFD")
    for sp in ax.spines.values():
        sp.set_edgecolor("#D9E1EC")
    ax.set_xticks([])
    ax.set_yticks([])
    span = XY.max(0) - XY.min(0) + 1e-6
    ax.set_xlim(XY[:, 0].min() - 0.22 * span[0], XY[:, 0].max() + 0.22 * span[0])
    ax.set_ylim(XY[:, 1].min() - 0.22 * span[1], XY[:, 1].max() + 0.22 * span[1])
    ax.set_xlabel("PC 1" if method == "PCA" else "t-SNE 1", color="#475569")
    ax.set_ylabel("PC 2" if method == "PCA" else "t-SNE 2", color="#475569")
    ax.legend(handles=[
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#94A3B8", markersize=9, label="database image"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=GOLD, markersize=11, label="your query image"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TXT, markersize=11, label="the change (text)"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor=SPACE, markersize=12, label="average"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=PROJ, markersize=10, label="Pic2Word query"),
        Line2D([0], [0], ls=":", color=PROJ, label="its top 3 matches"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=6, fontsize=9.5, frameon=False)
    fig.tight_layout()

    pt, pa, pp = XY[n], XY[n + 1], XY[n + 2]
    for i in top:
        ax.plot([pp[0], XY[i, 0]], [pp[1], XY[i, 1]], color=PROJ, lw=1.6, ls=":", zorder=2)
    ax.annotate("", xy=pp, xytext=XY[qi], arrowprops=dict(arrowstyle="->", color="#94A3B8", lw=1.4, ls="--"), zorder=2)
    cats = sorted({POOL[l]["cat"] for l in LABELS})
    cat_color = {c: PAIR_COLORS[k % len(PAIR_COLORS)] for k, c in enumerate(cats)}
    for i, l in enumerate(LABELS):
        ax.scatter(*XY[i], marker="s", s=40 if i != qi else 90, color=GOLD if i == qi else cat_color[POOL[l]["cat"]],
                   edgecolors="white", linewidths=1, zorder=4)
    ax.scatter(*pt, marker="o", s=170, color=TXT, edgecolors="white", linewidths=1.8, zorder=5)
    ax.scatter(*pa, marker="X", s=220, color=SPACE, edgecolors="white", linewidths=1.8, zorder=5)
    ax.scatter(*pp, marker="D", s=170, color=PROJ, edgecolors="white", linewidths=1.8, zorder=5)

    px_per_pt = fig.dpi / 72
    t_small, t_big = 30, 50
    anchors = ax.transData.transform(XY)
    names = [f'"{mod}"', "average", "Pic2Word"]
    sizes = [((t_big if i == qi else t_small) * px_per_pt + 10,) * 2 for i in range(n)]
    sizes += [((len(s) * 6.8 + 18) * px_per_pt, 22 * px_per_pt) for s in names]
    box = ax.get_window_extent()
    pos = place_labels(anchors, np.array(sizes, dtype=float), anchors, (box.x0, box.y0, box.x1, box.y1))
    inv = ax.transData.inverted()
    for i, l in enumerate(LABELS):
        size = t_big if i == qi else t_small
        c = GOLD if i == qi else cat_color[POOL[l]["cat"]]
        ax.add_artist(AnnotationBbox(
            OffsetImage(np.asarray(clip_crop(to_pil(POOL[l]["bytes"]), size)), zoom=1.0), XY[i],
            xybox=inv.transform(pos[i]), boxcoords="data", frameon=True,
            bboxprops=dict(edgecolor=c, linewidth=3 if i == qi else 1.6, boxstyle="round,pad=0.06"),
            arrowprops=dict(arrowstyle="-", color=c, lw=1, shrinkA=0, shrinkB=3), zorder=7 if i == qi else 6))
    for k, (name, color, p) in enumerate(zip(names, [TXT, SPACE, PROJ], [pt, pa, pp])):
        ax.annotate(name, p, xytext=inv.transform(pos[n + k]), textcoords="data", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color, zorder=8,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color, linewidth=1.2),
                    arrowprops=dict(arrowstyle="-", color=color, lw=1.2, shrinkA=0, shrinkB=6))

    left, right = st.columns([2.6, 1])
    with left:
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    with right:
        chips([(f"{float(D[top[0]] @ e_p2w):.2f}", "Pic2Word query ↔ its best match"),
               (f"{float(np.sort(np.delete(D @ e_img, qi))[-1]):.2f}", "query image ↔ nearest other image"),
               (f"{float(e_img @ e_p2w):.2f}", "query image ↔ Pic2Word query")])
        html(f'<div class="panel" style="font-size:14.5px">The composed query is a <b>text</b> vector, so it sits on the '
             f'text side of CLIP\'s modality gap, away from every image. Retrieval still works because only the '
             f'<b>ranking</b> matters.</div>')

    key_idea("The Pic2Word query is built by the text encoder, so it lands on the <b>text side</b> of the space. "
             "From there, its nearest images are the ones that match <b>both</b> the item and the change. "
             "Compare it with the <b>average</b>, which stays close to the query image.")

    code_panel("project the database and the queries", '''\
X = np.vstack([database, e_txt, e_avg, e_p2w])       # all 512-d unit vectors
xy = PCA(n_components=2).fit_transform(X)            # or TSNE(...) for small sets''')


# ═════════════════════════════════════════════════════════════════════════════
# NAVIGATION
# ═════════════════════════════════════════════════════════════════════════════
SECTIONS = {
    "1 · Big picture": sec_big_picture,
    "2 · Where a word lives": sec_word,
    "3 · Training the mapper": sec_training,
    "4 · Composed retrieval": sec_retrieval,
    "5 · One word, many sentences": sec_sentences,
    "6 · Embedding space": sec_space,
}
NAMES = list(SECTIONS)

if "sec" not in st.session_state:
    st.session_state.sec = NAMES[0]
if "last_sec" not in st.session_state:
    st.session_state.last_sec = NAMES[0]


def _keep_sec():
    if st.session_state.sec is None:          # segmented control was clicked off
        st.session_state.sec = st.session_state.last_sec
    st.session_state.last_sec = st.session_state.sec


def _go(delta):
    i = NAMES.index(st.session_state.sec) + delta
    st.session_state.sec = NAMES[max(0, min(len(NAMES) - 1, i))]
    st.session_state.last_sec = st.session_state.sec


try:
    nav_box = st.container(key="topnav")
except TypeError:                              # Streamlit < 1.39 has no container keys
    nav_box = st.container()
with nav_box:
    if hasattr(st, "segmented_control"):
        st.segmented_control("Section", NAMES, key="sec", on_change=_keep_sec, label_visibility="collapsed")
    else:
        st.radio("Section", NAMES, key="sec", on_change=_keep_sec, horizontal=True, label_visibility="collapsed")

current = st.session_state.sec or st.session_state.last_sec
SECTIONS[current]()

st.markdown("")
idx = NAMES.index(current)
b1, _, b2 = st.columns([1, 4, 1])
with b1:
    if idx > 0:
        st.button(f"← {NAMES[idx - 1][4:]}", on_click=_go, args=(-1,), key="prev")
with b2:
    if idx < len(NAMES) - 1:
        st.button(f"{NAMES[idx + 1][4:]} →", on_click=_go, args=(1,), key="next")

