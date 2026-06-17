"""
Pic2Word: Composed Image Retrieval -- Interactive Walkthrough
SEAS 8525: Computer Vision and Generative AI
Dr. Elbasheer -- Week 6

This app demonstrates the Pic2Word concept using CLIP as the backbone.
The composition is implemented as a weighted combination of image and
text embeddings in CLIP's shared space -- an approximation of the
learned Pic2Word mapper that faithfully demonstrates the core idea.

Setup:
    pip install streamlit torch torchvision transformers
    pip install matplotlib pillow numpy requests scikit-learn

Run:
    streamlit run pic2word_app.py
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import requests
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.manifold import TSNE
from io import BytesIO

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pic2Word Walkthrough | SEAS 8525",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SIDEBAR WIDTH STATE ───────────────────────────────────────────────────────
if "sidebar_width" not in st.session_state:
    st.session_state.sidebar_width = 280

# ── THEME ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    [data-testid="stSidebar"] {{
        min-width: {st.session_state.sidebar_width}px !important;
        max-width: {st.session_state.sidebar_width}px !important;
        width:     {st.session_state.sidebar_width}px !important;
        background: #1A2F4A;
        border-right: 2px solid #0D2B4E;
        transition: width 0.15s ease;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* Base */
    [data-testid="stAppViewContainer"] { background: #F7F9FC; color: #1A2332; }
    [data-testid="stSidebar"] {
        background: #1A2F4A;
        border-right: 2px solid #0D2B4E;
        transition: width 0.15s ease;
    }
    [data-testid="stSidebar"] * { color: #E8EFF7 !important; }
    [data-testid="stSidebar"] .stCodeBlock,
    [data-testid="stSidebar"] pre,
    [data-testid="stSidebar"] code {
        background: #0A1628 !important;
        color: #80CBC4 !important;
        border: 1px solid #1E3A5F !important;
        border-radius: 6px;
    }
    [data-testid="stSidebar"] .stRadio label { color: #CBD5E1 !important; }

    /* Main text */
    h1 { color: #0D47A1 !important; font-family: Georgia, serif; }
    h2 { color: #1565C0 !important; font-family: Georgia, serif; }
    h3 { color: #1976D2 !important; }
    p, li { color: #1A2332; font-size: 15px; line-height: 1.7; }

    /* Info boxes */
    .concept-box {
        background: #E3F2FD;
        border-left: 5px solid #0097A7;
        border-radius: 6px;
        padding: 16px 20px;
        margin: 12px 0;
        color: #0D2B4E;
        font-size: 14px;
        line-height: 1.7;
    }
    .highlight-box {
        background: #EDE7F6;
        border-left: 5px solid #5E35B1;
        border-radius: 6px;
        padding: 16px 20px;
        margin: 12px 0;
        color: #1A0050;
        font-size: 14px;
        line-height: 1.7;
    }
    .warning-box {
        background: #FFF8E1;
        border-left: 5px solid #F9A825;
        border-radius: 6px;
        padding: 14px 18px;
        margin: 12px 0;
        color: #3E2723;
        font-size: 13px;
    }
    .success-box {
        background: #E8F5E9;
        border-left: 5px solid #2E7D32;
        border-radius: 6px;
        padding: 14px 18px;
        margin: 12px 0;
        color: #1B3A1C;
        font-size: 13px;
    }
    .metric-card {
        background: #FFFFFF;
        border: 2px solid #BBDEFB;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin: 6px 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .metric-val {
        font-size: 26px;
        font-weight: bold;
        color: #0D47A1;
    }
    .metric-label {
        font-size: 12px;
        color: #607D8B;
        margin-top: 4px;
    }
    .section-tag {
        display: inline-block;
        background: #0097A7;
        color: white;
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    /* Code blocks: white background, readable */
    .stCodeBlock, pre, code {
        background: #F1F5F9 !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px;
    }
    .stButton > button {
        background: #1565C0;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: bold;
        padding: 8px 20px;
    }
    .stButton > button:hover { background: #1976D2; }
    hr { border-color: #BBDEFB; }
    [data-testid="stFileUploader"] { background: #FFFFFF; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── MODEL LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, processor, device


# ── GALLERY IMAGES ────────────────────────────────────────────────────────────
# Curated open-license images via Unsplash Source (free, no auth needed)
# Each entry: (label, category, url)
GALLERY = [
    # Shoes
    ("white sneakers",     "shoes",      "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=300"),
    ("red high heels",     "shoes",      "https://images.unsplash.com/photo-1543163521-1bf539c55dd2?w=300"),
    ("brown leather boots","shoes",      "https://images.unsplash.com/photo-1520639888713-7851133b1ed0?w=300"),
    ("black running shoes","shoes",      "https://images.unsplash.com/photo-1491553895911-0055eca6402d?w=300"),
    ("blue canvas shoes",  "shoes",      "https://images.unsplash.com/photo-1525966222134-fcfa99b8ae77?w=300"),
    # Bags
    ("black leather bag",  "bags",       "https://images.unsplash.com/photo-1548036328-c9fa89d128fa?w=300"),
    ("red handbag",        "bags",       "https://images.unsplash.com/photo-1584917865442-de89df76afd3?w=300"),
    ("brown backpack",     "bags",       "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=300"),
    ("beige tote bag",     "bags",       "https://images.unsplash.com/photo-1600857544200-b2f666a9a2ec?w=300"),
    ("blue denim bag",     "bags",       "https://images.unsplash.com/photo-1590874103328-eac38a683ce7?w=300"),
    # Chairs
    ("wooden chair",       "furniture",  "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=300"),
    ("red armchair",       "furniture",  "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=300"),
    ("white modern chair", "furniture",  "https://images.unsplash.com/photo-1567538096630-e0c55bd6374c?w=300"),
    ("black office chair", "furniture",  "https://images.unsplash.com/photo-1589384267710-7a170981ca78?w=300"),
    ("blue accent chair",  "furniture",  "https://images.unsplash.com/photo-1506439773649-6e0eb8cfb237?w=300"),
    # Cars
    ("red sports car",     "cars",       "https://images.unsplash.com/photo-1583121274602-3e2820c69888?w=300"),
    ("white SUV",          "cars",       "https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?w=300"),
    ("black sedan",        "cars",       "https://images.unsplash.com/photo-1502877338535-766e1452684a?w=300"),
    ("blue convertible",   "cars",       "https://images.unsplash.com/photo-1618843479313-40f8afb4b4d8?w=300"),
    ("yellow taxi",        "cars",       "https://images.unsplash.com/photo-1559416523-140ddc3d238c?w=300"),
    # Flowers
    ("red roses",          "flowers",    "https://images.unsplash.com/photo-1490750967868-88df5691cc54?w=300"),
    ("yellow sunflowers",  "flowers",    "https://images.unsplash.com/photo-1597848212624-a19eb35e2651?w=300"),
    ("purple lavender",    "flowers",    "https://images.unsplash.com/photo-1498462440456-0dba182e775b?w=300"),
    ("white daisies",      "flowers",    "https://images.unsplash.com/photo-1467736518471-8d521f1dc1e8?w=300"),
    ("pink tulips",        "flowers",    "https://images.unsplash.com/photo-1585952285284-5e68f9f09fc0?w=300"),
]


@st.cache_data(show_spinner=False)
def load_gallery_images():
    """Download and cache gallery images."""
    images, labels, categories, urls = [], [], [], []
    for label, category, url in GALLERY:
        try:
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                img = img.resize((224, 224))
                images.append(img)
                labels.append(label)
                categories.append(category)
                urls.append(url)
        except Exception:
            pass
    return images, labels, categories, urls


@st.cache_data(show_spinner=False)
def encode_gallery(_model, _processor, _device):
    """Encode all gallery images once and cache."""
    images, labels, categories, urls = load_gallery_images()
    if not images:
        return None, labels, categories, urls
    inputs = _processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        feats = _model.get_image_features(**inputs)
    feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy(), labels, categories, urls


def encode_image(model, processor, device, image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = model.get_image_features(**inputs)
    feat = feat.pooler_output if hasattr(feat, "pooler_output") else feat
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()


def encode_text(model, processor, device, text):
    inputs = processor(
        text=[text], return_tensors="pt", padding=True, truncation=True
    ).to(device)
    with torch.no_grad():
        feat = model.get_text_features(**inputs)
    feat = feat.pooler_output if hasattr(feat, "pooler_output") else feat
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()


def compose_query(img_emb, txt_emb, alpha=0.5):
    """
    Compose image and text embeddings.
    alpha=0.0 -> pure image retrieval
    alpha=1.0 -> pure text retrieval
    alpha=0.5 -> balanced composition (Pic2Word approximation)
    """
    composed = (1 - alpha) * img_emb + alpha * txt_emb
    composed = composed / np.linalg.norm(composed, axis=-1, keepdims=True)
    return composed


def retrieve_top_k(query_emb, gallery_emb, k=5):
    sims = np.dot(query_emb, gallery_emb.T).squeeze()
    top_k = np.argsort(sims)[::-1][:k]
    return top_k, sims[top_k]


def make_fig(w=8, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F7F9FC")
    for spine in ax.spines.values():
        spine.set_edgecolor("#CBD5E1")
    ax.tick_params(colors="#334155", labelsize=10)
    ax.xaxis.label.set_color("#334155")
    ax.yaxis.label.set_color("#334155")
    ax.title.set_color("#0D47A1")
    return fig, ax


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.session_state.sidebar_width = st.slider(
        "Panel width", min_value=200, max_value=440,
        value=st.session_state.sidebar_width, step=10,
        help="Drag to resize the sidebar"
    )
    st.markdown("## 🔎 Pic2Word Walkthrough")
    st.markdown("**SEAS 8525** · Computer Vision and Generative AI")
    st.markdown("*Dr. Elbasheer · Week 6*")
    st.divider()

    section = st.radio(
        "Jump to section",
        options=[
            "1 · What is Pic2Word?",
            "2 · The Mapper Concept",
            "3 · Composed Image Retrieval",
            "4 · Composition Weight Explorer",
            "5 · Embedding Space Visualization",
        ]
    )
    st.divider()

    st.markdown("#### Quick Setup")
    st.code(
        "pip install streamlit torch\n"
        "pip install torchvision transformers\n"
        "pip install matplotlib pillow\n"
        "pip install numpy requests\n"
        "pip install scikit-learn\n\n"
        "streamlit run pic2word_app.py",
        language="bash"
    )

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if device_str == "cuda":
        st.success(f"GPU ready: {torch.cuda.get_device_name(0)}")
    else:
        st.info("Running on CPU. CLIP is lightweight and fast.")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: WHAT IS PIC2WORD?
# ═════════════════════════════════════════════════════════════════════════════
if section.startswith("1"):
    st.markdown('<span class="section-tag">SECTION 1</span>', unsafe_allow_html=True)
    st.title("Pic2Word: Image as a Word in a Sentence")
    st.caption("Saito et al., 2023 · Composed Image Retrieval using CLIP")

    st.markdown("""<div class="concept-box">
    <strong>Core idea:</strong> Convert a query image into a single pseudo-word token
    that can be inserted directly into a text prompt. This lets you compose a search
    query that combines visual appearance from an image with semantic modification
    from text, for example: "find me something that looks like
    <em>[this image]</em> but in red."
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-val">1</div><div class="metric-label">Pseudo-word token per image</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-val">❄️ x2</div><div class="metric-label">Frozen CLIP encoders</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-val">MLP</div><div class="metric-label">Mapper architecture</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-val">0</div><div class="metric-label">Paired training examples needed</div></div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("The Problem Pic2Word Solves")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""<div class="highlight-box">
        <strong>Standard text search: too generic</strong><br><br>
        Query: "red shoes"<br>
        Problem: returns any red shoes, ignoring the specific
        style, shape, and design of the pair you had in mind.
        No way to express visual specificity in text alone.
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="highlight-box">
        <strong>Standard image search: no modification</strong><br><br>
        Query: [image of shoes]<br>
        Problem: returns visually similar shoes in the same color.
        No way to say "like these but different in this specific way."
        </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("""<div class="success-box">
        <strong>Pic2Word: composed query</strong><br><br>
        Query: [image of shoes] + "but in red"<br><br>
        The image provides visual specificity (shape, style, sole design,
        silhouette). The text provides the modification. Together they
        compose a query that neither could express alone.<br><br>
        This is the composed image retrieval problem.
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("How Pic2Word Works: Step by Step")

    st.code(
        "# Step 1: encode the query image with frozen CLIP image encoder\n"
        "image_embedding = clip.encode_image(query_image)\n"
        "# shape: [512] -- lies in CLIP's image embedding space\n\n"
        "# Step 2: Pic2Word mapper converts image embedding to a\n"
        "# pseudo-word token in CLIP's TEXT embedding space\n"
        "pseudo_word = pic2word_mapper(image_embedding)\n"
        "# shape: [512] -- now in text space, compatible with text tokens\n\n"
        "# Step 3: compose the query\n"
        '# \"find me [*] but in red\" where [*] = pseudo_word\n'
        'composed_query = clip.encode_text(f\"{pseudo_word} but in red\")\n'
        "# shape: [512] -- composed embedding\n\n"
        "# Step 4: retrieve from gallery by cosine similarity\n"
        "similarities = cosine_similarity(composed_query, gallery_embeddings)\n"
        "top_results  = gallery[argsort(similarities)[::-1][:5]]",
        language="python"
    )

    st.markdown("""<div class="warning-box">
    <strong>About this app:</strong> The official Pic2Word mapper weights require
    a specific training setup. This app implements the composed retrieval concept
    using a weighted combination of CLIP image and text embeddings, a faithful
    approximation that demonstrates the core idea with the same CLIP backbone.
    The key insight is identical: combine visual and textual signals in CLIP's
    shared embedding space to compose a richer query than either alone.
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: THE MAPPER CONCEPT
# ═════════════════════════════════════════════════════════════════════════════
elif section.startswith("2"):
    st.markdown('<span class="section-tag">SECTION 2</span>', unsafe_allow_html=True)
    st.title("The Mapper: Turning an Image into a Word")
    st.caption("How a small MLP bridges CLIP's image and text embedding spaces")

    with st.spinner("Loading CLIP..."):
        model, processor, device = load_clip()

    st.markdown("""<div class="concept-box">
    The Pic2Word mapper is a small multilayer perceptron (MLP) that takes a
    512-dimensional CLIP image embedding and outputs a 512-dimensional vector
    in CLIP's text embedding space. Once there, the pseudo-word token can be
    treated exactly like any other word embedding: inserted into a sentence,
    combined with other text tokens, and used for retrieval.
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("The Two Embedding Spaces")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""<div class="highlight-box">
        <strong>CLIP Image Space</strong><br><br>
        Populated by image encodings.<br>
        512-dimensional unit vectors.<br>
        Semantically rich: similar images cluster together.<br>
        But incompatible with raw text token embeddings.
        </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("""<div class="highlight-box">
        <strong>CLIP Text Space</strong><br><br>
        Populated by text encodings.<br>
        512-dimensional unit vectors.<br>
        Compositional: words combine to form sentence meanings.<br>
        The mapper projects image embeddings into this space.
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="success-box">
    <strong>Key insight:</strong> CLIP's contrastive training already aligns the
    image and text spaces; matching pairs are close together. But they are not
    identical. The mapper learns the residual transformation that converts an
    image embedding into the most compatible text-space representation, enabling
    it to compose naturally with other text tokens.
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("Mapper Architecture")

    st.code(
        "# Pic2Word mapper -- simplified PyTorch implementation\n"
        "import torch.nn as nn\n\n"
        "class Pic2WordMapper(nn.Module):\n"
        "    def __init__(self, embed_dim=512):\n"
        "        super().__init__()\n"
        "        self.mapper = nn.Sequential(\n"
        "            nn.Linear(embed_dim, embed_dim * 4),\n"
        "            nn.GELU(),\n"
        "            nn.Linear(embed_dim * 4, embed_dim),\n"
        "        )\n\n"
        "    def forward(self, image_embedding):\n"
        "        # image_embedding: [B, 512] from frozen CLIP image encoder\n"
        "        pseudo_word = self.mapper(image_embedding)\n"
        "        # L2 normalize to sit on the unit hypersphere\n"
        "        # -- same space as CLIP text embeddings\n"
        "        pseudo_word = pseudo_word / pseudo_word.norm(\n"
        "            dim=-1, keepdim=True\n"
        "        )\n"
        "        return pseudo_word\n"
        "        # output: [B, 512] -- ready to compose with text\n\n"
        "# Total trainable parameters: 512*2048 + 2048*512 = ~2M\n"
        "# Both CLIP encoders remain completely frozen",
        language="python"
    )

    st.divider()
    st.subheader("Live Demo: Image Embedding vs Text Embedding")
    st.markdown("Upload an image and type a description. See how close the image embedding already is to the text embedding; this is what CLIP's alignment gives us before the mapper.")

    col_a, col_b = st.columns(2)
    with col_a:
        uploaded = st.file_uploader("Upload image", type=["png","jpg","jpeg","webp"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, width='stretch')
    with col_b:
        desc = st.text_input("Text description:", value="a photo of shoes")
        modifier = st.text_input("Text modifier (composition):", value="but in red")

    if uploaded and st.button("Compare embeddings"):
        img = Image.open(uploaded).convert("RGB")
        img_emb = encode_image(model, processor, device, img)
        txt_emb = encode_text(model, processor, device, desc)
        mod_emb = encode_text(model, processor, device, modifier)
        composed = compose_query(img_emb, mod_emb, alpha=0.5)

        sim_img_txt  = float(np.dot(img_emb[0], txt_emb[0]))
        sim_img_mod  = float(np.dot(img_emb[0], mod_emb[0]))
        sim_composed = float(np.dot(composed[0], txt_emb[0]))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#1565C0">{sim_img_txt:.3f}</div><div class="metric-label">Image vs description similarity</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#E65100">{sim_img_mod:.3f}</div><div class="metric-label">Image vs modifier similarity</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#2E7D32">{sim_composed:.3f}</div><div class="metric-label">Composed query vs description</div></div>', unsafe_allow_html=True)

        st.markdown("""<div class="concept-box">
        <strong>What these numbers show:</strong> The image embedding and the text
        description embedding are already somewhat similar thanks to CLIP's alignment.
        The composed embedding balances the image's visual content with the text
        modifier; this is the approximation of what the Pic2Word mapper learns
        to do more precisely through training.
        </div>""", unsafe_allow_html=True)

        # Visualize the three embeddings (first 64 dims)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        fig.patch.set_facecolor("#FFFFFF")
        titles  = ["Image embedding", "Text modifier embedding", "Composed embedding"]
        embs    = [img_emb[0], mod_emb[0], composed[0]]
        colors  = ["#1565C0", "#E65100", "#2E7D32"]
        for ax, title, emb, color in zip(axes, titles, embs, colors):
            ax.set_facecolor("#F7F9FC")
            ax.bar(range(64), emb[:64],
                   color=[color if v > 0 else "#EF9A9A" for v in emb[:64]],
                   width=0.8)
            ax.axhline(0, color="#94A3B8", linewidth=0.8)
            ax.set_title(title, color="#0D47A1", fontsize=10)
            ax.set_xlabel("Dim index", color="#334155", fontsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#CBD5E1")
            ax.tick_params(colors="#334155", labelsize=7)
        plt.suptitle("First 64 dimensions of each embedding vector",
                     color="#334155", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: COMPOSED IMAGE RETRIEVAL
# ═════════════════════════════════════════════════════════════════════════════
elif section.startswith("3"):
    st.markdown('<span class="section-tag">SECTION 3</span>', unsafe_allow_html=True)
    st.title("Composed Image Retrieval")
    st.caption("Upload a query image, add a text modifier, retrieve matching results")

    with st.spinner("Loading CLIP..."):
        model, processor, device = load_clip()

    st.markdown("""<div class="concept-box">
    This is the core Pic2Word capability. You provide a query image and a text
    modifier that describes how you want the results to differ. The composed
    embedding combines both signals and retrieves the most relevant images
    from the gallery.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Loading gallery images..."):
        gallery_embs, gallery_labels, gallery_cats, gallery_urls = encode_gallery(
            model, processor, device
        )
        gallery_images, _, _, _ = load_gallery_images()

    if gallery_embs is None:
        st.error("Could not load gallery images. Check your internet connection.")
        st.stop()

    st.markdown(f"""<div class="success-box">
    Gallery loaded: <strong>{len(gallery_images)} images</strong> across
    <strong>{len(set(gallery_cats))} categories</strong>:
    {', '.join(sorted(set(gallery_cats)))}
    </div>""", unsafe_allow_html=True)

    st.divider()
    col_l, col_r = st.columns([1, 1.2])

    with col_l:
        st.subheader("Query Image")
        uploaded = st.file_uploader(
            "Upload your query image", type=["png","jpg","jpeg","webp"]
        )
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, width='stretch')

    with col_r:
        st.subheader("Text Modifier")
        st.markdown("Describe how you want the results to *differ* from the query image.")

        modifier = st.text_input(
            "Modifier:", value="but in red"
        )

        default_modifiers = [
            "but in red",
            "but in black",
            "but vintage style",
            "but outdoors",
            "but minimalist design",
            "but luxury version",
        ]
        st.markdown("**Quick modifiers:**")
        mod_cols = st.columns(3)
        for i, m in enumerate(default_modifiers):
            with mod_cols[i % 3]:
                if st.button(m, key=f"mod_{i}"):
                    modifier = m

        alpha = st.slider(
            "Composition weight (image vs text)",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="0.0 = pure image retrieval, 1.0 = pure text retrieval, 0.5 = balanced"
        )
        st.markdown(
            f"**Image influence: {(1-alpha)*100:.0f}%** | "
            f"**Text influence: {alpha*100:.0f}%**"
        )

        top_k = st.slider("Number of results", 3, 8, 5)

    if uploaded and st.button("Retrieve"):
        img = Image.open(uploaded).convert("RGB")

        with st.spinner("Composing query and retrieving..."):
            img_emb     = encode_image(model, processor, device, img)
            txt_emb     = encode_text(model, processor, device, modifier)
            composed    = compose_query(img_emb, txt_emb, alpha)
            top_idx, top_sims = retrieve_top_k(composed, gallery_embs, top_k)

        st.divider()
        st.subheader(f"Top {top_k} Results")
        st.markdown(
            f"Query: **[your image]** + **\"{modifier}\"** "
            f"(image {(1-alpha)*100:.0f}% / text {alpha*100:.0f}%)"
        )

        res_cols = st.columns(top_k)
        for i, (idx, sim) in enumerate(zip(top_idx, top_sims)):
            with res_cols[i]:
                st.image(gallery_images[idx], width='stretch')
                st.markdown(
                    f"**{gallery_labels[idx]}**\n\n"
                    f"Similarity: `{sim:.3f}`\n\n"
                    f"Category: *{gallery_cats[idx]}*"
                )

        st.divider()
        st.subheader("Similarity Breakdown")

        img_only_sims  = np.dot(img_emb, gallery_embs.T).squeeze()
        txt_only_sims  = np.dot(txt_emb, gallery_embs.T).squeeze()
        composed_sims  = np.dot(composed, gallery_embs.T).squeeze()

        fig, ax = make_fig(10, 4)
        x = np.arange(len(gallery_labels))
        w = 0.28
        ax.bar(x - w, img_only_sims,   width=w, label="Image only",   color="#1565C0", alpha=0.8)
        ax.bar(x,     txt_only_sims,   width=w, label="Text only",    color="#E65100", alpha=0.8)
        ax.bar(x + w, composed_sims,   width=w, label="Composed",     color="#2E7D32", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(gallery_labels, rotation=45, ha="right",
                           color="#334155", fontsize=8)
        ax.set_ylabel("Cosine similarity")
        ax.set_title(
            "Similarity scores: image only vs text only vs composed query"
        )
        ax.legend(labelcolor="#334155")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""<div class="concept-box">
        <strong>Reading this chart:</strong> The green bars (composed) blend the
        blue (image) and orange (text) signals. Notice how composition shifts
        retrieval toward gallery items that satisfy BOTH the visual appearance
        of your query image AND the semantic content of your modifier.
        This is what Pic2Word's learned mapper does, more precisely,
        by projecting the image into text space before composition.
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: COMPOSITION WEIGHT EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif section.startswith("4"):
    st.markdown('<span class="section-tag">SECTION 4</span>', unsafe_allow_html=True)
    st.title("Composition Weight Explorer")
    st.caption("How the image-text balance shifts retrieval results")

    with st.spinner("Loading CLIP..."):
        model, processor, device = load_clip()

    st.markdown("""<div class="concept-box">
    The composition weight controls how much influence the query image vs the
    text modifier has on the final retrieval. This section lets you see the
    same query at multiple weight settings simultaneously, showing how
    retrieval shifts as you move from pure image search to pure text search.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Loading gallery..."):
        gallery_embs, gallery_labels, gallery_cats, gallery_urls = encode_gallery(
            model, processor, device
        )
        gallery_images, _, _, _ = load_gallery_images()

    col_l, col_r = st.columns([1, 1])
    with col_l:
        uploaded = st.file_uploader(
            "Query image", type=["png","jpg","jpeg","webp"]
        )
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, width='stretch')
    with col_r:
        modifier = st.text_input("Text modifier:", value="but in red")
        top_k_exp = st.slider("Results per weight setting", 2, 5, 3)

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    alpha_labels = [
        "Pure image\n(alpha=0.0)",
        "Image dominant\n(alpha=0.25)",
        "Balanced\n(alpha=0.5)",
        "Text dominant\n(alpha=0.75)",
        "Pure text\n(alpha=1.0)",
    ]

    if uploaded and st.button("Explore All Weights"):
        img = Image.open(uploaded).convert("RGB")

        with st.spinner("Running retrieval at all weight settings..."):
            img_emb = encode_image(model, processor, device, img)
            txt_emb = encode_text(model, processor, device, modifier)

            all_results = []
            for alpha in alphas:
                composed = compose_query(img_emb, txt_emb, alpha)
                top_idx, top_sims = retrieve_top_k(
                    composed, gallery_embs, top_k_exp
                )
                all_results.append((top_idx, top_sims))

        st.divider()
        for i, (alpha, label, (top_idx, top_sims)) in enumerate(
            zip(alphas, alpha_labels, all_results)
        ):
            st.markdown(f"**{label.replace(chr(10), ' ')}**")
            cols = st.columns(top_k_exp)
            for j, (idx, sim) in enumerate(zip(top_idx, top_sims)):
                with cols[j]:
                    st.image(gallery_images[idx], width='stretch')
                    st.caption(f"{gallery_labels[idx]} ({sim:.3f})")
            st.divider()

        st.markdown("""<div class="concept-box">
        <strong>What to observe:</strong> At alpha=0.0 (pure image), results look
        most visually similar to your query image. At alpha=1.0 (pure text),
        results match your modifier text most closely. The balanced setting
        (alpha=0.5) is where Pic2Word's mapper operates, finding images
        that satisfy both constraints simultaneously. The learned mapper
        finds the optimal point in this space automatically during training.
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: EMBEDDING SPACE VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════
elif section.startswith("5"):
    st.markdown('<span class="section-tag">SECTION 5</span>', unsafe_allow_html=True)
    st.title("Embedding Space Visualization")
    st.caption("See how the composed query moves through CLIP's shared embedding space")

    with st.spinner("Loading CLIP..."):
        model, processor, device = load_clip()

    st.markdown("""<div class="concept-box">
    The composed query is a point in CLIP's 512-dimensional shared embedding space.
    Using t-SNE we project everything to 2D: the gallery images, the query image,
    the text modifier, and the composed query, to see how composition moves the
    query point toward items that satisfy both constraints.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Loading gallery..."):
        gallery_embs, gallery_labels, gallery_cats, gallery_urls = encode_gallery(
            model, processor, device
        )
        gallery_images, _, _, _ = load_gallery_images()

    col_l, col_r = st.columns([1, 1])
    with col_l:
        uploaded = st.file_uploader(
            "Query image", type=["png","jpg","jpeg","webp"]
        )
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, width='stretch')
    with col_r:
        modifier = st.text_input("Text modifier:", value="but in red")
        alpha_viz = st.slider(
            "Composition weight", 0.0, 1.0, 0.5, 0.05
        )

    if uploaded and st.button("Visualize Embedding Space"):
        img = Image.open(uploaded).convert("RGB")

        with st.spinner("Encoding and projecting..."):
            img_emb  = encode_image(model, processor, device, img)
            txt_emb  = encode_text(model, processor, device, modifier)
            composed = compose_query(img_emb, txt_emb, alpha_viz)

            all_embs = np.vstack([
                gallery_embs,
                img_emb,
                txt_emb,
                composed,
            ])

            tsne = TSNE(
                n_components=2, perplexity=min(10, len(gallery_embs) - 1),
                random_state=42, max_iter=1000,
                learning_rate="auto", init="pca"
            )
            coords = tsne.fit_transform(all_embs)

        n_gallery = len(gallery_embs)
        gallery_coords  = coords[:n_gallery]
        query_img_coord = coords[n_gallery]
        query_txt_coord = coords[n_gallery + 1]
        composed_coord  = coords[n_gallery + 2]

        cat_palette = {
            "shoes":     "#1565C0",
            "bags":      "#6A1B9A",
            "furniture": "#2E7D32",
            "cars":      "#E65100",
            "flowers":   "#BF360C",
        }

        fig, ax = make_fig(10, 7)

        # Gallery points
        for i, (x, y) in enumerate(gallery_coords):
            cat   = gallery_cats[i]
            color = cat_palette.get(cat, "#90A4AE")
            ax.scatter(x, y, c=color, s=120, alpha=0.7,
                       edgecolors="white", linewidth=0.8, zorder=3)
            ax.annotate(
                gallery_labels[i], (x, y),
                textcoords="offset points", xytext=(5, 3),
                color=color, fontsize=7, alpha=0.85
            )

        # Query image point
        ax.scatter(*query_img_coord, c="#FFB300", s=280, marker="*",
                   zorder=6, edgecolors="white", linewidth=1.5,
                   label="Query image")
        ax.annotate("YOUR IMAGE", query_img_coord,
                    textcoords="offset points", xytext=(8, 5),
                    color="#FFB300", fontsize=9, fontweight="bold")

        # Text modifier point
        ax.scatter(*query_txt_coord, c="#00897B", s=200, marker="^",
                   zorder=6, edgecolors="white", linewidth=1.5,
                   label=f'Text: "{modifier}"')
        ax.annotate(f'"{modifier}"', query_txt_coord,
                    textcoords="offset points", xytext=(8, 5),
                    color="#00897B", fontsize=9, fontweight="bold")

        # Composed query point
        ax.scatter(*composed_coord, c="#E53935", s=260, marker="D",
                   zorder=6, edgecolors="white", linewidth=1.5,
                   label="Composed query")
        ax.annotate("COMPOSED", composed_coord,
                    textcoords="offset points", xytext=(8, 5),
                    color="#E53935", fontsize=9, fontweight="bold")

        # Arrow from image to composed
        ax.annotate(
            "", xy=composed_coord, xytext=query_img_coord,
            arrowprops=dict(
                arrowstyle="->", color="#455A64",
                lw=1.5, linestyle="dashed"
            )
        )

        # Category legend
        from matplotlib.lines import Line2D
        legend_cats = [
            Line2D([0],[0], marker="o", color="w",
                   markerfacecolor=v, markersize=9, label=k)
            for k, v in cat_palette.items()
        ]
        legend_special = [
            Line2D([0],[0], marker="*", color="w",
                   markerfacecolor="#FFB300", markersize=12, label="Query image"),
            Line2D([0],[0], marker="^", color="w",
                   markerfacecolor="#00897B", markersize=10, label="Text modifier"),
            Line2D([0],[0], marker="D", color="w",
                   markerfacecolor="#E53935", markersize=10, label="Composed query"),
        ]
        ax.legend(
            handles=legend_cats + legend_special,
            loc="upper right", facecolor="#FFFFFF",
            edgecolor="#CBD5E1", labelcolor="#334155", fontsize=8
        )

        ax.set_title(
            "t-SNE projection of CLIP embedding space (512D to 2D)\n"
            "Dashed arrow shows how composition moves the query",
            color="#0D47A1", fontsize=11
        )
        ax.set_xlabel("t-SNE dimension 1")
        ax.set_ylabel("t-SNE dimension 2")
        ax.grid(True, color="#E2E8F0", alpha=0.6, linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""<div class="concept-box">
        <strong>What to look for:</strong> The dashed arrow shows how the composed
        query (red diamond) is pulled away from the query image (gold star) toward
        the text modifier (teal triangle). Ideally the composed point lands near
        gallery items that are visually similar to your image but semantically
        closer to your modifier. This movement in embedding space is exactly
        what Pic2Word's mapper learns to optimize during training.
        </div>""", unsafe_allow_html=True)

        # Top retrieved results
        top_idx, top_sims = retrieve_top_k(composed, gallery_embs, 5)
        st.divider()
        st.subheader("Top Retrieved Results")
        res_cols = st.columns(5)
        for i, (idx, sim) in enumerate(zip(top_idx, top_sims)):
            with res_cols[i]:
                st.image(gallery_images[idx], width='stretch')
                st.caption(f"{gallery_labels[idx]}\n{sim:.3f}")
