"""
Hotdog / Not Hotdog — WebApp Streamlit
Upload/drag&drop d'une image, prediction binaire (hot_dog vs not_hot_dog)
par le modele MobileNetV2 (transfer learning, sortie sigmoid).
"""

import os

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

# =====================================================
# CONFIG
# =====================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)

MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join(ROOT_DIR, "models", "model.keras")
)
IMAGE_SIZE = (224, 224)

st.set_page_config(
    page_title="Hotdog / Not Hotdog 🌭",
    page_icon="🌭",
    layout="centered",
)

# =====================================================
# STYLE (ludique + anime)
# =====================================================

st.markdown(
    """
    <style>
    @keyframes floaty {
        0%   { transform: translateY(0px) rotate(-2deg); }
        50%  { transform: translateY(-10px) rotate(2deg); }
        100% { transform: translateY(0px) rotate(-2deg); }
    }
    @keyframes pop-in {
        0%   { transform: scale(0.7); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }
    @keyframes gradient-shift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stApp {
        background: linear-gradient(120deg, #fff6e5, #ffe0b2, #ffd6d6, #fff6e5);
        background-size: 300% 300%;
        animation: gradient-shift 18s ease infinite;
    }

    .hero-emoji {
        text-align: center;
        font-size: 5rem;
        animation: floaty 3s ease-in-out infinite;
    }

    .hero-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 800;
        color: #7a3e00;
        margin-bottom: 0;
    }

    .hero-subtitle {
        text-align: center;
        color: #a15c00;
        font-size: 1.05rem;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }

    .result-card {
        border-radius: 20px;
        padding: 1.6rem 1.8rem;
        margin-top: 1rem;
        animation: pop-in 0.35s ease-out;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }

    .result-card.hotdog {
        background: linear-gradient(135deg, #ffdca8, #ff9d6c);
        border: 3px solid #ff7a00;
    }

    .result-card.not-hotdog {
        background: linear-gradient(135deg, #ffe9ea, #ffc9d0);
        border: 3px solid #ff5d73;
    }

    .result-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
        text-align: center;
    }

    .confidence-label {
        text-align: center;
        font-size: 0.95rem;
        color: #5a3200;
        margin-top: 0.6rem;
    }

    .footer-note {
        text-align: center;
        color: #9a7350;
        font-size: 0.85rem;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# HELPERS
# =====================================================


@st.cache_resource(show_spinner=False)
def load_model(model_path):
    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


def predict(model, image: Image.Image):
    """
    Retourne la probabilite (0-1) que l'image soit un hot_dog.
    """
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)
    return float(model.predict(array, verbose=0)[0][0])


# =====================================================
# HEADER
# =====================================================

st.markdown('<div class="hero-emoji">🌭</div>', unsafe_allow_html=True)
st.markdown('<p class="hero-title">Hotdog ou pas Hotdog ?</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Glisse une photo, le modèle te dit si c\'est un hotdog.</p>',
    unsafe_allow_html=True,
)

model = None
model_error = None
if os.path.exists(MODEL_PATH):
    try:
        with st.spinner("Chargement du modèle..."):
            model = load_model(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 - affichage utilisateur uniquement
        model_error = str(exc)
else:
    model_error = "missing"

if model is None:
    st.warning(
        "**Aucun modèle entraîné trouvé.**\n\n"
        f"Dépose le fichier du modèle entraîné (`.keras`) à cet emplacement :\n\n"
        f"`{MODEL_PATH}`\n\n"
        "Tu peux aussi indiquer un autre chemin via la variable d'environnement "
        "`MODEL_PATH` avant de lancer l'app."
    )
    if model_error and model_error != "missing":
        with st.expander("Détail de l'erreur de chargement"):
            st.code(model_error)

# =====================================================
# UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Dépose ou sélectionne une image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    disabled=model is None,
)

if uploaded_file is None:
    st.info("Aucune image sélectionnée pour le moment. 📷")
else:
    try:
        image = Image.open(uploaded_file)
    except UnidentifiedImageError:
        st.error("Ce fichier ne semble pas être une image valide. Réessaie avec un JPG ou un PNG.")
        st.stop()

    st.image(image, caption="Image envoyée", use_container_width=True)

    if model is not None:
        with st.spinner("Analyse en cours... 🔍"):
            prob_hotdog = predict(model, image)

        is_hotdog = prob_hotdog >= 0.5
        confidence = prob_hotdog if is_hotdog else 1.0 - prob_hotdog

        card_class = "hotdog" if is_hotdog else "not-hotdog"
        title = "🌭 HOTDOG ! 🌭" if is_hotdog else "🚫 PAS UN HOTDOG"

        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <p class="result-title">{title}</p>
                <p class="confidence-label">Confiance : {confidence:.1%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if is_hotdog:
            st.balloons()
        elif confidence < 0.6:
            st.caption(
                "⚠️ Le modèle n'est pas très confiant : l'image est peut-être ambiguë "
                "ou hors distribution (ni un hotdog, ni les 100 autres plats de Food-101)."
            )

st.markdown(
    '<p class="footer-note">Modèle Food-101 (hot_dog vs not_hot_dog) · MobileNetV2 (transfer learning) · '
    "Projet groupe — Jour 5</p>",
    unsafe_allow_html=True,
)
