"""
Hotdog / Not Hotdog — Food-101 edition
WebApp Streamlit : upload/drag&drop d'une image, prédiction du plat parmi
101 classes, avec un traitement spécial "HOTDOG !" vs "Pas un hotdog".
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
CLASSES_PATH = os.path.join(ROOT_DIR, "data", "meta", "classes.txt")
IMAGE_SIZE = (224, 224)
HOTDOG_CLASS = "hot_dog"

st.set_page_config(
    page_title="Hotdog / Not Hotdog 🌭",
    page_icon="🌭",
    layout="centered",
)

# =====================================================
# STYLE (ludique + animé)
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

    .result-dish {
        text-align: center;
        font-size: 1.3rem;
        margin-top: 0.3rem;
        text-transform: capitalize;
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


def load_class_names():
    if not os.path.exists(CLASSES_PATH):
        return []
    with open(CLASSES_PATH, "r") as f:
        return [line.strip() for line in f if line.strip()]


@st.cache_resource(show_spinner=False)
def load_model(model_path):
    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


def prettify(class_name: str) -> str:
    return class_name.replace("_", " ")


def predict(model, image: Image.Image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)
    preds = model.predict(array, verbose=0)[0]
    return preds


# =====================================================
# HEADER
# =====================================================

st.markdown('<div class="hero-emoji">🌭</div>', unsafe_allow_html=True)
st.markdown('<p class="hero-title">Hotdog ou pas Hotdog ?</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Glisse une photo de plat, on te dit si c\'est un hotdog...'
    " et sinon, on devine ce que c'est parmi 101 plats !</p>",
    unsafe_allow_html=True,
)

class_names = load_class_names()

if not class_names:
    st.error(
        f"Fichier des classes introuvable : `{CLASSES_PATH}`. "
        "Vérifie que le dataset Food-101 est bien présent dans `data/meta/classes.txt`."
    )
    st.stop()

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
            preds = predict(model, image)

        top_idx = int(np.argmax(preds))
        top_class = class_names[top_idx]
        confidence = float(preds[top_idx])
        is_hotdog = top_class == HOTDOG_CLASS

        card_class = "hotdog" if is_hotdog else "not-hotdog"
        title = "🌭 HOTDOG ! 🌭" if is_hotdog else "🚫 PAS UN HOTDOG"
        dish_line = (
            "" if is_hotdog else f'<p class="result-dish">C\'est plutôt : <b>{prettify(top_class)}</b></p>'
        )

        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <p class="result-title">{title}</p>
                {dish_line}
                <p class="confidence-label">Confiance : {confidence:.1%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if is_hotdog:
            st.balloons()
        elif confidence < 0.4:
            st.caption(
                "⚠️ Le modèle n'est pas très confiant : l'image est peut-être ambiguë "
                "ou hors des 101 classes connues."
            )

        top5_idx = np.argsort(preds)[::-1][:5]
        st.markdown("#### Top 5 des prédictions")
        for idx in top5_idx:
            label = prettify(class_names[idx])
            score = float(preds[idx])
            emoji = "🌭" if class_names[idx] == HOTDOG_CLASS else "🍽️"
            st.progress(min(max(score, 0.0), 1.0), text=f"{emoji} {label} — {score:.1%}")

st.markdown(
    '<p class="footer-note">Modèle Food-101 · MobileNetV2 (transfer learning) · '
    "Projet groupe — Jour 5</p>",
    unsafe_allow_html=True,
)
