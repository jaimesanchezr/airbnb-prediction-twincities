import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from nltk.stem import SnowballStemmer

st.set_page_config(page_title="Predicción de calificación — Twin Cities MSA")

MODELO_PATH = "modelo_nivel1_final.pkl"
ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]


@st.cache_resource
def cargar_modelo():
    pipeline = joblib.load(MODELO_PATH)
    vectorizador = pipeline.named_steps["preprocesador"].named_transformers_["texto"]
    clasificador = pipeline.named_steps["clasificador"]
    vocab = np.array(vectorizador.get_feature_names_out())
    coefs = clasificador.coef_[0]
    coefs_texto = coefs[: len(vocab)]
    es_lineal_con_proba = hasattr(clasificador, "predict_proba")
    return pipeline, vectorizador, vocab, coefs_texto, es_lineal_con_proba


pipeline, vectorizador, vocab, coefs_texto, es_lineal_con_proba = cargar_modelo()
stemmer = SnowballStemmer("english")


def limpiar_texto_v2(texto):
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúñ\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    palabras = texto.split()
    palabras_stem = [stemmer.stem(p) for p in palabras]
    return " ".join(palabras_stem)


def explicar(texto_limpio):
    x_tfidf = vectorizador.transform([texto_limpio]).toarray()[0]
    contrib = x_tfidf * coefs_texto
    presentes = np.nonzero(x_tfidf)[0]

    if len(presentes) == 0:
        return "No se identificaron términos relevantes en el texto ingresado."

    orden_desc = presentes[np.argsort(contrib[presentes])[::-1]]
    top_pos = [vocab[i] for i in orden_desc[:5] if contrib[i] > 0]
    top_neg = [vocab[i] for i in orden_desc[::-1][:5] if contrib[i] < 0]

    partes = []
    if top_pos:
        partes.append("A favor de 'excelente': " + ", ".join(top_pos))
    if top_neg:
        partes.append("En contra de 'excelente': " + ", ".join(top_neg))
    return " | ".join(partes) if partes else "Señal mixta, sin términos dominantes."


st.title("Predicción de calificación de alojamiento")
st.caption(
    "Modelo clásico (TF-IDF + clasificador lineal) entrenado sobre reseñas "
    "de Inside Airbnb (Twin Cities MSA, MN)."
)

texto_resenas = st.text_area(
    "Texto acumulado de las reseñas (en inglés)",
    height=200,
    placeholder="Great place, very clean, host was super responsive...",
)
room_type = st.selectbox("Tipo de alojamiento", ROOM_TYPES)

if st.button("Predecir"):
    if not texto_resenas or not texto_resenas.strip():
        st.warning("Ingresa al menos el texto de una reseña.")
    else:
        texto_limpio = limpiar_texto_v2(texto_resenas)
        df_input = pd.DataFrame(
            {"texto_limpio_v2": [texto_limpio], "room_type": [room_type]}
        )

        if es_lineal_con_proba:
            proba = pipeline.predict_proba(df_input)[0]
            pred = int(proba[1] >= 0.5)
            confianza = proba[pred]
            etiqueta = "Excelente" if pred == 1 else "No excelente"
            mensaje = f"{etiqueta} — probabilidad: {confianza:.1%}"
        else:
            score = pipeline.decision_function(df_input)[0]
            pred = int(score >= 0)
            confianza = 1 / (1 + np.exp(-abs(score)))
            etiqueta = "Excelente" if pred == 1 else "No excelente"
            mensaje = (
                f"{etiqueta} — confianza aproximada: {confianza:.1%} "
                f"(score={score:.2f}, no es una probabilidad calibrada)"
            )

        if pred == 1:
            st.success(mensaje)
        else:
            st.error(mensaje)

        st.subheader("Explicación de la predicción")
        st.write(explicar(texto_limpio))
