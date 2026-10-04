# Despliegue — Predicción de calificación (Twin Cities MSA)

Modelo desplegado: **Nivel 1 (clásico)** — TF-IDF + regresión logística
(`modelo_nivel1_final.pkl`). Justificación de esta elección: ver informe,
sección 6.3 "La decisión de despliegue".

**URL pública de la aplicación:**
https://airbnb-prediction-twincities-k5bpfachfdi7ccyz3zuob5.streamlit.app/

## Archivos de este despliegue

- `app.py` — aplicación Streamlit.
- `requirements.txt` — dependencias con versiones exactas.
- `runtime.txt` — fija Python 3.11 para el entorno de despliegue.
- `modelo_nivel1_final.pkl` — pipeline completo serializado con joblib
  (TF-IDF + ColumnTransformer + LogisticRegression, ya ajustado).

## Pasos para reproducir el despliegue (Streamlit Community Cloud, gratuito)

1. Crear un repositorio en GitHub (público) que contenga, en la raíz, los
   4 archivos listados arriba.
2. Crear una cuenta en https://share.streamlit.io usando el mismo usuario de
   GitHub (gratis, sin tarjeta de crédito).
3. Click en **New app** (o **Create app**).
4. Seleccionar:
   - **Repository**: el repositorio creado en el paso 1.
   - **Branch**: `main` (o la que corresponda).
   - **Main file path**: `app.py`.
5. Click en **Deploy**. Streamlit Cloud lee `runtime.txt`, instala Python 3.11
   y las dependencias de `requirements.txt`, y construye la app
   automáticamente (tarda 1-3 minutos la primera vez).
6. Una vez desplegada, la app queda disponible en una URL pública con el
   formato `https://<nombre-que-elijas>.streamlit.app`.
7. Verificar la app: pegar el texto acumulado de las reseñas de un
   alojamiento (en inglés), elegir el tipo de alojamiento, presionar
   **Predecir**, y confirmar que devuelve la predicción, la probabilidad y
   la explicación de términos influyentes.

## Ejecución local (para desarrollo o verificación antes de desplegar)

```bash
python -m venv venv
# Windows:  venv\Scripts\Activate.ps1
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Abre `http://localhost:8501` en el navegador.

## Notas de reproducibilidad

- **Versión de Python**: tanto en desarrollo local como en Streamlit Cloud
  es obligatorio usar **Python 3.11 o 3.12**, nunca 3.13+. Se confirmó en la
  práctica que con Python 3.14 la instalación de `scikit-learn==1.6.1` y de
  `pillow` (dependencia de Streamlit) falla al no existir wheels
  precompilados para esa versión, y pip intenta compilar desde el código
  fuente, lo cual requiere un compilador de C no disponible por defecto.
  `runtime.txt` resuelve esto en Streamlit Cloud; localmente, crear el
  entorno virtual explícitamente con `py -3.12 -m venv venv` (Windows) o
  `python3.12 -m venv venv` (Mac/Linux).
- El archivo `modelo_nivel1_final.pkl` **debe generarse con la misma versión
  de scikit-learn** fijada en `requirements.txt` (1.6.1) — la misma con la
  que se entrenó en el notebook original (Colab). Cargarlo con una versión
  más nueva (ej. 1.8.0) falla con
  `AttributeError: Can't get attribute '_RemainderColsList'`, porque
  scikit-learn cambió la implementación interna de `ColumnTransformer`
  entre versiones. Antes de regenerar el modelo, correr
  `import sklearn; print(sklearn.__version__)` en Colab y ajustar
  `requirements.txt` si no coincide.
- Si el modelo ganador de la búsqueda hubiera resultado `LinearSVC` en vez
  de `LogisticRegression`, la app lo detecta automáticamente
  (`decision_function` en vez de `predict_proba`) y lo indica como una
  confianza aproximada, no una probabilidad calibrada. En la versión final
  entregada, el ganador es `LogisticRegression`, por lo que la app reporta
  una probabilidad real.
- `room_type` en la app está limitado a las 4 categorías estándar de
  Inside Airbnb (`Entire home/apt`, `Private room`, `Shared room`,
  `Hotel room`).
- La limpieza de texto (`limpiar_texto_v2`) replica exactamente la función
  usada durante el entrenamiento (minúsculas, solo letras, stemming con
  `SnowballStemmer("english")`). Si esa función cambia en el notebook, debe
  actualizarse también en `app.py` para que la app sea consistente con el
  modelo.
- `@st.cache_resource` evita recargar el modelo en cada interacción del
  usuario — solo se carga una vez por sesión del servidor.
- Las apps gratuitas de Streamlit Cloud se "duermen" tras inactividad
  prolongada y tardan unos segundos en reactivarse la primera vez que se
  visitan después de un tiempo — comportamiento esperado, no un error.
