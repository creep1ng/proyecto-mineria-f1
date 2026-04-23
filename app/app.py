"""
F1 Race Finish Predictor - CRISP-DM Deployment
Aplicación Streamlit para predecir si un piloto terminará una carrera de Fórmula 1.

Ejecución:
    streamlit run app/app.py

El modelo se espera en:
    ../models/best_model_pipe.pkl  (relativo a este archivo)
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="F1 Race Finish Predictor",
    page_icon="🏎️",
    layout="wide",
)

st.title("🏎️ F1 Race Finish Predictor - CRISP-DM Deployment")
st.markdown(
    "Esta aplicación carga el pipeline de ML entrenado y predice la probabilidad "
    "de que un piloto **termine** la carrera (`finished = 1`) a partir de datos "
    "pre-carrera."
)

# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "best_model_pipe.pkl"
)

pipe = None
model_loaded = False
model_error = ""

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontró el modelo en: {os.path.abspath(MODEL_PATH)}\n"
            "Por favor, entrena y guarda el pipeline primero (fase 4 de CRISP-DM)."
        )
    pipe = joblib.load(MODEL_PATH)
    model_loaded = True
    st.success("✅ Modelo cargado exitosamente.")
except Exception as e:
    model_error = str(e)
    st.error(f"❌ Error al cargar el modelo:\n\n{model_error}")

# ---------------------------------------------------------------------------
# Inputs del usuario
# ---------------------------------------------------------------------------
st.sidebar.header("📋 Parámetros de entrada")


# Helper para inputs numéricos
def num_input(label, value, min_value=None, max_value=None, step=None, help_text=""):
    kwargs = {"help": help_text}
    if min_value is not None:
        kwargs["min_value"] = min_value
    if max_value is not None:
        kwargs["max_value"] = max_value
    if step is not None:
        kwargs["step"] = step
    return st.sidebar.number_input(label, value=value, **kwargs)


# --- Datos de parrilla y carrera ---
with st.sidebar.expander("Parrilla y carrera", expanded=True):
    grid = num_input(
        "grid (posición en parrilla)",
        1,
        1,
        30,
        1,
        "Posición de salida en la parrilla (1 = pole).",
    )
    laps = num_input("laps (vueltas programadas)", 57, 1, 100, 1)
    year = num_input("year", 2024, 1950, 2030, 1)
    round_num = num_input("round (número de carrera)", 1, 1, 25, 1)
    race_month = num_input("race_month (mes)", 3, 1, 12, 1)
    race_day_of_year = num_input("race_day_of_year (día del año)", 65, 1, 366, 1)
    season_progress = st.sidebar.slider(
        "season_progress (progreso de temporada)",
        0.0,
        1.0,
        0.15,
        0.01,
        help="Fracción de la temporada completada (0 = inicio, 1 = final).",
    )

# --- Datos del piloto ---
with st.sidebar.expander("Piloto", expanded=True):
    driver_age = num_input("driver_age (edad)", 26, 16, 60, 1)
    driver_race_count = num_input("driver_race_count (carreras previas)", 50, 0, 400, 1)
    driver_prev_finish_rate = st.sidebar.slider(
        "driver_prev_finish_rate",
        0.0,
        1.0,
        0.75,
        0.01,
        help="Tasa histórica de finalización del piloto.",
    )
    driver_last5_finish_rate = st.sidebar.slider(
        "driver_last5_finish_rate",
        0.0,
        1.0,
        0.80,
        0.01,
        help="Tasa de finalización en las últimas 5 carreras.",
    )
    driver_nationality_encoded = num_input("driver_nationality_encoded", 5, 0, 50, 1)

# --- Datos del constructor ---
with st.sidebar.expander("Constructor (equipo)", expanded=True):
    constructor_race_count = num_input(
        "constructor_race_count (carreras previas)", 200, 0, 1000, 1
    )
    constructor_prev_finish_rate = st.sidebar.slider(
        "constructor_prev_finish_rate", 0.0, 1.0, 0.70, 0.01
    )
    constructor_prev_avg_grid = num_input(
        "constructor_prev_avg_grid (parrilla promedio histórica)", 8.0, 1.0, 30.0, 0.5
    )
    constructor_last5_finish_rate = st.sidebar.slider(
        "constructor_last5_finish_rate", 0.0, 1.0, 0.75, 0.01
    )
    constructor_nationality_encoded = num_input(
        "constructor_nationality_encoded", 3, 0, 50, 1
    )

# --- Datos del circuito ---
with st.sidebar.expander("Circuito", expanded=True):
    circuit_finish_rate = st.sidebar.slider(
        "circuit_finish_rate",
        0.0,
        1.0,
        0.80,
        0.01,
        help="Tasa histórica de finalización en este circuito.",
    )
    circuit_avg_grid = num_input(
        "circuit_avg_grid (parrilla promedio histórica)", 10.0, 1.0, 30.0, 0.5
    )
    circuit_country_encoded = num_input("circuit_country_encoded", 10, 0, 100, 1)
    circuitRef_encoded = num_input("circuitRef_encoded", 8, 0, 200, 1)

# --- Clasificación (qualifying) ---
with st.sidebar.expander("Clasificación", expanded=True):
    has_qualifying = st.sidebar.selectbox(
        "has_qualifying",
        [0, 1],
        index=1,
        help="1 si hay datos de clasificación, 0 si no.",
    )
    front_row_start = st.sidebar.selectbox(
        "front_row_start (primera fila)", [0, 1], index=0
    )
    top10_start = st.sidebar.selectbox("top10_start (top 10 salida)", [0, 1], index=1)
    q1_seconds = num_input("q1_seconds (tiempo Q1 en segundos)", 88.5, 50.0, 120.0, 0.1)
    q2_seconds = num_input("q2_seconds (tiempo Q2 en segundos)", 87.9, 50.0, 120.0, 0.1)
    q3_seconds = num_input("q3_seconds (tiempo Q3 en segundos)", 87.4, 50.0, 120.0, 0.1)
    best_q_time = num_input(
        "best_q_time (mejor tiempo de clasificación)", 87.4, 50.0, 120.0, 0.1
    )
    grid_normalized = st.sidebar.slider(
        "grid_normalized",
        0.0,
        1.0,
        0.05,
        0.01,
        help="Posición de parrilla normalizada entre 0 y 1 (0 = pole, 1 = último).",
    )

# ---------------------------------------------------------------------------
# Botón de predicción
# ---------------------------------------------------------------------------
if st.sidebar.button("🔮 Predecir", type="primary"):
    if not model_loaded:
        st.error(
            "No se puede predecir porque el modelo no está disponible. "
            f"Error reportado: {model_error}"
        )
    else:
        input_df = pd.DataFrame(
            [
                {
                    "grid": grid,
                    "laps": laps,
                    "driver_age": driver_age,
                    "driver_race_count": driver_race_count,
                    "driver_prev_finish_rate": driver_prev_finish_rate,
                    "driver_last5_finish_rate": driver_last5_finish_rate,
                    "constructor_race_count": constructor_race_count,
                    "constructor_prev_finish_rate": constructor_prev_finish_rate,
                    "constructor_prev_avg_grid": constructor_prev_avg_grid,
                    "constructor_last5_finish_rate": constructor_last5_finish_rate,
                    "circuit_finish_rate": circuit_finish_rate,
                    "circuit_avg_grid": circuit_avg_grid,
                    "race_month": race_month,
                    "race_day_of_year": race_day_of_year,
                    "season_progress": season_progress,
                    "q1_seconds": q1_seconds,
                    "q2_seconds": q2_seconds,
                    "q3_seconds": q3_seconds,
                    "best_q_time": best_q_time,
                    "grid_normalized": grid_normalized,
                    "has_qualifying": has_qualifying,
                    "front_row_start": front_row_start,
                    "top10_start": top10_start,
                    "year": year,
                    "round": round_num,
                    "driver_nationality_encoded": driver_nationality_encoded,
                    "constructor_nationality_encoded": constructor_nationality_encoded,
                    "circuit_country_encoded": circuit_country_encoded,
                    "circuitRef_encoded": circuitRef_encoded,
                }
            ]
        )

        pred = pipe.predict(input_df)[0]
        prob = pipe.predict_proba(input_df)[0]
        prob_finish = prob[1] if len(prob) > 1 else prob[0]

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Probabilidad de terminar la carrera",
                value=f"{prob_finish * 100:.2f}%",
            )
        with col2:
            resultado = "✅ Sí termina" if pred == 1 else "❌ No termina"
            st.metric(label="Predicción", value=resultado)

        # Interpretación básica con feature importances si está disponible
        st.subheader("📊 Interpretación básica")

        importances = None
        feature_names = list(input_df.columns)

        # Intentar extraer feature_importances_ del paso del modelo dentro del pipeline
        if hasattr(pipe, "named_steps"):
            for name, estimator in pipe.named_steps.items():
                if hasattr(estimator, "feature_importances_"):
                    importances = estimator.feature_importances_
                    break
                # Soportar modelos envueltos en SelectFromModel, CalibratedClassifierCV, etc.
                if hasattr(estimator, "estimator") and hasattr(
                    estimator.estimator, "feature_importances_"
                ):
                    importances = estimator.estimator.feature_importances_
                    break
                if hasattr(estimator, "calibrated_classifiers_"):
                    # Tomar el primer clasificador calibrado
                    base_est = estimator.calibrated_classifiers_[0].estimator
                    if hasattr(base_est, "feature_importances_"):
                        importances = base_est.feature_importances_
                        break

        if importances is not None and len(importances) == len(feature_names):
            imp_df = pd.DataFrame(
                {"feature": feature_names, "importance": importances}
            ).sort_values("importance", ascending=False)

            top_n = min(10, len(imp_df))
            st.write(f"**Top {top_n} variables más importantes según el modelo:**")
            st.bar_chart(imp_df.set_index("feature").head(top_n))

            # Breve interpretación basada en los inputs actuales vs. importancias
            top_feature = imp_df.iloc[0]["feature"]
            st.info(
                f"El factor más influyente en la predicción es **{top_feature}**. "
                "Ajusta este parámetro en la barra lateral para observar cómo cambia el resultado."
            )
        else:
            st.write(
                "El modelo cargado no expone `feature_importances_` directamente, "
                "por lo que no se puede mostrar un ranking de importancia de variables. "
                "Esto es común en modelos como SVM o redes neuronales."
            )

        # Detalles técnicos opcionales
        with st.expander("Ver datos de entrada enviados al modelo"):
            st.dataframe(input_df.T.rename(columns={0: "valor"}))

# ---------------------------------------------------------------------------
# Pie de página con instrucciones
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    """
    ### 🚀 Instrucciones de ejecución

    Para iniciar esta aplicación localmente, abre una terminal en la raíz del proyecto y ejecuta:

    ```bash
    streamlit run app/app.py
    ```

    **Requisitos:**
    - Python 3.9+
    - `streamlit`, `pandas`, `numpy`, `joblib`, `scikit-learn` instalados.
    - El archivo de modelo `models/best_model_pipe.pkl` debe existir (generado en la fase de modelado).

    **Nota:** Si el modelo aún no existe, la app mostrará un mensaje instructivo explicando cómo generarlo.
    """
)
