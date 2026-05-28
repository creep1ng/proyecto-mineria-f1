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
import importlib.util
import types

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# Ensure pickle dependencies resolve to the reusable pipeline module, not this
# Streamlit script (`app/app.py`) when joblib imports `app.f1_pipeline`.
PIPELINE_MODULE = "app.f1_pipeline"
if PIPELINE_MODULE not in sys.modules:
    package = sys.modules.get("app")
    if package is None or not hasattr(package, "__path__"):
        package = types.ModuleType("app")
        package.__path__ = [APP_DIR]
        sys.modules["app"] = package

    spec = importlib.util.spec_from_file_location(
        PIPELINE_MODULE, os.path.join(APP_DIR, "f1_pipeline.py")
    )
    if spec is None or spec.loader is None:
        raise ImportError("No se pudo cargar app.f1_pipeline para deserializar el modelo.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[PIPELINE_MODULE] = module
    spec.loader.exec_module(module)

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
    "pre-carrera crudos. El pipeline aplica internamente ingeniería, selección, "
    "balanceo de entrenamiento, escalado y modelo."
)

# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(
    APP_DIR, "..", "models", "best_model_pipe.pkl"
)
TRAIN_REFERENCE_PATH = os.path.join(APP_DIR, "..", "output", "train_original.csv")

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


@st.cache_data
def load_training_reference(path):
    """Load observed training values used to constrain categorical inputs."""

    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


training_reference = load_training_reference(TRAIN_REFERENCE_PATH)
if training_reference.empty:
    st.warning(
        "No se encontró la referencia de entrenamiento. La app usará valores por defecto, "
        "pero los desplegables no podrán limitarse al dominio entrenado."
    )

# ---------------------------------------------------------------------------
# Inputs del usuario
# ---------------------------------------------------------------------------
st.sidebar.header("📋 Parámetros de entrada")


# Helper para inputs numéricos
def num_input(label, value, min_value=None, max_value=None, step=None, help_text="", key=None):
    kwargs = {"help": help_text}
    if key is not None:
        kwargs["key"] = key
    if min_value is not None:
        kwargs["min_value"] = min_value
    if max_value is not None:
        kwargs["max_value"] = max_value
    if step is not None:
        kwargs["step"] = step
    return st.sidebar.number_input(label, value=value, **kwargs)


def observed_values(column, fallback):
    if column not in training_reference.columns:
        return fallback
    values = training_reference[column].dropna().unique().tolist()
    return sorted(int(value) for value in values) or fallback


def option_index(options, default):
    return options.index(default) if default in options else 0


def encoded_select(label, column, default, help_text=""):
    options = observed_values(column, [default])
    return st.sidebar.selectbox(
        label,
        options,
        index=option_index(options, default),
        format_func=lambda value: f"{value} (observado en entrenamiento)",
        help=help_text,
        key=column,
    )


def build_profile_options(columns, sort_columns, ascending=False, limit=500):
    if training_reference.empty or any(
        col not in training_reference.columns for col in columns
    ):
        return pd.DataFrame(columns=columns)

    profiles = training_reference.loc[:, columns].drop_duplicates()
    profiles = (
        profiles.sort_values(sort_columns, ascending=ascending)
        .head(limit)
        .reset_index(drop=True)
    )
    return profiles


def profile_label(prefix, row, fields):
    parts = [
        f"{label}: {row[column]:.2f}"
        if isinstance(row[column], float)
        else f"{label}: {row[column]}"
        for label, column in fields
    ]
    return f"{prefix} {row.name + 1} · " + " · ".join(parts)


# --- Datos de parrilla y carrera ---
with st.sidebar.expander("Parrilla y carrera", expanded=True):
    grid = num_input(
        "grid (posición en parrilla)",
        1,
        1,
        30,
        1,
        "Posición de salida en la parrilla (1 = pole).",
        key="grid",
    )
    year = num_input("year", 2024, 1950, 2030, 1, key="year")
    round_num = num_input("round (ronda de la temporada)", 10, 1, 25, 1, key="round")
    top10_start = 1 if grid <= 10 else 0

# --- Datos del piloto ---
driver_profiles = build_profile_options(
    [
        "driver_race_count",
        "driver_prev_finish_rate",
        "driver_last5_finish_rate",
        "driver_nationality_encoded",
    ],
    ["driver_race_count", "driver_prev_finish_rate"],
)
with st.sidebar.expander("Piloto", expanded=True):
    if not driver_profiles.empty:
        driver_profile_index = st.sidebar.selectbox(
            "Piloto entrenado",
            range(len(driver_profiles)),
            format_func=lambda idx: profile_label(
                "Piloto",
                driver_profiles.iloc[idx],
                [
                    ("carreras", "driver_race_count"),
                    ("finish", "driver_prev_finish_rate"),
                    ("nac", "driver_nationality_encoded"),
                ],
            ),
            key="driver_profile",
        )
        selected_driver = driver_profiles.iloc[driver_profile_index]
        driver_race_default = int(selected_driver["driver_race_count"])
        driver_prev_finish_default = float(selected_driver["driver_prev_finish_rate"])
        driver_last5_finish_default = float(selected_driver["driver_last5_finish_rate"])
        driver_nationality_default = int(selected_driver["driver_nationality_encoded"])
    else:
        driver_race_default = 50
        driver_prev_finish_default = 0.75
        driver_last5_finish_default = 0.80
        driver_nationality_default = 5

    driver_race_count = num_input(
        "driver_race_count (carreras previas)",
        driver_race_default,
        0,
        400,
        1,
        key="driver_race_count",
    )
    driver_prev_finish_rate = st.sidebar.slider(
        "driver_prev_finish_rate",
        0.0,
        1.0,
        driver_prev_finish_default,
        0.01,
        help="Tasa histórica de finalización del piloto.",
        key="driver_prev_finish_rate",
    )
    driver_last5_finish_rate = st.sidebar.slider(
        "driver_last5_finish_rate",
        0.0,
        1.0,
        driver_last5_finish_default,
        0.01,
        help="Tasa de finalización en las últimas 5 carreras.",
        key="driver_last5_finish_rate",
    )
    driver_nationality_encoded = encoded_select(
        "Nacionalidad del piloto",
        "driver_nationality_encoded",
        driver_nationality_default,
        "Lista limitada a las nacionalidades codificadas presentes en entrenamiento.",
    )

# --- Datos del constructor ---
constructor_profiles = build_profile_options(
    [
        "constructor_race_count",
        "constructor_prev_finish_rate",
        "constructor_prev_avg_grid",
        "constructor_last5_finish_rate",
        "constructor_nationality_encoded",
    ],
    ["constructor_race_count", "constructor_prev_finish_rate"],
)
with st.sidebar.expander("Constructor (equipo)", expanded=True):
    if not constructor_profiles.empty:
        constructor_profile_index = st.sidebar.selectbox(
            "Constructor entrenado",
            range(len(constructor_profiles)),
            format_func=lambda idx: profile_label(
                "Constructor",
                constructor_profiles.iloc[idx],
                [
                    ("carreras", "constructor_race_count"),
                    ("finish", "constructor_prev_finish_rate"),
                    ("nac", "constructor_nationality_encoded"),
                ],
            ),
            key="constructor_profile",
        )
        selected_constructor = constructor_profiles.iloc[constructor_profile_index]
        constructor_race_default = int(selected_constructor["constructor_race_count"])
        constructor_prev_finish_default = float(
            selected_constructor["constructor_prev_finish_rate"]
        )
        constructor_prev_avg_grid_default = float(
            selected_constructor["constructor_prev_avg_grid"]
        )
        constructor_last5_finish_default = float(
            selected_constructor["constructor_last5_finish_rate"]
        )
        constructor_nationality_default = int(
            selected_constructor["constructor_nationality_encoded"]
        )
    else:
        constructor_race_default = 200
        constructor_prev_finish_default = 0.70
        constructor_prev_avg_grid_default = 8.0
        constructor_last5_finish_default = 0.75
        constructor_nationality_default = 5

    constructor_race_count = num_input(
        "constructor_race_count (carreras previas)",
        constructor_race_default,
        0,
        max(2500, constructor_race_default),
        1,
        key="constructor_race_count",
    )
    constructor_prev_finish_rate = st.sidebar.slider(
        "constructor_prev_finish_rate",
        0.0,
        1.0,
        constructor_prev_finish_default,
        0.01,
        key="constructor_prev_finish_rate",
    )
    constructor_prev_avg_grid = num_input(
        "constructor_prev_avg_grid (parrilla promedio histórica)",
        constructor_prev_avg_grid_default,
        0.0,
        40.0,
        0.5,
        key="constructor_prev_avg_grid",
    )
    constructor_last5_finish_rate = st.sidebar.slider(
        "constructor_last5_finish_rate",
        0.0,
        1.0,
        constructor_last5_finish_default,
        0.01,
        key="constructor_last5_finish_rate",
    )
    constructor_nationality_encoded = encoded_select(
        "Nacionalidad del constructor",
        "constructor_nationality_encoded",
        constructor_nationality_default,
        "Lista limitada a las nacionalidades de constructor presentes en entrenamiento.",
    )

# --- Datos del circuito ---
circuit_profiles = build_profile_options(
    [
        "circuit_finish_rate",
        "circuit_avg_grid",
        "circuit_country_encoded",
        "circuitRef_encoded",
    ],
    ["circuit_finish_rate", "circuit_avg_grid"],
)
with st.sidebar.expander("Circuito", expanded=True):
    if not circuit_profiles.empty:
        circuit_profile_index = st.sidebar.selectbox(
            "Circuito entrenado",
            range(len(circuit_profiles)),
            format_func=lambda idx: profile_label(
                "Circuito",
                circuit_profiles.iloc[idx],
                [
                    ("finish", "circuit_finish_rate"),
                    ("pais", "circuit_country_encoded"),
                    ("ref", "circuitRef_encoded"),
                ],
            ),
            key="circuit_profile",
        )
        selected_circuit = circuit_profiles.iloc[circuit_profile_index]
        circuit_finish_default = float(selected_circuit["circuit_finish_rate"])
        circuit_avg_grid_default = float(selected_circuit["circuit_avg_grid"])
        circuit_country_default = int(selected_circuit["circuit_country_encoded"])
        circuit_ref_default = int(selected_circuit["circuitRef_encoded"])
    else:
        circuit_finish_default = 0.80
        circuit_avg_grid_default = 10.0
        circuit_country_default = 10
        circuit_ref_default = 8

    circuit_finish_rate = st.sidebar.slider(
        "circuit_finish_rate",
        0.0,
        1.0,
        circuit_finish_default,
        0.01,
        help="Tasa histórica de finalización en este circuito.",
        key="circuit_finish_rate",
    )
    circuit_avg_grid = num_input(
        "circuit_avg_grid (parrilla promedio histórica)",
        circuit_avg_grid_default,
        1.0,
        30.0,
        0.5,
        key="circuit_avg_grid",
    )
    circuit_country_encoded = encoded_select(
        "País del circuito",
        "circuit_country_encoded",
        circuit_country_default,
        "Lista limitada a los países de circuito presentes en entrenamiento.",
    )
    circuitRef_encoded = encoded_select(
        "Circuito",
        "circuitRef_encoded",
        circuit_ref_default,
        "Lista limitada a los circuitos presentes en entrenamiento.",
    )

# --- Clasificación (qualifying) ---
qualifying_profiles = build_profile_options(
    ["has_qualifying", "q1_seconds", "q2_seconds", "q3_seconds"],
    ["has_qualifying", "q1_seconds"],
    [False, True],
)
with st.sidebar.expander("Clasificación", expanded=True):
    if not qualifying_profiles.empty:
        qualifying_profile_index = st.sidebar.selectbox(
            "Clasificación entrenada",
            range(len(qualifying_profiles)),
            format_func=lambda idx: profile_label(
                "Clasificación",
                qualifying_profiles.iloc[idx],
                [
                    ("hay", "has_qualifying"),
                    ("Q1", "q1_seconds"),
                    ("Q2", "q2_seconds"),
                    ("Q3", "q3_seconds"),
                ],
            ),
            key="qualifying_profile",
        )
        selected_qualifying = qualifying_profiles.iloc[qualifying_profile_index]
        has_qualifying_default = int(selected_qualifying["has_qualifying"])
        q1_default = float(selected_qualifying["q1_seconds"])
        q2_default = float(selected_qualifying["q2_seconds"])
        q3_default = float(selected_qualifying["q3_seconds"])
    else:
        has_qualifying_default = 1
        q1_default = 88.5
        q2_default = 87.9
        q3_default = 87.4

    has_qualifying_options = observed_values("has_qualifying", [0, 1])
    has_qualifying = st.sidebar.selectbox(
        "¿Hubo clasificación?",
        has_qualifying_options,
        index=option_index(has_qualifying_options, has_qualifying_default),
        format_func=lambda value: "Sí" if value == 1 else "No",
        help="Valores observados para `has_qualifying` durante entrenamiento.",
        key="has_qualifying",
    )
    q1_seconds = num_input(
        "q1_seconds (tiempo Q1 en segundos)",
        q1_default,
        50.0,
        1100.0,
        0.1,
        key="q1_seconds",
    )
    q2_seconds = num_input(
        "q2_seconds (tiempo Q2 en segundos)",
        q2_default,
        50.0,
        150.0,
        0.1,
        key="q2_seconds",
    )
    q3_seconds = num_input(
        "q3_seconds (tiempo Q3 en segundos)",
        q3_default,
        50.0,
        150.0,
        0.1,
        key="q3_seconds",
    )

# ---------------------------------------------------------------------------
# Botón de predicción
# ---------------------------------------------------------------------------
if st.sidebar.button("🔮 Predecir", type="primary", key="predict_button"):
    if not model_loaded:
        st.error(
            "No se puede predecir porque el modelo no está disponible. "
            f"Error reportado: {model_error}"
        )
    else:
        # Variables derivadas (ingeniería de características)
        # No se calculan manualmente: el pipeline entrenado contiene FeatureEngineer.

        input_df = pd.DataFrame(
            [
                {
                    "grid": grid,
                    "driver_race_count": driver_race_count,
                    "driver_prev_finish_rate": driver_prev_finish_rate,
                    "driver_last5_finish_rate": driver_last5_finish_rate,
                    "constructor_race_count": constructor_race_count,
                    "constructor_prev_finish_rate": constructor_prev_finish_rate,
                    "constructor_prev_avg_grid": constructor_prev_avg_grid,
                    "constructor_last5_finish_rate": constructor_last5_finish_rate,
                    "circuit_finish_rate": circuit_finish_rate,
                    "circuit_avg_grid": circuit_avg_grid,
                    "q1_seconds": q1_seconds,
                    "q2_seconds": q2_seconds,
                    "q3_seconds": q3_seconds,
                    "has_qualifying": has_qualifying,
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
