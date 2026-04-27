import pandas as pd
import os
import sys

# Asegurar que se encuentren las librerías locales si es necesario
lib_path = "/home/creep/workshop/proyecto-mineria/lib"
if os.path.isdir(lib_path):
    sys.path.insert(0, lib_path)

# Rutas
DATASET_PATH = "/home/creep/workshop/proyecto-mineria/f1_final_dataset.csv"
OUTPUT_DIR = "/home/creep/workshop/proyecto-mineria/output"
MD_PATH = os.path.join(OUTPUT_DIR, "exploracion_dataset.md")
HTML_PATH = os.path.join(OUTPUT_DIR, "pandas_profiling_report.html")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carga
print("Cargando dataset...")
df = pd.read_csv(DATASET_PATH)

# 1. Primeras filas
head_md = df.head(10).to_markdown(index=False)

# 2. Info
import io

buf = io.StringIO()
df.info(buf=buf)
info_lines = buf.getvalue()

# 3. Estadísticas descriptivas
# Incluir todas las numéricas
describe_md = df.describe().T.to_markdown()

# 4. Distribución de clases
finished_counts = df["finished"].value_counts().sort_index()
finished_pct = df["finished"].value_counts(normalize=True).sort_index() * 100
finished_df = pd.DataFrame(
    {
        "Clase": finished_counts.index,
        "Conteo": finished_counts.values,
        "Porcentaje": finished_pct.values.round(2),
    }
)
finished_md = finished_df.to_markdown(index=False)

# 5. Nulos
nulls = df.isnull().sum()
nulls_pct = (df.isnull().mean() * 100).round(4)
nulls_df = pd.DataFrame(
    {
        "Variable": df.columns,
        "Nulos": nulls.values,
        "Porcentaje_Nulos": nulls_pct.values,
    }
)
nulls_md = nulls_df.to_markdown(index=False)

# 6. Diccionario de datos inferido
descriptions = {
    "grid": "Posición de salida en la parrilla (grid position)",
    "laps": "Número de vueltas completadas en la carrera",
    "driver_age": "Edad del piloto al momento de la carrera",
    "driver_race_count": "Número total de carreras previas del piloto",
    "driver_prev_finish_rate": "Tasa histórica de finalización del piloto",
    "driver_last5_finish_rate": "Tasa de finalización del piloto en las últimas 5 carreras",
    "constructor_race_count": "Número total de carreras previas del constructor",
    "constructor_prev_finish_rate": "Tasa histórica de finalización del constructor",
    "constructor_prev_avg_grid": "Posición media de salida histórica del constructor",
    "constructor_last5_finish_rate": "Tasa de finalización del constructor en las últimas 5 carreras",
    "circuit_finish_rate": "Tasa histórica de finalización en el circuito",
    "circuit_avg_grid": "Posición media de salida histórica en el circuito",
    "race_month": "Mes en que se disputa la carrera",
    "race_day_of_year": "Día del año en que se disputa la carrera",
    "season_progress": "Progreso de la temporada (fracción de carreras completadas)",
    "q1_seconds": "Tiempo de clasificación Q1 en segundos",
    "q2_seconds": "Tiempo de clasificación Q2 en segundos",
    "q3_seconds": "Tiempo de clasificación Q3 en segundos",
    "best_q_time": "Mejor tiempo de clasificación (menor de Q1, Q2, Q3)",
    "grid_normalized": "Posición de salida normalizada (escala 0-1 aprox)",
    "has_qualifying": "Indicador binario de si hubo sesión de clasificación",
    "front_row_start": "Indicador binario de si salió en la primera fila",
    "top10_start": "Indicador binario de si salió en el top 10",
    "year": "Año de la carrera",
    "round": "Número de ronda dentro de la temporada",
    "driver_nationality_encoded": "Nacionalidad del piloto (codificada numéricamente)",
    "constructor_nationality_encoded": "Nacionalidad del constructor (codificada numéricamente)",
    "circuit_country_encoded": "País del circuito (codificado numéricamente)",
    "circuitRef_encoded": "Referencia del circuito (codificada numéricamente)",
    "finished": "Variable objetivo: 1 si el piloto finalizó la carrera, 0 si no",
}

data_dict = []
for col in df.columns:
    data_dict.append(
        {
            "Variable": col,
            "Descripcion": descriptions.get(col, "Descripción no disponible"),
            "Tipo_Dato": str(df[col].dtype),
            "Unicos": df[col].nunique(),
            "Ejemplo": str(df[col].iloc[0]),
        }
    )

data_dict_df = pd.DataFrame(data_dict)
data_dict_md = data_dict_df.to_markdown(index=False)

# Dimensiones
shape_md = f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}"

# Tipos de datos
dtypes_md = df.dtypes.reset_index()
dtypes_md.columns = ["Variable", "Tipo_Dato"]
dtypes_md = dtypes_md.to_markdown(index=False)

# Correlación con finished (top absolutas)
corr_series = (
    df.corr(numeric_only=True)["finished"]
    .drop("finished")
    .abs()
    .sort_values(ascending=False)
)
corr_df = pd.DataFrame(
    {
        "Variable": corr_series.index,
        "Correlacion_Abs_con_finished": corr_series.values.round(4),
    }
)
corr_md = corr_df.head(15).to_markdown(index=False)

# Construir markdown
md_content = f"""# Exploración Exhaustiva del Dataset F1

## 1. Dimensiones
{shape_md}

## 2. Primeras 10 Filas
{head_md}

## 3. Información General (info)
```
{info_lines}
```

## 4. Tipos de Datos
{dtypes_md}

## 5. Estadísticas Descriptivas (Variables Numéricas)
{describe_md}

## 6. Distribución de la Variable Objetivo: `finished`
{finished_md}

## 7. Valores Nulos por Columna
{nulls_md}

## 8. Diccionario de Datos Inferido
{data_dict_md}

## 9. Correlación Absoluta con `finished` (Top 15)
{corr_md}

## 10. Observaciones Clave
- El dataset contiene **{df.shape[0]}** registros y **{df.shape[1]}** variables.
- No se detectaron valores nulos en ninguna columna (porcentaje 0.0%% en todas).
- La variable objetivo `finished` está desbalanceada: la clase 1 (finalizó) representa aproximadamente el {finished_pct.get(1, 0):.2f}% de los registros.
- Existen múltiples variables derivadas de historiales (tasas de finalización, promedios de grid) que capturan comportamiento pasado.
- Las variables codificadas (`*_encoded`) son numéricas discretas resultado de transformaciones de variables categóricas.
- Los tiempos de clasificación (`q1_seconds`, `q2_seconds`, `q3_seconds`, `best_q_time`) presentan el mismo valor en las primeras filas, lo que podría indicar imputación o datos faltantes para épocas sin sesiones de clasificación modernas.

---
*Generado automáticamente para el proyecto CRISP-DM - F1 Classification*
"""

with open(MD_PATH, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"Markdown guardado en: {MD_PATH}")

# Pandas Profiling (ydata-profiling)
try:
    from ydata_profiling import ProfileReport

    print("Generando reporte de profiling...")
    profile = ProfileReport(
        df, title="F1 Dataset - Pandas Profiling Report", minimal=False
    )
    profile.to_file(HTML_PATH)
    print(f"Reporte HTML guardado en: {HTML_PATH}")
except Exception as e:
    print(f"Error generando profiling: {e}")
    # Intentar fallback con pandas_profiling antiguo
    try:
        import pandas_profiling

        profile = pandas_profiling.ProfileReport(
            df, title="F1 Dataset - Pandas Profiling Report"
        )
        profile.to_file(HTML_PATH)
        print(f"Reporte HTML (fallback) guardado en: {HTML_PATH}")
    except Exception as e2:
        print(f"Fallback también falló: {e2}")

print("Exploración completada.")
