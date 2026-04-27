import nbformat as nbf
import sys

nb_path = sys.argv[1]
nb = nbf.read(nb_path, as_version=4)

# Remove papermill error cells
nb.cells = [c for c in nb.cells if not any('papermill-error-cell' in str(v) for v in c.get('tags', []))]

# Replace the example data cell with correct columns
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and "example_data = pd.DataFrame" in cell.source:
        cell.source = """# Creamos 3 registros de ejemplo con las columnas procesadas finales
example_data = pd.DataFrame([
    {
        'grid': 1, 'driver_race_count': 120,
        'driver_prev_finish_rate': 0.85, 'driver_last5_finish_rate': 1.0,
        'constructor_race_count': 350, 'constructor_prev_finish_rate': 0.78,
        'constructor_prev_avg_grid': 2.5, 'constructor_last5_finish_rate': 0.8,
        'circuit_finish_rate': 0.88, 'circuit_avg_grid': 8.2,
        'q1_seconds': 88.5, 'q2_seconds': 87.9, 'q3_seconds': 87.4,
        'has_qualifying': 1, 'year': 2024,
        'driver_nationality_encoded': 5, 'circuit_country_encoded': 12, 'circuitRef_encoded': 7,
        'experience_ratio': 120 / 351,
        'grid_above_avg': 0,
        'avg_finish_rate': (0.85 + 0.78) / 2
    },
    {
        'grid': 15, 'driver_race_count': 35,
        'driver_prev_finish_rate': 0.55, 'driver_last5_finish_rate': 0.6,
        'constructor_race_count': 150, 'constructor_prev_finish_rate': 0.60,
        'constructor_prev_avg_grid': 12.0, 'constructor_last5_finish_rate': 0.5,
        'circuit_finish_rate': 0.75, 'circuit_avg_grid': 10.5,
        'q1_seconds': 91.2, 'q2_seconds': 90.8, 'q3_seconds': np.nan,
        'has_qualifying': 1, 'year': 2024,
        'driver_nationality_encoded': 8, 'circuit_country_encoded': 20, 'circuitRef_encoded': 15,
        'experience_ratio': 35 / 151,
        'grid_above_avg': 1,
        'avg_finish_rate': (0.55 + 0.60) / 2
    },
    {
        'grid': 20, 'driver_race_count': 10,
        'driver_prev_finish_rate': 0.30, 'driver_last5_finish_rate': 0.2,
        'constructor_race_count': 80, 'constructor_prev_finish_rate': 0.45,
        'constructor_prev_avg_grid': 16.5, 'constructor_last5_finish_rate': 0.3,
        'circuit_finish_rate': 0.65, 'circuit_avg_grid': 12.0,
        'q1_seconds': np.nan, 'q2_seconds': np.nan, 'q3_seconds': np.nan,
        'has_qualifying': 0, 'year': 2024,
        'driver_nationality_encoded': 2, 'circuit_country_encoded': 5, 'circuitRef_encoded': 22,
        'experience_ratio': 10 / 81,
        'grid_above_avg': 1,
        'avg_finish_rate': (0.30 + 0.45) / 2
    }
])

print('Datos de entrada (3 ejemplos):')
display(example_data)"""
        break

# Also update the markdown that lists expected variables
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'markdown' and 'Variables de entrada esperadas' in cell.source:
        cell.source = """## 5.1 Predicción de datos futuros

En esta sección mostramos cómo utilizar el pipeline cargado para predecir sobre nuevos registros.
El pipeline incluye el modelo final entrenado, por lo que los datos de entrada deben tener **las mismas columnas** que usó el modelo en entrenamiento (con excepción de la variable objetivo).

**Variables de entrada esperadas (después de preparación):**
- `grid`, `driver_race_count`, `driver_prev_finish_rate`, `driver_last5_finish_rate`
- `constructor_race_count`, `constructor_prev_finish_rate`, `constructor_prev_avg_grid`, `constructor_last5_finish_rate`
- `circuit_finish_rate`, `circuit_avg_grid`
- `q1_seconds`, `q2_seconds`, `q3_seconds`
- `has_qualifying`, `year`
- `driver_nationality_encoded`, `circuit_country_encoded`, `circuitRef_encoded`
- `experience_ratio`, `grid_above_avg`, `avg_finish_rate`"""
        break

nbf.write(nb, nb_path)
print(f"Notebook {nb_path} actualizado con columnas correctas.")
