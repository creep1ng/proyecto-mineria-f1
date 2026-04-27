import nbformat as nbf
import sys

nb_path = sys.argv[1]
nb = nbf.read(nb_path, as_version=4)

# Helper to find cell index by substring

def find_cell_index(substring):
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and substring in cell.source:
            return i
    return None

def find_markdown_index(substring):
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown' and substring in cell.source:
            return i
    return None

# 1. In data loading cell, drop laps immediately after loading
idx_load = find_cell_index("pd.read_csv(DATA_PATH)")
if idx_load is not None:
    src = nb.cells[idx_load].source
    # After df.head(), add drop
    old = "df.head()"
    new = """df.head()

# Eliminar 'laps' (variable conocida después de la carrera)
df = df.drop(columns=['laps'])
print(f"Columna 'laps' eliminada. Dimensiones actualizadas: {df.shape}")"""
    nb.cells[idx_load].source = src.replace(old, new)

# 2. Remove laps from diccionario markdown
idx_dict = find_markdown_index("`laps`")
if idx_dict is not None:
    src = nb.cells[idx_dict].source
    lines = src.split('\n')
    new_lines = [l for l in lines if '`laps`' not in l]
    nb.cells[idx_dict].source = '\n'.join(new_lines)

# 3. Remove laps from quality rules
idx_qual = find_cell_index("df['laps'] < 0")
if idx_qual is not None:
    src = nb.cells[idx_qual].source
    lines = src.split('\n')
    new_lines = [l for l in lines if "laps" not in l]
    nb.cells[idx_qual].source = '\n'.join(new_lines)

# 4. Remove laps from continuous_vars in outliers
idx_out = find_cell_index("continuous_vars =")
if idx_out is not None:
    src = nb.cells[idx_out].source
    src = src.replace("'laps', ", "")
    src = src.replace(", 'laps'", "")
    src = src.replace("'laps'", "")
    nb.cells[idx_out].source = src

# 5. Remove laps from winsorization
idx_win = find_cell_index("winsorize_cols =")
if idx_win is not None:
    src = nb.cells[idx_win].source
    src = src.replace("'laps', ", "")
    src = src.replace(", 'laps'", "")
    src = src.replace("'laps'", "")
    nb.cells[idx_win].source = src

# 6. Update variable groups - remove laps from any group
idx_groups = find_cell_index("var_groups = {")
if idx_groups is not None:
    src = nb.cells[idx_groups].source
    src = src.replace("'laps', ", "")
    src = src.replace(", 'laps'", "")
    src = src.replace("'laps'", "")
    nb.cells[idx_groups].source = src

# 7. Add ydata-profiling section after the pandas profiling reference markdown
idx_prof_ref = find_markdown_index("Reporte de Pandas Profiling")
if idx_prof_ref is not None:
    # Insert after this markdown cell
    new_cells = []
    new_cells.append(nbf.v4.new_markdown_cell("""### 2.6b Perfilamiento con ydata-profiling

Se genera un reporte exhaustivo con **ydata-profiling** para documentar la calidad, distribuciones, correlaciones y alertas del dataset. Este reporte se almacena en el repositorio para consulta."""))
    new_cells.append(nbf.v4.new_code_cell("""from ydata_profiling import ProfileReport

profile = ProfileReport(df, title='F1 Dataset - ydata-profiling Report', explorative=True)
report_path = f'{OUTPUT_DIR}/ydata_profiling_report.html'
profile.to_file(report_path)
print(f'Reporte ydata-profiling guardado en: {report_path}')"""))
    for c in reversed(new_cells):
        nb.cells.insert(idx_prof_ref + 1, c)

# 8. Update summary markdown - remove laps references, adjust counts
idx_sum = find_markdown_index("Resumen de la Preparación")
if idx_sum is not None:
    src = nb.cells[idx_sum].source
    src = src.replace("`laps`", "")
    src = src.replace("laps, ", "")
    src = src.replace(", laps", "")
    # Update counts: 29 features -> 28 features (since laps removed before count)
    src = src.replace("29 features", "28 features")
    src = src.replace("23,777 registros", "23,777 registros")
    # Update final shape counts if present
    src = src.replace("24,874 registros × 23 columnas", "24,874 registros × 22 columnas")
    src = src.replace("7,134 registros × 23 columnas", "7,134 registros × 22 columnas")
    src = src.replace("22 features + finished", "21 features + finished")
    src = src.replace("23 columnas (22 features + finished)", "22 columnas (21 features + finished)")
    nb.cells[idx_sum].source = src

# 9. Remove laps from documentation of variables eliminadas
idx_drop_doc = find_markdown_index("Variables Eliminadas")
if idx_drop_doc is not None:
    src = nb.cells[idx_drop_doc].source
    lines = src.split('\n')
    new_lines = [l for l in lines if 'laps' not in l.lower()]
    nb.cells[idx_drop_doc].source = '\n'.join(new_lines)

# 10. Add note about laps removal in business understanding or data understanding
idx_bus = find_markdown_index("Reglas de Calidad desde el Negocio")
if idx_bus is not None:
    src = nb.cells[idx_bus].source
    src = src.replace(
        "4. **`laps`**: No negativa.",
        "4. **`laps`**: Eliminada del dataset (conocida post-carrera)."
    )
    nb.cells[idx_bus].source = src

nbf.write(nb, nb_path)
print(f"Notebook {nb_path} modificado exitosamente.")
