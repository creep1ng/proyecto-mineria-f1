import nbformat as nbf
import sys

nb_path = sys.argv[1]
nb = nbf.read(nb_path, as_version=4)

for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'markdown':
        # Remove laps from variable lists
        cell.source = cell.source.replace("`laps`, ", "")
        cell.source = cell.source.replace(", `laps`", "")
        cell.source = cell.source.replace("`laps`", "")
        cell.source = cell.source.replace("laps, ", "")
        cell.source = cell.source.replace(", laps", "")
    if cell.cell_type == 'code':
        # Remove laps from example data dicts
        cell.source = cell.source.replace("'laps': laps, ", "")
        cell.source = cell.source.replace("'laps': 57, ", "")
        cell.source = cell.source.replace("'laps': 50, ", "")
        cell.source = cell.source.replace("laps = num_input(\"laps (vueltas programadas)\", 57, 1, 100, 1)\n", "")
        # Remove laps from DataFrame construction in app.py equivalent
        if "input_df = pd.DataFrame" in cell.source:
            cell.source = cell.source.replace("'laps': laps,\n", "")

nbf.write(nb, nb_path)
print(f"Notebook {nb_path} modificado exitosamente.")
