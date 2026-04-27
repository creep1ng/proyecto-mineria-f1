import nbformat as nbf
import sys

nb_path = sys.argv[1]
nb = nbf.read(nb_path, as_version=4)

# Remove papermill error cells
nb.cells = [c for c in nb.cells if not any('papermill-error-cell' in str(v) for v in c.get('tags', []))]

# Fix the first code cell to use absolute path
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and "MODEL_PATH = '../models/best_model_pipe.pkl'" in cell.source:
        cell.source = cell.source.replace(
            "MODEL_PATH = '../models/best_model_pipe.pkl'",
            "MODEL_PATH = '/home/creep/workshop/proyecto-mineria/models/best_model_pipe.pkl'"
        )
        break

nbf.write(nb, nb_path)
print(f"Notebook {nb_path} restaurado y corregido.")
