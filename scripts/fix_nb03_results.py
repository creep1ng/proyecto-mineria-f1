import nbformat as nbf
import sys

nb_path = sys.argv[1]
nb = nbf.read(nb_path, as_version=4)

# Fix the results display line
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and "example_data[['grid', 'driver_age', 'year', 'round']]" in cell.source:
        cell.source = cell.source.replace(
            "example_data[['grid', 'driver_age', 'year', 'round']]",
            "example_data[['grid', 'year']]"
        )
        break

nbf.write(nb, nb_path)
print(f"Notebook {nb_path} corregido.")
