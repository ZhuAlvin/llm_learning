#!/usr/bin/env python3
"""Build 10.4 notebook by reading cell files from cells_10_4/ directory."""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))
CELLS_DIR = os.path.join(BASE, "cells_10_4")

# Read existing notebook
with open(os.path.join(BASE, "04_edge_deployment.ipynb"), "r") as f:
    nb = json.load(f)

# Apply cell replacements
ci = 0
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell_file = os.path.join(CELLS_DIR, f"cell_{ci:02d}.py")
        if os.path.exists(cell_file):
            with open(cell_file, "r") as f:
                source = f.read()
            cell["source"] = source
            cell["outputs"] = []
            cell["execution_count"] = None
        ci += 1

print(f"Total code cells: {ci}")
print(f"Cell files found in {CELLS_DIR}:")

import glob
cell_files = sorted(glob.glob(os.path.join(CELLS_DIR, "cell_*.py")))
for cf in cell_files:
    print(f"  {os.path.basename(cf)}")

out_path = os.path.join(BASE, "04_edge_deployment.ipynb")
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nWritten: {out_path}")
print(f"Replaced {len(cell_files)} code cells")
