import json

with open("notebooks/NFL_Prediction_Pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Cells: {len(nb['cells'])}")
print(f"nbformat: {nb['nbformat']}")
for i, c in enumerate(nb["cells"]):
    src = c["source"][0][:60] if c["source"] else "(empty)"
    print(f"  {i+1:2d}. {c['cell_type']:8s} | {src}")
print("Notebook is valid.")
