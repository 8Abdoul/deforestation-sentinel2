"""Semaine 1 - Verification des tuiles : comptage, proportion de foret, apercu visuel.
Lancer apres 02_tiling.py :  python src\\03_explore.py"""
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TILES = Path("data/tiles")
files = sorted((TILES / "images").glob("*.npy"))
print("Nombre total de tuiles :", len(files))

for tag in ("y2020", "y2024"):
    fs = [f for f in files if f.name.startswith(tag)]
    if not fs:
        continue
    masks = np.stack([np.load(TILES / "masks" / f.name) for f in fs])
    print(f"{tag} : {len(fs)} tuiles, foret = {masks.mean() * 100:.1f} % des pixels")

sample = random.sample([f for f in files if f.name.startswith("y2020")], k=6)
fig, axes = plt.subplots(2, 6, figsize=(18, 6))
for k, f in enumerate(sample):
    img = np.load(f)                      # (4, 256, 256) : B2, B3, B4, B8
    mask = np.load(TILES / "masks" / f.name)
    rgb = np.clip(img[[3, 2, 1]].transpose(1, 2, 0) / 0.3, 0, 1)   # fausses couleurs B8, B4, B3
    axes[0, k].imshow(rgb); axes[0, k].set_title(f.stem, fontsize=8); axes[0, k].axis("off")
    axes[1, k].imshow(mask, cmap="Greens", vmin=0, vmax=1); axes[1, k].axis("off")
Path("notebooks").mkdir(exist_ok=True)
fig.savefig("notebooks/apercu_tuiles.png", dpi=120, bbox_inches="tight")
print("Image enregistree : notebooks/apercu_tuiles.png")
