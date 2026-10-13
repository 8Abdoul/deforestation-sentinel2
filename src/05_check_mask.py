"""Diagnostic des masques : compare l'image (couleurs reelles + vegetation NDVI) au masque Hansen.
Lancer apres 02_tiling.py :  python src\\05_check_mask.py"""
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TILES = Path("data/tiles")
files = sorted((TILES / "images").glob("y2020_*.npy"))


def ndvi(img):                                  # img : (4, H, W) = B2, B3, B4, B8
    return (img[3] - img[2]) / (img[3] + img[2] + 1e-6)


# --- 1) Chiffres sur toutes les tuiles 2020 ---
veg_px = mask_px = both_px = total = 0
for f in files:
    img, mask = np.load(f), np.load(TILES / "masks" / f.name).astype(bool)
    veg = ndvi(img) > 0.6
    veg_px += veg.sum(); mask_px += mask.sum(); both_px += (veg & mask).sum(); total += mask.size
print(f"Tuiles 2020 : {len(files)}")
print(f"Pixels tres verts (NDVI > 0.6) : {100 * veg_px / total:.1f} %")
print(f"Pixels 'foret' selon le masque : {100 * mask_px / total:.1f} %")
if mask_px:
    print(f"Parmi les pixels du masque, % tres verts : {100 * both_px / mask_px:.1f} %")
if veg_px:
    print(f"Parmi les pixels tres verts, % dans le masque : {100 * both_px / veg_px:.1f} %")

# --- 2) Apercu : couleurs reelles / fausses couleurs / NDVI > 0.6 / masque ---
random.seed(1)
sample = random.sample(files, k=4)
fig, axes = plt.subplots(4, 4, figsize=(14, 14))
titles = ["Couleurs reelles", "Fausses couleurs", "NDVI > 0.6 (tres vert)", "Masque WorldCover (foret)"]
for r, f in enumerate(sample):
    img, mask = np.load(f), np.load(TILES / "masks" / f.name)
    true = np.clip(img[[2, 1, 0]].transpose(1, 2, 0) / 0.2, 0, 1)
    false = np.clip(img[[3, 2, 1]].transpose(1, 2, 0) / 0.3, 0, 1)
    panels = [true, false, ndvi(img) > 0.6, mask]
    for c, p in enumerate(panels):
        axes[r, c].imshow(p, cmap=None if c < 2 else "Greens", vmin=0, vmax=1)
        axes[r, c].axis("off")
        if r == 0:
            axes[r, c].set_title(titles[c], fontsize=11)
    axes[r, 0].text(2, 12, f.stem, color="white", fontsize=8, backgroundcolor="black")
Path("notebooks").mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig("notebooks/diagnostic_masque.png", dpi=110)
print("Image enregistree : notebooks/diagnostic_masque.png")
