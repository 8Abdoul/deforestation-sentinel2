"""Semaine 3 - Detection de changement 2020 -> 2024 avec le modele entraine (models/unet_best.pth).
Prerequis : pip install scipy   (en plus de torch, segmentation-models-pytorch, rasterio, matplotlib)
Lancer :    python src\\07_change_detection.py
Resultats : notebooks/changement_2020_2024.png, data/loss_pred_2020_2024.npy et chiffres dans le terminal.
Attention : un composite 2024 trouble (fumee) ou different de 2020 cree de fausses pertes. Compare toujours avec Hansen."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from scipy import ndimage

TILES = Path("data/tiles")
SIZE, THR, MIN_PIXELS = 256, 0.5, 50          # 50 pixels de 10 m x 10 m = 0.5 hectare
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def position(f):                               # y2020_r003_c005.npy -> (3, 5)
    _, r, c = f.stem.split("_")
    return int(r[1:]), int(c[1:])


model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=4, classes=1)
model.load_state_dict(torch.load("models/unet_best.pth", map_location=DEVICE))
model.to(DEVICE).eval()


def predict(f):
    x = torch.from_numpy(np.load(f)).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return (torch.sigmoid(model(x))[0, 0] > THR).cpu().numpy()


pos20 = {position(f): f for f in (TILES / "images").glob("y2020_*.npy")}
pos24 = {position(f): f for f in (TILES / "images").glob("y2024_*.npy")}
common = sorted(set(pos20) & set(pos24))
n_r, n_c = max(r for r, _ in common) + 1, max(c for _, c in common) + 1
forest20 = np.zeros((n_r * SIZE, n_c * SIZE), bool)
forest24 = np.zeros_like(forest20)
valid = np.zeros_like(forest20)
for (r, c) in common:
    sl = (slice(r * SIZE, (r + 1) * SIZE), slice(c * SIZE, (c + 1) * SIZE))
    forest20[sl], forest24[sl], valid[sl] = predict(pos20[(r, c)]), predict(pos24[(r, c)]), True
print(f"{len(common)} tuiles communes aux deux annees")

# Foret perdue = foret en 2020 et non-foret en 2024, nettoyee du bruit
loss = forest20 & ~forest24 & valid
loss = ndimage.binary_opening(loss, structure=np.ones((3, 3)))
labels, n = ndimage.label(loss)
sizes = ndimage.sum(loss, labels, range(1, n + 1))
loss = np.isin(labels, [i + 1 for i, s in enumerate(sizes) if s >= MIN_PIXELS])
np.save("data/loss_pred_2020_2024.npy", loss)
print(f"Foret perdue predite : {loss.sum() * 0.01:.0f} hectares ({100 * loss.sum() / valid.sum():.2f} % de la zone)")

# Verification independante avec Hansen (pertes 2021-2024)
with rasterio.open("data/raw/hansen_loss_2021_2024.tif") as src:
    h = src.read(1).astype(bool)
H, W = min(h.shape[0], loss.shape[0]), min(h.shape[1], loss.shape[1])
h, lp, v = h[:H, :W], loss[:H, :W], valid[:H, :W]
h = h & v
inter = (h & lp).sum()
print(f"Hansen : {h.sum() * 0.01:.0f} hectares de pertes 2021-2024")
print(f"Precision : {100 * inter / max(lp.sum(), 1):.1f} %   Rappel : {100 * inter / max(h.sum(), 1):.1f} %   "
      f"IoU : {100 * inter / max((h | lp).sum(), 1):.1f} %")

fig, ax = plt.subplots(1, 4, figsize=(20, 5.5))
for a, img, t in zip(ax, [forest20, forest24, loss, h], ["Foret 2020 (predite)", "Foret 2024 (predite)",
                                                          "Perte predite 2020-2024", "Pertes Hansen 2021-2024"]):
    a.imshow(img, cmap="Reds" if "erte" in t else "Greens"); a.set_title(t); a.axis("off")
fig.tight_layout()
Path("notebooks").mkdir(exist_ok=True)
fig.savefig("notebooks/changement_2020_2024.png", dpi=110)
print("Image enregistree : notebooks/changement_2020_2024.png")
