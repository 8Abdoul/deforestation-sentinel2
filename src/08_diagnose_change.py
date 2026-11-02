"""Diagnostic : pourquoi la detection de changement 2020 -> 2024 ne trouve-t-elle rien ?
Lancer :  python src\\08_diagnose_change.py
Sortie : chiffres dans le terminal + notebooks/diagnostic_changement.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch

TILES, SIZE = Path("data/tiles"), 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
norm = np.load("models/norm.npz")
MEAN, STD, THR = norm["mean"][:, None, None], norm["std"][:, None, None], float(norm["thr"])
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=4, classes=1)
model.load_state_dict(torch.load("models/unet_best.pth", map_location=DEVICE))
model.to(DEVICE).eval()


def position(f):
    _, r, c = f.stem.split("_")
    return int(r[1:]), int(c[1:])


def forest_pred(img):
    x = torch.from_numpy((img - MEAN) / STD).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return (torch.sigmoid(model(x))[0, 0] > THR).cpu().numpy()


def ndvi(img):
    return (img[3] - img[2]) / (img[3] + img[2] + 1e-6)


with rasterio.open("data/raw/hansen_loss_2021_2024.tif") as src:
    hansen = src.read(1).astype(bool)

pos20 = {position(f): f for f in (TILES / "images").glob("y2020_*.npy")}
pos24 = {position(f): f for f in (TILES / "images").glob("y2024_*.npy")}
acc = dict(n=0, h=0, f20=0, f24=0, h_f20=0, h_f24=0, raw_loss=0, nd20_h=0.0, nd24_h=0.0, nd20=0.0, nd24=0.0)
band20, band24, best = np.zeros(4), np.zeros(4), (0, None)
for (r, c), f20 in pos20.items():
    if (r, c) not in pos24:
        continue
    i, j = r * SIZE, c * SIZE
    h = hansen[i:i + SIZE, j:j + SIZE]
    if h.shape != (SIZE, SIZE):
        continue
    a, b = np.load(f20), np.load(pos24[(r, c)])
    p20, p24 = forest_pred(a), forest_pred(b)
    n20, n24 = ndvi(a), ndvi(b)
    acc["n"] += SIZE * SIZE; acc["h"] += h.sum()
    acc["f20"] += p20.sum(); acc["f24"] += p24.sum()
    acc["h_f20"] += (p20 & h).sum(); acc["h_f24"] += (p24 & h).sum()
    acc["raw_loss"] += (p20 & ~p24).sum()
    acc["nd20_h"] += n20[h].sum(); acc["nd24_h"] += n24[h].sum()
    acc["nd20"] += n20.sum(); acc["nd24"] += n24.sum()
    band20 += a.mean(axis=(1, 2)); band24 += b.mean(axis=(1, 2))
    if h.sum() > best[0]:
        best = (h.sum(), (a, b, h, p24))

N, H = acc["n"], max(acc["h"], 1)
k = len(pos20)
print(f"Pixels Hansen 'perte 2021-2024' : {acc['h']} ({100 * acc['h'] / N:.2f} % de la zone)")
print(f"Foret predite : 2020 = {100 * acc['f20'] / N:.1f} %   2024 = {100 * acc['f24'] / N:.1f} %")
print(f"Perte brute predite (avant nettoyage) : {acc['raw_loss']} pixels")
print(f"Dans les zones de perte Hansen : predit foret en 2020 = {100 * acc['h_f20'] / H:.1f} %   en 2024 = {100 * acc['h_f24'] / H:.1f} %")
print(f"NDVI moyen dans les zones de perte Hansen : 2020 = {acc['nd20_h'] / H:.3f}   2024 = {acc['nd24_h'] / H:.3f}")
print(f"NDVI moyen de toute la zone : 2020 = {acc['nd20'] / N:.3f}   2024 = {acc['nd24'] / N:.3f}")
print("Reflectance moyenne par bande 2020 :", (band20 / k).round(3), " 2024 :", (band24 / k).round(3))

if best[1] is not None:
    a, b, h, p24 = best[1]
    fig, ax = plt.subplots(1, 4, figsize=(18, 5))
    ax[0].imshow(np.clip(a[[2, 1, 0]].transpose(1, 2, 0) / 0.2, 0, 1)); ax[0].set_title("2020 (couleurs reelles)")
    ax[1].imshow(np.clip(b[[2, 1, 0]].transpose(1, 2, 0) / 0.2, 0, 1)); ax[1].set_title("2024 (couleurs reelles)")
    ax[2].imshow(h, cmap="Reds"); ax[2].set_title("Perte Hansen 2021-2024")
    ax[3].imshow(p24, cmap="Greens"); ax[3].set_title("Foret predite 2024")
    for x in ax:
        x.axis("off")
    fig.tight_layout()
    Path("notebooks").mkdir(exist_ok=True)
    fig.savefig("notebooks/diagnostic_changement.png", dpi=110)
    print("Image enregistree : notebooks/diagnostic_changement.png")
