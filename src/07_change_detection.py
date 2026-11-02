"""Semaine 3 - Detection de changement 2020 -> 2024 (version 3).
Prerequis : pip install scipy
Lancer :    python src\\07_change_detection.py
Principe : perte = foret predite en 2020 ET non-foret predit en 2024 ET baisse du NDVI > SEUIL (4 seuils testes),
           puis nettoyage morphologique. Comparaison avec Hansen 2021-2024 en pixels exacts et avec tolerance de 3 pixels.
Resultats : notebooks/changement_2020_2024.png, data/loss_pred_2020_2024.npy.
Attention : choisir le seuil NDVI en comparant a Hansen = reglage sur la reference ; a signaler dans le README."""
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
SIZE, MIN_PIXELS, TOL = 256, 50, 3            # 50 pixels de 10 m = 0.5 ha ; tolerance = 3 pixels (30 m, resolution de Hansen)
DROPS = [0.0, 0.1, 0.2, 0.3]                  # baisses de NDVI testees
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def position(f):
    _, r, c = f.stem.split("_")
    return int(r[1:]), int(c[1:])


model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=4, classes=1)
model.load_state_dict(torch.load("models/unet_best.pth", map_location=DEVICE))
model.to(DEVICE).eval()
norm = np.load("models/norm.npz")                 # cree par 04_train.py (v2)
MEAN, STD, THR = norm["mean"][:, None, None], norm["std"][:, None, None], float(norm["thr"])


def predict(img):
    x = torch.from_numpy((img - MEAN) / STD).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return (torch.sigmoid(model(x))[0, 0] > THR).cpu().numpy()


def ndvi(img):
    return (img[3] - img[2]) / (img[3] + img[2] + 1e-6)


pos20 = {position(f): f for f in (TILES / "images").glob("y2020_*.npy")}
pos24 = {position(f): f for f in (TILES / "images").glob("y2024_*.npy")}
common = sorted(set(pos20) & set(pos24))
n_r, n_c = max(r for r, _ in common) + 1, max(c for _, c in common) + 1
shape = (n_r * SIZE, n_c * SIZE)
forest20, forest24, valid = (np.zeros(shape, bool) for _ in range(3))
nd20, nd24 = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
for (r, c) in common:
    sl = (slice(r * SIZE, (r + 1) * SIZE), slice(c * SIZE, (c + 1) * SIZE))
    a, b = np.load(pos20[(r, c)]), np.load(pos24[(r, c)])
    forest20[sl], forest24[sl], valid[sl] = predict(a), predict(b), True
    nd20[sl], nd24[sl] = ndvi(a), ndvi(b)
print(f"{len(common)} tuiles communes aux deux annees")
print(f"Foret predite : 2020 = {100 * forest20.sum() / valid.sum():.1f} %   2024 = {100 * forest24.sum() / valid.sum():.1f} %")

with rasterio.open("data/raw/hansen_loss_2021_2024.tif") as src:
    hansen = src.read(1).astype(bool)
H, W = min(hansen.shape[0], shape[0]), min(hansen.shape[1], shape[1])
forest20, forest24, valid, nd20, nd24 = (x[:H, :W] for x in (forest20, forest24, valid, nd20, nd24))
hansen = hansen[:H, :W] & valid
hansen_tol = ndimage.binary_dilation(hansen, iterations=TOL)
print(f"Hansen : {hansen.sum() * 0.01:.0f} hectares de pertes 2021-2024")


def detect(drop):
    loss = forest20 & ~forest24 & valid & ((nd20 - nd24) > drop)
    loss = ndimage.binary_opening(loss, structure=np.ones((3, 3)))
    labels, n = ndimage.label(loss)
    if n == 0:
        return loss
    sizes = ndimage.sum(loss, labels, range(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= MIN_PIXELS
    return keep[labels]


best = None
print("\nBaisse NDVI | hectares | precision | rappel |  IoU  | precision (tol.) | rappel (tol.)")
for drop in DROPS:
    loss = detect(drop)
    inter = (loss & hansen).sum()
    prec, rec = inter / max(loss.sum(), 1), inter / max(hansen.sum(), 1)
    iou = inter / max((loss | hansen).sum(), 1)
    loss_tol = ndimage.binary_dilation(loss, iterations=TOL)
    prec_t, rec_t = (loss & hansen_tol).sum() / max(loss.sum(), 1), (hansen & loss_tol).sum() / max(hansen.sum(), 1)
    print(f"   > {drop:.1f}    | {loss.sum() * 0.01:7.0f}  |  {100 * prec:5.1f} %  | {100 * rec:5.1f} % | {100 * iou:4.1f} % |     {100 * prec_t:5.1f} %      |   {100 * rec_t:5.1f} %")
    if best is None or iou > best[0]:
        best = (iou, drop, loss)

iou, drop, loss = best
np.save("data/loss_pred_2020_2024.npy", loss)
print(f"\nSeuil retenu : baisse NDVI > {drop} (IoU {100 * iou:.1f} %) -> {loss.sum() * 0.01:.0f} hectares de perte predite")

fig, ax = plt.subplots(1, 4, figsize=(20, 5.5))
for a, img, t in zip(ax, [forest20, forest24, loss, hansen], ["Foret 2020 (predite)", "Foret 2024 (predite)",
                                                              f"Perte predite (NDVI > {drop})", "Pertes Hansen 2021-2024"]):
    a.imshow(img, cmap="Reds" if "erte" in t else "Greens", vmin=0, vmax=1); a.set_title(t); a.axis("off")
fig.tight_layout()
Path("notebooks").mkdir(exist_ok=True)
fig.savefig("notebooks/changement_2020_2024.png", dpi=110)
print("Image enregistree : notebooks/changement_2020_2024.png")