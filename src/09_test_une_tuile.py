"""Test isole du modele sur quelques tuiles 2020 (sans assemblage ni 2024).
Lancer :  python src\\09_test_une_tuile.py"""
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch

norm = np.load("models/norm.npz")
MEAN, STD = norm["mean"][:, None, None], norm["std"][:, None, None]
print("mean", norm["mean"], "std", norm["std"], "seuil", float(norm["thr"]))

model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=4, classes=1)
print("chargement des poids :", model.load_state_dict(torch.load("models/unet_best.pth", map_location="cpu")))
model.eval()

files = sorted(Path("data/tiles/images").glob("y2020_*.npy"))[::28][:6]
for f in files:
    img = np.load(f)
    mask = np.load(Path("data/tiles/masks") / f.name)
    x = torch.from_numpy((img - MEAN) / STD).float().unsqueeze(0)
    with torch.no_grad():
        p = torch.sigmoid(model(x))[0, 0].numpy()
    print(f"{f.name} | image {img.dtype} min={img.min():.3f} max={img.max():.3f} | proba min={p.min():.3f} "
          f"moy={p.mean():.3f} max={p.max():.3f} | % foret predit={100 * (p > 0.5).mean():.1f} | % foret masque={100 * mask.mean():.1f}")
