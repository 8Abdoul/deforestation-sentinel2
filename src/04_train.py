"""Semaine 2 - Premier entrainement U-Net (foret / non-foret) sur les tuiles 2020.
Prerequis : pip install torch segmentation-models-pytorch
Lancer :    python src\\04_train.py
Remarque : GPU conseille (Google Colab ou Kaggle si ton PC n'en a pas)."""
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

TILES = Path("data/tiles")
BATCH, EPOCHS, LR = 8, 15, 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def position(f):                              # y2020_r003_c005.npy -> (3, 5)
    _, r, c = f.stem.split("_")
    return int(r[1:]), int(c[1:])


def split_by_zone(files):
    """Split par bandes de lignes : nord = train, centre = val, sud = test (pas de fuite spatiale)."""
    max_r = max(position(f)[0] for f in files)
    train = [f for f in files if position(f)[0] <= 0.6 * max_r]
    val = [f for f in files if 0.6 * max_r < position(f)[0] <= 0.8 * max_r]
    test = [f for f in files if position(f)[0] > 0.8 * max_r]
    return train, val, test


class TileDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        x = torch.from_numpy(np.load(f)).float()                                  # (4, 256, 256)
        y = torch.from_numpy(np.load(TILES / "masks" / f.name)).float().unsqueeze(0)  # (1, 256, 256)
        return x, y


def iou_score(logits, y, thr=0.5):
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * y).sum().item()
    union = (pred + y).clamp(0, 1).sum().item()
    return inter / union if union > 0 else 1.0


def evaluate(model, loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for x, y in loader:
            scores.append(iou_score(model(x.to(DEVICE)), y.to(DEVICE)))
    return float(np.mean(scores))


def main():
    files = sorted((TILES / "images").glob("y2020_*.npy"))
    train_f, val_f, test_f = split_by_zone(files)
    print(f"train={len(train_f)}  val={len(val_f)}  test={len(test_f)}  device={DEVICE}")
    train_dl = DataLoader(TileDataset(train_f), batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(TileDataset(val_f), batch_size=BATCH)
    test_dl = DataLoader(TileDataset(test_f), batch_size=BATCH)

    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=4, classes=1).to(DEVICE)
    dice, bce = smp.losses.DiceLoss(mode="binary"), nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    Path("models").mkdir(exist_ok=True)
    best = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = dice(out, y) + bce(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        val_iou = evaluate(model, val_dl)
        print(f"epoque {epoch:02d}  loss={total / len(train_dl):.4f}  IoU val={val_iou:.3f}")
        if val_iou > best:
            best = val_iou
            torch.save(model.state_dict(), "models/unet_best.pth")

    model.load_state_dict(torch.load("models/unet_best.pth", map_location=DEVICE))
    print(f"IoU test (meilleur modele) = {evaluate(model, test_dl):.3f}")


if __name__ == "__main__":
    main()
