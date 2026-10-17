"""Semaine 2 - Entrainement U-Net (foret / non-foret) sur les tuiles 2020.   (version 2)
Prerequis : pip install torch segmentation-models-pytorch
Lancer :    python src\\04_train.py
Nouveautes v2 : normalisation des entrees, IoU train/val a chaque epoque (pour diagnostiquer),
                proportion de foret par zone, choix du seuil sur la validation, sauvegarde de models/norm.npz."""
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

TILES = Path("data/tiles")
BATCH, EPOCHS, LR = 8, 12, 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def position(f):                              # y2020_r003_c005.npy -> (3, 5)
    _, r, c = f.stem.split("_")
    return int(r[1:]), int(c[1:])


def split_by_zone(files):
    """Nord = train, centre = val, sud = test (pas de fuite spatiale)."""
    max_r = max(position(f)[0] for f in files)
    train = [f for f in files if position(f)[0] <= 0.6 * max_r]
    val = [f for f in files if 0.6 * max_r < position(f)[0] <= 0.8 * max_r]
    test = [f for f in files if position(f)[0] > 0.8 * max_r]
    return train, val, test


def compute_norm(files):
    s, s2, n = np.zeros(4), np.zeros(4), 0
    for f in files:
        x = np.load(f).astype("float64")
        s += x.sum(axis=(1, 2)); s2 += (x ** 2).sum(axis=(1, 2)); n += x.shape[1] * x.shape[2]
    mean = s / n
    return mean.astype("float32"), np.sqrt(s2 / n - mean ** 2).astype("float32")


class TileDataset(Dataset):
    def __init__(self, files, mean, std):
        self.files, self.mean, self.std = files, mean[:, None, None], std[:, None, None]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        x = torch.from_numpy((np.load(f) - self.mean) / self.std).float()             # (4, 256, 256)
        y = torch.from_numpy(np.load(TILES / "masks" / f.name)).float().unsqueeze(0)  # (1, 256, 256)
        return x, y


def collect(model, loader):
    model.eval()
    logits, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            logits.append(model(x.to(DEVICE)).cpu()); ys.append(y)
    return torch.cat(logits), torch.cat(ys)


def iou(logits, y, thr=0.5):
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * y).sum().item()
    union = ((pred + y) > 0).float().sum().item()
    return inter / union if union > 0 else 1.0


def forest_pct(files):
    return 100 * np.mean([np.load(TILES / "masks" / f.name).mean() for f in files])


def main():
    files = sorted((TILES / "images").glob("y2020_*.npy"))
    train_f, val_f, test_f = split_by_zone(files)
    print(f"train={len(train_f)}  val={len(val_f)}  test={len(test_f)}  device={DEVICE}")
    print(f"% de foret : train={forest_pct(train_f):.1f}  val={forest_pct(val_f):.1f}  test={forest_pct(test_f):.1f}")

    mean, std = compute_norm(train_f)
    print("moyenne par bande :", mean.round(3), " ecart-type :", std.round(3))
    mk = lambda fs, shuffle=False: DataLoader(TileDataset(fs, mean, std), batch_size=BATCH, shuffle=shuffle)
    train_dl, train_eval_dl, val_dl, test_dl = mk(train_f, True), mk(train_f), mk(val_f), mk(test_f)

    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=4, classes=1).to(DEVICE)
    dice, bce = smp.losses.DiceLoss(mode="binary"), nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    Path("models").mkdir(exist_ok=True)
    best = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = dice(out, y) + bce(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        tr_iou = iou(*collect(model, train_eval_dl))
        va_iou = iou(*collect(model, val_dl))
        print(f"epoque {epoch:02d}  loss={total / len(train_dl):.4f}  IoU train={tr_iou:.3f}  IoU val={va_iou:.3f}")
        if va_iou > best:
            best = va_iou
            torch.save(model.state_dict(), "models/unet_best.pth")

    model.load_state_dict(torch.load("models/unet_best.pth", map_location=DEVICE))
    v_logits, v_y = collect(model, val_dl)
    thrs = [0.3, 0.4, 0.5, 0.6, 0.7]
    scores = [iou(v_logits, v_y, t) for t in thrs]
    best_thr = thrs[int(np.argmax(scores))]
    print("IoU val selon le seuil :", {t: round(s, 3) for t, s in zip(thrs, scores)}, "-> seuil retenu", best_thr)
    t_logits, t_y = collect(model, test_dl)
    print(f"IoU test : seuil 0.5 = {iou(t_logits, t_y, 0.5):.3f}   seuil {best_thr} = {iou(t_logits, t_y, best_thr):.3f}")
    np.savez("models/norm.npz", mean=mean, std=std, thr=best_thr)


if __name__ == "__main__":
    main()