"""
Semaine 1 - Decoupage des GeoTIFF exportes en tuiles 256 x 256.

Prerequis : pip install rasterio numpy
Place les fichiers telecharges depuis Google Drive dans data/raw/ :
    s2_2020.tif, forest_2020.tif, s2_2024.tif, forest_2024.tif
"""
from pathlib import Path

import numpy as np
import rasterio

SIZE = 256
MAX_INVALID = 0.20   # on ignore les tuiles avec plus de 20 % de pixels vides (nuages masques)


def tile_pair(img_path, mask_path, out_dir, tag):
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    with rasterio.open(img_path) as src_img, rasterio.open(mask_path) as src_mask:
        img = src_img.read().astype("float32")          # (4, H, W), reflectance x 10000
        mask = src_mask.read(1).astype("uint8")         # (H, W)

    # Les deux exports peuvent differer d'un pixel : on recadre au plus petit commun
    h = min(img.shape[1], mask.shape[0])
    w = min(img.shape[2], mask.shape[1])
    img, mask = img[:, :h, :w], mask[:h, :w]

    invalid = ~np.isfinite(img).all(axis=0) | (img.sum(axis=0) == 0)
    img = np.nan_to_num(img, nan=0.0) / 10000.0
    img = np.clip(img, 0.0, 1.0)

    kept = 0
    for i in range(0, h - SIZE + 1, SIZE):
        for j in range(0, w - SIZE + 1, SIZE):
            if invalid[i:i + SIZE, j:j + SIZE].mean() > MAX_INVALID:
                continue
            name = f"{tag}_r{i // SIZE:03d}_c{j // SIZE:03d}.npy"   # position conservee pour le split par zone
            np.save(out / "images" / name, img[:, i:i + SIZE, j:j + SIZE])
            np.save(out / "masks" / name, mask[i:i + SIZE, j:j + SIZE])
            kept += 1
    print(f"{tag} : {kept} tuiles conservees")


if __name__ == "__main__":
    for year in (2020, 2024):
        tile_pair(f"data/raw/s2_{year}.tif", f"data/raw/forest_{year}.tif", "data/tiles", f"y{year}")
