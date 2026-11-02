"""Cree deux petits GeoTIFF de test (512 x 512, 2020 et 2024, meme zone) pour essayer l'API.
Lancer depuis la racine : python src/make_test_tiles.py -> data/test_2020.tif et data/test_2024.tif"""
import rasterio
from rasterio.windows import Window

win = Window(1000, 1000, 512, 512)
for year in (2020, 2024):
    with rasterio.open(f"data/raw/s2_{year}.tif") as src:
        profile = src.profile.copy()
        profile.update(width=512, height=512, transform=src.window_transform(win))
        data = src.read(window=win)
    with rasterio.open(f"data/test_{year}.tif", "w", **profile) as dst:
        dst.write(data)
    print("cree : data/test_%d.tif" % year)
