"""Semaine 4 - Carte interactive (Folium) : fond satellite + pertes predites + pertes Hansen.
Prerequis : pip install folium   (apres avoir lance 07_change_detection.py qui cree data/loss_pred_2020_2024.npy)
Lancer :    python src\\10_make_map.py     ->  docs/map.html  (a ouvrir dans le navigateur)"""
from pathlib import Path

import folium
import numpy as np
import rasterio
from PIL import Image

loss = np.load("data/loss_pred_2020_2024.npy")                 # (H, W) booleen, grille des tuiles (origine en haut a gauche)
H, W = loss.shape
with rasterio.open("data/raw/s2_2020.tif") as src:
    t = src.transform
with rasterio.open("data/raw/hansen_loss_2021_2024.tif") as src:
    hansen = src.read(1).astype(bool)[:H, :W]

west, north = t.c, t.f
east, south = west + W * t.a, north + H * t.e                  # t.e est negatif
bounds = [[south, west], [north, east]]

Path("docs/overlays").mkdir(parents=True, exist_ok=True)


def save_overlay(mask, rgb, path):
    rgba = np.zeros(mask.shape + (4,), np.uint8)
    rgba[mask] = (*rgb, 200)
    Image.fromarray(rgba, "RGBA").save(path)


save_overlay(loss, (230, 30, 30), "docs/overlays/perte_predite.png")
save_overlay(hansen, (255, 165, 0), "docs/overlays/perte_hansen.png")

m = folium.Map(location=[-9.91, -63.04], zoom_start=12, tiles=None)
folium.TileLayer("OpenStreetMap", name="Carte").add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery", name="Satellite (Esri)").add_to(m)
folium.raster_layers.ImageOverlay("docs/overlays/perte_predite.png", bounds=bounds, opacity=0.85,
                                  name="Perte predite 2020-2024 (U-Net)").add_to(m)
folium.raster_layers.ImageOverlay("docs/overlays/perte_hansen.png", bounds=bounds, opacity=0.85,
                                  name="Pertes Hansen 2021-2024 (reference)", show=False).add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

info = f"""<div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999; background: white; padding: 10px 14px;
border: 1px solid #888; border-radius: 6px; font-size: 13px; max-width: 290px;">
<b>Deforestation - region d'Ariquemes (Bresil)</b><br>
Perte predite 2020-2024 : <b>{loss.sum() * 0.01:.0f} ha</b> ({100 * loss.mean():.2f} % de la zone)<br>
Reference Hansen 2021-2024 : {hansen.sum() * 0.01:.0f} ha<br>
<span style="color:#e61e1e">&#9632;</span> perte predite &nbsp; <span style="color:#ffa500">&#9632;</span> Hansen<br>
<small>Modele U-Net sur Sentinel-2 (10 m). Resultat indicatif.</small></div>"""
m.get_root().html.add_child(folium.Element(info))
m.save("docs/map.html")
print("Carte enregistree : docs/map.html")
