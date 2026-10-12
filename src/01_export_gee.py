"""
Semaine 1 - Export des images Sentinel-2 et des masques de foret (Hansen)
depuis Google Earth Engine vers Google Drive.

Prerequis :
    pip install earthengine-api
    1) Creer un projet Google Cloud (gratuit) et activer l'API Earth Engine
    2) Enregistrer le projet sur https://earthengine.google.com (usage non commercial)
    3) Premiere execution : ee.Authenticate() ouvre le navigateur pour te connecter
"""
import ee

PROJECT = "deforestation-sentinel2"
# ee.Authenticate()                 # deja fait une fois avec : earthengine authenticate
ee.Initialize(project=PROJECT)

# --- Zone d'etude : region d'Ariquemes (Rondonia, Bresil), environ 33 km x 33 km ---
# Format : [ouest, sud, est, nord]. Ajuste-la si tu veux une autre zone.
ROI = ee.Geometry.Rectangle([-63.2, -10.1, -62.9, -9.8])

# Hansen v1.13 (version 2025) couvre les pertes de foret jusqu'en 2025.
# L'ancienne version 2023_v1_11 est obsolete.
YEAR_BEFORE, YEAR_AFTER = 2020, 2024
HANSEN = ee.Image("UMD/hansen/global_forest_change_2025_v1_13")


def sentinel2_composite(year):
    """Composite median de la saison seche (juin-septembre), nuages masques."""
    cloud_score = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(ROI)
        .filterDate(f"{year}-06-01", f"{year}-09-30")
        .linkCollection(cloud_score, ["cs_cdf"])
        .map(lambda img: img.updateMask(img.select("cs_cdf").gte(0.6)))
    )
    # B2 = bleu, B3 = vert, B4 = rouge, B8 = proche infrarouge (10 m de resolution)
    return s2.select(["B2", "B3", "B4", "B8"]).median().clip(ROI)


def forest_mask(year):
    """1 = foret, 0 = non-foret, pour une annee donnee (couvert >= 30 % en 2000, sans perte avant)."""
    tree = HANSEN.select("treecover2000").gte(30)
    lossyear = HANSEN.select("lossyear")           # 0 = pas de perte, 1..25 = 2001..2025
    lost_before = lossyear.gt(0).And(lossyear.lte(year - 2000))
    return tree.And(lost_before.Not()).rename("forest").clip(ROI).toByte()


def export(image, name):
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=name,
        folder="deforestation",
        fileNamePrefix=name,
        region=ROI,
        scale=10,
        crs="EPSG:4326",
        maxPixels=1e10,
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"Export lance : {name}  (suivi : https://code.earthengine.google.com/tasks)")


for year in (YEAR_BEFORE, YEAR_AFTER):
    export(sentinel2_composite(year), f"s2_{year}")
    export(forest_mask(year), f"forest_{year}")
