"""Alternative de masque : ESA WorldCover 2020 (10 m) + pertes Hansen 2021-2024 pour verification.
Lancer :  python src\\06_export_worldcover.py
Fichiers produits dans Google Drive (dossier 'deforestation') :
    wc_forest_2020.tif      1 = arbres (classe 10 de WorldCover), 0 = autre
    hansen_loss_2021_2024.tif  1 = perte de couvert arbore detectee par Hansen entre 2021 et 2024
"""
import ee

ee.Initialize(project="deforestation-sentinel2")

ROI = ee.Geometry.Rectangle([-63.2, -10.1, -62.9, -9.8])     # meme zone que 01_export_gee.py

worldcover = ee.ImageCollection("ESA/WorldCover/v100").first()   # carte de 2020, bande 'Map'
forest_2020 = worldcover.select("Map").eq(10).rename("forest").clip(ROI).toByte()

hansen = ee.Image("UMD/hansen/global_forest_change_2025_v1_13")
lossyear = hansen.select("lossyear")
loss_21_24 = lossyear.gte(21).And(lossyear.lte(24)).rename("loss").clip(ROI).toByte()


def export(image, name):
    ee.batch.Export.image.toDrive(
        image=image, description=name, folder="deforestation", fileNamePrefix=name,
        region=ROI, scale=10, crs="EPSG:4326", maxPixels=1e10, fileFormat="GeoTIFF",
    ).start()
    print("Export lance :", name)


export(forest_2020, "wc_forest_2020")
export(loss_21_24, "hansen_loss_2021_2024")
