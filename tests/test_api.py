"""Tests de l'API.   Lancer depuis la racine du projet :  python -m pytest
Prerequis : pip install pytest httpx   (et models/unet_best.pth + models/norm.npz presents)."""
import numpy as np
from fastapi.testclient import TestClient
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from app.main import app


def make_tif(h=64, w=64, bands=4, seed=0):
    rng = np.random.default_rng(seed)
    data = (rng.random((bands, h, w)) * 3000).astype("float32")
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", height=h, width=w, count=bands, dtype="float32",
                      crs="EPSG:4326", transform=from_origin(0, 0, 0.0001, 0.0001)) as dst:
            dst.write(data)
        return mem.read()


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict_returns_percentage():
    with TestClient(app) as client:
        r = client.post("/predict", files={"file": ("t.tif", make_tif(), "image/tiff")})
        assert r.status_code == 200
        assert 0 <= r.json()["forest_percent"] <= 100


def test_predict_rejects_invalid_file():
    with TestClient(app) as client:
        r = client.post("/predict", files={"file": ("x.tif", b"pas un geotiff", "image/tiff")})
        assert r.status_code == 400


def test_predict_rejects_wrong_band_count():
    with TestClient(app) as client:
        r = client.post("/predict", files={"file": ("t.tif", make_tif(bands=3), "image/tiff")})
        assert r.status_code == 422


def test_change_and_shape_mismatch():
    with TestClient(app) as client:
        ok = client.post("/change", files={"image_2020": ("a.tif", make_tif(seed=1), "image/tiff"),
                                           "image_2024": ("b.tif", make_tif(seed=2), "image/tiff")})
        assert ok.status_code == 200 and ok.json()["loss_hectares"] >= 0
        bad = client.post("/change", files={"image_2020": ("a.tif", make_tif(64, 64), "image/tiff"),
                                            "image_2024": ("b.tif", make_tif(96, 96), "image/tiff")})
        assert bad.status_code == 422
