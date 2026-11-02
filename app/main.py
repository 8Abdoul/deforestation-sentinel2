"""API FastAPI : segmentation de foret et detection de changement (U-Net sur Sentinel-2).
A placer dans  app/main.py   (avec un fichier vide app/__init__.py).
Lancer en local :  uvicorn app.main:app --reload     puis ouvrir http://127.0.0.1:8000/docs
Entrees : GeoTIFF a 4 bandes (B2, B3, B4, B8) en reflectance x 10000, comme les exports Earth Engine (pixels de 10 m)."""
import io
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from PIL import Image
from rasterio.io import MemoryFile

MODEL_PATH, NORM_PATH, MAP_PATH = Path("models/unet_best.pth"), Path("models/norm.npz"), Path("docs/map.html")
HA_PER_PIXEL = 0.01          # pixel Sentinel-2 de 10 m x 10 m
MAX_PIXELS = 4000 * 4000
STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    norm = np.load(NORM_PATH)
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=4, classes=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    STATE.update(model=model, mean=norm["mean"][:, None, None], std=norm["std"][:, None, None], thr=float(norm["thr"]))
    yield
    STATE.clear()


app = FastAPI(title="Surveillance de la deforestation", version="1.0",
              description="Segmentation foret / non-foret (U-Net) et detection de perte entre deux dates, sur Sentinel-2.",
              lifespan=lifespan)


def read_image(data: bytes) -> np.ndarray:
    """GeoTIFF 4 bandes -> tableau (4, H, W) en reflectance 0-1, recadre a un multiple de 32."""
    try:
        with MemoryFile(data) as mem, mem.open() as src:
            if src.count < 4:
                raise HTTPException(422, f"4 bandes attendues (B2, B3, B4, B8), {src.count} trouvee(s).")
            if src.width * src.height > MAX_PIXELS:
                raise HTTPException(413, "Image trop grande (maximum 4000 x 4000 pixels).")
            img = src.read([1, 2, 3, 4]).astype("float32")
    except rasterio.errors.RasterioIOError:
        raise HTTPException(400, "Fichier illisible : un GeoTIFF est attendu.")
    img = np.clip(np.nan_to_num(img) / 10000.0, 0.0, 1.0)
    h, w = (img.shape[1] // 32) * 32, (img.shape[2] // 32) * 32
    if h == 0 or w == 0:
        raise HTTPException(422, "Image trop petite (minimum 32 x 32 pixels).")
    return img[:, :h, :w]


def predict_forest(img: np.ndarray, block: int = 256) -> np.ndarray:
    x = (img - STATE["mean"]) / STATE["std"]
    out = np.zeros(img.shape[1:], bool)
    with torch.no_grad():
        for i in range(0, img.shape[1], block):
            for j in range(0, img.shape[2], block):
                tile = torch.from_numpy(x[:, i:i + block, j:j + block]).float().unsqueeze(0)
                out[i:i + block, j:j + block] = (torch.sigmoid(STATE["model"](tile))[0, 0] > STATE["thr"]).numpy()
    return out


def ndvi(img: np.ndarray) -> np.ndarray:
    return (img[3] - img[2]) / (img[3] + img[2] + 1e-6)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in STATE}


@app.post("/predict")
async def predict(file: UploadFile = File(..., description="GeoTIFF 4 bandes (B2, B3, B4, B8)")):
    """Pourcentage de foret predit sur l'image."""
    img = read_image(await file.read())
    mask = await run_in_threadpool(predict_forest, img)
    return {"height": int(mask.shape[0]), "width": int(mask.shape[1]),
            "forest_percent": round(100 * float(mask.mean()), 2)}


@app.post("/predict/mask")
async def predict_mask(file: UploadFile = File(...)):
    """Masque de foret predit, au format PNG (blanc = foret)."""
    img = read_image(await file.read())
    mask = await run_in_threadpool(predict_forest, img)
    buf = io.BytesIO()
    Image.fromarray((mask * 255).astype(np.uint8)).save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/change")
async def change(image_2020: UploadFile = File(...), image_2024: UploadFile = File(...), ndvi_drop: float = 0.2):
    """Perte de foret entre deux dates : foret avant, non-foret apres, et baisse du NDVI superieure au seuil."""
    a, b = read_image(await image_2020.read()), read_image(await image_2024.read())
    if a.shape != b.shape:
        raise HTTPException(422, f"Les deux images doivent avoir la meme taille ({a.shape[1:]} contre {b.shape[1:]}).")
    f20, f24 = await run_in_threadpool(predict_forest, a), await run_in_threadpool(predict_forest, b)
    loss = f20 & ~f24 & ((ndvi(a) - ndvi(b)) > ndvi_drop)
    return {"forest_percent_before": round(100 * float(f20.mean()), 2),
            "forest_percent_after": round(100 * float(f24.mean()), 2),
            "loss_hectares": round(float(loss.sum()) * HA_PER_PIXEL, 1),
            "loss_percent": round(100 * float(loss.mean()), 2),
            "ndvi_drop": ndvi_drop,
            "note": "Estimation brute, sans nettoyage morphologique (voir src/07_change_detection.py)."}


@app.get("/map")
def map_page():
    """Carte interactive generee par src/10_make_map.py."""
    if not MAP_PATH.exists():
        raise HTTPException(404, "Carte introuvable : lance python src/10_make_map.py")
    return FileResponse(MAP_PATH, media_type="text/html")
