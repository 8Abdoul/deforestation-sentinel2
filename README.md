# Surveillance de la déforestation par images satellites (Sentinel-2 + U-Net)

Pipeline de détection de la perte de forêt entre 2020 et 2024 dans la région d'Ariquemes (Rondônia, Brésil) : segmentation U-Net sur images Sentinel-2, détection de changement et carte interactive.

![Validation des masques](docs/diagnostic_masque.png)

## Résultats
| Étape | Résultat |
|---|---|
| Segmentation forêt / non-forêt (zone de test jamais vue) | IoU = 0,879 (validation : 0,913) |
| Perte prédite 2020-2024 | ~2 100 à 2 700 ha selon le seuil NDVI (Hansen : 2 619 ha) |
| Accord avec Hansen 2021-2024, pixel exact | précision 49 %, rappel 40 %, IoU 28 % (seuil NDVI 0,2) |
| Accord avec Hansen, tolérance de 30 m | précision 66 %, rappel 53 % |

## Méthode
1. **Données** : composites médians Sentinel-2 (bandes B2, B3, B4, B8, 10 m) de la saison sèche 2020 et 2024 via Google Earth Engine, nuages masqués (Cloud Score+). Zone d'environ 33 km x 33 km, découpée en tuiles de 256 x 256.
2. **Étiquettes** : ESA WorldCover 2020 (classe « arbres », 10 m).
3. **Modèle** : U-Net (encodeur ResNet34 pré-entraîné, PyTorch / segmentation-models-pytorch), perte Dice + BCE, entrées normalisées.
4. **Split par zone** (nord = entraînement, centre = validation, sud = test) pour éviter la fuite spatiale entre tuiles voisines.
5. **Changement** : perte = forêt prédite en 2020, non-forêt prédit en 2024 et baisse du NDVI, puis nettoyage morphologique (objets > 0,5 ha).
6. **Carte** : Folium, couches de perte prédite et de référence Hansen sur fond satellite.

## Validation des données (point important)
Le masque forestier Hansen, utilisé au départ, ne marquait que 2,4 % de forêt contre ~34 % de pixels très verts (NDVI > 0,6) : inutilisable. Il a été remplacé par WorldCover, qui recouvre 95 % des pixels très verts (diagnostic visuel et chiffré ci-dessus). Hansen est conservé comme référence indépendante pour les pertes 2021-2024.

## Limites
- Une seule région et deux dates : rien ne prouve la généralisation à d'autres zones ou années.
- Étiquettes WorldCover = estimation, pas une vérité terrain.
- Hansen est à 30 m et couvre l'année complète, alors que les composites s'arrêtent en septembre.
- Détection par différence de deux classifications : les erreurs des deux dates s'additionnent.
- Le seuil NDVI est réglé en comparant à Hansen : l'évaluation est donc un peu optimiste.

## Utilisation
```powershell
python -m venv .venv ; .venv\Scripts\Activate.ps1 ; pip install -r requirements.txt
python src\01_export_gee.py      # export Sentinel-2 (Google Earth Engine, usage non commercial)
python src\06_export_worldcover.py
python src\02_tiling.py
python src\04_train.py
python src\07_change_detection.py
python src\10_make_map.py        # carte : docs/map.html
```
## API et Docker

L'API (FastAPI) expose :
- `GET /health` : état du service
- `POST /predict` : pourcentage de forêt sur un GeoTIFF 4 bandes (B2, B3, B4, B8)
- `POST /predict/mask` : masque de forêt en PNG
- `POST /change` : perte de forêt entre deux dates
- `GET /map` : carte interactive

Tests : `python -m pytest` (5 tests). 

Conteneur :
docker build -t deforestation-api .
docker run -p 8000:8000 deforestation-api
# puis http://localhost:8000/docs

Les poids du modèle (`models/unet_best.pth`) ne sont pas versionnés : lancer `python src/04_train.py` pour les régénérer avant de construire l'image.

![Health](docs/api_health.png)
![Prédiction](docs/api_predict.png)

## À venir
 Piste d'amélioration : augmentation de données, canal NDVI, plusieurs régions.
