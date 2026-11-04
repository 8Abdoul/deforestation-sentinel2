FROM python:3.12-slim

WORKDIR /code

RUN pip install --upgrade pip

# PyTorch version CPU (lourd : delais longs, reessais et reprise en cas de coupure)
RUN pip install --no-cache-dir --default-timeout=300 --retries 10 --resume-retries 10 \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install --no-cache-dir --default-timeout=300 --retries 10 -r requirements-api.txt

# Bibliotheques systeme manquantes dans l'image slim (necessaires a rasterio et PyTorch)
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY app ./app
COPY models ./models
COPY docs/map.html ./docs/map.html

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
