FROM python:3.12-slim

WORKDIR /code

# PyTorch version CPU (beaucoup plus legere que la version GPU)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app ./app
COPY models ./models
COPY docs/map.html ./docs/map.html

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
