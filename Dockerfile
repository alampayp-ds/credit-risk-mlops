# Imagen de la API de riesgo crediticio (FastAPI + Uvicorn)
#   docker build -t riesgo-crediticio-api:1.1.2 .
#   docker run -p 8000:8000 riesgo-crediticio-api:1.1.2

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

# XGBoost necesita OpenMP (libgomp) en Linux
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias primero, para reutilizar la capa en caché si solo cambia el código
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código necesario para cargar el modelo y servir la API, y el modelo entrenado
COPY mlops_pipeline/src/ft_engineering.py \
     mlops_pipeline/src/model_training_evaluation.py \
     mlops_pipeline/src/model_deploy.py \
     mlops_pipeline/src/modelo_riesgo_crediticio.joblib \
     ./src/

RUN useradd --create-home api
USER api

WORKDIR /app/src
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
