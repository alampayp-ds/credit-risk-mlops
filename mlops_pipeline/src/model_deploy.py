"""Despliegue del modelo de riesgo crediticio como API con FastAPI.

Carga el mejor modelo guardado por `model_training_evaluation.py` (archivo
.joblib con el Pipeline de limpieza + preprocesamiento + XGBoost y su umbral)
y expone el endpoint `/predict` para predicciones por lotes en JSON o CSV.

Ejecutar desde la raíz del repo:
    python mlops_pipeline/src/model_deploy.py
Documentación interactiva: http://localhost:8000/docs
"""

import io
import json
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from ft_engineering import COLUMNAS_NOMINALES, COLUMNAS_NUMERICAS, COLUMNAS_ORDINALES
from model_training_evaluation import RUTA_MODELO

HOST = "0.0.0.0"
PUERTO = 8000
DECISIONES = {0: "no paga", 1: "paga"}

# Registro real de la base, usado como ejemplo en /docs
EJEMPLO_REGISTRO = {
    "tipo_credito": 4, "capital_prestado": 3692160.0, "plazo_meses": 10, "edad_cliente": 42,
    "tipo_laboral": "Independiente", "salario_cliente": 8000000, "total_otros_prestamos": 2500000,
    "cuota_pactada": 341296, "puntaje_datacredito": 695.0, "cant_creditosvigentes": 10, "huella_consulta": 5,
    "saldo_mora": 0.0, "saldo_total": 51258.0, "saldo_principal": 51258.0, "saldo_mora_codeudor": 0.0,
    "creditos_sectorFinanciero": 5, "creditos_sectorCooperativo": 0, "creditos_sectorReal": 0,
    "promedio_ingresos_datacredito": 908526.0, "tendencia_ingresos": "Estable",
}

# Todas las columnas son obligatorias pero aceptan null: el modelo imputa los faltantes.
# Las columnas adicionales (puntaje, fecha_prestamo, Pago_atiempo...) se ignoran.
Registro = create_model(
    "Registro",
    __config__=ConfigDict(extra="ignore", json_schema_extra={"example": EJEMPLO_REGISTRO}),
    **{columna: (float | None, ...) for columna in COLUMNAS_NUMERICAS},
    tipo_credito=(int | str | None, ...),  # código de tipo de crédito
    tipo_laboral=(str | None, ...),
    tendencia_ingresos=(str | None, ...),
)


class SolicitudPrediccion(BaseModel):
    registros: list[Registro] = Field(min_length=1)


class Prediccion(BaseModel):
    indice: int
    proba_impago: float
    prediccion: int = Field(description="Pago_atiempo predicho: 1 = paga a tiempo, 0 = no paga")
    decision: str


class RespuestaPrediccion(BaseModel):
    modelo: str
    version: str
    umbral_impago: float
    n_registros: int
    predicciones: list[Prediccion]


def cargar_modelo(ruta=RUTA_MODELO):
    return joblib.load(ruta)


def predecir(artefacto, registros):
    """Probabilidad de impago y decisión para cada registro validado."""
    columnas = artefacto["columnas_entrada"]
    datos = pd.DataFrame([registro.model_dump() for registro in registros], columns=columnas)
    datos[COLUMNAS_NUMERICAS] = datos[COLUMNAS_NUMERICAS].astype(float)  # None -> NaN

    modelo = artefacto["modelo"]
    # El Pipeline se ajustó con todas las columnas del dataset; las que descarta
    # (puntaje, fecha_prestamo) se completan vacías y no influyen en la predicción
    datos = datos.reindex(columns=modelo.feature_names_in_)
    indice_impago = list(modelo.classes_).index(artefacto["clase_impago"])
    proba_impago = modelo.predict_proba(datos)[:, indice_impago]
    prediccion = np.where(proba_impago >= artefacto["umbral_impago"], artefacto["clase_impago"],
                          artefacto["clase_pago"])
    return [
        Prediccion(indice=i, proba_impago=round(float(p), 6), prediccion=int(c), decision=DECISIONES[int(c)])
        for i, (p, c) in enumerate(zip(proba_impago, prediccion))
    ]


def validar_registros(registros):
    """Valida la lista de registros; los errores se devuelven como 422 indicando fila y campo."""
    try:
        return SolicitudPrediccion(registros=registros).registros
    except ValidationError as error:
        raise RequestValidationError(error.errors(include_url=False)) from error


def leer_csv(contenido):
    try:
        datos = pd.read_csv(io.BytesIO(contenido))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el CSV: {error}") from error
    faltantes = sorted(set(COLUMNAS_NUMERICAS + COLUMNAS_NOMINALES + COLUMNAS_ORDINALES) - set(datos.columns))
    if faltantes:
        raise HTTPException(status_code=422, detail=f"Faltan columnas en el CSV: {', '.join(faltantes)}")
    # to_json convierte NaN en null y los tipos de numpy en tipos nativos
    return json.loads(datos.to_json(orient="records"))


@asynccontextmanager
async def ciclo_de_vida(app):
    app.state.artefacto = cargar_modelo()
    yield


app = FastAPI(
    title="API de riesgo crediticio",
    description="Predice si un cliente pagará a tiempo su crédito (`Pago_atiempo`). "
                "Acepta lotes de registros en JSON o CSV.",
    version="1.1.2",
    lifespan=ciclo_de_vida,
)


@app.get("/", summary="Información del modelo")
def informacion():
    artefacto = app.state.artefacto
    return {
        "modelo": artefacto["nombre"],
        "version": artefacto["version"],
        "fecha_entrenamiento": artefacto["fecha_entrenamiento"],
        "umbral_impago": artefacto["umbral_impago"],
        "metricas_test": artefacto["metricas_test"],
        "columnas_entrada": artefacto["columnas_entrada"],
    }


@app.get("/health", summary="Estado del servicio")
def salud():
    return {"estado": "ok", "modelo_cargado": app.state.artefacto is not None}


@app.post(
    "/predict",
    summary="Predicción por lotes (JSON o CSV)",
    response_model=RespuestaPrediccion,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {"schema": {
                    "type": "object",
                    "required": ["registros"],
                    "properties": {"registros": {"type": "array", "items": Registro.model_json_schema()}},
                    "example": {"registros": [EJEMPLO_REGISTRO]},
                }},
                "multipart/form-data": {"schema": {
                    "type": "object",
                    "required": ["archivo"],
                    "properties": {"archivo": {"type": "string", "format": "binary"}},
                }},
                "text/csv": {"schema": {"type": "string"}},
            },
        }
    },
)
async def predict(request: Request):
    """Predicción por lotes.

    - **JSON**: `{"registros": [{...}, {...}]}`
    - **CSV**: archivo en el campo `archivo` (multipart/form-data) o el CSV en el cuerpo (text/csv).
      Debe tener una fila por registro y las columnas de entrada como encabezado.
    """
    tipo_contenido = request.headers.get("content-type", "")
    if tipo_contenido.startswith("application/json"):
        try:
            cuerpo = await request.json()
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=400, detail="El cuerpo no es un JSON válido") from error
        if not isinstance(cuerpo, dict) or "registros" not in cuerpo:
            raise HTTPException(status_code=422, detail='El JSON debe tener la forma {"registros": [...]}')
        registros = cuerpo["registros"]
    elif tipo_contenido.startswith("multipart/form-data"):
        formulario = await request.form()
        archivo = formulario.get("archivo")
        if archivo is None or isinstance(archivo, str):
            raise HTTPException(status_code=422, detail="Envía el CSV en el campo 'archivo'")
        registros = leer_csv(await archivo.read())
    elif tipo_contenido.startswith("text/csv"):
        registros = leer_csv(await request.body())
    else:
        raise HTTPException(status_code=415, detail="Usa application/json, multipart/form-data o text/csv")

    artefacto = app.state.artefacto
    predicciones = predecir(artefacto, validar_registros(registros))
    return RespuestaPrediccion(
        modelo=artefacto["nombre"],
        version=artefacto["version"],
        umbral_impago=artefacto["umbral_impago"],
        n_registros=len(predicciones),
        predicciones=predicciones,
    )


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PUERTO)
