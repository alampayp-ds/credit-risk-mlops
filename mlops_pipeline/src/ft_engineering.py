"""Ingeniería de características para el modelo de riesgo de crédito.

Se construye el pipeline de preprocesamiento a partir de las decisiones del EDA
(`comprension_eda.ipynb`):

1. Limpieza: valores fuera de rango plausible pasan a NaN, categorías inválidas
   de `tendencia_ingresos` pasan a NaN y los códigos raros de `tipo_credito` se
   agrupan en "Otros".
2. ColumnTransformer con tres grupos: numéricas, categóricas nominales y
   categóricas ordinales.

Se excluyen `puntaje` (fuga de información) y `fecha_prestamo` (solo para
ordenar o separar en el tiempo).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder

TARGET = "Pago_atiempo"
SEMILLA = 42

COLUMNAS_NUMERICAS = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "cuota_pactada", "puntaje_datacredito",
    "cant_creditosvigentes", "huella_consulta", "saldo_mora", "saldo_total",
    "saldo_principal", "saldo_mora_codeudor", "creditos_sectorFinanciero",
    "creditos_sectorCooperativo", "creditos_sectorReal", "promedio_ingresos_datacredito",
]
COLUMNAS_NOMINALES = ["tipo_credito", "tipo_laboral"]
COLUMNAS_ORDINALES = ["tendencia_ingresos"]

# Orden de menor a mayor para cada variable ordinal
ORDEN_TENDENCIA_INGRESOS = ["Decreciente", "Estable", "Creciente"]

# Rangos plausibles (inclusivos); fuera de ellos el valor pasa a NaN.
# None significa sin límite por ese lado.
LIMITES_PLAUSIBLES = {
    "edad_cliente": (18, 100),           # 121-123 años = clientes sin historial
    "puntaje_datacredito": (150, 950),   # escala de Datacrédito
    "salario_cliente": (0, None),
    "total_otros_prestamos": (0, None),
    "plazo_meses": (1, 60),
}

# Códigos de tipo_credito con muy pocos registros (7 y 68: 3 casos en total)
CODIGOS_CREDITO_RAROS = ["7", "68"]


def limpiar_datos(df):
    """Aplica las reglas de limpieza del EDA sin modificar el DataFrame original."""
    df = df.copy()

    for columna, (minimo, maximo) in LIMITES_PLAUSIBLES.items():
        fuera_de_rango = pd.Series(False, index=df.index)
        if minimo is not None:
            fuera_de_rango |= df[columna] < minimo
        if maximo is not None:
            fuera_de_rango |= df[columna] > maximo
        df[columna] = df[columna].mask(fuera_de_rango)

    tendencia_valida = df["tendencia_ingresos"].isin(ORDEN_TENDENCIA_INGRESOS)
    df["tendencia_ingresos"] = df["tendencia_ingresos"].astype(object).where(tendencia_valida, np.nan)

    # tipo_credito es un código, no una magnitud
    tipo_credito = df["tipo_credito"].astype(str)
    df["tipo_credito"] = tipo_credito.where(~tipo_credito.isin(CODIGOS_CREDITO_RAROS), "Otros").astype(object)
    df["tipo_laboral"] = df["tipo_laboral"].astype(object)

    return df


def construir_preprocesador():
    """Pipeline completo: limpieza + ColumnTransformer de tres grupos."""
    pipeline_numerico = Pipeline([
        # add_indicator crea las flags de faltante (incluida la de "sin historial")
        ("imputador", SimpleImputer(strategy="median", add_indicator=True)),
    ])
    pipeline_nominal = Pipeline([
        ("imputador", SimpleImputer(strategy="constant", fill_value="Sin dato")),
        ("codificador", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pipeline_ordinal = Pipeline([
        # El faltante de tendencia_ingresos ya queda marcado por el indicador
        # de promedio_ingresos_datacredito, por eso se imputa con la moda
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("codificador", OrdinalEncoder(categories=[ORDEN_TENDENCIA_INGRESOS])),
    ])

    transformador_columnas = ColumnTransformer(
        [
            ("numericas", pipeline_numerico, COLUMNAS_NUMERICAS),
            ("nominales", pipeline_nominal, COLUMNAS_NOMINALES),
            ("ordinales", pipeline_ordinal, COLUMNAS_ORDINALES),
        ],
        remainder="drop",  # descarta puntaje y fecha_prestamo
    )

    return Pipeline([
        ("limpieza", FunctionTransformer(limpiar_datos, feature_names_out="one-to-one")),
        ("preprocesamiento", transformador_columnas),
    ])


def cargar_datos():
    # Raíz del repo: primera carpeta (actual o superior) que contiene requirements.txt
    inicio = Path(__file__).resolve().parent
    project_root = next(p for p in [inicio, *inicio.parents] if (p / "requirements.txt").exists())
    return pd.read_csv(project_root / "Base_de_datos.csv")


def generar_conjuntos(test_size=0.2):
    """Carga el dataset y devuelve X_train, X_test, y_train, y_test (split estratificado).

    Los conjuntos se devuelven sin transformar: el preprocesador se ajusta
    después, dentro del Pipeline de cada modelo, solo con train.
    """
    df = cargar_datos()
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEMILLA)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generar_conjuntos()

    preprocesador = construir_preprocesador()
    X_train_transformado = preprocesador.fit_transform(X_train)
    X_test_transformado = preprocesador.transform(X_test)
    nombres_columnas = preprocesador.get_feature_names_out()

    print(f"Datos originales:   {(len(X_train) + len(X_test), X_train.shape[1])}")
    print(f"Train original:     {X_train.shape} -> transformado: {X_train_transformado.shape}")
    print(f"Test original:      {X_test.shape} -> transformado: {X_test_transformado.shape}")
    print(f"Nulos en train transformado: {np.isnan(X_train_transformado).sum()}")
    print(f"Nulos en test transformado:  {np.isnan(X_test_transformado).sum()}")

    print(f"\nColumnas resultantes ({len(nombres_columnas)}):")
    for nombre in nombres_columnas:
        print(f"  {nombre}")
