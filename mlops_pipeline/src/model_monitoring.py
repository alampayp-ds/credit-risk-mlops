"""Monitoreo del modelo de riesgo de crédito y detección de data drift.

El trabajo arma una tabla con los datos y los pronósticos del modelo y la
muestrea por periodos mensuales según `fecha_prestamo`:

1. Referencia (histórico): préstamos hasta `FIN_REFERENCIA`. Con ellos se
   entrena el mejor modelo del avance 2 (XGBoost) y se ajusta su umbral. Sus
   pronósticos son out-of-fold, para no compararse contra predicciones hechas
   sobre los mismos datos de entrenamiento.
2. Producción: cada mes posterior se puntúa con ese modelo. Los meses con
   menos de `MIN_REGISTROS_PERIODO` préstamos se agrupan con los siguientes.

Para cada periodo y variable se calcula PSI, divergencia de Jensen-Shannon,
KS (numéricas) y chi-cuadrado (categóricas), se asigna un semáforo y se
generan alertas y recomendaciones. La app `app_monitoreo.py` visualiza estos
resultados.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp

from ft_engineering import (COLUMNAS_NOMINALES, COLUMNAS_NUMERICAS, COLUMNAS_ORDINALES, TARGET, cargar_datos,
                            construir_preprocesador, limpiar_datos)
from model_training_evaluation import (CLASE_IMPAGO, CLASE_PAGO, LIMITE_DESBALANCE, build_model, definir_modelos,
                                       predecir_impago_oof, summarize_classification, umbral_optimo)

MODELO_MONITOREADO = "XGBoost"
FIN_REFERENCIA = pd.Period("2025-03", freq="M")
MIN_REGISTROS_PERIODO = 100
ETIQUETA_REFERENCIA = "Referencia"

VARIABLES_NUMERICAS = COLUMNAS_NUMERICAS + ["proba_impago"]
VARIABLES_CATEGORICAS = COLUMNAS_NOMINALES + COLUMNAS_ORDINALES
N_BINS_PSI = 10
EPSILON = 1e-4  # evita log(0) en bins vacíos

# Semáforo: (amarillo desde, rojo desde)
UMBRALES_PSI = (0.10, 0.25)
UMBRALES_JS = (0.10, 0.20)
ALFA = 0.05  # KS y chi-cuadrado solo llegan a amarillo: con n grande casi todo es significativo

# Reglas de recomendación
MIN_VARIABLES_ROJAS_REENTRENAR = 3
CAIDA_MAXIMA_ROC_AUC = 0.05
MIN_IMPAGOS_ROC_AUC = 10  # con menos impagos el ROC-AUC del periodo es puro ruido
SALTO_PSI = 0.10            # aumento de PSI respecto del periodo anterior
PERIODOS_TENDENCIA = 3      # aumentos consecutivos de PSI

ESTADOS = ["verde", "amarillo", "rojo"]


# --- Tabla de datos con pronósticos ---------------------------------------

def asignar_periodos(fechas):
    """Etiqueta de periodo para cada fecha: referencia o mes (agrupando meses chicos)."""
    meses = fechas.dt.to_period("M")
    etiquetas = pd.Series(ETIQUETA_REFERENCIA, index=fechas.index)

    grupos, actual = [], []
    for mes, cantidad in meses[meses > FIN_REFERENCIA].value_counts().sort_index().items():
        actual.append((mes, cantidad))
        if sum(c for _, c in actual) >= MIN_REGISTROS_PERIODO:
            grupos.append(actual)
            actual = []
    if actual:  # el último grupo no llegó al mínimo: se une al anterior
        if grupos:
            grupos[-1] += actual
        else:
            grupos.append(actual)

    for grupo in grupos:
        inicio, fin = grupo[0][0], grupo[-1][0]
        etiqueta = str(inicio) if inicio == fin else f"{inicio} a {fin}"
        etiquetas[meses.isin([mes for mes, _ in grupo])] = etiqueta
    return etiquetas


def generar_tabla_pronosticos():
    """Datos + periodo + probabilidad de impago + predicción, y el umbral usado."""
    df = cargar_datos()
    df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"])
    df["periodo"] = asignar_periodos(df["fecha_prestamo"])
    df = df.sort_values("fecha_prestamo").reset_index(drop=True)

    es_referencia = df["periodo"] == ETIQUETA_REFERENCIA
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_ref, y_ref = X[es_referencia], y[es_referencia]

    balancear = y_ref.value_counts(normalize=True).min() < LIMITE_DESBALANCE
    estimador, sample_weight = definir_modelos(y_ref, balancear)[MODELO_MONITOREADO]
    preprocesador = construir_preprocesador()

    modelo = build_model(preprocesador, estimador, X_ref, y_ref, sample_weight)
    proba_ref = predecir_impago_oof(preprocesador, estimador, X_ref, y_ref, sample_weight)
    umbral = umbral_optimo(y_ref, proba_ref)

    indice_impago = list(modelo.classes_).index(CLASE_IMPAGO)
    df["proba_impago"] = np.nan
    df.loc[es_referencia, "proba_impago"] = proba_ref
    df.loc[~es_referencia, "proba_impago"] = modelo.predict_proba(X[~es_referencia])[:, indice_impago]
    df["prediccion"] = np.where(df["proba_impago"] >= umbral, CLASE_IMPAGO, CLASE_PAGO)
    return df, umbral


def orden_periodos(tabla):
    """Referencia primero y luego los periodos de producción en orden cronológico."""
    produccion = tabla.loc[tabla["periodo"] != ETIQUETA_REFERENCIA].groupby("periodo")["fecha_prestamo"].min()
    return [ETIQUETA_REFERENCIA] + produccion.sort_values().index.tolist()


# --- Métricas de drift ------------------------------------------------------

def distribuciones(referencia, actual, categorica):
    """Proporciones por bin de referencia y actual. Los faltantes forman su propio bin."""
    if categorica:
        referencia = referencia.astype(object).fillna("Sin dato")
        actual = actual.astype(object).fillna("Sin dato")
        categorias = sorted(set(referencia) | set(actual), key=str)
        conteo_ref = referencia.value_counts().reindex(categorias, fill_value=0).to_numpy()
        conteo_act = actual.value_counts().reindex(categorias, fill_value=0).to_numpy()
    else:
        bordes = np.unique(np.nanquantile(referencia, np.linspace(0, 1, N_BINS_PSI + 1)))
        bordes = np.concatenate([[-np.inf], bordes[1:-1], [np.inf]])

        def contar(valores):
            return np.append(np.histogram(valores.dropna(), bins=bordes)[0], valores.isna().sum())

        conteo_ref, conteo_act = contar(referencia), contar(actual)
    return conteo_ref, conteo_act


def psi(conteo_ref, conteo_act):
    p = np.clip(conteo_ref / conteo_ref.sum(), EPSILON, None)
    q = np.clip(conteo_act / conteo_act.sum(), EPSILON, None)
    return float(np.sum((q - p) * np.log(q / p)))


def divergencia_js(conteo_ref, conteo_act):
    # scipy devuelve la distancia (raíz de la divergencia)
    return float(jensenshannon(conteo_ref, conteo_act, base=2) ** 2)


def semaforo(valor, umbrales):
    amarillo, rojo = umbrales
    return "rojo" if valor > rojo else "amarillo" if valor > amarillo else "verde"


def peor_estado(*estados):
    return max(estados, key=ESTADOS.index)


def comparar_variable(referencia, actual, categorica):
    """Métricas de drift de una variable entre referencia y un periodo."""
    conteo_ref, conteo_act = distribuciones(referencia, actual, categorica)
    resultado = {
        "tipo": "categórica" if categorica else "numérica",
        "psi": psi(conteo_ref, conteo_act),
        "js": divergencia_js(conteo_ref, conteo_act),
        "ks_estadistico": np.nan,
        "p_valor": np.nan,
    }
    if categorica:
        presentes = (conteo_ref + conteo_act) > 0
        resultado["p_valor"] = chi2_contingency(np.vstack([conteo_ref, conteo_act])[:, presentes])[1]
    elif referencia.notna().any() and actual.notna().any():
        prueba = ks_2samp(referencia.dropna(), actual.dropna())
        resultado["ks_estadistico"], resultado["p_valor"] = prueba.statistic, prueba.pvalue

    resultado["estado"] = peor_estado(
        semaforo(resultado["psi"], UMBRALES_PSI),
        semaforo(resultado["js"], UMBRALES_JS),
        "amarillo" if resultado["p_valor"] < ALFA else "verde",
    )
    return resultado


def calcular_drift(tabla):
    """Una fila por (periodo, variable) con PSI, JS, KS/chi-cuadrado, semáforo y tendencia."""
    datos = limpiar_datos(tabla)
    periodos = orden_periodos(tabla)[1:]
    referencia = datos[datos["periodo"] == ETIQUETA_REFERENCIA]

    filas = []
    for orden, periodo in enumerate(periodos):
        actual = datos[datos["periodo"] == periodo]
        for variable in VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS:
            filas.append({
                "periodo": periodo,
                "orden_periodo": orden,
                "variable": variable,
                "n_actual": len(actual),
                **comparar_variable(referencia[variable], actual[variable], variable in VARIABLES_CATEGORICAS),
            })

    drift = pd.DataFrame(filas).sort_values(["variable", "orden_periodo"])
    cambio = drift.groupby("variable")["psi"].diff()
    drift["cambio_psi"] = cambio
    drift["salto_abrupto"] = cambio > SALTO_PSI
    # Tendencia: PSI en aumento durante PERIODOS_TENDENCIA periodos seguidos
    aumentos = (cambio > 0).astype(int)
    drift["tendencia_creciente"] = (
        aumentos.groupby(drift["variable"]).rolling(PERIODOS_TENDENCIA).sum().reset_index(level=0, drop=True)
        == PERIODOS_TENDENCIA
    )
    return drift.sort_values(["orden_periodo", "psi"], ascending=[True, False]).reset_index(drop=True)


# --- Desempeño y alertas ---------------------------------------------------

def calcular_desempeno(tabla):
    """Métricas del modelo por periodo (la etiqueta real está disponible en esta base)."""
    filas = []
    for periodo in orden_periodos(tabla):
        datos = tabla[tabla["periodo"] == periodo]
        metricas = summarize_classification(datos[TARGET], datos["prediccion"], 1 - datos["proba_impago"])
        filas.append({
            "periodo": periodo,
            "n": len(datos),
            "n_impagos": int((datos[TARGET] == CLASE_IMPAGO).sum()),
            "tasa_impago_real": (datos[TARGET] == CLASE_IMPAGO).mean(),
            "tasa_impago_predicha": (datos["prediccion"] == CLASE_IMPAGO).mean(),
            "proba_impago_media": datos["proba_impago"].mean(),
            **metricas,
        })
    return pd.DataFrame(filas)


def generar_alertas(drift, desempeno):
    """Alertas por periodo con su nivel y recomendación."""
    roc_auc_ref = desempeno.set_index("periodo").loc[ETIQUETA_REFERENCIA, "roc_auc"]
    alertas = []
    for periodo, grupo in drift.groupby("periodo", sort=False):
        rojas = grupo.loc[grupo["estado"] == "rojo", "variable"].tolist()
        psi_proba = grupo.set_index("variable").loc["proba_impago", "psi"]
        roc_auc, n_impagos = desempeno.set_index("periodo").loc[periodo, ["roc_auc", "n_impagos"]]
        caida = roc_auc_ref - roc_auc

        motivos = []
        if len(rojas) >= MIN_VARIABLES_ROJAS_REENTRENAR:
            motivos.append(f"{len(rojas)} variables en rojo")
        if psi_proba > UMBRALES_PSI[1]:
            motivos.append(f"PSI de la probabilidad predicha = {psi_proba:.2f}")
        if caida > CAIDA_MAXIMA_ROC_AUC and n_impagos >= MIN_IMPAGOS_ROC_AUC:
            motivos.append(f"ROC-AUC cae {caida:.3f} frente a la referencia ({roc_auc:.3f} vs {roc_auc_ref:.3f})")
        if motivos:
            alertas.append({
                "periodo": periodo, "nivel": "crítico",
                "mensaje": "Drift que puede comprometer la precisión del modelo: " + "; ".join(motivos) + ".",
                "recomendacion": "Reentrenar el modelo con datos recientes y recalibrar el umbral.",
            })
        elif rojas:
            alertas.append({
                "periodo": periodo, "nivel": "advertencia",
                "mensaje": f"Drift alto en: {', '.join(rojas)}.",
                "recomendacion": "Revisar la calidad y el origen de estas variables antes de decidir un reentrenamiento.",
            })

        saltos = grupo.loc[grupo["salto_abrupto"], "variable"].tolist()
        if saltos:
            alertas.append({
                "periodo": periodo, "nivel": "advertencia",
                "mensaje": f"Cambio abrupto de PSI (> {SALTO_PSI}) respecto del periodo anterior en: {', '.join(saltos)}.",
                "recomendacion": "Investigar cambios recientes en la captura de datos o en la política de originación.",
            })
        tendencias = grupo.loc[grupo["tendencia_creciente"], "variable"].tolist()
        if tendencias:
            alertas.append({
                "periodo": periodo, "nivel": "informativo",
                "mensaje": f"PSI en aumento durante {PERIODOS_TENDENCIA} periodos seguidos en: {', '.join(tendencias)}.",
                "recomendacion": "Vigilar estas variables en los próximos periodos.",
            })
    return pd.DataFrame(alertas, columns=["periodo", "nivel", "mensaje", "recomendacion"])


def ejecutar_monitoreo():
    """Corre el trabajo completo y devuelve todos los resultados."""
    tabla, umbral = generar_tabla_pronosticos()
    drift = calcular_drift(tabla)
    desempeno = calcular_desempeno(tabla)
    return {
        "tabla": tabla,
        "umbral": umbral,
        "periodos": orden_periodos(tabla),
        "drift": drift,
        "desempeno": desempeno,
        "alertas": generar_alertas(drift, desempeno),
    }


if __name__ == "__main__":
    resultados = ejecutar_monitoreo()
    tabla, drift, desempeno, alertas = (resultados[k] for k in ["tabla", "drift", "desempeno", "alertas"])

    print(f"Modelo monitoreado: {MODELO_MONITOREADO} (umbral de impago = {resultados['umbral']:.3f})")
    print(f"Referencia: préstamos hasta {FIN_REFERENCIA} ({(tabla['periodo'] == ETIQUETA_REFERENCIA).sum()} registros)")

    print("\nDesempeño por periodo:")
    columnas = ["periodo", "n", "n_impagos", "tasa_impago_real", "tasa_impago_predicha", "proba_impago_media", "f1", "roc_auc"]
    print(desempeno[columnas].round(3).to_string(index=False))

    print("\nVariables por estado del semáforo:")
    conteo = pd.crosstab(drift["periodo"], drift["estado"]).reindex(index=resultados["periodos"][1:],
                                                                      columns=ESTADOS, fill_value=0)
    print(conteo.to_string())

    print("\nPSI por periodo (variables con algún periodo en amarillo o rojo):")
    psi_tabla = drift.pivot(index="variable", columns="periodo", values="psi")[resultados["periodos"][1:]]
    print(psi_tabla[psi_tabla.max(axis=1) > UMBRALES_PSI[0]].round(3).to_string())

    print("\nAlertas:")
    for alerta in alertas.itertuples():
        print(f"  [{alerta.nivel}] {alerta.periodo}: {alerta.mensaje}\n      -> {alerta.recomendacion}")
