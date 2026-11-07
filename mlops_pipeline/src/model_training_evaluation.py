"""Entrenamiento y evaluación de modelos de riesgo de crédito.

Usa el preprocesador de `ft_engineering.py` dentro de un Pipeline de sklearn,
de modo que se ajusta solo con el conjunto de entrenamiento.

Cada modelo se evalúa con el umbral por defecto (0.5) y con un umbral de
probabilidad de impago ajustado por validación cruzada solo sobre train.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, average_precision_score,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ft_engineering import SEMILLA, TARGET, construir_preprocesador, generar_conjuntos

CLASE_IMPAGO = 0
CLASE_PAGO = 1

# Si la clase minoritaria está por debajo de este porcentaje, se compensa el desbalance
LIMITE_DESBALANCE = 0.2
UMBRAL_POR_DEFECTO = 0.5
N_FOLDS = 5


def crear_pipeline(preprocessor, estimator):
    """Pipeline sin entrenar: copia del preprocesador + copia del estimador."""
    return Pipeline([
        ("preprocesador", clone(preprocessor)),
        ("estimador", clone(estimator)),
    ])


def parametros_fit(sample_weight):
    return {} if sample_weight is None else {"estimador__sample_weight": sample_weight}


def build_model(preprocessor, estimator, X_train, y_train, sample_weight=None):
    """Arma el Pipeline preprocesador + estimador y lo entrena con train."""
    modelo = crear_pipeline(preprocessor, estimator)
    modelo.fit(X_train, y_train, **parametros_fit(sample_weight))
    return modelo


def predecir_impago_oof(preprocessor, estimator, X_train, y_train, sample_weight=None):
    """Probabilidad de impago out-of-fold (validación cruzada) para cada fila de train."""
    validacion = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)
    proba = cross_val_predict(crear_pipeline(preprocessor, estimator), X_train, y_train, cv=validacion,
                              method="predict_proba", params=parametros_fit(sample_weight))
    # predict_proba ordena las columnas por clase ascendente
    return proba[:, list(np.unique(y_train)).index(CLASE_IMPAGO)]


def umbral_optimo(y_true, proba_impago):
    """Umbral de probabilidad de impago que maximiza el F1 de impago."""
    precision, recall, umbrales = precision_recall_curve(np.asarray(y_true) == CLASE_IMPAGO, proba_impago)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return umbrales[np.argmax(f1[:-1])]


def optimizar_umbral(preprocessor, estimator, X_train, y_train, sample_weight=None):
    """Umbral de probabilidad de impago que maximiza el F1 de impago.

    Usa predicciones out-of-fold de validación cruzada sobre train, así que el
    conjunto de test no interviene en la elección.
    """
    proba_impago = predecir_impago_oof(preprocessor, estimator, X_train, y_train, sample_weight)
    return umbral_optimo(y_train, proba_impago)


def summarize_classification(y_true, y_pred, y_proba=None, pos_label=CLASE_PAGO):
    """Accuracy, precision, recall, F1, ROC-AUC y PR-AUC respecto de `pos_label`.

    `y_proba` es la probabilidad de la clase `pos_label`.
    """
    resumen = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=pos_label),
        "f1": f1_score(y_true, y_pred, pos_label=pos_label),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }
    if y_proba is not None:
        es_positiva = np.asarray(y_true) == pos_label
        resumen["roc_auc"] = roc_auc_score(es_positiva, y_proba)
        resumen["pr_auc"] = average_precision_score(es_positiva, y_proba)
    return resumen


def definir_modelos(y_train, balancear):
    """Devuelve {nombre: (estimador, sample_weight)} con la compensación del desbalance."""
    class_weight = "balanced" if balancear else None
    # scale_pos_weight pondera la clase 1; como la minoritaria es la 0, el peso es n(0) / n(1)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if balancear else 1
    pesos_gradient_boosting = compute_sample_weight("balanced", y_train) if balancear else None

    return {
        # El preprocesador no escala; la regresión logística lo necesita para converger
        "LogisticRegression": (
            Pipeline([
                ("escalador", StandardScaler()),
                ("clasificador", LogisticRegression(class_weight=class_weight, max_iter=2000)),
            ]),
            None,
        ),
        "RandomForest": (
            RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight=class_weight,
                                   n_jobs=-1, random_state=SEMILLA),
            None,
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=SEMILLA),
            pesos_gradient_boosting,
        ),
        "XGBoost": (
            XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                          scale_pos_weight=scale_pos_weight, eval_metric="logloss",
                          random_state=SEMILLA, n_jobs=-1),
            None,
        ),
    }


def graficar_comparacion(tabla, proba_impago_por_modelo, umbral_por_modelo, y_test):
    """Gráficos comparativos en test: métricas, curvas ROC y matrices de confusión.

    `tabla` trae solo las filas con umbral ajustado, en el orden de los modelos.
    """
    modelos = tabla.index.tolist()

    # 1. Métricas por modelo (umbral ajustado)
    metricas = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    eje = tabla[metricas].plot.bar(figsize=(11, 5), rot=0, width=0.8)
    eje.set_ylim(0, 1.05)
    eje.set_ylabel("Valor en test")
    eje.set_xlabel("")
    eje.set_title(f"Métricas por modelo, umbral ajustado (clase positiva: {TARGET} = {CLASE_PAGO})", pad=30)
    eje.legend(ncols=len(metricas), loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    for barras in eje.containers:
        eje.bar_label(barras, fmt="%.2f", fontsize=7)
    plt.tight_layout()

    # 2. Curvas ROC superpuestas (no dependen del umbral)
    _, eje = plt.subplots(figsize=(6.5, 6))
    for nombre in modelos:
        RocCurveDisplay.from_predictions(y_test, 1 - proba_impago_por_modelo[nombre], pos_label=CLASE_PAGO,
                                         name=nombre, ax=eje)
    eje.plot([0, 1], [0, 1], ls="--", color="gray", label="Azar (AUC = 0.50)")
    eje.set_title("Curvas ROC en test")
    eje.set_xlabel("Tasa de falsos positivos (impagos predichos como 'paga')")
    eje.set_ylabel("Tasa de verdaderos positivos (pagos predichos como 'paga')")
    eje.legend(loc="lower right")
    plt.tight_layout()

    # 3. Matrices de confusión (umbral ajustado)
    fig, ejes = plt.subplots(1, len(modelos), figsize=(4.2 * len(modelos), 4))
    for eje, nombre in zip(ejes, modelos):
        umbral = umbral_por_modelo[nombre]
        y_pred = np.where(proba_impago_por_modelo[nombre] >= umbral, CLASE_IMPAGO, CLASE_PAGO)
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, labels=[CLASE_PAGO, CLASE_IMPAGO], display_labels=["Paga (1)", "No paga (0)"],
            cmap="Blues", colorbar=False, ax=eje,
        )
        eje.set_title(f"{nombre}\n(umbral impago = {umbral:.3f})", fontsize=10)
        eje.set_xlabel("Predicción")
        eje.set_ylabel("Valor real")
    fig.suptitle("Matrices de confusión en test")
    plt.tight_layout()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generar_conjuntos()

    proporcion_clases = y_train.value_counts(normalize=True)
    balancear = proporcion_clases.min() < LIMITE_DESBALANCE
    print("Distribución de la variable objetivo en train:")
    print(proporcion_clases.mul(100).round(2).to_string())
    print(f"¿Desbalanceada (minoritaria < {LIMITE_DESBALANCE:.0%})?: {balancear}\n")

    preprocesador = construir_preprocesador()
    resultados = {}
    proba_impago_por_modelo, umbral_por_modelo = {}, {}
    for nombre, (estimador, sample_weight) in definir_modelos(y_train, balancear).items():
        modelo = build_model(preprocesador, estimador, X_train, y_train, sample_weight)
        umbral_ajustado = optimizar_umbral(preprocesador, estimador, X_train, y_train, sample_weight)
        print(f"Entrenado: {nombre} (umbral ajustado por CV: {umbral_ajustado:.3f})")

        indice_impago = list(modelo.classes_).index(CLASE_IMPAGO)
        proba_impago = modelo.predict_proba(X_test)[:, indice_impago]
        proba_impago_por_modelo[nombre] = proba_impago
        umbral_por_modelo[nombre] = umbral_ajustado
        for etiqueta_umbral, umbral in [("0.5", UMBRAL_POR_DEFECTO), ("ajustado", umbral_ajustado)]:
            y_pred = np.where(proba_impago >= umbral, CLASE_IMPAGO, CLASE_PAGO)
            resultados[(nombre, etiqueta_umbral)] = {
                **summarize_classification(y_test, y_pred, 1 - proba_impago),
                "umbral_impago": umbral,
            }

    tabla = pd.DataFrame(resultados).T.rename_axis(["modelo", "umbral"])
    # Modelos ordenados por F1 con el umbral ajustado, cada uno con sus dos umbrales juntos
    orden = tabla.xs("ajustado", level="umbral")["f1"].sort_values(ascending=False).index
    tabla = tabla.reindex(orden, level="modelo")

    print(f"\nMétricas en test (clase positiva = pago a tiempo, {TARGET} = {CLASE_PAGO}):")
    print(tabla.round(3).to_string())

    tabla_ajustado = tabla.xs("ajustado", level="umbral")
    mejor = tabla_ajustado.iloc[0]
    print(f"\nMejor modelo: {tabla_ajustado.index[0]} con umbral de impago {mejor['umbral_impago']:.3f} "
          f"(mayor F1 en test con umbral ajustado: F1={mejor['f1']:.3f}, ROC-AUC={mejor['roc_auc']:.3f})")

    graficar_comparacion(tabla_ajustado, proba_impago_por_modelo, umbral_por_modelo, y_test)
    plt.show()
