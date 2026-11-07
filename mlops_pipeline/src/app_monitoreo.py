"""Dashboard de monitoreo y data drift del modelo de riesgo de crédito.

Ejecutar desde la raíz del repo:
    streamlit run mlops_pipeline/src/app_monitoreo.py
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ft_engineering import TARGET, limpiar_datos
from model_monitoring import (ALFA, ESTADOS, ETIQUETA_REFERENCIA, FIN_REFERENCIA, MODELO_MONITOREADO, N_BINS_PSI,
                              MIN_IMPAGOS_ROC_AUC, UMBRALES_JS, UMBRALES_PSI, VARIABLES_CATEGORICAS, VARIABLES_NUMERICAS,
                              ejecutar_monitoreo)

ICONOS = {"verde": "🟢", "amarillo": "🟡", "rojo": "🔴"}
COLORES = {"verde": "#2e9e5b", "amarillo": "#e6a700", "rojo": "#d64545"}
NIVELES_ALERTA = {"crítico": st.error, "advertencia": st.warning, "informativo": st.info}

st.set_page_config(page_title="Monitoreo de drift", page_icon="📉", layout="wide")


@st.cache_data(show_spinner="Entrenando el modelo y calculando el drift...")
def cargar_resultados():
    return ejecutar_monitoreo()


resultados = cargar_resultados()
tabla, drift, desempeno, alertas = (resultados[k] for k in ["tabla", "drift", "desempeno", "alertas"])
periodos_produccion = resultados["periodos"][1:]
escala_periodos = alt.Scale(domain=periodos_produccion)

# --- Barra lateral ----------------------------------------------------------
st.sidebar.title("Filtros")
periodo = st.sidebar.selectbox("Periodo a analizar", periodos_produccion, index=len(periodos_produccion) - 1)
st.sidebar.markdown(
    f"""
**Modelo:** {MODELO_MONITOREADO}
**Umbral de impago:** {resultados['umbral']:.3f}
**Referencia:** préstamos hasta {FIN_REFERENCIA}

**Semáforo**
- PSI: 🟡 > {UMBRALES_PSI[0]} · 🔴 > {UMBRALES_PSI[1]}
- Jensen-Shannon: 🟡 > {UMBRALES_JS[0]} · 🔴 > {UMBRALES_JS[1]}
- KS / chi-cuadrado: 🟡 si p < {ALFA}
"""
)

st.title("📉 Monitoreo del modelo de riesgo de crédito")
st.caption(f"Distribución histórica (referencia) frente a cada periodo de producción. Periodo seleccionado: **{periodo}**")

drift_periodo = drift[drift["periodo"] == periodo]
fila_desempeno = desempeno.set_index("periodo").loc[periodo]
fila_referencia = desempeno.set_index("periodo").loc[ETIQUETA_REFERENCIA]
conteo_estados = drift_periodo["estado"].value_counts()

tab_resumen, tab_metricas, tab_distribuciones, tab_temporal, tab_datos = st.tabs(
    ["Resumen y alertas", "Métricas de drift", "Distribuciones", "Análisis temporal", "Datos y pronósticos"]
)

# --- Resumen y alertas ---------------------------------------------------------
with tab_resumen:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Registros del periodo", f"{int(fila_desempeno['n']):,}")
    c2.metric("🔴 Variables en rojo", int(conteo_estados.get("rojo", 0)))
    c3.metric("🟡 Variables en amarillo", int(conteo_estados.get("amarillo", 0)))
    c4.metric("ROC-AUC", f"{fila_desempeno['roc_auc']:.3f}",
              f"{fila_desempeno['roc_auc'] - fila_referencia['roc_auc']:+.3f} vs referencia")
    psi_proba = drift_periodo.set_index("variable").loc["proba_impago", "psi"]
    c5.metric("PSI de la probabilidad predicha", f"{psi_proba:.3f}")

    st.subheader(f"Alertas y recomendaciones · {periodo}")
    alertas_periodo = alertas[alertas["periodo"] == periodo]
    if alertas_periodo.empty:
        st.success("Sin desviaciones significativas en este periodo. No se requiere acción.")
    for alerta in alertas_periodo.itertuples():
        NIVELES_ALERTA[alerta.nivel](f"**{alerta.nivel.capitalize()}:** {alerta.mensaje}\n\n➡️ {alerta.recomendacion}")
    if fila_desempeno["n_impagos"] < MIN_IMPAGOS_ROC_AUC:
        st.caption(f"Este periodo tiene solo {int(fila_desempeno['n_impagos'])} impagos: "
                   "su ROC-AUC es poco confiable y no dispara alertas.")

    st.subheader("Historial de alertas")
    resumen_alertas = pd.crosstab(alertas["periodo"], alertas["nivel"]).reindex(periodos_produccion, fill_value=0)
    st.dataframe(resumen_alertas, width="stretch")

# --- Métricas de drift ---------------------------------------------------------
with tab_metricas:
    st.subheader(f"Métricas por variable · {periodo}")
    vista = drift_periodo.assign(semaforo=drift_periodo["estado"].map(ICONOS) + " " + drift_periodo["estado"])
    st.dataframe(
        vista[["semaforo", "variable", "tipo", "psi", "js", "ks_estadistico", "p_valor", "cambio_psi"]],
        hide_index=True,
        width="stretch",
        column_config={
            "semaforo": "Estado",
            "variable": "Variable",
            "tipo": "Tipo",
            "psi": st.column_config.ProgressColumn("PSI (riesgo)", format="%.3f", min_value=0,
                                                   max_value=max(0.5, vista["psi"].max())),
            "js": st.column_config.ProgressColumn("Jensen-Shannon", format="%.3f", min_value=0,
                                                  max_value=max(0.3, vista["js"].max())),
            "ks_estadistico": st.column_config.NumberColumn("KS", format="%.3f"),
            "p_valor": st.column_config.NumberColumn("p-valor (KS / χ²)", format="%.4f"),
            "cambio_psi": st.column_config.NumberColumn("Δ PSI vs periodo anterior", format="%+.3f"),
        },
    )

    barras = alt.Chart(vista).mark_bar().encode(
        x=alt.X("psi:Q", title="PSI"),
        y=alt.Y("variable:N", sort="-x", title=None),
        color=alt.Color("estado:N", scale=alt.Scale(domain=ESTADOS, range=[COLORES[e] for e in ESTADOS]),
                        title="Estado"),
        tooltip=["variable", alt.Tooltip("psi:Q", format=".3f"), "estado"],
    )
    reglas = alt.Chart(pd.DataFrame({"psi": UMBRALES_PSI})).mark_rule(strokeDash=[4, 4]).encode(x="psi:Q")
    st.altair_chart((barras + reglas).properties(height=520, title="Barras de riesgo: PSI por variable"),
                    width="stretch")

# --- Distribuciones ------------------------------------------------------------
with tab_distribuciones:
    variables = drift_periodo["variable"].tolist()  # ordenadas de mayor a menor PSI
    variable = st.selectbox("Variable", variables, format_func=lambda v: f"{ICONOS[drift_periodo.set_index('variable').loc[v, 'estado']]} {v}")

    datos = limpiar_datos(tabla[tabla["periodo"].isin([ETIQUETA_REFERENCIA, periodo])])
    datos["grupo"] = np.where(datos["periodo"] == ETIQUETA_REFERENCIA, "Histórico (referencia)", f"Actual ({periodo})")

    if variable in VARIABLES_CATEGORICAS:
        valores = datos[variable].astype(object).fillna("Sin dato").astype(str)
        proporciones = (valores.groupby(datos["grupo"]).value_counts(normalize=True)
                        .rename("proporcion").reset_index().rename(columns={variable: "categoria"}))
        grafico = alt.Chart(proporciones).mark_bar().encode(
            x=alt.X("categoria:N", title=variable),
            xOffset="grupo:N",
            y=alt.Y("proporcion:Q", title="Proporción", axis=alt.Axis(format="%")),
            color=alt.Color("grupo:N", title=None),
            tooltip=["grupo", "categoria", alt.Tooltip("proporcion:Q", format=".1%")],
        )
    else:
        # Bins por cuantiles de la referencia, los mismos que usa el PSI
        referencia = datos.loc[datos["periodo"] == ETIQUETA_REFERENCIA, variable]
        bordes = np.unique(np.nanquantile(referencia, np.linspace(0, 1, N_BINS_PSI + 1)))
        etiquetas = [f"{i + 1:02d}. {bordes[i]:,.3g} – {bordes[i + 1]:,.3g}" for i in range(len(bordes) - 1)]
        bins = pd.cut(datos[variable].clip(bordes[0], bordes[-1]), bordes, labels=etiquetas, include_lowest=True)
        bins = bins.cat.add_categories("Sin dato").fillna("Sin dato").astype(str)
        proporciones = (bins.groupby(datos["grupo"]).value_counts(normalize=True)
                        .rename("proporcion").reset_index().rename(columns={variable: "bin"}))
        grafico = alt.Chart(proporciones).mark_bar().encode(
            x=alt.X("bin:N", title=f"{variable} (bins por cuantiles de la referencia)", sort=etiquetas + ["Sin dato"]),
            xOffset="grupo:N",
            y=alt.Y("proporcion:Q", title="Proporción", axis=alt.Axis(format="%")),
            color=alt.Color("grupo:N", title=None),
            tooltip=["grupo", "bin", alt.Tooltip("proporcion:Q", format=".1%")],
        )
        st.caption("En la referencia cada bin tiene cerca del 10 % de los datos: las barras que se alejan de esa "
                   "altura en el periodo actual muestran dónde está el drift.")
    st.altair_chart(grafico.properties(height=420, title=f"Histórico vs actual: {variable}"), width="stretch")

    metricas_variable = drift_periodo.set_index("variable").loc[variable]
    m1, m2, m3 = st.columns(3)
    m1.metric("PSI", f"{metricas_variable['psi']:.3f}")
    m2.metric("Jensen-Shannon", f"{metricas_variable['js']:.3f}")
    prueba = "χ²" if variable in VARIABLES_CATEGORICAS else "KS"
    m3.metric(f"p-valor {prueba}", f"{metricas_variable['p_valor']:.4f}")

# --- Análisis temporal ---------------------------------------------------------
with tab_temporal:
    st.subheader("Evolución del PSI por variable")
    mapa = alt.Chart(drift).mark_rect().encode(
        x=alt.X("periodo:N", scale=escala_periodos, title="Periodo"),
        y=alt.Y("variable:N", title=None),
        color=alt.Color("psi:Q", scale=alt.Scale(scheme="orangered", domain=[0, 0.5], clamp=True), title="PSI"),
        tooltip=["periodo", "variable", alt.Tooltip("psi:Q", format=".3f"), "estado"],
    )
    st.altair_chart(mapa.properties(height=520), width="stretch")

    mas_drift = drift.groupby("variable")["psi"].max().sort_values(ascending=False).index.tolist()
    seleccion = st.multiselect("Variables a comparar", VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS,
                               default=mas_drift[:4])
    lineas = alt.Chart(drift[drift["variable"].isin(seleccion)]).mark_line(point=True).encode(
        x=alt.X("periodo:N", scale=escala_periodos, title="Periodo"),
        y=alt.Y("psi:Q", title="PSI"),
        color=alt.Color("variable:N", title="Variable"),
        tooltip=["periodo", "variable", alt.Tooltip("psi:Q", format=".3f"),
                 alt.Tooltip("cambio_psi:Q", format="+.3f", title="Δ PSI")],
    )
    umbrales = alt.Chart(pd.DataFrame({"psi": UMBRALES_PSI, "nivel": ["amarillo", "rojo"]})).mark_rule(
        strokeDash=[4, 4]).encode(y="psi:Q", color=alt.value("gray"))
    st.altair_chart((lineas + umbrales).properties(height=380), width="stretch")

    st.subheader("Cambios abruptos y tendencias")
    cambios = drift[drift["salto_abrupto"] | drift["tendencia_creciente"]]
    st.dataframe(
        cambios[["periodo", "variable", "psi", "cambio_psi", "salto_abrupto", "tendencia_creciente"]],
        hide_index=True, width="stretch",
        column_config={
            "psi": st.column_config.NumberColumn("PSI", format="%.3f"),
            "cambio_psi": st.column_config.NumberColumn("Δ PSI", format="%+.3f"),
            "salto_abrupto": "Salto abrupto",
            "tendencia_creciente": "Tendencia creciente",
        },
    )

    st.subheader("Desempeño del modelo en el tiempo")
    desempeno_largo = desempeno.melt(id_vars=["periodo", "n_impagos"],
                                     value_vars=["tasa_impago_real", "tasa_impago_predicha", "roc_auc"],
                                     var_name="metrica", value_name="valor")
    orden_todos = resultados["periodos"]
    grafico_desempeno = alt.Chart(desempeno_largo).mark_line(point=True).encode(
        x=alt.X("periodo:N", scale=alt.Scale(domain=orden_todos), title="Periodo"),
        y=alt.Y("valor:Q", title=None),
        color=alt.Color("metrica:N", title="Métrica"),
        tooltip=["periodo", "metrica", alt.Tooltip("valor:Q", format=".3f"), "n_impagos"],
    )
    st.altair_chart(grafico_desempeno.properties(height=340), width="stretch")
    st.caption("ROC-AUC de la referencia calculado out-of-fold. Los periodos con pocos impagos tienen un ROC-AUC muy variable.")

# --- Datos y pronósticos -------------------------------------------------------
with tab_datos:
    st.subheader(f"Datos con pronósticos · {periodo}")
    columnas = ["fecha_prestamo", "periodo", "proba_impago", "prediccion", TARGET] + \
               [c for c in tabla.columns if c not in {"fecha_prestamo", "periodo", "proba_impago", "prediccion", TARGET}]
    datos_periodo = tabla.loc[tabla["periodo"] == periodo, columnas]
    st.dataframe(datos_periodo, hide_index=True, width="stretch",
                 column_config={"proba_impago": st.column_config.NumberColumn("proba_impago", format="%.3f")})
    st.download_button("Descargar CSV del periodo", datos_periodo.to_csv(index=False).encode("utf-8"),
                       file_name=f"pronosticos_{periodo.replace(' ', '_')}.csv", mime="text/csv")
