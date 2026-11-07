# Despliegue de proyecto: modelo de riesgo crediticio

Modelo predictivo que estima si un cliente **pagará a tiempo su crédito** (`Pago_atiempo`), junto con el pipeline completo de datos, entrenamiento, evaluación y monitoreo de data drift.

## Caso de negocio

Proyecto desarrollado para el equipo de Datos y Analítica de una empresa financiera. El objetivo es construir un modelo de aprendizaje automático que, a partir de información histórica de créditos, anticipe el comportamiento de nuevos usuarios: si pagarán a tiempo o no. Por eso el modelo solo usa variables disponibles al momento de la solicitud. Un buen modelo permite:

- **Reducir pérdidas por impago**, rechazando o ajustando las condiciones (monto, plazo, garantías) de las solicitudes más riesgosas.
- **Aprobar con más confianza** a los clientes de bajo riesgo.
- **Detectar a tiempo** cuándo la población de clientes cambia y el modelo deja de ser confiable.

El reto principal es el **fuerte desbalance**: el 95.25 % de los clientes paga a tiempo y solo el 4.75 % (511 casos) no lo hace. Un modelo que diga "paga" para todos obtiene 95 % de accuracy y F1 = 0.976 para la clase 1 sin aportar nada, así que el valor está en identificar a la minoría que no paga.

## Datos

`Base_de_datos.csv`: 10,763 créditos y 23 columnas, sin filas duplicadas, otorgados entre noviembre de 2024 y abril de 2026.

| Grupo | Variables |
|---|---|
| Crédito | `tipo_credito`, `fecha_prestamo`, `capital_prestado`, `plazo_meses`, `cuota_pactada` |
| Cliente | `edad_cliente`, `tipo_laboral`, `salario_cliente`, `total_otros_prestamos` |
| Central de riesgo (Datacrédito) | `puntaje_datacredito`, `cant_creditosvigentes`, `huella_consulta`, `saldo_mora`, `saldo_total`, `saldo_principal`, `saldo_mora_codeudor`, `creditos_sectorFinanciero`, `creditos_sectorCooperativo`, `creditos_sectorReal`, `promedio_ingresos_datacredito`, `tendencia_ingresos` |
| Score interno | `puntaje` |
| Objetivo | `Pago_atiempo` (1 = pagó a tiempo, 0 = no pagó a tiempo) |

## Estructura del proyecto

La estructura de carpetas es fija porque el despliegue a producción se valida automáticamente con Jenkins.

```
credit-risk-mlops/
├── Base_de_datos.csv
├── readme.md
├── requirements.txt
└── mlops_pipeline/src/
    ├── cargar_datos.ipynb             
    ├── comprension_eda.ipynb          
    ├── ft_engineering.py              
    ├── model_training_evaluation.py   
    ├── model_monitoring.py            
    ├── app_monitoreo.py               
    └── model_deploy.py                
```

## Instalación y ejecución

Requiere Python 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En macOS, XGBoost necesita la librería OpenMP:

```bash
brew install libomp
```

Todos los comandos se ejecutan desde la raíz del repositorio:

```bash
python mlops_pipeline/src/ft_engineering.py               # verifica el preprocesamiento
python mlops_pipeline/src/model_training_evaluation.py    # compara modelos y muestra gráficos
python mlops_pipeline/src/model_monitoring.py             # reporte de drift en la terminal
streamlit run mlops_pipeline/src/app_monitoreo.py         # dashboard en http://localhost:8501
```

La primera vez que se usa Streamlit, pide un correo opcional: basta con dejarlo vacío y presionar Enter.

## Proceso

### Avance 1: carga y análisis exploratorio

- `cargar_datos.ipynb` convierte el Excel original a `Base_de_datos.csv`.
- `comprension_eda.ipynb` analiza la estructura, la calidad y las relaciones de los datos.

Hallazgos principales:

- **Fuga de información en `puntaje`:** separa las clases de forma perfecta (AUC = 1.0). Ningún cliente con impago supera 62.67 y ningún cliente que pagó baja de 63.81. Además, las demás variables explican apenas un 3.6 % de su variación (R² = 0.036), así que lo más probable es que se calcule después de observar el pago. **Se excluye del modelo.**
- **Segmento sin historial:** 150 clientes con edad de 121 a 123 años y `puntaje_datacredito` ≤ 0. Son valores por defecto para clientes sin historial en la central de riesgo, no errores aleatorios.
- **Faltantes informativos:** `promedio_ingresos_datacredito` y `tendencia_ingresos` faltan en el ~27 % de los registros, siempre en los mismos clientes (sin información de ingresos en Datacrédito).
- **Valores imposibles:** salarios de hasta 22,000 millones, plazos de 90 meses, puntajes negativos.
- **Poder predictivo bajo de las variables legítimas:** las más discriminantes son `puntaje_datacredito`, `huella_consulta`, `promedio_ingresos_datacredito` y `edad_cliente`, con separaciones pequeñas.

### Avance 2: ingeniería de características y modelamiento

`**ft_engineering.py**`

1. **Limpieza:** los valores fuera de rango plausible pasan a faltante. Los rangos son edad 18–100, `puntaje_datacredito` 150–950, salario y otros préstamos ≥ 0, y plazo 1–60 meses. Las categorías inválidas de `tendencia_ingresos` también pasan a faltante, y los códigos raros de `tipo_credito` (7 y 68) se agrupan en "Otros".
2. **ColumnTransformer** con tres grupos:
   - **Numéricas (17):** imputación por mediana más indicadores de faltante, que capturan el segmento sin historial.
   - **Nominales** (`tipo_credito`, `tipo_laboral`): imputación "Sin dato" + OneHotEncoder.
   - **Ordinal** (`tendencia_ingresos`): imputación por moda + OrdinalEncoder (Decreciente < Estable < Creciente).
3. Se excluyen `puntaje` (fuga) y `fecha_prestamo`.
4. `generar_conjuntos()` devuelve train y test (80/20) con split estratificado y semilla fija. El preprocesador se ajusta solo con train, dentro del Pipeline de cada modelo.

`**model_training_evaluation.py**`

- Compara Regresión Logística, Random Forest, Gradient Boosting y XGBoost.
- Compensa el desbalance con `class_weight="balanced"`, `sample_weight` (Gradient Boosting) y `scale_pos_weight` (XGBoost).
- Ajusta el umbral de decisión de cada modelo con predicciones out-of-fold de validación cruzada sobre train, sin usar test.

Resultados en test con el umbral ajustado (clase positiva: `Pago_atiempo` = 1):

| Modelo | Umbral de impago | F1 | ROC-AUC |
|---|---|---|---|
| **XGBoost** | 0.677 | **0.960** | 0.664 |
| Random Forest | 0.443 | 0.948 | 0.673 |
| Gradient Boosting | 0.606 | 0.935 | 0.663 |
| Regresión Logística | 0.586 | 0.883 | 0.688 |

**Modelo seleccionado: XGBoost** (mayor F1). Las diferencias de ROC-AUC entre modelos están dentro de la variación de la validación cruzada (desviación estándar de 0.02 a 0.04).

**Conclusión:** el desempeño está limitado por los datos, no por el algoritmo. Probar nuevas variables, ajustar hiperparámetros y comparar 14 modelos con PyCaret no dio mejoras significativas. Un F1 de ~0.97 solo se alcanza incluyendo `puntaje`, lo que sería fuga de información.

### Avance 3: monitoreo y data drift

`**model_monitoring.py**` arma la tabla de datos con los pronósticos del modelo y la muestrea por periodos según `fecha_prestamo`:

- **Referencia (histórico):** préstamos hasta marzo de 2025 (5,528 registros). Con ellos se entrena XGBoost y se ajusta el umbral. Sus pronósticos son out-of-fold.
- **Producción:** cada mes desde abril de 2025 se puntúa con ese modelo. Los meses con menos de 100 préstamos se agrupan (feb–abr 2026).

Para cada periodo se comparan las 20 variables de entrada y la probabilidad predicha contra la referencia:

| Métrica | Variables | 🟡 Amarillo | 🔴 Rojo |
|---|---|---|---|
| PSI (Population Stability Index, 10 bins por cuantiles) | todas | > 0.10 | > 0.25 |
| Divergencia de Jensen-Shannon (base 2) | todas | > 0.10 | > 0.20 |
| Kolmogorov-Smirnov | numéricas | p < 0.05 | — |
| Chi-cuadrado | categóricas | p < 0.05 | — |

Las pruebas KS y chi-cuadrado llegan como máximo a amarillo, porque con muestras grandes casi cualquier diferencia resulta significativa.

**Alertas automáticas:**

- **Crítica (reentrenar):** 3 o más variables en rojo, PSI de la probabilidad predicha > 0.25, o caída del ROC-AUC mayor a 0.05 respecto de la referencia. Esta última solo cuenta si el periodo tiene al menos 10 impagos.
- **Advertencia (revisar variables):** alguna variable en rojo, o un salto de PSI mayor a 0.10 respecto del periodo anterior.
- **Informativa:** PSI en aumento durante 3 periodos seguidos.

`**app_monitoreo.py**` (Streamlit) tiene cinco pestañas:

1. **Resumen y alertas:** indicadores del periodo, mensajes automáticos y recomendaciones.
2. **Métricas de drift:** tabla con semáforo y barras de riesgo por variable.
3. **Distribuciones:** comparación histórico vs actual de cada variable.
4. **Análisis temporal:** mapa de calor y evolución del PSI, cambios abruptos, tendencias y desempeño del modelo en el tiempo.
5. **Datos y pronósticos:** tabla del periodo con descarga en CSV.

#### Hallazgos del monitoreo

- `**promedio_ingresos_datacredito**` tiene el drift más fuerte y creciente: su PSI pasa de 0.25 en abril de 2025 a 1.10 en 2026. Los ingresos se concentran en valores ligados al salario mínimo, que suben con el tiempo, así que el modelo ve ingresos "más altos" que no lo son en términos reales.
- `**plazo_meses**` tuvo un cambio temporal entre mayo y agosto de 2025 (PSI de 0.39 a 0.55) que después se normalizó. Apunta a un cambio pasajero en la política de plazos.
- `**total_otros_prestamos**`, `**capital_prestado**` y `**cuota_pactada**` pasan a rojo desde mediados o finales de 2025.
- **El modelo sobreestima el impago:** la tasa predicha sube de 10 % a 13–17 %, mientras la real baja de 5 % a 2–4 %.
- **El ROC-AUC se mantiene en 0.67–0.74** en los meses con suficientes impagos, con una caída a 0.52 en agosto de 2025. En los meses con menos de 10 impagos, el AUC no es confiable.

#### Recomendaciones

1. **Reentrenar** el modelo con datos más recientes: el monitoreo genera alertas críticas desde junio de 2025, en 6 de los 11 periodos.
2. **Expresar los montos en términos reales** (deflactados o relativos al salario mínimo), para que la inflación no se confunda con un cambio en el perfil de los clientes.
3. **Recalibrar el umbral** de decisión, porque la tasa de impago real cambió frente a la del periodo de entrenamiento.
4. **Validar con un split temporal** en lugar de aleatorio, ya que la población cambia con el tiempo.
5. **Confirmar con el negocio** cómo se calcula `puntaje` antes de considerarlo en futuras versiones.

## Versiones

| Versión | Contenido |
|---|---|
| v1.0.0 | Estructura inicial del proyecto |
| v1.0.1 | Avance 1: carga de datos y análisis exploratorio |
| v1.1.0 | Avance 2: ingeniería de características y modelamiento |
| v1.1.1 | Avance 3: monitoreo de data drift, app en Streamlit y documentación |
