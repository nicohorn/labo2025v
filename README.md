# Experimentos de Feature Engineering para Predicción de Churn Bancario

## Guía de Reproducibilidad

### Prerequisitos

#### Hardware Recomendado
- **CPU:** Mínimo 8 cores (recomendado 20+ cores para paralelización)
- **RAM:** Mínimo 16 GB (recomendado 32 GB)
- **Almacenamiento:** ~5 GB libres (para dataset + resultados)

#### Software Requerido

**1. R (versión 4.4.2 o superior)**
- **Windows:** Descargar de [CRAN](https://cran.r-project.org/bin/windows/base/)
- **Linux/Mac:** Instalar desde repositorio oficial

**2. Git**
- Para clonar el repositorio

**3. Paquetes de R**

Ejecutar en R:
```r
# Paquetes principales
install.packages(c(
  "data.table",      # Manipulación de datos
  "lightgbm",        # Modelo de gradient boosting
  "DiceKriging",     # Para Bayesian Optimization
  "mlr",             # Machine Learning framework
  "ParamHelpers",    # Definición de hiperparámetros
  "mlrMBO",          # Bayesian Optimization
  "smoof",           # Funciones de optimización
  "checkmate",       # Validación de argumentos
  "yaml",            # Lectura/escritura YAML
  "parallel",        # Paralelización
  "scales"           # Formateo de números
))
```

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/nicohorn/labo2025v
cd labo2025v
```

### Paso 2: Verificar el Dataset

El dataset está incluido en el repositorio:
```bash
ls -lh datasets/gerencial_competencia_2025.csv.gz
# Debería mostrar: ~16 MB
```

### Paso 3: Ejecutar Experimentos

#### Opción A: Ejecutar TODOS los experimentos

```bash
# Windows (PowerShell o CMD)
Rscript src/rscripts/workflow_610_BASELINE_z610_paralelo.R
Rscript src/rscripts/workflow_616_BASELINE_z610.R
Rscript src/rscripts/workflow_617_NO_TREND.R
Rscript src/rscripts/workflow_618_ONLY_TREND.R

# Linux/Mac
Rscript src/rscripts/workflow_610_BASELINE_z610_paralelo.R
Rscript src/rscripts/workflow_616_BASELINE_z610.R
Rscript src/rscripts/workflow_617_NO_TREND.R
Rscript src/rscripts/workflow_618_ONLY_TREND.R
```

**Tiempo estimado total:** ~4-5 horas (con paralelización en máquina de 20 cores)

#### Opción B: Ejecutar experimento individual

```bash
# Solo el experimento SOLO TREND (el más interesante)
Rscript src/rscripts/workflow_618_ONLY_TREND.R
```

**Tiempo estimado:** ~3-4 horas (5 seeds en paralelo)

### Paso 4: Monitorear Ejecución

Durante la ejecución, puedes monitorear el progreso:

```bash
# Ver logs en tiempo real
tail -f exp/exp_only_trend/workflow_only_trend_*.log

# Ver health checks
tail -f exp/exp_only_trend/health_*.txt

# Ver progreso de cada seed
cat exp/exp_only_trend/progress_*.txt
```

### Paso 5: Verificar Resultados

Una vez completada la ejecución:

```bash
# Ver resumen de resultados
cat exp/exp_only_trend/resumen_only_trend_exp6180.txt

# Ver todas las carpetas generadas
ls exp/exp_only_trend/
# Deberías ver: WF6180_seed1_ONLY_TREND/, WF6181_seed2_ONLY_TREND/, etc.
```

Cada carpeta de seed contiene:
- `modelo.txt` - Modelo entrenado
- `impo.txt` - Importancia de features
- `BO_log.txt` - Log de Bayesian Optimization
- `ganancias.txt` - Curva de ganancia
- `curva_de_ganancia.pdf` - Gráfico de ganancia
- `PARAM.yml` - Parámetros utilizados

### Paso 6: Generar Análisis (Opcional)

```bash
# Análisis de feature importance
Rscript src/workflows/analyze_fe_importance.R

# Generar reporte comparativo en PDF
Rscript src/workflows/generate_comparison_report.R
```

### Troubleshooting

#### Error: "Package 'X' not found"
```r
# Instalar paquete faltante
install.packages("nombre_paquete")
```

#### Error: "Cannot allocate memory"
- Reducir número de workers en el script (línea ~73)
- Aumentar RAM disponible
- Ejecutar experimentos de forma secuencial

#### Error: "Dataset not found"
- Verificar que `datasets/gerencial_competencia_2025.csv.gz` existe
- Verificar ruta absoluta en scripts (Windows: `C:/`, Linux: `/home/`)

#### Ejecución muy lenta
- Verificar número de cores disponibles: `parallel::detectCores()`
- Ajustar `num_workers` y `threads_per_worker` en scripts
- Considerar ejecutar solo 1-2 seeds como prueba

### Resultados Esperados

Si todo funciona correctamente, deberías obtener:

**WF618 (SOLO TREND):**
```
Ganancia promedio: ~$11,535,660
Envíos promedio: ~1,429
AUC promedio: ~0.9948
Duración promedio: ~44 min/seed
```

**WF616 (COMPLETO):**
```
Ganancia promedio: ~$13,558,860
Envíos promedio: ~863
AUC promedio: ~0.9988
Duración promedio: ~50 min/seed
```

---

## Descripción del Proyecto

Este proyecto implementa y evalúa diferentes estrategias de **Feature Engineering Histórico** para predecir BAJA+2 (clientes que abandonarán un banco en 2 meses) usando **LightGBM** y **Bayesian Optimization**.

### Objetivo

Determinar qué componentes del Feature Engineering Histórico son realmente necesarios para maximizar la ganancia en la predicción de churn bancario.

---

## Dataset

- **Archivo:** `datasets/gerencial_competencia_2025.csv.gz`
- **Tamaño:** 16 MB comprimido
- **Registros:** 273,666 filas
- **Clientes únicos:** 17,745
- **Variables originales:** 32
- **Período temporal:** Mayo 2020 - Septiembre 2021 (17 meses)

---

## Experimentos Realizados

### 1. **z610 (BÁSICO) - Línea Base**
```
Features: lag1, lag2, delta1, delta2 (~145)
Ganancia: $5,943,360
Envíos: ~1,449
Tiempo: ~2 min/seed
```
**Workflow:** `src/rscripts/workflow_610_BASELINE_z610_paralelo.R`

---

### 2. **WF617 (SIN TREND) - Ablation Study**
```
Features: Lags + Deltas + Rolling + Ratios + Volatilidad (~631)
         TODO EXCEPTO TRENDS
Ganancia: $5,716,920 (42.2% vs WF616)
Envíos: ~1,291
Tiempo: ~134 min/seed
```
**Resultado clave:** Sin TREND, el FE avanzado rinde PEOR que el básico.

**Workflow:** `src/rscripts/workflow_617_NO_TREND.R`

---

### 3. **WF618 (SOLO TREND) - Isolation Study**
```
Features: SOLO trend_3 y trend_6 (~87)
Ganancia: $11,535,660 (85.1% vs WF616) 
Envíos: ~1,429
Tiempo: ~44 min/seed
```
**Resultado clave:** Con SOLO TREND (13% de features) se alcanza 85% de la ganancia.

**Workflow:** `src/rscripts/workflow_618_ONLY_TREND.R`

---

### 4. **WF616 (COMPLETO) - Feature Engineering Completo**
```
Features: TODO (Lags + Deltas + Rolling + TRENDS + Ratios + Volatilidad) (~689)
Ganancia: $13,558,860 (100%)
Envíos: ~863
AUC: ~0.9988
Tiempo: ~50 min/seed
```
**Workflow:** `src/rscripts/workflow_616_BASELINE_z610.R`

---

## Hallazgos Principales

### **TREND es el componente crítico**

| Experimento | Features | Ganancia | % vs Completo | Eficiencia |
|------------|----------|----------|---------------|------------|
| z610 (básico) | 145 | $5.9M | 43.8% | - |
| WF617 (sin TREND) | 631 | $5.7M | 42.2% | Peor que básico |
| WF618 (solo TREND) | 87 | $11.5M | **85.1%** |  13% features → 85% ganancia |
| WF616 (completo) | 689 | $13.6M | 100% | Máxima precisión |

### **Contribución por Tipo de Feature**

```
TREND:              61.2% 
Rolling Stats:      15.1%  
Lag Avanzado (3,6):  5.1%  
Lag Básico (1,2):    4.4%
Original:            3.8%
Delta Básico (1,2):  3.3%
Delta Avanzado (3):  2.1%
Volatilidad (CV):    1.9%
Ratio:               1.8%
Volatilidad (Range): 1.4%
```

### **Top 3 Variables Más Importantes**

1. `Visa_fechaalta_trend_6`: 22.4% de ganancia
2. `internet_trend_6`: 8.9% de ganancia
3. `Master_fechaalta_trend_6`: 7.4% de ganancia

**Todas son TRENDS** - capturan trayectorias sostenidas de abandono.

---

## Tipos de Variables de Feature Engineering

### **1. LAGS (Rezagos)**
Valores de meses anteriores.
```
mcaja_ahorro_lag1 = saldo hace 1 mes
mcaja_ahorro_lag6 = saldo hace 6 meses
```

### **2. DELTAS (Diferencias)**
Cambios absolutos.
```
mcaja_ahorro_delta1 = actual - lag1
mcaja_ahorro_delta6 = actual - lag6
```

### **3. ROLLING STATISTICS**
Estadísticas móviles en ventanas de tiempo.
```
mcaja_ahorro_roll_mean_3 = promedio últimos 3 meses
mcaja_ahorro_roll_sd_3 = volatilidad
```

### **4. TRENDS (Tendencias)  MÁS IMPORTANTE**
Pendiente de regresión lineal sobre ventanas.
```
mcaja_ahorro_trend_6 = slope de regresión en últimos 6 meses
```
**Captura trayectorias sostenidas, no eventos puntuales.**

### **5. RATIOS (Proporciones)**
Cambios relativos.
```
mcaja_ahorro_ratio_vs_lag6 = actual / lag6
```

### **6. VOLATILIDAD**
Métricas de estabilidad.
```
mcaja_ahorro_cv_3 = desviación estándar / media
mcaja_ahorro_range_norm_3 = (max - min) / media
```

---

## Estructura del Proyecto

```
labo2025v/
├── datasets/
│   └── gerencial_competencia_2025.csv.gz
├── src/
│   ├── rscripts/
│   │   ├── workflow_610_BASELINE_z610_paralelo.R
│   │   ├── workflow_616_BASELINE_z610.R
│   │   ├── workflow_617_NO_TREND.R
│   │   └── workflow_618_ONLY_TREND.R
│   └── workflows/
│       ├── analyze_fe_importance.R
│       └── generate_comparison_report.R
├── exp/
│   ├── exp_z610_baseline/
│   ├── exp_baseline/
│   ├── exp_no_trend/
│   └── exp_only_trend/
├── RESUMEN_4_EXPERIMENTOS.txt
├── RESUMEN_FEATURE_ENGINEERING_EXPERIMENTOS.md
├── GUION_VIDEO_EXPERIMENTOS_FE_FINAL.md
└── README.md
```

---

## Cómo Ejecutar

### Prerequisitos

```r
install.packages(c("data.table", "lightgbm", "DiceKriging",
                   "mlr", "ParamHelpers", "mlrMBO", "smoof",
                   "checkmate", "yaml"))
```

### Ejecutar Experimentos

```bash
# Baseline z610
Rscript src/rscripts/workflow_610_BASELINE_z610_paralelo.R

# FE Completo
Rscript src/rscripts/workflow_616_BASELINE_z610.R

# Sin TREND
Rscript src/rscripts/workflow_617_NO_TREND.R

# Solo TREND
Rscript src/rscripts/workflow_618_ONLY_TREND.R
```

### Analizar Resultados

```bash
# Análisis de feature importance
Rscript src/workflows/analyze_fe_importance.R

# Generar reporte comparativo
Rscript src/workflows/generate_comparison_report.R
```

---

## Características Técnicas

### Paralelización
- **Workers:** 5 (uno por seed)
- **Threads por worker:** 4
- **Total cores utilizados:** 20

### Logging y Monitoreo
- **Logs estructurados:** Timestamps, niveles (INFO, SUCCESS, WARNING, ERROR)
- **Health checks:** Monitoreo de ejecución
- **Progress tracking:** Archivo de progreso por seed

### Bayesian Optimization
- **Iteraciones:** 10 (z610) / 100 (WF616, WF617, WF618)
- **Parámetros optimizados:** num_leaves, min_data_in_leaf
- **Métrica:** AUC

### Training Strategy
- **Training:** Mayo 2020 - Abril 2021
- **Validation:** Mayo 2021
- **Final Train:** Mayo 2020 - Mayo 2021
- **Predicción:** Julio 2021 (BAJA+2)

### Catastrophe Analysis
13 variables seteadas a NA en Junio 2020 (simula crisis COVID).

---

## Resultados y Análisis

Ver documentación completa en:
- **Resumen ejecutivo:** `RESUMEN_4_EXPERIMENTOS.txt`
- **Feature Engineering detallado:** `RESUMEN_FEATURE_ENGINEERING_EXPERIMENTOS.md`
- **Guión de video:** `GUION_VIDEO_EXPERIMENTOS_FE_FINAL.md`

---

## Conclusiones

1. **TREND features son críticas:** Con solo 58 features (8% del total) se alcanza 85% de la ganancia.

2. **Sin TREND, el FE avanzado no aporta valor:** WF617 (631 features sin TREND) rinde peor que z610 básico (145 features).

3. **Trade-off validado:**
   - **Producción rápida:** Usar solo TREND (85% ganancia, 13% features, 44 min)
   - **Máxima precisión:** Usar FE completo (100% ganancia, 100% features, 50 min)

4. **Lección clave:** Para predecir churn bancario, no basta con ver "cuánto tiene el cliente hoy". Necesitamos ver **hacia dónde va** → por eso TREND es el rey.

---

## Autor

Proyecto desarrollado para Laboratorio de Implementación I 2025, Universidad Austral.

Aimé Giorlando y Nicolás Horn

**Fecha:** Noviembre 2025
