# ============================================================================
# ANÁLISIS DE ABLACIÓN DE FEATURES HISTÓRICAS
# ============================================================================
# Este script realiza experimentos de ABLACIÓN para medir directamente
# el impacto de cada tipo de feature en la GANANCIA FINAL.
#
# ABLACIÓN: Consiste en entrenar modelos eliminando grupos de features
# y comparar la ganancia resultante vs el modelo completo.
#
# Esto te dice exactamente cuánta ganancia aporta cada grupo de features.
# ============================================================================

require("data.table")
require("lightgbm")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

PARAM <- list()
PARAM$dataset <- "gerencial_competencia_2025.csv.gz"
PARAM$semilla <- 102191

# Hiperparámetros a usar (usa los mejores del modelo completo)
PARAM$lgbm <- list(
  objective = "binary",
  metric = "auc",
  first_metric_only = TRUE,
  boost_from_average = TRUE,
  feature_pre_filter = FALSE,
  verbosity = -100,
  force_row_wise = TRUE,
  seed = PARAM$semilla,
  max_bin = 31,
  learning_rate = 0.03,
  feature_fraction = 0.5,
  num_iterations = 500,  # Reducido para experimentos rápidos
  num_leaves = 64,       # Ajusta según tus mejores hiperparámetros
  min_data_in_leaf = 1000
)

# ============================================================================
# CARGAR Y PREPARAR DATASET
# ============================================================================

cat("\n========================================\n")
cat("ANÁLISIS DE ABLACIÓN\n")
cat("========================================\n\n")

cat("Cargando dataset...\n")
dataset <- fread(paste0("/content/datasets/", PARAM$dataset))

# Aplicar Catastrophe Analysis
dataset[foto_mes == 202006, internet := NA]
dataset[foto_mes == 202006, mrentabilidad := NA]
dataset[foto_mes == 202006, mrentabilidad_annual := NA]
dataset[foto_mes == 202006, mcomisiones := NA]
dataset[foto_mes == 202006, mactivos_margen := NA]
dataset[foto_mes == 202006, mpasivos_margen := NA]
dataset[foto_mes == 202006, mcuentas_saldo := NA]
dataset[foto_mes == 202006, ctarjeta_visa_transacciones := NA]
dataset[foto_mes == 202006, mtarjeta_visa_consumo := NA]
dataset[foto_mes == 202006, mtarjeta_master_consumo := NA]
dataset[foto_mes == 202006, ccallcenter_transacciones := NA]
dataset[foto_mes == 202006, chomebanking_transacciones := NA]

# Feature Engineering Intra-mes
if ("foto_mes" %in% colnames(dataset))
  dataset[, kmes := foto_mes %% 100]

if (all(c("mpayroll", "cliente_edad") %in% colnames(dataset)))
  dataset[, mpayroll_sobre_edad := mpayroll / cliente_edad]

# ============================================================================
# GENERAR TODAS LAS FEATURES HISTÓRICAS
# ============================================================================

cat("\nGenerando features históricas...\n")

setorder(dataset, numero_de_cliente, foto_mes)

cols_lagueables <- copy(setdiff(
  colnames(dataset),
  c("numero_de_cliente", "foto_mes", "clase_ternaria")
))

cat("Variables base:", length(cols_lagueables), "\n")

# === LAGS ===
cat("  Generando lags...\n")
for (lag in c(1, 2, 3, 6)) {
  dataset[,
          paste0(cols_lagueables, "_lag", lag) := shift(.SD, lag, NA, "lag"),
          by = numero_de_cliente,
          .SDcols = cols_lagueables
  ]
}

# === DELTAS ===
cat("  Generando deltas...\n")
for (vcol in cols_lagueables) {
  for (lag in 1:3) {
    dataset[, paste0(vcol, "_delta", lag) := get(vcol) - get(paste0(vcol, "_lag", lag))]
  }
}

# === ROLLING ===
cat("  Generando rolling statistics (puede tardar)...\n")
for (vcol in cols_lagueables) {
  # Rolling 3 meses
  dataset[, paste0(vcol, "_roll3_mean") := frollmean(get(vcol), 3, align = "right", na.rm = TRUE),
          by = numero_de_cliente]
  dataset[, paste0(vcol, "_roll3_sd") := frollapply(get(vcol), 3, sd, align = "right", na.rm = TRUE),
          by = numero_de_cliente]

  # Rolling 6 meses
  dataset[, paste0(vcol, "_roll6_mean") := frollmean(get(vcol), 6, align = "right", na.rm = TRUE),
          by = numero_de_cliente]
}

# === RATIOS ===
cat("  Generando ratios...\n")
for (vcol in cols_lagueables) {
  dataset[, paste0(vcol, "_ratio_vs_roll3") :=
            ifelse(get(paste0(vcol, "_roll3_mean")) != 0,
                   get(vcol) / get(paste0(vcol, "_roll3_mean")),
                   NA)]
}

cat("\nTotal de columnas después de FE:", ncol(dataset), "\n")

# ============================================================================
# PREPARAR DATOS PARA MODELADO
# ============================================================================

dataset[, clase01 := ifelse(clase_ternaria %in% c("BAJA+1", "BAJA+2"), 1, 0)]

# Training y validation
train_meses <- c(202104, 202103, 202102, 202101,
                 202012, 202011, 202010, 202009, 202008, 202007)
validate_mes <- c(202105)
future_mes <- c(202107)

set.seed(PARAM$semilla, kind = "L'Ecuyer-CMRG")
dataset[, azar := runif(nrow(dataset))]

dataset[, fold_train := foto_mes %in% train_meses]
dataset[, fold_validate := foto_mes %in% validate_mes]
dataset[, fold_future := foto_mes %in% future_mes]

# ============================================================================
# FUNCIÓN AUXILIAR: ENTRENAR Y EVALUAR
# ============================================================================

entrenar_y_evaluar <- function(cols_incluidas, nombre_experimento) {

  cat("\n--- Experimento:", nombre_experimento, "---\n")
  cat("Features usadas:", length(cols_incluidas), "\n")

  # Crear datasets
  dtrain <- lgb.Dataset(
    data = data.matrix(dataset[fold_train == TRUE, cols_incluidas, with = FALSE]),
    label = dataset[fold_train == TRUE, clase01],
    free_raw_data = FALSE
  )

  dvalidate <- lgb.Dataset(
    data = data.matrix(dataset[fold_validate == TRUE, cols_incluidas, with = FALSE]),
    label = dataset[fold_validate == TRUE, clase01],
    free_raw_data = FALSE
  )

  # Entrenar
  modelo <- lgb.train(
    data = dtrain,
    valids = list(valid = dvalidate),
    param = PARAM$lgbm,
    verbose = -100
  )

  # Predecir en future
  prediccion <- predict(
    modelo,
    data.matrix(dataset[fold_future == TRUE, cols_incluidas, with = FALSE])
  )

  # Calcular ganancia
  tb_pred <- data.table(
    numero_de_cliente = dataset[fold_future == TRUE, numero_de_cliente],
    prob = prediccion,
    clase_ternaria = dataset[fold_future == TRUE, clase_ternaria]
  )

  tb_pred[, ganancia := -3000.0]
  tb_pred[clase_ternaria == "BAJA+2", ganancia := 117000.0]

  setorder(tb_pred, -prob)
  tb_pred[, gan_acum := cumsum(ganancia)]
  tb_pred[, gan_suavizada := frollmean(gan_acum, 400, align = "center", na.rm = TRUE, hasNA = TRUE)]

  ganancia_max <- max(tb_pred$gan_suavizada, na.rm = TRUE)
  envios <- which.max(tb_pred$gan_suavizada)

  cat("Ganancia máxima:", ganancia_max, "\n")
  cat("Envíos óptimos:", envios, "\n")

  rm(dtrain, dvalidate, modelo)
  gc(full = TRUE, verbose = FALSE)

  return(list(
    experimento = nombre_experimento,
    num_features = length(cols_incluidas),
    ganancia = ganancia_max,
    envios = envios
  ))
}

# ============================================================================
# EXPERIMENTOS DE ABLACIÓN
# ============================================================================

cat("\n\n========================================\n")
cat("INICIANDO EXPERIMENTOS DE ABLACIÓN\n")
cat("========================================\n\n")

resultados_ablacion <- list()

# Obtener todas las columnas disponibles (sin clase_ternaria, clase01, azar)
todas_cols <- setdiff(colnames(dataset),
                      c("numero_de_cliente", "foto_mes", "clase_ternaria",
                        "clase01", "azar", "fold_train", "fold_validate", "fold_future"))

# Identificar columnas por tipo
cols_originales <- todas_cols[!grepl("_(lag|delta|roll|trend|ratio|cv|range)", todas_cols)]
cols_lag <- todas_cols[grepl("_lag[0-9]$", todas_cols)]
cols_delta <- todas_cols[grepl("_delta[0-9]$", todas_cols)]
cols_rolling <- todas_cols[grepl("_roll[0-9]_(mean|sd|max|min)$", todas_cols)]
cols_ratio <- todas_cols[grepl("_ratio_", todas_cols)]

cat("Distribución de features:\n")
cat("  Originales:", length(cols_originales), "\n")
cat("  Lags:", length(cols_lag), "\n")
cat("  Deltas:", length(cols_delta), "\n")
cat("  Rolling:", length(cols_rolling), "\n")
cat("  Ratios:", length(cols_ratio), "\n\n")

# ============================================================================
# EXPERIMENTO 1: Modelo completo (baseline)
# ============================================================================

resultado <- entrenar_y_evaluar(todas_cols, "COMPLETO (baseline)")
resultados_ablacion[[length(resultados_ablacion) + 1]] <- resultado

# ============================================================================
# EXPERIMENTO 2: Solo variables originales (sin FE histórico)
# ============================================================================

resultado <- entrenar_y_evaluar(cols_originales, "SOLO_ORIGINALES (sin FE histórico)")
resultados_ablacion[[length(resultados_ablacion) + 1]] <- resultado

# ============================================================================
# EXPERIMENTO 3: Originales + Lags
# ============================================================================

resultado <- entrenar_y_evaluar(c(cols_originales, cols_lag), "ORIGINALES + LAGS")
resultados_ablacion[[length(resultados_ablacion) + 1]] <- resultado

# ============================================================================
# EXPERIMENTO 4: Originales + Deltas
# ============================================================================

resultado <- entrenar_y_evaluar(c(cols_originales, cols_delta), "ORIGINALES + DELTAS")
resultados_ablacion[[length(resultados_ablacion) + 1]] <- resultado

# ============================================================================
# EXPERIMENTO 5: Originales + Rolling
# ============================================================================

resultado <- entrenar_y_evaluar(c(cols_originales, cols_rolling), "ORIGINALES + ROLLING")
resultados_ablacion[[length(resultados_ablacion) + 1]] <- resultado

# ============================================================================
# EXPERIMENTO 6: Originales + Ratios
# ============================================================================

resultado <- entrenar_y_evaluar(c(cols_originales, cols_ratio), "ORIGINALES + RATIOS")
resultados_ablacion[[length(resultados_ablacion) + 1]] <- resultado

# ============================================================================
# EXPERIMENTO 7: Sin Lags (ablación)
# ============================================================================

cols_sin_lags <- setdiff(todas_cols, cols_lag)
resultado <- entrenar_y_evaluar(cols_sin_lags, "SIN_LAGS (ablación)")
resultados_ablacion[[length(resultados_ablacion) + 1]] <- resultado

# ============================================================================
# EXPERIMENTO 8: Sin Deltas (ablación)
# ============================================================================

cols_sin_deltas <- setdiff(todas_cols, cols_delta)
resultado <- entrenar_y_evaluar(cols_sin_deltas, "SIN_DELTAS (ablación)")
resultados_ablacion[[length(resultados_ablacion) + 1]] <- resultado

# ============================================================================
# EXPERIMENTO 9: Sin Rolling (ablación)
# ============================================================================

cols_sin_rolling <- setdiff(todas_cols, cols_rolling)
resultado <- entrenar_y_evaluar(cols_sin_rolling, "SIN_ROLLING (ablación)")
resultados_ablacion[[length(resultados_ablacion) + 1]] <- resultado

# ============================================================================
# CONSOLIDAR RESULTADOS
# ============================================================================

cat("\n\n========================================\n")
cat("RESULTADOS CONSOLIDADOS\n")
cat("========================================\n\n")

tb_resultados <- rbindlist(resultados_ablacion)
setorder(tb_resultados, -ganancia)

# Calcular delta respecto al baseline
ganancia_baseline <- tb_resultados[experimento == "COMPLETO (baseline)", ganancia]
tb_resultados[, delta_vs_baseline := ganancia - ganancia_baseline]
tb_resultados[, delta_pct := (delta_vs_baseline / ganancia_baseline) * 100]

print(tb_resultados)

cat("\n\nINTERPRETACIÓN:\n\n")

cat("1. IMPACTO DEL FE HISTÓRICO:\n")
ganancia_solo_orig <- tb_resultados[experimento == "SOLO_ORIGINALES (sin FE histórico)", ganancia]
mejora_fe <- ganancia_baseline - ganancia_solo_orig
mejora_fe_pct <- (mejora_fe / ganancia_solo_orig) * 100
cat("   El FE histórico aporta:", mejora_fe, "\n")
cat("   Mejora porcentual:", round(mejora_fe_pct, 2), "%\n\n")

cat("2. CONTRIBUCIÓN POR TIPO (incremental):\n")
for (tipo in c("LAGS", "DELTAS", "ROLLING", "RATIOS")) {
  exp_name <- paste0("ORIGINALES + ", tipo)
  if (exp_name %in% tb_resultados$experimento) {
    gan <- tb_resultados[experimento == exp_name, ganancia]
    delta <- gan - ganancia_solo_orig
    delta_pct <- (delta / ganancia_solo_orig) * 100
    cat("   ", tipo, ":", delta, "(", round(delta_pct, 2), "%)\n")
  }
}

cat("\n3. PÉRDIDA POR ABLACIÓN (cuánto cae la ganancia al eliminar):\n")
for (tipo in c("LAGS", "DELTAS", "ROLLING")) {
  exp_name <- paste0("SIN_", tipo, " (ablación)")
  if (exp_name %in% tb_resultados$experimento) {
    gan <- tb_resultados[experimento == exp_name, ganancia]
    perdida <- ganancia_baseline - gan
    perdida_pct <- (perdida / ganancia_baseline) * 100
    cat("   Sin", tipo, ": pierde", perdida, "(", round(perdida_pct, 2), "%)\n")
  }
}

# Guardar resultados
fwrite(tb_resultados, "ablacion_resultados.txt", sep = "\t")

cat("\n\nArchivo generado: ablacion_resultados.txt\n")

cat("\n========================================\n")
cat("RECOMENDACIONES BASADAS EN ABLACIÓN:\n")
cat("========================================\n\n")

# Identificar el tipo más valioso
tipos <- c("LAGS", "DELTAS", "ROLLING", "RATIOS")
mejores <- c()
for (tipo in tipos) {
  exp_name <- paste0("ORIGINALES + ", tipo)
  if (exp_name %in% tb_resultados$experimento) {
    gan <- tb_resultados[experimento == exp_name, ganancia]
    mejores <- c(mejores, list(list(tipo = tipo, ganancia = gan)))
  }
}

cat("✓ PRIORIZA estos tipos de features en futuras iteraciones:\n")
cat("  (ordenados por impacto individual)\n\n")

mejores_dt <- rbindlist(mejores)
setorder(mejores_dt, -ganancia)
print(mejores_dt)

cat("\n\n✓ CONSIDERA ELIMINAR features que causan poca pérdida en ablación\n")
cat("  para reducir dimensionalidad sin perder performance\n\n")

cat("========================================\n")
cat("ANÁLISIS DE ABLACIÓN COMPLETADO\n")
cat("========================================\n")
