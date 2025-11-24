# ============================================================================
# WORKFLOW 616 - FEATURE ENGINEERING HISTÓRICO SOLO (VERSIÓN LOCAL WINDOWS)
# ============================================================================
# Versión adaptada para ejecución local en Windows
# SOLO Feature Engineering Histórico (sin intra-mes)
#
# ANTES DE EJECUTAR:
# 1. Verifica que el dataset esté en: C:/Users/User/Documents/labo2025v/datasets/
# 2. Ejecuta setup/install_packages.R si es primera vez
# 3. Ejecuta setup/test_environment.R para verificar entorno
# 4. Ajusta PARAM_GLOBAL$semillas para pruebas (usa solo 1 semilla primero)
# ============================================================================

format(Sys.time(), "%a %b %d %X %Y")

# ============================================================================
# LIMPIEZA INICIAL
# ============================================================================

rm(list = ls(all.names = TRUE))
gc(full = TRUE, verbose = FALSE)

# ============================================================================
# CONFIGURACIÓN DE RUTAS (AJUSTAR SEGÚN TU SISTEMA)
# ============================================================================

BASE_DIR <- "C:/Users/User/Documents/labo2025v"
DATASETS_DIR <- file.path(BASE_DIR, "datasets")
EXP_DIR <- file.path(BASE_DIR, "exp")

# Verificar que existan los directorios
if (!dir.exists(DATASETS_DIR)) {
  stop("ERROR: No existe ", DATASETS_DIR, "\nCrea el directorio o ajusta BASE_DIR")
}

if (!dir.exists(EXP_DIR)) {
  dir.create(EXP_DIR, recursive = TRUE)
  cat("Creado directorio de experimentos:", EXP_DIR, "\n")
}

# ============================================================================
# CARGAR LIBRERÍAS
# ============================================================================

cat("\nCargando librerías...\n")

if (!require("data.table")) {
  stop("Falta data.table. Ejecuta: install.packages('data.table')")
}

if (!require("R.utils")) {
  install.packages("R.utils")
  require("R.utils")
}

cat("Librerías cargadas exitosamente.\n\n")

# ============================================================================
# PARÁMETROS GLOBALES
# ============================================================================

PARAM_GLOBAL <- list()
PARAM_GLOBAL$experimento_base <- 6160
PARAM_GLOBAL$dataset <- "gerencial_competencia_2025.csv.gz"

# IMPORTANTE: Para primera ejecución de prueba, usa solo 1 semilla
# Descomenta la siguiente línea para prueba rápida:
# PARAM_GLOBAL$semillas <- c(102191)

# Para ejecución completa, usa las 5 semillas:
PARAM_GLOBAL$semillas <- c(922081) #c(153929, 838969, 922081, 795581, 194609)

# Lista para almacenar resultados
resultados_totales <- list()

cat("========================================\n")
cat("CONFIGURACIÓN DEL EXPERIMENTO\n")
cat("========================================\n")
cat("Experimento base:", PARAM_GLOBAL$experimento_base, "\n")
cat("Dataset:", PARAM_GLOBAL$dataset, "\n")
cat("Semillas:", length(PARAM_GLOBAL$semillas), "\n")
cat("Directorio de datos:", DATASETS_DIR, "\n")
cat("Directorio de experimentos:", EXP_DIR, "\n")
cat("========================================\n\n")

# Verificar que existe el dataset
dataset_path <- file.path(DATASETS_DIR, PARAM_GLOBAL$dataset)
if (!file.exists(dataset_path)) {
  stop("ERROR: No se encuentra el dataset en: ", dataset_path,
       "\n\nDescárgalo desde: https://storage.googleapis.com/open-courses/austral2025-af91/gerencial_competencia_2025.csv.gz")
}

# ============================================================================
# LOOP PRINCIPAL - ITERACIÓN SOBRE SEMILLAS
# ============================================================================

for (seed_idx in 1:length(PARAM_GLOBAL$semillas)) {

  cat("\n\n========================================\n")
  cat("PROCESANDO SEMILLA ", seed_idx, " de ", length(PARAM_GLOBAL$semillas), "\n")
  cat("Semilla: ", PARAM_GLOBAL$semillas[seed_idx], "\n")
  cat("========================================\n\n")

  # Inicializar PARAM para esta semilla
  PARAM <- list()
  PARAM$semilla_primigenia <- PARAM_GLOBAL$semillas[seed_idx]
  PARAM$experimento <- PARAM_GLOBAL$experimento_base + seed_idx - 1
  PARAM$dataset <- PARAM_GLOBAL$dataset
  PARAM$out <- list()
  PARAM$out$lgbm <- list()

  # ===================================================================
  # Carpeta del Experimento
  # ===================================================================

  experimento_folder <- paste0("WF", PARAM$experimento, "_seed", seed_idx, "_FE_historico_SOLO")
  experimento_path <- file.path(EXP_DIR, experimento_folder)

  dir.create(experimento_path, showWarnings = FALSE, recursive = TRUE)
  setwd(experimento_path)

  cat("Carpeta de trabajo: ", experimento_path, "\n")

  # ===================================================================
  # Preprocesamiento del dataset
  # ===================================================================

  cat("\nLeyendo dataset...\n")
  dataset <- fread(dataset_path)
  cat("Dataset cargado:", nrow(dataset), "filas,", ncol(dataset), "columnas\n")

  # Catastrophe Analysis
  cat("Aplicando Catastrophe Analysis para foto_mes 202006...\n")
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

  # ===================================================================
  # Feature Engineering INTRA-MES: DESHABILITADO
  # ===================================================================

  cat("\n⚠️ Feature Engineering INTRA-MES: DESHABILITADO\n")
  cat("Esta versión usa ÚNICAMENTE features históricas (entre meses)\n\n")

  # ===================================================================
  # Feature Engineering HISTÓRICO AVANZADO
  # ===================================================================

  cat("\n========================================\n")
  cat("INICIANDO FEATURE ENGINEERING HISTÓRICO AVANZADO\n")
  cat("========================================\n\n")

  # Ordenar dataset por cliente y mes
  setorder(dataset, numero_de_cliente, foto_mes)

  # Definir columnas a laguear
  cols_lagueables <- copy(setdiff(
    colnames(dataset),
    c("numero_de_cliente", "foto_mes", "clase_ternaria")
  ))

  cat("Número de variables base para FE histórico:", length(cols_lagueables), "\n\n")

  # ===================================================================
  # TÉCNICA 1: LAGS MÚLTIPLES
  # ===================================================================

  cat("[1/6] Generando LAGS MÚLTIPLES (1, 2, 3 y 6 meses)...\n")

  dataset[,
          paste0(cols_lagueables, "_lag1") := shift(.SD, 1, NA, "lag"),
          by = numero_de_cliente,
          .SDcols = cols_lagueables
  ]

  dataset[,
          paste0(cols_lagueables, "_lag2") := shift(.SD, 2, NA, "lag"),
          by = numero_de_cliente,
          .SDcols = cols_lagueables
  ]

  dataset[,
          paste0(cols_lagueables, "_lag3") := shift(.SD, 3, NA, "lag"),
          by = numero_de_cliente,
          .SDcols = cols_lagueables
  ]

  dataset[,
          paste0(cols_lagueables, "_lag6") := shift(.SD, 6, NA, "lag"),
          by = numero_de_cliente,
          .SDcols = cols_lagueables
  ]

  cat("   ✓ Lags generados: ", length(cols_lagueables) * 4, " nuevas variables\n\n")

  # ===================================================================
  # TÉCNICA 2: DELTAS
  # ===================================================================

  cat("[2/6] Generando DELTAS (cambios entre períodos)...\n")

  for (vcol in cols_lagueables)
  {
    dataset[, paste0(vcol, "_delta1") := get(vcol) - get(paste0(vcol, "_lag1"))]
    dataset[, paste0(vcol, "_delta2") := get(vcol) - get(paste0(vcol, "_lag2"))]
    dataset[, paste0(vcol, "_delta3") := get(vcol) - get(paste0(vcol, "_lag3"))]
  }

  cat("   ✓ Deltas generados: ", length(cols_lagueables) * 3, " nuevas variables\n\n")

  # ===================================================================
  # TÉCNICA 3: VENTANAS MÓVILES
  # ===================================================================

  cat("[3/6] Generando VENTANAS MÓVILES (rolling stats)...\n")
  cat("   Esto puede tardar varios minutos...\n")

  for (vcol in cols_lagueables)
  {
    # Ventana de 3 meses
    dataset[, paste0(vcol, "_roll3_mean") := frollmean(
      x = get(vcol),
      n = 3,
      align = "right",
      na.rm = TRUE
    ), by = numero_de_cliente]

    dataset[, paste0(vcol, "_roll3_max") := frollapply(
      x = get(vcol),
      n = 3,
      FUN = max,
      align = "right",
      na.rm = TRUE
    ), by = numero_de_cliente]

    dataset[, paste0(vcol, "_roll3_min") := frollapply(
      x = get(vcol),
      n = 3,
      FUN = min,
      align = "right",
      na.rm = TRUE
    ), by = numero_de_cliente]

    dataset[, paste0(vcol, "_roll3_sd") := frollapply(
      x = get(vcol),
      n = 3,
      FUN = sd,
      align = "right",
      na.rm = TRUE
    ), by = numero_de_cliente]

    # Ventana de 6 meses
    dataset[, paste0(vcol, "_roll6_mean") := frollmean(
      x = get(vcol),
      n = 6,
      align = "right",
      na.rm = TRUE
    ), by = numero_de_cliente]

    dataset[, paste0(vcol, "_roll6_sd") := frollapply(
      x = get(vcol),
      n = 6,
      FUN = sd,
      align = "right",
      na.rm = TRUE
    ), by = numero_de_cliente]
  }

  cat("   ✓ Ventanas móviles generadas: ", length(cols_lagueables) * 6, " nuevas variables\n\n")

  # ===================================================================
  # TÉCNICA 4: TENDENCIAS
  # ===================================================================

  cat("[4/6] Generando TENDENCIAS (slopes)...\n")

  calc_slope <- function(y) {
    if (all(is.na(y))) return(NA)
    x <- 1:length(y)
    valid <- !is.na(y)
    if (sum(valid) < 2) return(NA)
    tryCatch({
      coef(lm(y[valid] ~ x[valid]))[2]
    }, error = function(e) NA)
  }

  for (vcol in cols_lagueables)
  {
    dataset[, paste0(vcol, "_trend3") := frollapply(
      x = get(vcol),
      n = 3,
      FUN = calc_slope,
      align = "right"
    ), by = numero_de_cliente]

    dataset[, paste0(vcol, "_trend6") := frollapply(
      x = get(vcol),
      n = 6,
      FUN = calc_slope,
      align = "right"
    ), by = numero_de_cliente]
  }

  cat("   ✓ Tendencias generadas: ", length(cols_lagueables) * 2, " nuevas variables\n\n")

  # ===================================================================
  # TÉCNICA 5: RATIOS HISTÓRICOS
  # ===================================================================

  cat("[5/6] Generando RATIOS HISTÓRICOS...\n")

  for (vcol in cols_lagueables)
  {
    dataset[, paste0(vcol, "_ratio_vs_roll3") :=
              ifelse(get(paste0(vcol, "_roll3_mean")) != 0,
                     get(vcol) / get(paste0(vcol, "_roll3_mean")),
                     NA)]

    dataset[, paste0(vcol, "_ratio_vs_roll6") :=
              ifelse(get(paste0(vcol, "_roll6_mean")) != 0,
                     get(vcol) / get(paste0(vcol, "_roll6_mean")),
                     NA)]

    dataset[, paste0(vcol, "_ratio_vs_lag6") :=
              ifelse(get(paste0(vcol, "_lag6")) != 0,
                     get(vcol) / get(paste0(vcol, "_lag6")),
                     NA)]
  }

  cat("   ✓ Ratios generados: ", length(cols_lagueables) * 3, " nuevas variables\n\n")

  # ===================================================================
  # TÉCNICA 6: VOLATILIDAD Y ESTABILIDAD
  # ===================================================================

  cat("[6/6] Generando métricas de VOLATILIDAD Y ESTABILIDAD...\n")

  for (vcol in cols_lagueables)
  {
    dataset[, paste0(vcol, "_cv3") :=
              ifelse(get(paste0(vcol, "_roll3_mean")) != 0,
                     get(paste0(vcol, "_roll3_sd")) / abs(get(paste0(vcol, "_roll3_mean"))),
                     NA)]

    dataset[, paste0(vcol, "_cv6") :=
              ifelse(get(paste0(vcol, "_roll6_mean")) != 0,
                     get(paste0(vcol, "_roll6_sd")) / abs(get(paste0(vcol, "_roll6_mean"))),
                     NA)]

    dataset[, paste0(vcol, "_range3") :=
              get(paste0(vcol, "_roll3_max")) - get(paste0(vcol, "_roll3_min"))]
  }

  cat("   ✓ Métricas de volatilidad generadas: ", length(cols_lagueables) * 3, " nuevas variables\n\n")

  # ===================================================================
  # RESUMEN DE FEATURE ENGINEERING
  # ===================================================================

  total_features_nuevas <- length(cols_lagueables) * (4 + 3 + 6 + 2 + 3 + 3)

  cat("\n========================================\n")
  cat("RESUMEN FEATURE ENGINEERING HISTÓRICO\n")
  cat("========================================\n")
  cat("Variables base:", length(cols_lagueables), "\n")
  cat("\nNuevas features generadas:\n")
  cat("  - Lags (1,2,3,6):", length(cols_lagueables) * 4, "\n")
  cat("  - Deltas (1,2,3):", length(cols_lagueables) * 3, "\n")
  cat("  - Rolling stats:", length(cols_lagueables) * 6, "\n")
  cat("  - Tendencias:", length(cols_lagueables) * 2, "\n")
  cat("  - Ratios:", length(cols_lagueables) * 3, "\n")
  cat("  - Volatilidad:", length(cols_lagueables) * 3, "\n")
  cat("\nTOTAL NUEVAS FEATURES:", total_features_nuevas, "\n")
  cat("TOTAL FEATURES EN DATASET:", ncol(dataset), "\n")
  cat("\n⚠️ NO se incluyeron features intra-mes\n")
  cat("========================================\n\n")

  # ===================================================================
  # Modelado - Training Strategy
  # ===================================================================

  cat("Configurando Training Strategy...\n")

  PARAM$trainingstrategy <- list()
  PARAM$trainingstrategy$validate <- c(202105)

  PARAM$trainingstrategy$training <- c(
    202104, 202103, 202102, 202101,
    202012, 202011, 202010, 202009, 202008, 202007,
    202006, 202005
  )

  PARAM$trainingstrategy$training_pct <- 1.0
  PARAM$trainingstrategy$positivos <- c("BAJA+1", "BAJA+2")

  dataset[, clase01 := ifelse(clase_ternaria %in% PARAM$trainingstrategy$positivos, 1, 0)]

  campos_buenos <- copy(setdiff(
    colnames(dataset), c("clase_ternaria", "clase01", "azar")
  ))

  set.seed(PARAM$semilla_primigenia, kind = "L'Ecuyer-CMRG")
  dataset[, azar := runif(nrow(dataset))]

  dataset[, fold_train := foto_mes %in% PARAM$trainingstrategy$training &
            (clase_ternaria %in% c("BAJA+1", "BAJA+2") |
               azar < PARAM$trainingstrategy$training_pct)]

  cat("Features para modelado:", length(campos_buenos), "\n")
  cat("Registros de entrenamiento:", sum(dataset$fold_train), "\n")
  cat("Registros de validación:", sum(dataset$foto_mes %in% PARAM$trainingstrategy$validate), "\n\n")

  # ===================================================================
  # LightGBM
  # ===================================================================

  if (!require("lightgbm")) {
    cat("Instalando LightGBM...\n")
    install.packages("lightgbm")
    require("lightgbm")
  }

  cat("Creando datasets de LightGBM...\n")

  dtrain <- lgb.Dataset(
    data = data.matrix(dataset[fold_train == TRUE, campos_buenos, with = FALSE]),
    label = dataset[fold_train == TRUE, clase01],
    free_raw_data = TRUE
  )

  dvalidate <- lgb.Dataset(
    data = data.matrix(dataset[foto_mes %in% PARAM$trainingstrategy$validate, campos_buenos, with = FALSE]),
    label = dataset[foto_mes %in% PARAM$trainingstrategy$validate, clase01],
    free_raw_data = TRUE
  )

  # ===================================================================
  # Hyperparameter Tuning
  # ===================================================================

  cat("\nIniciando Hyperparameter Tuning...\n")

  if (!require("DiceKriging")) {
    install.packages("DiceKriging")
    require("DiceKriging")
  }

  if (!require("mlrMBO")) {
    install.packages("mlrMBO")
    require("mlrMBO")
  }

  PARAM$hipeparametertuning <- list()
  PARAM$hipeparametertuning$num_interations <- 10  # Cambia a 3 para pruebas rápidas
  PARAM$lgbm <- list()

  PARAM$lgbm$param_fijos <- list(
    objective = "binary",
    metric = "auc",
    first_metric_only = TRUE,
    boost_from_average = TRUE,
    feature_pre_filter = FALSE,
    verbosity = -100,
    force_row_wise = TRUE,
    seed = PARAM$semilla_primigenia,
    max_bin = 31,
    learning_rate = 0.03,
    feature_fraction = 0.5,
    num_iterations = 2048,
    early_stopping_rounds = 200
  )

  PARAM$hipeparametertuning$hs <- makeParamSet(
    makeIntegerParam("num_leaves", lower = 2L, upper = 256L),
    makeIntegerParam("min_data_in_leaf", lower = 2L, upper = 8192L)
  )

  EstimarGanancia_AUC_lightgbm <- function(x) {

    param_completo <- modifyList(PARAM$lgbm$param_fijos, x)

    modelo_train <- lgb.train(
      data = dtrain,
      valids = list(valid = dvalidate),
      eval = "auc",
      param = param_completo,
      verbose = -100
    )

    AUC <- modelo_train$record_evals$valid$auc$eval[[modelo_train$best_iter]]
    attr(AUC, "extras") <- list("num_iterations" = modelo_train$best_iter)

    rm(modelo_train)
    gc(full = TRUE, verbose = FALSE)

    message(format(Sys.time(), "%a %b %d %X %Y"), " AUC ", AUC)

    return(AUC)
  }

  configureMlr(show.learner.output = FALSE)

  obj.fun <- makeSingleObjectiveFunction(
    fn = EstimarGanancia_AUC_lightgbm,
    minimize = FALSE,
    noisy = FALSE,
    par.set = PARAM$hipeparametertuning$hs,
    has.simple.signature = FALSE
  )

  ctrl <- makeMBOControl(
    save.on.disk.at.time = 600,
    save.file.path = "HT.RDATA"
  )

  ctrl <- setMBOControlTermination(
    ctrl,
    iters = PARAM$hipeparametertuning$num_interations
  )

  ctrl <- setMBOControlInfill(ctrl, crit = makeMBOInfillCritEI())

  surr.km <- makeLearner(
    "regr.km",
    predict.type = "se",
    covtype = "matern3_2",
    control = list(trace = TRUE)
  )

  if (!file.exists("HT.RDATA")) {
    bayesiana_salida <- mbo(obj.fun, learner = surr.km, control = ctrl)
  } else {
    bayesiana_salida <- mboContinue("HT.RDATA")
  }

  tb_bayesiana <- as.data.table(bayesiana_salida$opt.path)
  setorder(tb_bayesiana, -y, -num_iterations)

  fwrite(tb_bayesiana,
         file = "BO_log.txt",
         sep = "\t"
  )

  PARAM$out$lgbm$mejores_hiperparametros <- tb_bayesiana[
    1,
    setdiff(colnames(tb_bayesiana),
            c("y", "dob", "eol", "error.message", "exec.time", "ei", "error.model",
              "train.time", "prop.type", "propose.time", "se", "mean", "iter")),
    with = FALSE
  ]

  cat("\nMejores hiperparámetros:\n")
  print(PARAM$out$lgbm$mejores_hiperparametros)

  # ===================================================================
  # Producción - Modelo Final
  # ===================================================================

  cat("\nEntrenando modelo final...\n")

  PARAM$trainingstrategy$final_train <- c(
    202105, 202104, 202103, 202102, 202101,
    202012, 202011, 202010, 202009, 202008, 202007,
    202006, 202005
  )

  dataset[, fold_final_train := foto_mes %in% PARAM$trainingstrategy$final_train]

  dfinal_train <- lgb.Dataset(
    data = data.matrix(dataset[fold_final_train == TRUE, campos_buenos, with = FALSE]),
    label = dataset[fold_final_train == TRUE, clase01],
    free_raw_data = TRUE
  )

  fijos <- copy(PARAM$lgbm$param_fijos)
  fijos$num_iterations <- NULL
  fijos$early_stopping_rounds <- NULL

  param_final <- c(fijos, PARAM$out$lgbm$mejores_hiperparametros)

  final_model <- lgb.train(
    data = dfinal_train,
    param = param_final,
    verbose = -100
  )

  lgb.save(final_model, "modelo.txt")

  tb_importancia <- as.data.table(lgb.importance(final_model))
  fwrite(tb_importancia,
         file = "impo.txt",
         sep = "\t"
  )

  cat("Modelo guardado. Top 10 features más importantes:\n")
  print(head(tb_importancia, 10))

  # ===================================================================
  # Scoring
  # ===================================================================

  cat("\nGenerando predicciones...\n")

  PARAM$trainingstrategy$future <- c(202107)
  dfuture <- dataset[foto_mes %in% PARAM$trainingstrategy$future]

  prediccion <- predict(
    final_model,
    data.matrix(dfuture[, campos_buenos, with = FALSE])
  )

  tb_prediccion <- dfuture[, list(numero_de_cliente)]
  tb_prediccion[, prob := prediccion]

  fwrite(tb_prediccion,
         file = "prediccion.txt",
         sep = "\t"
  )

  # ===================================================================
  # Curva de Ganancia
  # ===================================================================

  tb_prediccion[, clase_ternaria := dfuture$clase_ternaria]
  tb_prediccion[, ganancia := -3000.0]
  tb_prediccion[clase_ternaria == "BAJA+2", ganancia := 117000.0]

  setorder(tb_prediccion, -prob)
  tb_prediccion[, gan_acum := cumsum(ganancia)]

  tb_prediccion[,
                gan_suavizada := frollmean(
                  x = gan_acum,
                  n = 400,
                  align = "center",
                  na.rm = TRUE,
                  hasNA = TRUE
                )
  ]

  resultado <- list()
  resultado$ganancia_suavizada_max <- max(tb_prediccion$gan_suavizada, na.rm = TRUE)
  options(digits = 8)
  resultado$envios <- which.max(tb_prediccion$gan_suavizada)
  resultado$semilla <- PARAM$semilla_primigenia
  resultado$seed_idx <- seed_idx

  cat("\n")
  print(resultado)

  fwrite(tb_prediccion,
         file = "ganancias.txt",
         sep = "\t"
  )

  tb_prediccion[, envios := .I]

  pdf("curva_de_ganancia.pdf")

  plot(
    x = tb_prediccion$envios,
    y = tb_prediccion$gan_acum,
    type = "l",
    col = "gray",
    xlim = c(0, 6000),
    ylim = c(0, 8000000),
    main = paste0("Seed ", seed_idx, " (SOLO FE Hist) - Gan= ", as.integer(resultado$ganancia_suavizada_max), " envios= ", resultado$envios),
    xlab = "Envios",
    ylab = "Ganancia",
    panel.first = grid()
  )

  dev.off()

  if (!require("yaml")) {
    install.packages("yaml")
    require("yaml")
  }

  PARAM$resultado <- resultado

  write_yaml(PARAM, file = "PARAM.yml")

  # ===================================================================
  # Guardar resultado y limpiar
  # ===================================================================

  if (!exists("resultados_totales")) resultados_totales <- list()
  resultados_totales[[seed_idx]] <- resultado

  rm(dataset, dtrain, dvalidate, dfinal_train, final_model, tb_prediccion)
  gc(full = TRUE, verbose = FALSE)

  cat("\n========================================\n")
  cat("Semilla ", seed_idx, " completada exitosamente\n")
  cat("Ganancia: ", resultado$ganancia_suavizada_max, "\n")
  cat("Envíos: ", resultado$envios, "\n")
  cat("========================================\n\n")

} # Fin del loop sobre las semillas

cat("\n\n***************************************\n")
cat("TODAS LAS SEMILLAS PROCESADAS\n")
cat("***************************************\n")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

setwd(EXP_DIR)

tb_resumen <- data.table(
  seed_idx = sapply(resultados_totales, function(x) x$seed_idx),
  semilla = sapply(resultados_totales, function(x) x$semilla),
  ganancia = sapply(resultados_totales, function(x) x$ganancia_suavizada_max),
  envios = sapply(resultados_totales, function(x) x$envios)
)

tb_resumen[, rank := rank(-ganancia)]

cat("\n\n========================================\n")
cat("RESUMEN FINAL DE LAS SEMILLAS\n")
cat("(SOLO FEATURE ENGINEERING HISTÓRICO)\n")
cat("========================================\n\n")
print(tb_resumen)

cat("\nESTADÍSTICAS:\n")
cat("Ganancia promedio: ", mean(tb_resumen$ganancia), "\n")
cat("Ganancia máxima: ", max(tb_resumen$ganancia), "\n")
cat("Ganancia mínima: ", min(tb_resumen$ganancia), "\n")
cat("Desviación estándar: ", sd(tb_resumen$ganancia), "\n")
cat("Coeficiente de variación: ", sd(tb_resumen$ganancia) / mean(tb_resumen$ganancia) * 100, "%\n")
cat("Mejor semilla: ", tb_resumen[rank == 1, semilla], " (seed_idx ", tb_resumen[rank == 1, seed_idx], ")\n")

fwrite(tb_resumen,
       file = paste0("resumen_seeds_exp", PARAM_GLOBAL$experimento_base, "_FE_historico_SOLO.txt"),
       sep = "\t"
)

saveRDS(resultados_totales,
        file = paste0("resultados_completos_exp", PARAM_GLOBAL$experimento_base, "_FE_historico_SOLO.rds")
)

cat("\nArchivos guardados en:", EXP_DIR, "\n")

cat("\n========================================\n")
cat("WORKFLOW COMPLETADO\n")
cat("========================================\n")

format(Sys.time(), "%a %b %d %X %Y")
