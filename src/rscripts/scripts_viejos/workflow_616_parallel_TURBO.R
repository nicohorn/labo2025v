# ============================================================================
# WORKFLOW 616 - EJECUCIÓN PARALELA TURBO (PC ALTA GAMA)
# ============================================================================
# Versión optimizada para PCs con muchos cores (16+) y RAM abundante (32+ GB)
#
# CONFIGURACIÓN ESPECÍFICA:
# - CPU: 20 cores
# - RAM: 64 GB
# - Estrategia: Máxima paralelización
#
# OPTIMIZACIONES:
# - Usa 18-19 cores (deja 1-2 para el sistema)
# - Buffer de memoria optimizado
# - Sin límites artificiales
# ============================================================================

format(Sys.time(), "%a %b %d %X %Y")

cat("\n========================================\n")
cat("🚀 WORKFLOW PARALELO TURBO MODE 🚀\n")
cat("========================================\n\n")

# ============================================================================
# CONFIGURACIÓN PARA PC ALTA GAMA
# ============================================================================

BASE_DIR <- "C:/Users/User/Documents/labo2025v"
DATASETS_DIR <- file.path(BASE_DIR, "datasets")
EXP_DIR <- file.path(BASE_DIR, "exp")

# CONFIGURACIÓN AGRESIVA DE CORES
# Con 20 cores físicos, usa 18 (deja 2 para el sistema)
NUM_CORES <- 18

cat("========================================\n")
cat("CONFIGURACIÓN HARDWARE\n")
cat("========================================\n")
cat("Cores totales detectados:", parallel::detectCores(), "\n")
cat("Cores a usar:", NUM_CORES, "\n")
cat("Cores libres para sistema:", parallel::detectCores() - NUM_CORES, "\n")
cat("RAM total: ~64 GB\n")
cat("RAM estimada por proceso: ~4-5 GB\n")
cat("RAM total estimada:", NUM_CORES * 5, "GB (", round((NUM_CORES * 5)/64 * 100, 1), "% de tu RAM)\n")
cat("========================================\n\n")

# Configuración del experimento
PARAM_GLOBAL <- list()
PARAM_GLOBAL$experimento_base <- 6160
PARAM_GLOBAL$dataset <- "gerencial_competencia_2025.csv.gz"

# TODAS LAS SEMILLAS EN PARALELO
PARAM_GLOBAL$semillas <- c(102191, 200207, 300313, 400419, 500523)

cat("Experimento base:", PARAM_GLOBAL$experimento_base, "\n")
cat("Dataset:", PARAM_GLOBAL$dataset, "\n")
cat("Semillas a procesar:", length(PARAM_GLOBAL$semillas), "\n")
cat("Todas las semillas correrán SIMULTÁNEAMENTE\n\n")

# ============================================================================
# VERIFICACIONES PREVIAS
# ============================================================================

cat("Verificando entorno...\n")

dataset_path <- file.path(DATASETS_DIR, PARAM_GLOBAL$dataset)
if (!file.exists(dataset_path)) {
  stop("ERROR: Dataset no encontrado en: ", dataset_path)
}
cat("✓ Dataset encontrado\n")

required_packages <- c("data.table", "lightgbm", "mlrMBO", "DiceKriging", "parallel", "yaml", "R.utils")
missing_packages <- c()

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    missing_packages <- c(missing_packages, pkg)
  }
}

if (length(missing_packages) > 0) {
  stop("ERROR: Faltan paquetes: ", paste(missing_packages, collapse = ", "))
}
cat("✓ Todos los paquetes instalados\n")

cat("\n========================================\n")
cat("ESTIMACIÓN DE TIEMPO\n")
cat("========================================\n")
cat("Tiempo estimado con tu hardware:\n")
cat("  - Por semilla (secuencial): ~2 horas\n")
cat("  - Total secuencial: ~10 horas\n")
cat("  - Total PARALELO (", NUM_CORES, " cores): ~2-2.5 horas ⚡\n")
cat("  - AHORRO: ~8 horas (80%)\n")
cat("========================================\n\n")

readline("Presiona ENTER para iniciar la ejecución TURBO... ")

cat("\n🚀 INICIANDO MODO TURBO 🚀\n\n")

# ============================================================================
# FUNCIÓN PARA EJECUTAR UNA SEMILLA (OPTIMIZADA)
# ============================================================================

ejecutar_semilla <- function(seed_idx, semilla, PARAM_GLOBAL, BASE_DIR, DATASETS_DIR, EXP_DIR) {

  # Cargar librerías en el worker
  suppressPackageStartupMessages({
    require("data.table")
    require("R.utils")
    require("lightgbm")
    require("mlrMBO")
    require("DiceKriging")
    require("mlr")
    require("ParamHelpers")
    require("yaml")
  })

  # Configurar threads de data.table para no saturar
  # Con 18 procesos paralelos, cada uno usa 1 thread
  setDTthreads(1)

  PARAM <- list()
  PARAM$semilla_primigenia <- semilla
  PARAM$experimento <- PARAM_GLOBAL$experimento_base + seed_idx - 1
  PARAM$dataset <- PARAM_GLOBAL$dataset
  PARAM$out <- list()
  PARAM$out$lgbm <- list()

  inicio <- Sys.time()
  cat("\n🔥 [Semilla", seed_idx, "] INICIANDO -", format(inicio), "\n")

  tryCatch({

    # ===================================================================
    # Setup
    # ===================================================================

    experimento_folder <- paste0("WF", PARAM$experimento, "_seed", seed_idx, "_FE_historico_SOLO_TURBO")
    experimento_path <- file.path(EXP_DIR, experimento_folder)

    dir.create(experimento_path, showWarnings = FALSE, recursive = TRUE)
    setwd(experimento_path)

    # ===================================================================
    # Cargar Dataset
    # ===================================================================

    cat("[Semilla", seed_idx, "] Cargando dataset...\n")
    dataset_path <- file.path(DATASETS_DIR, PARAM$dataset)
    dataset <- fread(dataset_path, showProgress = FALSE)

    # Catastrophe Analysis
    dataset[foto_mes == 202006, `:=`(
      internet = NA, mrentabilidad = NA, mrentabilidad_annual = NA,
      mcomisiones = NA, mactivos_margen = NA, mpasivos_margen = NA,
      mcuentas_saldo = NA, ctarjeta_visa_transacciones = NA,
      mtarjeta_visa_consumo = NA, mtarjeta_master_consumo = NA,
      ccallcenter_transacciones = NA, chomebanking_transacciones = NA
    )]

    # ===================================================================
    # Feature Engineering Histórico Completo
    # ===================================================================

    cat("[Semilla", seed_idx, "] FE Histórico...\n")

    setorder(dataset, numero_de_cliente, foto_mes)

    cols_lagueables <- setdiff(
      colnames(dataset),
      c("numero_de_cliente", "foto_mes", "clase_ternaria")
    )

    # LAGS
    for (lag in c(1, 2, 3, 6)) {
      dataset[,
              paste0(cols_lagueables, "_lag", lag) := shift(.SD, lag, NA, "lag"),
              by = numero_de_cliente,
              .SDcols = cols_lagueables]
    }

    # DELTAS
    for (vcol in cols_lagueables) {
      dataset[, paste0(vcol, "_delta1") := get(vcol) - get(paste0(vcol, "_lag1"))]
      dataset[, paste0(vcol, "_delta2") := get(vcol) - get(paste0(vcol, "_lag2"))]
      dataset[, paste0(vcol, "_delta3") := get(vcol) - get(paste0(vcol, "_lag3"))]
    }

    # ROLLING STATS
    for (vcol in cols_lagueables) {
      dataset[, paste0(vcol, "_roll3_mean") := frollmean(get(vcol), 3, align = "right", na.rm = TRUE),
              by = numero_de_cliente]
      dataset[, paste0(vcol, "_roll3_max") := frollapply(get(vcol), 3, max, align = "right", na.rm = TRUE),
              by = numero_de_cliente]
      dataset[, paste0(vcol, "_roll3_min") := frollapply(get(vcol), 3, min, align = "right", na.rm = TRUE),
              by = numero_de_cliente]
      dataset[, paste0(vcol, "_roll3_sd") := frollapply(get(vcol), 3, sd, align = "right", na.rm = TRUE),
              by = numero_de_cliente]
      dataset[, paste0(vcol, "_roll6_mean") := frollmean(get(vcol), 6, align = "right", na.rm = TRUE),
              by = numero_de_cliente]
      dataset[, paste0(vcol, "_roll6_sd") := frollapply(get(vcol), 6, sd, align = "right", na.rm = TRUE),
              by = numero_de_cliente]
    }

    # TENDENCIAS
    calc_slope <- function(y) {
      if (all(is.na(y))) return(NA)
      x <- 1:length(y)
      valid <- !is.na(y)
      if (sum(valid) < 2) return(NA)
      tryCatch(coef(lm(y[valid] ~ x[valid]))[2], error = function(e) NA)
    }

    for (vcol in cols_lagueables) {
      dataset[, paste0(vcol, "_trend3") := frollapply(get(vcol), 3, calc_slope, align = "right"),
              by = numero_de_cliente]
      dataset[, paste0(vcol, "_trend6") := frollapply(get(vcol), 6, calc_slope, align = "right"),
              by = numero_de_cliente]
    }

    # RATIOS
    for (vcol in cols_lagueables) {
      dataset[, paste0(vcol, "_ratio_vs_roll3") :=
                ifelse(get(paste0(vcol, "_roll3_mean")) != 0,
                       get(vcol) / get(paste0(vcol, "_roll3_mean")), NA)]
      dataset[, paste0(vcol, "_ratio_vs_roll6") :=
                ifelse(get(paste0(vcol, "_roll6_mean")) != 0,
                       get(vcol) / get(paste0(vcol, "_roll6_mean")), NA)]
      dataset[, paste0(vcol, "_ratio_vs_lag6") :=
                ifelse(get(paste0(vcol, "_lag6")) != 0,
                       get(vcol) / get(paste0(vcol, "_lag6")), NA)]
    }

    # VOLATILIDAD
    for (vcol in cols_lagueables) {
      dataset[, paste0(vcol, "_cv3") :=
                ifelse(get(paste0(vcol, "_roll3_mean")) != 0,
                       get(paste0(vcol, "_roll3_sd")) / abs(get(paste0(vcol, "_roll3_mean"))), NA)]
      dataset[, paste0(vcol, "_cv6") :=
                ifelse(get(paste0(vcol, "_roll6_mean")) != 0,
                       get(paste0(vcol, "_roll6_sd")) / abs(get(paste0(vcol, "_roll6_mean"))), NA)]
      dataset[, paste0(vcol, "_range3") :=
                get(paste0(vcol, "_roll3_max")) - get(paste0(vcol, "_roll3_min"))]
    }

    cat("[Semilla", seed_idx, "] ✓ FE completado -", ncol(dataset), "features\n")

    # ===================================================================
    # Training Strategy
    # ===================================================================

    PARAM$trainingstrategy <- list(
      validate = c(202105),
      training = c(202104, 202103, 202102, 202101, 202012, 202011,
                   202010, 202009, 202008, 202007, 202006, 202005),
      training_pct = 1.0,
      positivos = c("BAJA+1", "BAJA+2")
    )

    dataset[, clase01 := ifelse(clase_ternaria %in% PARAM$trainingstrategy$positivos, 1, 0)]

    campos_buenos <- setdiff(colnames(dataset), c("clase_ternaria", "clase01", "azar"))

    set.seed(PARAM$semilla_primigenia, kind = "L'Ecuyer-CMRG")
    dataset[, azar := runif(nrow(dataset))]

    dataset[, fold_train := foto_mes %in% PARAM$trainingstrategy$training &
              (clase_ternaria %in% PARAM$trainingstrategy$positivos |
                 azar < PARAM$trainingstrategy$training_pct)]

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

    cat("[Semilla", seed_idx, "] Hyperparameter Tuning...\n")

    PARAM$hipeparametertuning <- list(num_interations = 10)
    PARAM$lgbm <- list(
      param_fijos = list(
        objective = "binary", metric = "auc", first_metric_only = TRUE,
        boost_from_average = TRUE, feature_pre_filter = FALSE,
        verbosity = -100, force_row_wise = TRUE,
        seed = PARAM$semilla_primigenia, max_bin = 31,
        learning_rate = 0.03, feature_fraction = 0.5,
        num_iterations = 2048, early_stopping_rounds = 200,
        num_threads = 1  # 1 thread por proceso para no saturar
      )
    )

    PARAM$hipeparametertuning$hs <- makeParamSet(
      makeIntegerParam("num_leaves", lower = 2L, upper = 256L),
      makeIntegerParam("min_data_in_leaf", lower = 2L, upper = 8192L)
    )

    EstimarGanancia_AUC_lightgbm <- function(x) {
      param_completo <- modifyList(PARAM$lgbm$param_fijos, x)
      modelo_train <- lgb.train(
        data = dtrain, valids = list(valid = dvalidate),
        eval = "auc", param = param_completo, verbose = -100
      )
      AUC <- modelo_train$record_evals$valid$auc$eval[[modelo_train$best_iter]]
      attr(AUC, "extras") <- list("num_iterations" = modelo_train$best_iter)
      rm(modelo_train)
      gc(full = TRUE, verbose = FALSE)
      return(AUC)
    }

    configureMlr(show.learner.output = FALSE)

    obj.fun <- makeSingleObjectiveFunction(
      fn = EstimarGanancia_AUC_lightgbm, minimize = FALSE,
      noisy = FALSE, par.set = PARAM$hipeparametertuning$hs,
      has.simple.signature = FALSE
    )

    ctrl <- makeMBOControl(save.on.disk.at.time = 600, save.file.path = "HT.RDATA")
    ctrl <- setMBOControlTermination(ctrl, iters = PARAM$hipeparametertuning$num_interations)
    ctrl <- setMBOControlInfill(ctrl, crit = makeMBOInfillCritEI())

    surr.km <- makeLearner("regr.km", predict.type = "se",
                           covtype = "matern3_2", control = list(trace = FALSE))

    bayesiana_salida <- mbo(obj.fun, learner = surr.km, control = ctrl)

    tb_bayesiana <- as.data.table(bayesiana_salida$opt.path)
    setorder(tb_bayesiana, -y, -num_iterations)
    fwrite(tb_bayesiana, file = "BO_log.txt", sep = "\t")

    PARAM$out$lgbm$mejores_hiperparametros <- tb_bayesiana[1,
      setdiff(colnames(tb_bayesiana),
              c("y", "dob", "eol", "error.message", "exec.time", "ei", "error.model",
                "train.time", "prop.type", "propose.time", "se", "mean", "iter")),
      with = FALSE]

    # ===================================================================
    # Modelo Final
    # ===================================================================

    cat("[Semilla", seed_idx, "] Entrenando modelo final...\n")

    PARAM$trainingstrategy$final_train <- c(202105, 202104, 202103, 202102, 202101,
                                            202012, 202011, 202010, 202009, 202008,
                                            202007, 202006, 202005)

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

    final_model <- lgb.train(data = dfinal_train, param = param_final, verbose = -100)

    lgb.save(final_model, "modelo.txt")

    tb_importancia <- as.data.table(lgb.importance(final_model))
    fwrite(tb_importancia, file = "impo.txt", sep = "\t")

    # ===================================================================
    # Scoring y Ganancia
    # ===================================================================

    PARAM$trainingstrategy$future <- c(202107)
    dfuture <- dataset[foto_mes %in% PARAM$trainingstrategy$future]

    prediccion <- predict(final_model, data.matrix(dfuture[, campos_buenos, with = FALSE]))

    tb_prediccion <- dfuture[, list(numero_de_cliente)]
    tb_prediccion[, prob := prediccion]
    fwrite(tb_prediccion, file = "prediccion.txt", sep = "\t")

    tb_prediccion[, clase_ternaria := dfuture$clase_ternaria]
    tb_prediccion[, ganancia := -3000.0]
    tb_prediccion[clase_ternaria == "BAJA+2", ganancia := 117000.0]

    setorder(tb_prediccion, -prob)
    tb_prediccion[, gan_acum := cumsum(ganancia)]
    tb_prediccion[, gan_suavizada := frollmean(gan_acum, 400, align = "center",
                                                na.rm = TRUE, hasNA = TRUE)]

    resultado <- list(
      ganancia_suavizada_max = max(tb_prediccion$gan_suavizada, na.rm = TRUE),
      envios = which.max(tb_prediccion$gan_suavizada),
      semilla = PARAM$semilla_primigenia,
      seed_idx = seed_idx
    )

    fwrite(tb_prediccion, file = "ganancias.txt", sep = "\t")

    tb_prediccion[, envios := .I]

    pdf("curva_de_ganancia.pdf")
    plot(x = tb_prediccion$envios, y = tb_prediccion$gan_acum,
         type = "l", col = "gray", xlim = c(0, 6000), ylim = c(0, 8000000),
         main = paste0("Seed ", seed_idx, " TURBO - Gan= ", as.integer(resultado$ganancia_suavizada_max)),
         xlab = "Envios", ylab = "Ganancia", panel.first = grid())
    dev.off()

    PARAM$resultado <- resultado
    write_yaml(PARAM, file = "PARAM.yml")

    rm(dataset, dtrain, dvalidate, dfinal_train, final_model, tb_prediccion)
    gc(full = TRUE, verbose = FALSE)

    fin <- Sys.time()
    duracion <- as.numeric(difftime(fin, inicio, units = "mins"))

    cat("\n✅ [Semilla", seed_idx, "] COMPLETADA en", round(duracion, 1), "min\n")
    cat("   Ganancia:", resultado$ganancia_suavizada_max, "\n")
    cat("   Envíos:", resultado$envios, "\n\n")

    return(list(
      success = TRUE, seed_idx = seed_idx, semilla = semilla,
      ganancia = resultado$ganancia_suavizada_max,
      envios = resultado$envios, duracion_min = duracion
    ))

  }, error = function(e) {
    cat("\n❌ [Semilla", seed_idx, "] ERROR:", e$message, "\n")
    return(list(success = FALSE, seed_idx = seed_idx, semilla = semilla, error = e$message))
  })
}

# ============================================================================
# EJECUCIÓN EN PARALELO
# ============================================================================

require("parallel")

cat("\n🚀 Creando cluster con", NUM_CORES, "cores...\n")
cl <- makeCluster(NUM_CORES)

clusterExport(cl, c("PARAM_GLOBAL", "BASE_DIR", "DATASETS_DIR", "EXP_DIR"))

inicio_total <- Sys.time()

cat("\n⚡ TODAS LAS SEMILLAS CORRIENDO EN PARALELO ⚡\n")
cat("Hora de inicio:", format(inicio_total), "\n")
cat("Hora estimada de finalización:", format(inicio_total + as.difftime(2.5, units = "hours")), "\n\n")

resultados <- parLapply(cl, 1:length(PARAM_GLOBAL$semillas), function(i) {
  ejecutar_semilla(i, PARAM_GLOBAL$semillas[i], PARAM_GLOBAL, BASE_DIR, DATASETS_DIR, EXP_DIR)
})

stopCluster(cl)

fin_total <- Sys.time()
duracion_total <- as.numeric(difftime(fin_total, inicio_total, units = "hours"))

# ============================================================================
# RESUMEN FINAL
# ============================================================================

cat("\n\n========================================\n")
cat("🏁 TURBO MODE COMPLETADO 🏁\n")
cat("========================================\n\n")

cat("⏱  Tiempo total:", round(duracion_total, 2), "horas\n")
cat("⏱  Tiempo ahorrado vs secuencial: ~", round(10 - duracion_total, 1), "horas\n\n")

resultados_exitosos <- Filter(function(x) x$success, resultados)
resultados_fallidos <- Filter(function(x) !x$success, resultados)

if (length(resultados_exitosos) > 0) {

  tb_resumen <- rbindlist(lapply(resultados_exitosos, function(r) {
    data.table(seed_idx = r$seed_idx, semilla = r$semilla,
               ganancia = r$ganancia, envios = r$envios,
               duracion_min = r$duracion_min)
  }))

  setorder(tb_resumen, -ganancia)
  tb_resumen[, rank := .I]

  cat("RESULTADOS:\n\n")
  print(tb_resumen)

  cat("\n📊 ESTADÍSTICAS:\n")
  cat("Ganancia promedio:", mean(tb_resumen$ganancia), "\n")
  cat("Ganancia máxima:", max(tb_resumen$ganancia), "⭐\n")
  cat("Ganancia mínima:", min(tb_resumen$ganancia), "\n")
  cat("Desv estándar:", sd(tb_resumen$ganancia), "\n")
  cat("🏆 Mejor semilla:", tb_resumen[rank == 1, semilla], "\n")

  setwd(EXP_DIR)
  fwrite(tb_resumen, file = paste0("resumen_TURBO_exp", PARAM_GLOBAL$experimento_base, ".txt"), sep = "\t")
  saveRDS(resultados, file = paste0("resultados_TURBO_exp", PARAM_GLOBAL$experimento_base, ".rds"))

  cat("\n📁 Archivos guardados en:", EXP_DIR, "\n")
}

if (length(resultados_fallidos) > 0) {
  cat("\n⚠ SEMILLAS FALLIDAS:\n")
  for (r in resultados_fallidos) {
    cat("  - Semilla", r$seed_idx, ":", r$error, "\n")
  }
}

cat("\n========================================\n")
cat("🎉 EJECUCIÓN TURBO FINALIZADA 🎉\n")
cat("========================================\n")

format(Sys.time(), "%a %b %d %X %Y")
