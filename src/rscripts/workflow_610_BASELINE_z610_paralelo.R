################################################################################
# WORKFLOW 610 BASELINE z610 - VERSION PARALELA PARA WINDOWS
#
# Adaptación del notebook 610_WorkFlow_01_gerencial_julio_5_semillas.ipynb
# para ejecución local en Windows con paralelización
#
# Características:
# - FE Histórico BÁSICO (lag1, lag2, delta1, delta2)
# - Lógica EXACTA del notebook z610
# - Paralelización con 5 workers
# - Logging completo
# - Health checks
# - 5 semillas
################################################################################

# === CONFIGURACIÓN GLOBAL ===
BASE_DIR <- "C:/Users/User/Documents/labo2025v"
EXP_DIR <- file.path(BASE_DIR, "exp", "exp_z610_baseline")
DATASET_PATH <- file.path(BASE_DIR, "datasets", "gerencial_competencia_2025.csv.gz")

# Crear directorio de experimento
dir.create(EXP_DIR, recursive = TRUE, showWarnings = FALSE)
setwd(EXP_DIR)

# === PARÁMETROS GLOBALES ===
PARAM_GLOBAL <- list()
PARAM_GLOBAL$experimento_base <- 6310
PARAM_GLOBAL$dataset <- "gerencial_competencia_2025.csv.gz"
PARAM_GLOBAL$semillas <- c(153929, 838969, 922081, 795581, 194609)

# === LOGGING SETUP ===
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
log_file <- file.path(EXP_DIR, paste0("workflow_z610_", timestamp, ".log"))
progress_file <- file.path(EXP_DIR, paste0("progress_", timestamp, ".txt"))
health_file <- file.path(EXP_DIR, paste0("health_", timestamp, ".txt"))

log_msg <- function(msg, level = "INFO") {
  timestamp <- format(Sys.time(), "[%Y-%m-%d %H:%M:%S]")
  full_msg <- paste(timestamp, paste0("[", level, "]"), msg)
  cat(full_msg, "\n")
  cat(full_msg, "\n", file = log_file, append = TRUE)
}

health_check <- function(msg) {
  timestamp <- format(Sys.time(), "%H:%M:%S")
  health_msg <- paste0("💊 ", timestamp, " | ", msg, "\n")
  cat(health_msg)
  cat(health_msg, file = health_file, append = TRUE)
}

seed_log <- function(msg, level = "INFO", seed_idx = NULL) {
  prefix <- if(!is.null(seed_idx)) paste0("[SEED ", seed_idx, "] ") else ""
  log_msg(paste0(prefix, msg), level)
}

# === BANNER ===
cat("\n")
cat("========================================\n")
cat(" WORKFLOW 610 BASELINE (z610 logic)\n")
cat("   Paralelo + Caching + Logging\n")
cat("========================================\n")
cat("\n")

log_msg("============================================================")
log_msg("WORKFLOW 610 BASELINE z610 INICIADO")
log_msg("============================================================")
log_msg(paste("Log file:", log_file))
log_msg(paste("Progress file:", progress_file))
log_msg(paste("Health file:", health_file))

# === CONFIGURACIÓN HARDWARE ===
num_cores <- parallel::detectCores()
num_workers <- 5  # Una por seed
threads_per_worker <- max(1, floor((num_cores - 3) / num_workers))

log_msg("CONFIGURACIÓN HARDWARE")
log_msg(paste("  Cores totales detectados:", num_cores))
log_msg(paste("  Workers paralelos:", num_workers))
log_msg(paste("  Threads por worker:", threads_per_worker))
log_msg(paste("  Cores totales en uso:", num_workers * threads_per_worker))

log_msg(paste("Semillas a ejecutar:", length(PARAM_GLOBAL$semillas)))
log_msg(paste("Semillas:", paste(PARAM_GLOBAL$semillas, collapse = ", ")))

# === CARGA DE PAQUETES ===
log_msg("Cargando paquetes...")

suppressPackageStartupMessages({
  require("data.table")
  require("lightgbm")
  require("DiceKriging")
  require("mlr")
  require("ParamHelpers")
  require("mlrMBO")
  require("smoof")
  require("checkmate")
  require("yaml")
  require("parallel")
})

setDTthreads(threads_per_worker)
log_msg(paste("✅ data.table threads configurados:", threads_per_worker), level = "SUCCESS")
log_msg("✅ Paquetes cargados correctamente", level = "SUCCESS")

health_check("Sistema iniciado, paquetes cargados")

# === VERIFICAR DATASET ===
log_msg("Verificando dataset...")
if (!file.exists(DATASET_PATH)) {
  log_msg(paste("ERROR: Dataset no encontrado en", DATASET_PATH), level = "ERROR")
  stop("Dataset no encontrado")
}
log_msg(paste("✅ Dataset encontrado:", round(file.size(DATASET_PATH) / 1024^2, 2), "MB"), level = "SUCCESS")

# ===================================================================
# FUNCIÓN PARA EJECUTAR UNA SEED
# ===================================================================

ejecutar_seed <- function(seed_idx, semilla, experimento_base) {

  seed_log(paste("Iniciando ejecución - Semilla:", semilla), seed_idx = seed_idx)
  tiempo_inicio <- Sys.time()

  # Configurar threads para este worker
  setDTthreads(threads_per_worker)

  # === PARÁMETROS DE ESTA SEED ===
  PARAM <- list()
  PARAM$semilla_primigenia <- semilla
  PARAM$experimento <- experimento_base + seed_idx - 1
  PARAM$dataset <- PARAM_GLOBAL$dataset
  PARAM$out <- list()
  PARAM$out$lgbm <- list()

  # === CARPETA DEL EXPERIMENTO ===
  experimento_folder <- paste0("WF", PARAM$experimento, "_seed", seed_idx, "_z610_baseline")
  exp_path <- file.path(EXP_DIR, experimento_folder)
  dir.create(exp_path, showWarnings = FALSE)

  seed_log(paste("Carpeta:", experimento_folder), seed_idx = seed_idx)

  # === LECTURA Y PREPROCESAMIENTO ===
  seed_log("Cargando dataset...", seed_idx = seed_idx)
  dataset <- fread(DATASET_PATH)

  seed_log(paste("Dataset cargado:", nrow(dataset), "filas,", ncol(dataset), "columnas"), seed_idx = seed_idx)

  # === CATASTROPHE ANALYSIS ===
  seed_log("Aplicando Catastrophe Analysis (202006)...", seed_idx = seed_idx)

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

  seed_log("✅ Catastrophe Analysis aplicado (13 variables)", seed_idx = seed_idx, level = "SUCCESS")

  # === FEATURE ENGINEERING INTRAMES ===
  seed_log("Feature Engineering intrames...", seed_idx = seed_idx)

  atributos_presentes <- function(patributos) {
    atributos <- unique(patributos)
    comun <- intersect(atributos, colnames(dataset))
    return(length(atributos) == length(comun))
  }

  if (atributos_presentes(c("foto_mes")))
    dataset[, kmes := foto_mes %% 100]

  if (atributos_presentes(c("mpayroll", "cliente_edad")))
    dataset[, mpayroll_sobre_edad := mpayroll / cliente_edad]

  ncol_pre_fe <- ncol(dataset)
  seed_log(paste("Variables base:", ncol_pre_fe), seed_idx = seed_idx)

  # === FEATURE ENGINEERING HISTÓRICO (BÁSICO - z610) ===
  seed_log("Feature Engineering Histórico BÁSICO (lag1, lag2, delta1, delta2)...", seed_idx = seed_idx)

  cols_lagueables <- copy(setdiff(
    colnames(dataset),
    c("numero_de_cliente", "foto_mes", "clase_ternaria")
  ))

  # Lag 1
  dataset[,
    paste0(cols_lagueables, "_lag1") := shift(.SD, 1, NA, "lag"),
    by = numero_de_cliente,
    .SDcols = cols_lagueables
  ]

  # Lag 2
  dataset[,
    paste0(cols_lagueables, "_lag2") := shift(.SD, 2, NA, "lag"),
    by = numero_de_cliente,
    .SDcols = cols_lagueables
  ]

  # Deltas
  for (vcol in cols_lagueables) {
    dataset[, paste0(vcol, "_delta1") := get(vcol) - get(paste0(vcol, "_lag1"))]
    dataset[, paste0(vcol, "_delta2") := get(vcol) - get(paste0(vcol, "_lag2"))]
  }

  ncol_post_fe <- ncol(dataset)
  seed_log(paste("✅ FE Histórico completado:", ncol_post_fe - ncol_pre_fe, "nuevas features"), seed_idx = seed_idx, level = "SUCCESS")
  seed_log(paste("Total features:", ncol_post_fe), seed_idx = seed_idx)

  # === TRAINING STRATEGY ===
  seed_log("Configurando Training Strategy...", seed_idx = seed_idx)

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

  seed_log(paste("Training set:", sum(dataset$fold_train), "registros"), seed_idx = seed_idx)
  seed_log(paste("Validation set:", sum(dataset$foto_mes %in% PARAM$trainingstrategy$validate), "registros"), seed_idx = seed_idx)

  # === CREAR DATASETS LGBM ===
  seed_log("Creando datasets LightGBM...", seed_idx = seed_idx)

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

  seed_log("✅ Datasets LightGBM creados", seed_idx = seed_idx, level = "SUCCESS")

  # === HYPERPARAMETER TUNING ===
  seed_log("Iniciando Bayesian Optimization (10 iteraciones)...", seed_idx = seed_idx)

  PARAM$hipeparametertuning <- list()
  PARAM$hipeparametertuning$num_interations <- 10
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
    save.file.path = file.path(exp_path, "HT.RDATA")
  )

  ctrl <- setMBOControlTermination(ctrl, iters = PARAM$hipeparametertuning$num_interations)
  ctrl <- setMBOControlInfill(ctrl, crit = makeMBOInfillCritEI())

  surr.km <- makeLearner(
    "regr.km",
    predict.type = "se",
    covtype = "matern3_2",
    control = list(trace = FALSE)
  )

  ht_file <- file.path(exp_path, "HT.RDATA")

  if (!file.exists(ht_file)) {
    bayesiana_salida <- mbo(obj.fun, learner = surr.km, control = ctrl)
  } else {
    bayesiana_salida <- mboContinue(ht_file)
  }

  tb_bayesiana <- as.data.table(bayesiana_salida$opt.path)
  setorder(tb_bayesiana, -y, -num_iterations)

  fwrite(tb_bayesiana,
    file = file.path(exp_path, "BO_log.txt"),
    sep = "\t"
  )

  PARAM$out$lgbm$mejores_hiperparametros <- tb_bayesiana[
    1,
    setdiff(colnames(tb_bayesiana),
      c("y", "dob", "eol", "error.message", "exec.time", "ei", "error.model",
        "train.time", "prop.type", "propose.time", "se", "mean", "iter")),
    with = FALSE
  ]

  mejor_auc <- tb_bayesiana[1, y]
  seed_log(paste("✅ Bayesian Optimization completado. Mejor AUC:", round(mejor_auc, 6)), seed_idx = seed_idx, level = "SUCCESS")
  seed_log(paste("Mejores hiperparámetros: num_leaves =", PARAM$out$lgbm$mejores_hiperparametros$num_leaves,
                 ", min_data_in_leaf =", PARAM$out$lgbm$mejores_hiperparametros$min_data_in_leaf), seed_idx = seed_idx)

  # === PRODUCCIÓN - FINAL TRAIN ===
  seed_log("Entrenando modelo final...", seed_idx = seed_idx)

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

  lgb.save(final_model, file.path(exp_path, "modelo.txt"))

  tb_importancia <- as.data.table(lgb.importance(final_model))
  fwrite(tb_importancia,
    file = file.path(exp_path, "impo.txt"),
    sep = "\t"
  )

  seed_log("✅ Modelo final entrenado y guardado", seed_idx = seed_idx, level = "SUCCESS")

  # === SCORING ===
  seed_log("Scoring en futuro (202107)...", seed_idx = seed_idx)

  PARAM$trainingstrategy$future <- c(202107)
  dfuture <- dataset[foto_mes %in% PARAM$trainingstrategy$future]

  prediccion <- predict(
    final_model,
    data.matrix(dfuture[, campos_buenos, with = FALSE])
  )

  tb_prediccion <- dfuture[, list(numero_de_cliente)]
  tb_prediccion[, prob := prediccion]

  fwrite(tb_prediccion,
    file = file.path(exp_path, "prediccion.txt"),
    sep = "\t"
  )

  # === CURVA DE GANANCIA ===
  seed_log("Calculando curva de ganancia...", seed_idx = seed_idx)

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

  ganancia_max <- max(tb_prediccion$gan_suavizada, na.rm = TRUE)
  envios_optimos <- which.max(tb_prediccion$gan_suavizada)

  fwrite(tb_prediccion,
    file = file.path(exp_path, "ganancias.txt"),
    sep = "\t"
  )

  tb_prediccion[, envios := .I]

  pdf(file.path(exp_path, "curva_de_ganancia.pdf"))

  plot(
    x = tb_prediccion$envios,
    y = tb_prediccion$gan_acum,
    type = "l",
    col = "gray",
    xlim = c(0, 6000),
    ylim = c(0, 8000000),
    main = paste0("Seed ", seed_idx, " - Gan= ", as.integer(ganancia_max), " envios= ", envios_optimos),
    xlab = "Envios",
    ylab = "Ganancia",
    panel.first = grid()
  )

  lines(
    x = tb_prediccion$envios,
    y = tb_prediccion$gan_suavizada,
    col = "red",
    lwd = 2
  )

  dev.off()

  # === GUARDAR PARAM ===
  resultado <- list()
  resultado$ganancia <- ganancia_max
  resultado$envios <- envios_optimos
  resultado$semilla <- PARAM$semilla_primigenia
  resultado$seed_idx <- seed_idx
  resultado$mejor_auc <- mejor_auc

  PARAM$resultado <- resultado

  write_yaml(PARAM, file = file.path(exp_path, "PARAM.yml"))

  # === RESUMEN ===
  tiempo_fin <- Sys.time()
  duracion <- as.numeric(difftime(tiempo_fin, tiempo_inicio, units = "mins"))

  seed_log("============================================================", seed_idx = seed_idx)
  seed_log("SEED COMPLETADA EXITOSAMENTE", seed_idx = seed_idx, level = "SUCCESS")
  seed_log(paste("Ganancia:", scales::dollar(ganancia_max)), seed_idx = seed_idx)
  seed_log(paste("Envíos óptimos:", envios_optimos), seed_idx = seed_idx)
  seed_log(paste("Mejor AUC:", round(mejor_auc, 6)), seed_idx = seed_idx)
  seed_log(paste("Duración:", round(duracion, 1), "minutos"), seed_idx = seed_idx)
  seed_log("============================================================", seed_idx = seed_idx)

  # Limpiar memoria
  rm(dataset, dtrain, dvalidate, dfinal_train, final_model, tb_prediccion)
  gc(full = TRUE, verbose = FALSE)

  return(list(
    seed_idx = seed_idx,
    semilla = semilla,
    ganancia = ganancia_max,
    envios = envios_optimos,
    mejor_auc = mejor_auc,
    duracion_min = duracion,
    status = "OK"
  ))
}

# ===================================================================
# EJECUCIÓN PARALELA
# ===================================================================

log_msg(paste("Creando cluster con", num_workers, "workers..."))

cl <- makeCluster(num_workers)

# Exportar variables y funciones necesarias
clusterExport(cl, c(
  "PARAM_GLOBAL", "EXP_DIR", "DATASET_PATH", "threads_per_worker",
  "log_msg", "seed_log", "ejecutar_seed", "log_file"
))

# Cargar paquetes en cada worker
clusterEvalQ(cl, {
  suppressPackageStartupMessages({
    require("data.table")
    require("lightgbm")
    require("DiceKriging")
    require("mlr")
    require("ParamHelpers")
    require("mlrMBO")
    require("smoof")
    require("checkmate")
    require("yaml")
  })
})

log_msg("✅ Cluster creado y configurado", level = "SUCCESS")
health_check("Cluster paralelo creado con 5 workers")

log_msg("============================================================")
log_msg("✅ EJECUCIÓN PARALELA INICIADA", level = "SUCCESS")
log_msg(paste("Hora inicio:", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
log_msg(paste("Hora estimada fin:", format(Sys.time() + 60*60, "%Y-%m-%d %H:%M:%S")))  # ~1 hora estimado
log_msg("============================================================")

cat("\n⚡ TODAS LAS", length(PARAM_GLOBAL$semillas), "SEMILLAS CORRIENDO EN PARALELO ⚡\n\n")

# Ejecutar en paralelo
resultados <- parLapply(cl, 1:length(PARAM_GLOBAL$semillas), function(i) {
  ejecutar_seed(i, PARAM_GLOBAL$semillas[i], PARAM_GLOBAL$experimento_base)
})

# Cerrar cluster
stopCluster(cl)

log_msg("✅ Ejecución paralela completada", level = "SUCCESS")
health_check("Todas las semillas procesadas exitosamente")

# ===================================================================
# RESUMEN FINAL
# ===================================================================

log_msg("Generando resumen final...")

tb_resumen <- rbindlist(resultados)
setorder(tb_resumen, -ganancia)
tb_resumen[, rank := .I]

log_msg("============================================================")
log_msg("RESUMEN FINAL - WORKFLOW 610 BASELINE z610")
log_msg("============================================================")
cat("\n")
print(tb_resumen)
cat("\n")

log_msg("ESTADÍSTICAS:")
log_msg(paste("  Ganancia promedio:", scales::dollar(mean(tb_resumen$ganancia))))
log_msg(paste("  Ganancia máxima:", scales::dollar(max(tb_resumen$ganancia))))
log_msg(paste("  Ganancia mínima:", scales::dollar(min(tb_resumen$ganancia))))
log_msg(paste("  Desviación estándar:", scales::dollar(sd(tb_resumen$ganancia))))
log_msg(paste("  Coeficiente de variación:", round(sd(tb_resumen$ganancia)/mean(tb_resumen$ganancia)*100, 2), "%"))
log_msg(paste("  Envíos promedio:", round(mean(tb_resumen$envios))))
log_msg(paste("  AUC promedio:", round(mean(tb_resumen$mejor_auc), 6)))
log_msg(paste("  Duración promedio:", round(mean(tb_resumen$duracion_min), 1), "minutos"))
log_msg(paste("  Mejor semilla:", tb_resumen[rank == 1, semilla], "(seed_idx", tb_resumen[rank == 1, seed_idx], ")"))

# Guardar resumen
fwrite(tb_resumen,
  file = file.path(EXP_DIR, paste0("resumen_z610_baseline_exp", PARAM_GLOBAL$experimento_base, ".txt")),
  sep = "\t"
)

saveRDS(resultados,
  file = file.path(EXP_DIR, paste0("resultados_z610_baseline_exp", PARAM_GLOBAL$experimento_base, ".rds"))
)

log_msg("Archivos guardados:")
log_msg(paste("  -", paste0("resumen_z610_baseline_exp", PARAM_GLOBAL$experimento_base, ".txt")))
log_msg(paste("  -", paste0("resultados_z610_baseline_exp", PARAM_GLOBAL$experimento_base, ".rds")))
log_msg("Cada semilla tiene su carpeta individual con resultados detallados.")

log_msg("============================================================")
log_msg("WORKFLOW 610 BASELINE FINALIZADO EXITOSAMENTE", level = "SUCCESS")
log_msg(paste("Tiempo total:", round(difftime(Sys.time(),
  as.POSIXct(strptime(timestamp, "%Y%m%d_%H%M%S")), units = "mins"), 1), "minutos"))
log_msg("============================================================")

cat("\n")
cat("✅ WORKFLOW 610 BASELINE COMPLETADO\n")
cat("📊 Revisa los resultados en:", EXP_DIR, "\n")
cat("\n")
