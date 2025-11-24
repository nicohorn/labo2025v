# ============================================================================
# WORKFLOW 616 - TURBO MODE CON LOGGING COMPLETO
# ============================================================================
# Versión con logging detallado para monitoreo en tiempo real
#
# CARACTERÍSTICAS DE LOGGING:
# - Log general en: exp/workflow_turbo_TIMESTAMP.log
# - Log por semilla en: exp/WF*/semilla_X.log
# - Timestamps en cada operación
# - Resumen de progreso en tiempo real
# - Log de errores detallado
# ============================================================================

format(Sys.time(), "%a %b %d %X %Y")

# ============================================================================
# SISTEMA DE LOGGING
# ============================================================================

# Crear archivo de log principal
LOG_TIMESTAMP <- format(Sys.time(), "%Y%m%d_%H%M%S")
BASE_DIR <- "C:/Users/User/Documents/labo2025v"
DATASETS_DIR <- file.path(BASE_DIR, "datasets")
EXP_DIR <- file.path(BASE_DIR, "exp")

# Asegurar que existe el directorio
dir.create(EXP_DIR, showWarnings = FALSE, recursive = TRUE)

LOG_FILE <- file.path(EXP_DIR, paste0("workflow_turbo_", LOG_TIMESTAMP, ".log"))
PROGRESS_FILE <- file.path(EXP_DIR, paste0("progress_", LOG_TIMESTAMP, ".txt"))

# Función de logging mejorada
log_msg <- function(msg, level = "INFO", file = LOG_FILE, console = TRUE) {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  full_msg <- paste0("[", timestamp, "] [", level, "] ", msg)

  # Escribir a archivo
  cat(full_msg, "\n", file = file, append = TRUE)

  # Mostrar en consola
  if (console) {
    if (level == "ERROR") {
      cat("❌", full_msg, "\n")
    } else if (level == "SUCCESS") {
      cat("✅", full_msg, "\n")
    } else if (level == "WARNING") {
      cat("⚠️ ", full_msg, "\n")
    } else if (level == "PROGRESS") {
      cat("⏳", full_msg, "\n")
    } else {
      cat(full_msg, "\n")
    }
  }
}

# Función para actualizar progreso
update_progress <- function(seed_idx, stage, details = "") {
  progress_msg <- paste0(
    "Semilla ", seed_idx, " | ",
    "Stage: ", stage, " | ",
    "Time: ", format(Sys.time(), "%H:%M:%S")
  )

  if (details != "") {
    progress_msg <- paste0(progress_msg, " | ", details)
  }

  # Escribir a archivo de progreso
  cat(progress_msg, "\n", file = PROGRESS_FILE, append = TRUE)

  log_msg(progress_msg, level = "PROGRESS", console = TRUE)
}

# Banner inicial
cat("\n")
cat("========================================\n")
cat(" WORKFLOW PARALELO \n")
cat("   CON LOGGING \n")
cat("========================================\n\n")

log_msg(paste(rep("=", 60), collapse = ""))
log_msg("WORKFLOW TURBO MODE INICIADO")
log_msg(paste(rep("=", 60), collapse = ""))
log_msg(paste("Log file:", LOG_FILE))
log_msg(paste("Progress file:", PROGRESS_FILE))

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

NUM_CORES <- 18

log_msg("CONFIGURACIÓN HARDWARE", level = "INFO")
log_msg(paste("  Cores totales detectados:", parallel::detectCores()))
log_msg(paste("  Cores a usar:", NUM_CORES))
log_msg(paste("  Cores libres:", parallel::detectCores() - NUM_CORES))
log_msg("  RAM total: ~64 GB")
log_msg(paste("  RAM estimada uso:", NUM_CORES * 5, "GB"))

PARAM_GLOBAL <- list()
PARAM_GLOBAL$experimento_base <- 6160
PARAM_GLOBAL$dataset <- "gerencial_competencia_2025.csv.gz"
PARAM_GLOBAL$semillas <- c(153929, 838969, 922081, 795581, 194609)

log_msg("CONFIGURACIÓN EXPERIMENTO", level = "INFO")
log_msg(paste("  Experimento base:", PARAM_GLOBAL$experimento_base))
log_msg(paste("  Dataset:", PARAM_GLOBAL$dataset))
log_msg(paste("  Semillas:", length(PARAM_GLOBAL$semillas)))
log_msg(paste("  Semillas:", paste(PARAM_GLOBAL$semillas, collapse = ", ")))

# ============================================================================
# VERIFICACIONES PREVIAS
# ============================================================================

log_msg("VERIFICANDO ENTORNO", level = "INFO")

dataset_path <- file.path(DATASETS_DIR, PARAM_GLOBAL$dataset)
if (!file.exists(dataset_path)) {
  log_msg(paste("Dataset no encontrado:", dataset_path), level = "ERROR")
  stop("ERROR: Dataset no encontrado")
} else {
  file_size <- round(file.info(dataset_path)$size / 1024^2, 2)
  log_msg(paste("Dataset encontrado:", file_size, "MB"), level = "SUCCESS")
}

required_packages <- c("data.table", "lightgbm", "mlrMBO", "DiceKriging",
                       "parallel", "yaml", "R.utils")
missing_packages <- c()

log_msg("Verificando paquetes...", level = "INFO")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    missing_packages <- c(missing_packages, pkg)
    log_msg(paste("  FALTA:", pkg), level = "ERROR")
  } else {
    version <- packageVersion(pkg)
    log_msg(paste("  OK:", pkg, "-", version), level = "INFO")
  }
}

if (length(missing_packages) > 0) {
  log_msg(paste("Faltan paquetes:", paste(missing_packages, collapse = ", ")),
          level = "ERROR")
  stop("ERROR: Faltan paquetes")
} else {
  log_msg("Todos los paquetes instalados", level = "SUCCESS")
}

log_msg("ESTIMACIÓN DE TIEMPO", level = "INFO")
log_msg("  Por semilla (secuencial): ~2 horas")
log_msg("  Total secuencial: ~10 horas")
log_msg(paste("  Total PARALELO (", NUM_CORES, " cores): ~2-2.5 horas"))
log_msg("  AHORRO: ~8 horas (80%)")

cat("\n")
cat("📊 LOGS DISPONIBLES:\n")
cat("  Log principal:", LOG_FILE, "\n")
cat("  Progreso:", PROGRESS_FILE, "\n")
cat("\n")
cat("💡 TIP: Puedes abrir estos archivos en otra ventana\n")
cat("   para ver el progreso en tiempo real\n\n")

readline("Presiona ENTER para iniciar la ejecución TURBO... ")

log_msg("USUARIO CONFIRMÓ INICIO", level = "INFO")
log_msg("INICIANDO EJECUCIÓN EN PARALELO", level = "SUCCESS")

# ============================================================================
# FUNCIÓN PARA EJECUTAR UNA SEMILLA (CON LOGGING)
# ============================================================================

ejecutar_semilla <- function(seed_idx, semilla, PARAM_GLOBAL, BASE_DIR,
                             DATASETS_DIR, EXP_DIR, LOG_FILE) {

  # Log específico de esta semilla
  experimento_folder <- paste0("WF", PARAM_GLOBAL$experimento_base + seed_idx - 1,
                               "_seed", seed_idx, "_FE_historico_SOLO_TURBO")
  experimento_path <- file.path(EXP_DIR, experimento_folder)
  dir.create(experimento_path, showWarnings = FALSE, recursive = TRUE)

  seed_log_file <- file.path(experimento_path, paste0("semilla_", seed_idx, ".log"))

  # Función de logging local
  log_seed <- function(msg, level = "INFO") {
    timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
    full_msg <- paste0("[", timestamp, "] [Semilla ", seed_idx, "] [", level, "] ", msg)

    # Log local de la semilla
    cat(full_msg, "\n", file = seed_log_file, append = TRUE)

    # Log general
    cat(full_msg, "\n", file = LOG_FILE, append = TRUE)

    # Consola (solo mensajes importantes)
    if (level %in% c("ERROR", "SUCCESS", "START", "FINISH")) {
      if (level == "ERROR") {
        cat("❌ [Semilla", seed_idx, "]", msg, "\n")
      } else if (level == "SUCCESS") {
        cat("✅ [Semilla", seed_idx, "]", msg, "\n")
      } else if (level == "START") {
        cat("🚀 [Semilla", seed_idx, "]", msg, "\n")
      } else if (level == "FINISH") {
        cat("🏁 [Semilla", seed_idx, "]", msg, "\n")
      }
    }
  }

  # Cargar librerías
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

  setDTthreads(1)

  PARAM <- list()
  PARAM$semilla_primigenia <- semilla
  PARAM$experimento <- PARAM_GLOBAL$experimento_base + seed_idx - 1
  PARAM$dataset <- PARAM_GLOBAL$dataset
  PARAM$out <- list()
  PARAM$out$lgbm <- list()

  inicio <- Sys.time()
  log_seed(paste("INICIANDO -", format(inicio)), level = "START")
  log_seed(paste("Semilla:", semilla), level = "INFO")
  log_seed(paste("Experimento:", PARAM$experimento), level = "INFO")
  log_seed(paste("Carpeta:", experimento_folder), level = "INFO")

  tryCatch({

    setwd(experimento_path)

    # ===================================================================
    # CARGAR DATASET
    # ===================================================================

    update_progress(seed_idx, "CARGANDO_DATASET", "")
    log_seed("Cargando dataset...", level = "INFO")

    inicio_carga <- Sys.time()
    dataset_path <- file.path(DATASETS_DIR, PARAM$dataset)
    dataset <- fread(dataset_path, showProgress = FALSE)
    fin_carga <- Sys.time()

    tiempo_carga <- as.numeric(difftime(fin_carga, inicio_carga, units = "secs"))
    log_seed(paste("Dataset cargado en", round(tiempo_carga, 1), "segundos"), level = "SUCCESS")
    log_seed(paste("Dimensiones:", nrow(dataset), "filas x", ncol(dataset), "cols"), level = "INFO")

    # Catastrophe Analysis
    log_seed("Aplicando Catastrophe Analysis...", level = "INFO")
    dataset[foto_mes == 202006, `:=`(
      internet = NA, mrentabilidad = NA, mrentabilidad_annual = NA,
      mcomisiones = NA, mactivos_margen = NA, mpasivos_margen = NA,
      mcuentas_saldo = NA, ctarjeta_visa_transacciones = NA,
      mtarjeta_visa_consumo = NA, mtarjeta_master_consumo = NA,
      ccallcenter_transacciones = NA, chomebanking_transacciones = NA
    )]
    log_seed("Catastrophe Analysis completado", level = "SUCCESS")

    # ===================================================================
    # FEATURE ENGINEERING
    # ===================================================================

    update_progress(seed_idx, "FEATURE_ENGINEERING", "")
    log_seed("Iniciando Feature Engineering Histórico...", level = "INFO")

    inicio_fe <- Sys.time()

    setorder(dataset, numero_de_cliente, foto_mes)

    cols_lagueables <- setdiff(
      colnames(dataset),
      c("numero_de_cliente", "foto_mes", "clase_ternaria")
    )

    log_seed(paste("Variables base:", length(cols_lagueables)), level = "INFO")

    # LAGS
    log_seed("Generando lags (1,2,3,6)...", level = "INFO")
    for (lag in c(1, 2, 3, 6)) {
      dataset[,
              paste0(cols_lagueables, "_lag", lag) := shift(.SD, lag, NA, "lag"),
              by = numero_de_cliente,
              .SDcols = cols_lagueables]
    }
    log_seed(paste("Lags generados:", length(cols_lagueables) * 4, "variables"), level = "SUCCESS")

    # DELTAS
    log_seed("Generando deltas...", level = "INFO")
    for (vcol in cols_lagueables) {
      dataset[, paste0(vcol, "_delta1") := get(vcol) - get(paste0(vcol, "_lag1"))]
      dataset[, paste0(vcol, "_delta2") := get(vcol) - get(paste0(vcol, "_lag2"))]
      dataset[, paste0(vcol, "_delta3") := get(vcol) - get(paste0(vcol, "_lag3"))]
    }
    log_seed(paste("Deltas generados:", length(cols_lagueables) * 3, "variables"), level = "SUCCESS")

    # ROLLING STATS
    log_seed("Generando rolling statistics...", level = "INFO")
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
    log_seed(paste("Rolling stats generados:", length(cols_lagueables) * 6, "variables"), level = "SUCCESS")

    # TENDENCIAS
    log_seed("Generando tendencias...", level = "INFO")
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
    log_seed(paste("Tendencias generadas:", length(cols_lagueables) * 2, "variables"), level = "SUCCESS")

    # RATIOS
    log_seed("Generando ratios históricos...", level = "INFO")
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
    log_seed(paste("Ratios generados:", length(cols_lagueables) * 3, "variables"), level = "SUCCESS")

    # VOLATILIDAD
    log_seed("Generando métricas de volatilidad...", level = "INFO")
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
    log_seed(paste("Volatilidad generada:", length(cols_lagueables) * 3, "variables"), level = "SUCCESS")

    fin_fe <- Sys.time()
    tiempo_fe <- as.numeric(difftime(fin_fe, inicio_fe, units = "mins"))

    log_seed(paste("FE COMPLETADO en", round(tiempo_fe, 1), "minutos"), level = "SUCCESS")
    log_seed(paste("Total features:", ncol(dataset)), level = "INFO")

    total_fe <- length(cols_lagueables) * (4 + 3 + 6 + 2 + 3 + 3)
    log_seed(paste("Nuevas features creadas:", total_fe), level = "INFO")

    # ===================================================================
    # TRAINING STRATEGY
    # ===================================================================

    update_progress(seed_idx, "PREPARANDO_DATOS", "")
    log_seed("Configurando training strategy...", level = "INFO")

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

    log_seed(paste("Features para modelado:", length(campos_buenos)), level = "INFO")
    log_seed(paste("Registros training:", sum(dataset$fold_train)), level = "INFO")
    log_seed(paste("Registros validation:", sum(dataset$foto_mes %in% PARAM$trainingstrategy$validate)), level = "INFO")

    log_seed("Creando datasets LightGBM...", level = "INFO")

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

    log_seed("Datasets LightGBM creados", level = "SUCCESS")

    # ===================================================================
    # HYPERPARAMETER TUNING
    # ===================================================================

    update_progress(seed_idx, "HYPERPARAMETER_TUNING", "10 iteraciones")
    log_seed("Iniciando Hyperparameter Tuning...", level = "INFO")
    log_seed("Iteraciones de BO: 10", level = "INFO")

    inicio_ht <- Sys.time()

    PARAM$hipeparametertuning <- list(num_interations = 10)
    PARAM$lgbm <- list(
      param_fijos = list(
        objective = "binary", metric = "auc", first_metric_only = TRUE,
        boost_from_average = TRUE, feature_pre_filter = FALSE,
        verbosity = -100, force_row_wise = TRUE,
        seed = PARAM$semilla_primigenia, max_bin = 31,
        learning_rate = 0.03, feature_fraction = 0.5,
        num_iterations = 2048, early_stopping_rounds = 200,
        num_threads = 1
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

    fin_ht <- Sys.time()
    tiempo_ht <- as.numeric(difftime(fin_ht, inicio_ht, units = "mins"))

    log_seed(paste("Hyperparameter Tuning completado en", round(tiempo_ht, 1), "minutos"), level = "SUCCESS")
    log_seed(paste("Mejor AUC:", round(tb_bayesiana[1, y], 4)), level = "INFO")
    log_seed(paste("Mejores hiperparámetros:",
                   "num_leaves=", PARAM$out$lgbm$mejores_hiperparametros$num_leaves,
                   "min_data_in_leaf=", PARAM$out$lgbm$mejores_hiperparametros$min_data_in_leaf),
             level = "INFO")

    # ===================================================================
    # MODELO FINAL
    # ===================================================================

    update_progress(seed_idx, "ENTRENANDO_MODELO_FINAL", "")
    log_seed("Entrenando modelo final...", level = "INFO")

    inicio_train <- Sys.time()

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

    fin_train <- Sys.time()
    tiempo_train <- as.numeric(difftime(fin_train, inicio_train, units = "mins"))

    log_seed(paste("Modelo final entrenado en", round(tiempo_train, 1), "minutos"), level = "SUCCESS")
    log_seed(paste("Top 3 features:", paste(head(tb_importancia$Feature, 3), collapse = ", ")), level = "INFO")

    # ===================================================================
    # SCORING Y GANANCIA
    # ===================================================================

    update_progress(seed_idx, "SCORING", "")
    log_seed("Generando predicciones...", level = "INFO")

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

    log_seed(paste("Predicciones generadas:", nrow(tb_prediccion), "clientes"), level = "SUCCESS")

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

    log_seed("Curva de ganancia guardada", level = "SUCCESS")

    # Limpiar memoria
    rm(dataset, dtrain, dvalidate, dfinal_train, final_model, tb_prediccion)
    gc(full = TRUE, verbose = FALSE)

    fin <- Sys.time()
    duracion <- as.numeric(difftime(fin, inicio, units = "mins"))

    log_seed(paste(rep("=", 50), collapse = ""), level = "INFO")
    log_seed("SEMILLA COMPLETADA EXITOSAMENTE", level = "FINISH")
    log_seed(paste("Tiempo total:", round(duracion, 1), "minutos"), level = "INFO")
    log_seed(paste("Ganancia:", resultado$ganancia_suavizada_max), level = "INFO")
    log_seed(paste("Envíos óptimos:", resultado$envios), level = "INFO")
    log_seed(paste(rep("=", 50), collapse = ""), level = "INFO")

    update_progress(seed_idx, "COMPLETADO",
                   paste("Ganancia:", resultado$ganancia_suavizada_max,
                         "Tiempo:", round(duracion, 1), "min"))

    return(list(
      success = TRUE, seed_idx = seed_idx, semilla = semilla,
      ganancia = resultado$ganancia_suavizada_max,
      envios = resultado$envios, duracion_min = duracion,
      log_file = seed_log_file
    ))

  }, error = function(e) {
    log_seed(paste("ERROR CRÍTICO:", e$message), level = "ERROR")
    log_seed(paste("Traceback:", paste(sys.calls(), collapse = " -> ")), level = "ERROR")

    update_progress(seed_idx, "ERROR", e$message)

    return(list(success = FALSE, seed_idx = seed_idx, semilla = semilla,
                error = e$message, log_file = seed_log_file))
  })
}

# ============================================================================
# EJECUCIÓN EN PARALELO
# ============================================================================

require("parallel")

log_msg(paste("Creando cluster con", NUM_CORES, "cores..."), level = "INFO")
cl <- makeCluster(NUM_CORES)

clusterExport(cl, c("PARAM_GLOBAL", "BASE_DIR", "DATASETS_DIR", "EXP_DIR",
                    "LOG_FILE", "PROGRESS_FILE", "log_msg", "update_progress",
                    "ejecutar_semilla"))

inicio_total <- Sys.time()

log_msg(paste(rep("=", 60), collapse = ""), level = "INFO")
log_msg("EJECUCIÓN EN PARALELO INICIADA", level = "SUCCESS")
log_msg(paste("Hora inicio:", format(inicio_total)), level = "INFO")
log_msg(paste("Hora estimada fin:", format(inicio_total + as.difftime(2.5, units = "hours"))), level = "INFO")
log_msg(paste(rep("=", 60), collapse = ""), level = "INFO")

cat("\n⚡ TODAS LAS", length(PARAM_GLOBAL$semillas), "SEMILLAS CORRIENDO EN PARALELO ⚡\n\n")
cat("📊 Monitorea el progreso en tiempo real:\n")
cat("   Log principal:", LOG_FILE, "\n")
cat("   Progreso:", PROGRESS_FILE, "\n")
cat("   Logs individuales: exp/WF*/semilla_X.log\n\n")
cat("💡 TIP: Abre estos archivos en Notepad++ o similar\n")
cat("   para ver actualizaciones en tiempo real\n\n")

resultados <- parLapply(cl, 1:length(PARAM_GLOBAL$semillas), function(i) {
  ejecutar_semilla(i, PARAM_GLOBAL$semillas[i], PARAM_GLOBAL,
                   BASE_DIR, DATASETS_DIR, EXP_DIR, LOG_FILE)
})

stopCluster(cl)

fin_total <- Sys.time()
duracion_total <- as.numeric(difftime(fin_total, inicio_total, units = "hours"))

# ============================================================================
# RESUMEN FINAL
# ============================================================================

log_msg(paste(rep("=", 60), collapse = ""), level = "INFO")
log_msg("EJECUCIÓN PARALELA FINALIZADA", level = "SUCCESS")
log_msg(paste(rep("=", 60), collapse = ""), level = "INFO")

cat("\n\n")
cat("========================================\n")
cat("🏁 TURBO MODE COMPLETADO 🏁\n")
cat("========================================\n\n")

log_msg(paste("Tiempo total ejecución:", round(duracion_total, 2), "horas"), level = "INFO")
log_msg(paste("Tiempo ahorrado vs secuencial: ~", round(10 - duracion_total, 1), "horas"), level = "INFO")

resultados_exitosos <- Filter(function(x) x$success, resultados)
resultados_fallidos <- Filter(function(x) !x$success, resultados)

log_msg(paste("Semillas exitosas:", length(resultados_exitosos)), level = "SUCCESS")
log_msg(paste("Semillas fallidas:", length(resultados_fallidos)), level = ifelse(length(resultados_fallidos) > 0, "WARNING", "INFO"))

if (length(resultados_exitosos) > 0) {

  tb_resumen <- rbindlist(lapply(resultados_exitosos, function(r) {
    data.table(seed_idx = r$seed_idx, semilla = r$semilla,
               ganancia = r$ganancia, envios = r$envios,
               duracion_min = r$duracion_min)
  }))

  setorder(tb_resumen, -ganancia)
  tb_resumen[, rank := .I]

  cat("\n📊 RESULTADOS:\n\n")
  print(tb_resumen)

  log_msg("ESTADÍSTICAS FINALES", level = "INFO")
  log_msg(paste("  Ganancia promedio:", mean(tb_resumen$ganancia)), level = "INFO")
  log_msg(paste("  Ganancia máxima:", max(tb_resumen$ganancia)), level = "SUCCESS")
  log_msg(paste("  Ganancia mínima:", min(tb_resumen$ganancia)), level = "INFO")
  log_msg(paste("  Desviación estándar:", sd(tb_resumen$ganancia)), level = "INFO")
  log_msg(paste("  Mejor semilla:", tb_resumen[rank == 1, semilla]), level = "SUCCESS")
  log_msg(paste("  Duración promedio por semilla:", round(mean(tb_resumen$duracion_min), 1), "min"), level = "INFO")

  setwd(EXP_DIR)
  fwrite(tb_resumen, file = paste0("resumen_TURBO_", LOG_TIMESTAMP, ".txt"), sep = "\t")
  saveRDS(resultados, file = paste0("resultados_TURBO_", LOG_TIMESTAMP, ".rds"))

  log_msg(paste("Resultados guardados en:", EXP_DIR), level = "SUCCESS")

  cat("\n📁 Archivos generados:\n")
  cat("  - resumen_TURBO_", LOG_TIMESTAMP, ".txt\n")
  cat("  - resultados_TURBO_", LOG_TIMESTAMP, ".rds\n")
  cat("  - ", LOG_FILE, "\n")
  cat("  - ", PROGRESS_FILE, "\n")
}

if (length(resultados_fallidos) > 0) {
  cat("\n⚠ SEMILLAS FALLIDAS:\n")
  log_msg("SEMILLAS FALLIDAS", level = "ERROR")
  for (r in resultados_fallidos) {
    msg <- paste("Semilla", r$seed_idx, ":", r$error)
    cat("  -", msg, "\n")
    log_msg(msg, level = "ERROR")
    log_msg(paste("  Log detallado en:", r$log_file), level = "INFO")
  }
}

log_msg(paste(rep("=", 60), collapse = ""), level = "INFO")
log_msg("WORKFLOW TURBO MODE FINALIZADO", level = "SUCCESS")
log_msg(paste("Tiempo total:", round(duracion_total, 2), "horas"), level = "INFO")
log_msg(paste(rep("=", 60), collapse = ""), level = "INFO")

cat("\n========================================\n")
cat("🎉 EJECUCIÓN COMPLETADA 🎉\n")
cat("========================================\n\n")

format(Sys.time(), "%a %b %d %X %Y")
