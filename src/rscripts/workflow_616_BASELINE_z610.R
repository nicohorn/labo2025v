# ============================================================================
# WORKFLOW 616 BASELINE - LÓGICA EXACTA DE z610 + PARALELIZACIÓN
# ============================================================================
#
# FILOSOFÍA:
# - Lógica IDÉNTICA a z610_WorkFlow_01_gerencial_julio.ipynb
# - ÚNICO cambio: Feature Engineering Histórico AVANZADO (lags 1,2,3,6 + deltas + rolling + trends + ratios + volatilidad)
# - Mantiene: Catastrophe Analysis, Training Strategy, Hyperparameter Tuning, Scoring
# - Agrega: Paralelización, Caching, Logging, Health Checks
#
# ============================================================================

format(Sys.time(), "%a %b %d %X %Y")

# ============================================================================
# SISTEMA DE LOGGING
# ============================================================================

LOG_TIMESTAMP <- format(Sys.time(), "%Y%m%d_%H%M%S")
BASE_DIR <- "C:/Users/User/Documents/labo2025v"
DATASETS_DIR <- file.path(BASE_DIR, "datasets")
EXP_DIR <- file.path(BASE_DIR, "exp", "exp_baseline")

dir.create(EXP_DIR, showWarnings = FALSE, recursive = TRUE)

LOG_FILE <- file.path(EXP_DIR, paste0("workflow_baseline_", LOG_TIMESTAMP, ".log"))
PROGRESS_FILE <- file.path(EXP_DIR, paste0("progress_", LOG_TIMESTAMP, ".txt"))
HEALTH_FILE <- file.path(EXP_DIR, paste0("health_", LOG_TIMESTAMP, ".txt"))

log_msg <- function(msg, level = "INFO", file = LOG_FILE, console = TRUE) {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  full_msg <- paste0("[", timestamp, "] [", level, "] ", msg)

  tryCatch({
    cat(full_msg, "\n", file = file, append = TRUE)
  }, error = function(e) {
    cat("ERROR ESCRIBIENDO LOG:", e$message, "\n", file = stderr())
  })

  if (console) {
    if (level == "ERROR") {
      cat("❌", full_msg, "\n")
    } else if (level == "SUCCESS") {
      cat("✅", full_msg, "\n")
    } else if (level == "WARNING") {
      cat("⚠️ ", full_msg, "\n")
    } else if (level == "PROGRESS") {
      cat("⏳", full_msg, "\n")
    } else if (level == "HEALTH") {
      cat("💊", full_msg, "\n")
    } else {
      cat(full_msg, "\n")
    }
  }
}

update_progress <- function(seed_idx, stage, details = "") {
  progress_msg <- paste0(
    "Semilla ", seed_idx, " | ",
    "Stage: ", stage, " | ",
    "Time: ", format(Sys.time(), "%H:%M:%S")
  )

  if (details != "") {
    progress_msg <- paste0(progress_msg, " | ", details)
  }

  cat(progress_msg, "\n", file = PROGRESS_FILE, append = TRUE)
  log_msg(progress_msg, level = "PROGRESS", console = TRUE)
}

health_check <- function(mensaje = "Health check") {
  health_msg <- paste0(
    "HEALTH CHECK | ",
    "Time: ", format(Sys.time(), "%H:%M:%S"), " | ",
    mensaje
  )

  cat(health_msg, "\n", file = HEALTH_FILE, append = TRUE)
  log_msg(health_msg, level = "HEALTH", console = TRUE)
}

# Banner
cat("\n")
cat("========================================\n")
cat(" WORKFLOW 616 BASELINE (z610 logic)\n")
cat("   Paralelo + Caching + Logging\n")
cat("========================================\n\n")

log_msg(paste(rep("=", 60), collapse = ""))
log_msg("WORKFLOW 616 BASELINE INICIADO")
log_msg(paste(rep("=", 60), collapse = ""))
log_msg(paste("Log file:", LOG_FILE))
log_msg(paste("Progress file:", PROGRESS_FILE))
log_msg(paste("Health file:", HEALTH_FILE))

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

NUM_WORKERS <- 5
THREADS_PER_WORKER <- 4

log_msg("CONFIGURACIÓN HARDWARE", level = "INFO")
log_msg(paste("  Cores totales detectados:", parallel::detectCores()))
log_msg(paste("  Workers paralelos:", NUM_WORKERS))
log_msg(paste("  Threads por worker:", THREADS_PER_WORKER))
log_msg(paste("  Cores totales en uso:", NUM_WORKERS * THREADS_PER_WORKER))

PARAM_GLOBAL <- list()
PARAM_GLOBAL$experimento <- "WF616"
PARAM_GLOBAL$semillas <- c(153929, 838969, 922081, 795581, 194609)

log_msg(paste("Semillas a ejecutar:", length(PARAM_GLOBAL$semillas)))
log_msg(paste("Semillas:", paste(PARAM_GLOBAL$semillas, collapse = ", ")))

# ============================================================================
# CARGAR PAQUETES
# ============================================================================

log_msg("Cargando paquetes...", level = "INFO")

require("data.table")
require("lightgbm")
require("DiceKriging")
require("mlr")
require("mlrMBO")
require("ParamHelpers")

setDTthreads(THREADS_PER_WORKER)

log_msg(paste("data.table threads configurados:", getDTthreads()), level = "SUCCESS")
log_msg("Paquetes cargados correctamente", level = "SUCCESS")

health_check("Sistema iniciado, paquetes cargados")

# ============================================================================
# VERIFICAR DATASET
# ============================================================================

dataset_file <- file.path(DATASETS_DIR, "gerencial_competencia_2025.csv.gz")

log_msg("Verificando dataset...", level = "INFO")
if (file.exists(dataset_file)) {
  file_size <- file.info(dataset_file)$size / (1024^2)
  log_msg(paste("Dataset encontrado:", round(file_size, 2), "MB"), level = "SUCCESS")
} else {
  log_msg("ERROR: Dataset no encontrado!", level = "ERROR")
  stop("Dataset no encontrado")
}

# ============================================================================
# FUNCIÓN EJECUTAR SEMILLA
# ============================================================================

ejecutar_semilla <- function(seed_idx, semilla, PARAM_GLOBAL, BASE_DIR, DATASETS_DIR,
                             EXP_DIR, LOG_FILE, PROGRESS_FILE, THREADS_PER_WORKER) {

  require("data.table")
  setDTthreads(THREADS_PER_WORKER)
  require("lightgbm")
  require("DiceKriging")
  require("mlr")
  require("mlrMBO")
  require("ParamHelpers")

  tryCatch({
    inicio_seed <- Sys.time()

    # Crear directorio
    exp_folder <- paste0("WF616", seed_idx - 1, "_seed", seed_idx, "_BASELINE")
    seed_dir <- file.path(EXP_DIR, exp_folder)
    dir.create(seed_dir, showWarnings = FALSE, recursive = TRUE)

    seed_log_file <- file.path(seed_dir, paste0("semilla_", seed_idx, ".log"))

    seed_log <- function(msg, level = "INFO") {
      timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
      full_msg <- paste0("[", timestamp, "] [", level, "] ", msg)
      cat(full_msg, "\n", file = seed_log_file, append = TRUE)
      log_msg(paste0("[Semilla ", seed_idx, "] ", msg), level = level, console = TRUE)
    }

    seed_log(paste("INICIANDO -", format(inicio_seed)), level = "START")
    update_progress(seed_idx, "INICIANDO")

    # ========================================================================
    # VERIFICAR CACHE DE DATASET
    # ========================================================================

    dataset_cache_file <- file.path(seed_dir, "dataset_con_FE.rds")

    if (file.exists(dataset_cache_file)) {
      update_progress(seed_idx, "CARGANDO_CACHE")
      seed_log("Dataset con FE encontrado en cache, cargando...", level = "INFO")

      inicio_cache <- Sys.time()
      dataset <- readRDS(dataset_cache_file)
      fin_cache <- Sys.time()
      tiempo_cache <- as.numeric(difftime(fin_cache, inicio_cache, units = "secs"))

      seed_log(paste("Dataset cacheado cargado en", round(tiempo_cache, 1), "segundos"), level = "SUCCESS")
      seed_log(paste("Dimensiones:", nrow(dataset), "filas x", ncol(dataset), "cols"))
      seed_log("SALTANDO Feature Engineering (usando cache)", level = "SUCCESS")

    } else {

      # ======================================================================
      # CARGA DATASET ORIGINAL
      # ======================================================================

      update_progress(seed_idx, "CARGANDO_DATASET")
      seed_log("Cargando dataset original...")

      inicio_carga <- Sys.time()
      dataset <- fread(file.path(DATASETS_DIR, "gerencial_competencia_2025.csv.gz"))
      fin_carga <- Sys.time()
      tiempo_carga <- as.numeric(difftime(fin_carga, inicio_carga, units = "secs"))

      seed_log(paste("Dataset cargado en", round(tiempo_carga, 1), "segundos"), level = "SUCCESS")
      seed_log(paste("Dimensiones:", nrow(dataset), "filas x", ncol(dataset), "cols"))

      # ======================================================================
      # CATASTROPHE ANALYSIS - LÓGICA EXACTA DE z610
      # ======================================================================

      seed_log("Aplicando Catastrophe Analysis (z610 method)...", level = "INFO")

      # Asignar NA a variables específicas en foto_mes 202006
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

      seed_log("Catastrophe Analysis completado (13 variables en 202006 → NA)", level = "SUCCESS")

      # ======================================================================
      # FEATURE ENGINEERING HISTÓRICO AVANZADO
      # ======================================================================

      update_progress(seed_idx, "FEATURE_ENGINEERING")
      seed_log("Iniciando Feature Engineering Histórico AVANZADO...")

      inicio_fe <- Sys.time()
      cols_antes_fe <- ncol(dataset)

      # Variables base
      cols_lagueables <- setdiff(
        colnames(dataset),
        c("numero_de_cliente", "foto_mes", "clase_ternaria")
      )

      seed_log(paste("Variables base:", length(cols_lagueables)))

      # Ordenar por cliente y mes
      setorder(dataset, numero_de_cliente, foto_mes)

      # === LAGS (1, 2, 3, 6) ===
      seed_log("Generando lags (1,2,3,6)...")
      inicio_lags <- Sys.time()

      for (lag in c(1, 2, 3, 6)) {
        dataset[,
                paste0(cols_lagueables, "_lag", lag) := shift(.SD, lag, NA, "lag"),
                by = numero_de_cliente,
                .SDcols = cols_lagueables]
      }

      fin_lags <- Sys.time()
      cols_lags <- sum(grepl("_lag[0-9]$", names(dataset)))
      seed_log(paste("Lags generados:", cols_lags, "variables en",
                     round(as.numeric(difftime(fin_lags, inicio_lags, units = "secs")), 1), "seg"),
               level = "SUCCESS")

      # === DELTAS (1, 2, 3) ===
      seed_log("Generando deltas...")
      inicio_deltas <- Sys.time()
      cols_antes_deltas <- ncol(dataset)

      for (lag in c(1, 2, 3)) {
        for (col in cols_lagueables) {
          lag_col <- paste0(col, "_lag", lag)
          if (lag_col %in% names(dataset)) {
            delta_col <- paste0(col, "_delta", lag)
            dataset[, (delta_col) := get(col) - get(lag_col)]
          }
        }
      }

      fin_deltas <- Sys.time()
      cols_deltas <- ncol(dataset) - cols_antes_deltas
      seed_log(paste("Deltas generados:", cols_deltas, "variables en",
                     round(as.numeric(difftime(fin_deltas, inicio_deltas, units = "secs")), 1), "seg"),
               level = "SUCCESS")

      # === ROLLING STATISTICS (ventanas 3, 6) ===
      seed_log("Generando rolling statistics...")
      inicio_rolling <- Sys.time()
      cols_antes_rolling <- ncol(dataset)

      for (ventana in c(3, 6)) {
        for (col in cols_lagueables) {
          # Media móvil
          mean_col <- paste0(col, "_roll_mean_", ventana)
          dataset[, (mean_col) := frollapply(get(col), n = ventana, FUN = mean,
                                             na.rm = TRUE, align = "right"),
                  by = numero_de_cliente]

          # Máximo móvil
          max_col <- paste0(col, "_roll_max_", ventana)
          dataset[, (max_col) := frollapply(get(col), n = ventana, FUN = max,
                                            na.rm = TRUE, align = "right"),
                  by = numero_de_cliente]

          # Mínimo móvil
          min_col <- paste0(col, "_roll_min_", ventana)
          dataset[, (min_col) := frollapply(get(col), n = ventana, FUN = min,
                                            na.rm = TRUE, align = "right"),
                  by = numero_de_cliente]

          # Desviación estándar móvil
          sd_col <- paste0(col, "_roll_sd_", ventana)
          dataset[, (sd_col) := frollapply(get(col), n = ventana, FUN = sd,
                                           na.rm = TRUE, align = "right"),
                  by = numero_de_cliente]
        }
      }

      fin_rolling <- Sys.time()
      cols_rolling <- ncol(dataset) - cols_antes_rolling
      seed_log(paste("Rolling stats generados:", cols_rolling, "variables en",
                     round(as.numeric(difftime(fin_rolling, inicio_rolling, units = "mins")), 1), "min"),
               level = "SUCCESS")

      # === TENDENCIAS (ventanas 3, 6) ===
      seed_log("Generando tendencias...")
      inicio_trends <- Sys.time()
      cols_antes_trends <- ncol(dataset)

      for (ventana in c(3, 6)) {
        for (col in cols_lagueables) {
          trend_col <- paste0(col, "_trend_", ventana)

          dataset[, (trend_col) := {
            if (.N >= ventana) {
              valores <- tail(get(col), ventana)
              if (all(is.na(valores))) {
                NA_real_
              } else {
                x <- 1:ventana
                y <- valores
                validos <- !is.na(y)
                if (sum(validos) >= 2) {
                  coef(lm(y[validos] ~ x[validos]))[2]
                } else {
                  NA_real_
                }
              }
            } else {
              NA_real_
            }
          }, by = numero_de_cliente]
        }
      }

      fin_trends <- Sys.time()
      cols_trends <- ncol(dataset) - cols_antes_trends
      seed_log(paste("Tendencias generadas:", cols_trends, "variables en",
                     round(as.numeric(difftime(fin_trends, inicio_trends, units = "mins")), 1), "min"),
               level = "SUCCESS")

      # === RATIOS HISTÓRICOS ===
      seed_log("Generando ratios históricos...")
      inicio_ratios <- Sys.time()
      cols_antes_ratios <- ncol(dataset)

      for (col in cols_lagueables) {
        # Ratio vs lag1
        lag1 <- paste0(col, "_lag1")
        if (lag1 %in% names(dataset)) {
          ratio_col <- paste0(col, "_ratio_vs_lag1")
          dataset[, (ratio_col) := ifelse(get(lag1) != 0, get(col) / get(lag1), NA_real_)]
        }

        # Ratio vs media rolling 3
        mean3 <- paste0(col, "_roll_mean_3")
        if (mean3 %in% names(dataset)) {
          ratio_mean_col <- paste0(col, "_ratio_vs_mean3")
          dataset[, (ratio_mean_col) := ifelse(get(mean3) != 0, get(col) / get(mean3), NA_real_)]
        }
      }

      fin_ratios <- Sys.time()
      cols_ratios <- ncol(dataset) - cols_antes_ratios
      seed_log(paste("Ratios generados:", cols_ratios, "variables en",
                     round(as.numeric(difftime(fin_ratios, inicio_ratios, units = "secs")), 1), "seg"),
               level = "SUCCESS")

      # === VOLATILIDAD ===
      seed_log("Generando métricas de volatilidad...")
      inicio_vol <- Sys.time()
      cols_antes_vol <- ncol(dataset)

      for (ventana in c(3, 6)) {
        for (col in cols_lagueables) {
          mean_col <- paste0(col, "_roll_mean_", ventana)
          sd_col <- paste0(col, "_roll_sd_", ventana)

          if (mean_col %in% names(dataset) && sd_col %in% names(dataset)) {
            # Coeficiente de variación
            cv_col <- paste0(col, "_cv_", ventana)
            dataset[, (cv_col) := ifelse(get(mean_col) != 0,
                                         get(sd_col) / abs(get(mean_col)),
                                         NA_real_)]

            # Rango normalizado
            max_col <- paste0(col, "_roll_max_", ventana)
            min_col <- paste0(col, "_roll_min_", ventana)

            if (max_col %in% names(dataset) && min_col %in% names(dataset)) {
              range_col <- paste0(col, "_range_norm_", ventana)
              dataset[, (range_col) := ifelse(get(mean_col) != 0,
                                              (get(max_col) - get(min_col)) / abs(get(mean_col)),
                                              NA_real_)]
            }
          }
        }
      }

      fin_vol <- Sys.time()
      cols_vol <- ncol(dataset) - cols_antes_vol
      seed_log(paste("Volatilidad generada:", cols_vol, "variables en",
                     round(as.numeric(difftime(fin_vol, inicio_vol, units = "secs")), 1), "seg"),
               level = "SUCCESS")

      fin_fe <- Sys.time()
      cols_despues_fe <- ncol(dataset)
      tiempo_fe <- as.numeric(difftime(fin_fe, inicio_fe, units = "mins"))

      seed_log(paste("FE COMPLETADO en", round(tiempo_fe, 1), "minutos"), level = "SUCCESS")
      seed_log(paste("Total features:", cols_despues_fe, "(agregadas:", cols_despues_fe - cols_antes_fe, ")"))

      # Guardar dataset con FE en cache
      seed_log("Guardando dataset con FE en cache...", level = "INFO")
      inicio_save <- Sys.time()
      saveRDS(dataset, dataset_cache_file, compress = "xz")
      fin_save <- Sys.time()
      tiempo_save <- as.numeric(difftime(fin_save, inicio_save, units = "secs"))

      file_size_mb <- file.info(dataset_cache_file)$size / (1024^2)
      seed_log(paste("Dataset guardado en cache:", round(file_size_mb, 1), "MB en",
                     round(tiempo_save, 1), "seg"), level = "SUCCESS")

    }  # Fin cache check

    # ========================================================================
    # TRAINING STRATEGY - LÓGICA EXACTA DE z610
    # ========================================================================

    update_progress(seed_idx, "TRAINING_STRATEGY")
    seed_log("Configurando Training Strategy (z610 logic)...")

    # Clase binaria
    dataset[, clase01 := ifelse(clase_ternaria %in% c("BAJA+1", "BAJA+2"), 1, 0)]

    # Training months (z610: 202005-202104)
    training_months <- c(
      202104, 202103, 202102, 202101,
      202012, 202011, 202010, 202009, 202008, 202007,
      202006, 202005
    )

    # Validation month (z610: 202105)
    validate_month <- 202105

    # Final train months (z610: 202005-202105)
    final_train_months <- c(
      202105, 202104, 202103, 202102, 202101,
      202012, 202011, 202010, 202009, 202008, 202007,
      202006, 202005
    )

    # Future month (z610: 202107)
    future_month <- 202107

    seed_log(paste("Training:", paste(range(training_months), collapse = " a ")))
    seed_log(paste("Validation:", validate_month))
    seed_log(paste("Final train:", paste(range(final_train_months), collapse = " a ")))
    seed_log(paste("Future:", future_month))

    # Campos buenos (sin clase_ternaria, clase01)
    campos_buenos <- setdiff(
      colnames(dataset),
      c("numero_de_cliente", "foto_mes", "clase_ternaria", "clase01")
    )

    seed_log(paste("Features para modelo:", length(campos_buenos)))

    # Crear datasets LightGBM
    dtrain <- lgb.Dataset(
      data = data.matrix(dataset[foto_mes %in% training_months, campos_buenos, with = FALSE]),
      label = dataset[foto_mes %in% training_months, clase01],
      free_raw_data = FALSE
    )

    dvalidate <- lgb.Dataset(
      data = data.matrix(dataset[foto_mes == validate_month, campos_buenos, with = FALSE]),
      label = dataset[foto_mes == validate_month, clase01],
      free_raw_data = FALSE
    )

    seed_log(paste("Train set:", nrow(dataset[foto_mes %in% training_months]), "filas"))
    seed_log(paste("Validation set:", nrow(dataset[foto_mes == validate_month]), "filas"))

    # ========================================================================
    # HYPERPARAMETER TUNING - LÓGICA EXACTA DE z610
    # ========================================================================

    update_progress(seed_idx, "HYPERPARAMETER_TUNING", "10 iteraciones")
    seed_log("Iniciando Bayesian Optimization (z610 logic: AUC)...")

    inicio_bo <- Sys.time()

    set.seed(semilla)

    # Parámetros FIJOS (z610)
    param_fijos <- list(
      objective = "binary",
      metric = "auc",
      first_metric_only = TRUE,
      boost_from_average = TRUE,
      feature_pre_filter = FALSE,
      verbosity = -100,
      force_row_wise = TRUE,
      seed = semilla,
      max_bin = 31,
      learning_rate = 0.03,
      feature_fraction = 0.5,
      num_iterations = 2048,
      early_stopping_rounds = 200
    )

    # Parámetros a OPTIMIZAR (z610: solo num_leaves y min_data_in_leaf)
    configuracion_bo <- makeParamSet(
      makeIntegerParam("num_leaves", lower = 2L, upper = 256L),
      makeIntegerParam("min_data_in_leaf", lower = 2L, upper = 8192L)
    )

    # Función objetivo (z610: optimiza AUC en validation)
    EstimarGanancia_AUC_lightgbm <- function(x) {

      param_completo <- modifyList(param_fijos, x)

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

    # Configurar BO
    configureMlr(show.learner.output = FALSE)

    obj.fun <- makeSingleObjectiveFunction(
      fn = EstimarGanancia_AUC_lightgbm,
      minimize = FALSE,
      noisy = FALSE,
      par.set = configuracion_bo,
      has.simple.signature = FALSE
    )

    # Checkpoint de BO
    bo_checkpoint_file <- file.path(seed_dir, "BO_checkpoint.RData")

    ctrl <- makeMBOControl(
      save.on.disk.at.time = 300,
      save.file.path = bo_checkpoint_file
    )
    ctrl <- setMBOControlTermination(ctrl, iters = 10L)
    ctrl <- setMBOControlInfill(ctrl, crit = makeMBOInfillCritEI())

    surr.km <- makeLearner(
      "regr.km",
      predict.type = "se",
      covtype = "matern3_2",
      control = list(trace = FALSE)
    )

    # Ejecutar BO
    if (file.exists(bo_checkpoint_file)) {
      seed_log("Checkpoint de BO encontrado, resumiendo...", level = "INFO")
      bayesiana_salida <- mboContinue(bo_checkpoint_file)
    } else {
      bayesiana_salida <- mbo(obj.fun, learner = surr.km, control = ctrl)
    }

    fin_bo <- Sys.time()
    tiempo_bo <- as.numeric(difftime(fin_bo, inicio_bo, units = "mins"))

    # Extraer mejores hiperparámetros
    tb_bayesiana <- as.data.table(bayesiana_salida$opt.path)
    setorder(tb_bayesiana, -y, -num_iterations)

    fwrite(tb_bayesiana, file.path(seed_dir, "BO_log.txt"), sep = "\t")

    mejores_hiperparametros <- tb_bayesiana[
      1,
      setdiff(colnames(tb_bayesiana),
              c("y", "dob", "eol", "error.message", "exec.time", "ei", "error.model",
                "train.time", "prop.type", "propose.time", "se", "mean", "iter")),
      with = FALSE
    ]

    mejor_auc <- tb_bayesiana[1, y]

    seed_log(paste("Hyperparameter Tuning completado en", round(tiempo_bo, 1), "min"),
             level = "SUCCESS")
    seed_log(paste("Mejor AUC:", round(mejor_auc, 6)))
    seed_log(paste("Mejores params:",
                   "num_leaves =", mejores_hiperparametros$num_leaves,
                   "min_data_in_leaf =", mejores_hiperparametros$min_data_in_leaf,
                   "num_iterations =", mejores_hiperparametros$num_iterations))

    # ========================================================================
    # ENTRENAMIENTO MODELO FINAL - LÓGICA z610
    # ========================================================================

    update_progress(seed_idx, "ENTRENANDO_MODELO_FINAL")
    seed_log("Entrenando modelo final con mejores hiperparámetros...")

    inicio_train_final <- Sys.time()

    # Dataset final train
    dfinal_train <- lgb.Dataset(
      data = data.matrix(dataset[foto_mes %in% final_train_months, campos_buenos, with = FALSE]),
      label = dataset[foto_mes %in% final_train_months, clase01],
      free_raw_data = FALSE
    )

    seed_log(paste("Final train set:", nrow(dataset[foto_mes %in% final_train_months]), "filas"))

    # Parámetros finales (fijos + mejores encontrados)
    param_fijos_final <- param_fijos
    param_fijos_final$num_iterations <- NULL
    param_fijos_final$early_stopping_rounds <- NULL

    param_final <- c(param_fijos_final, mejores_hiperparametros)

    set.seed(semilla)

    final_model <- lgb.train(
      data = dfinal_train,
      param = param_final,
      verbose = -100
    )

    fin_train_final <- Sys.time()
    tiempo_train_final <- as.numeric(difftime(fin_train_final, inicio_train_final, units = "mins"))

    seed_log(paste("Modelo final entrenado en", round(tiempo_train_final, 1), "min"),
             level = "SUCCESS")

    # Guardar modelo
    lgb.save(final_model, file.path(seed_dir, "modelo.txt"))

    # Importancia de variables
    tb_importancia <- as.data.table(lgb.importance(final_model))
    fwrite(tb_importancia, file.path(seed_dir, "impo.txt"), sep = "\t")

    # ========================================================================
    # SCORING - LÓGICA EXACTA DE z610
    # ========================================================================

    update_progress(seed_idx, "SCORING")
    seed_log("Generando predicciones...")

    dfuture <- dataset[foto_mes == future_month]

    seed_log(paste("Future set:", nrow(dfuture), "filas"))

    # Predicciones
    prediccion <- predict(
      final_model,
      data.matrix(dfuture[, campos_buenos, with = FALSE])
    )

    # Tabla de predicciones
    tb_prediccion <- dfuture[, list(numero_de_cliente)]
    tb_prediccion[, prob := prediccion]

    fwrite(tb_prediccion, file.path(seed_dir, "prediccion.txt"), sep = "\t")

    # Calcular ganancia - LÓGICA z610
    tb_prediccion[, clase_ternaria := dfuture$clase_ternaria]

    # Ganancias (z610: 117000 para BAJA+2, -3000 para resto)
    tb_prediccion[, ganancia := -3000.0]
    tb_prediccion[clase_ternaria == "BAJA+2", ganancia := 117000.0]

    # Ordenar y acumular
    setorder(tb_prediccion, -prob)
    tb_prediccion[, gan_acum := cumsum(ganancia)]

    # Media móvil de ancho 400 (z610)
    tb_prediccion[,
                  gan_suavizada := frollmean(
                    x = gan_acum,
                    n = 400,
                    align = "center",
                    na.rm = TRUE,
                    hasNA = TRUE
                  )]

    # Ganancia máxima suavizada
    ganancia_suavizada_max <- max(tb_prediccion$gan_suavizada, na.rm = TRUE)
    envios_optimos <- which.max(tb_prediccion$gan_suavizada)

    seed_log(paste("Ganancia máxima suavizada:", formatC(ganancia_suavizada_max, format = "f", big.mark = ",", digits = 0)))
    seed_log(paste("Envíos óptimos:", envios_optimos))

    # Guardar ganancias
    fwrite(tb_prediccion, file.path(seed_dir, "ganancias.txt"), sep = "\t")

    # Crear submission con envíos óptimos
    tb_prediccion[, envios := .I]
    submission <- tb_prediccion[envios <= envios_optimos, .(numero_de_cliente)]
    fwrite(submission, file.path(seed_dir, paste0("submission_", seed_idx, ".csv")))

    seed_log(paste("Submission generado:", nrow(submission), "envíos"), level = "SUCCESS")

    # ========================================================================
    # FIN
    # ========================================================================

    fin_seed <- Sys.time()
    duracion_total <- as.numeric(difftime(fin_seed, inicio_seed, units = "mins"))

    update_progress(seed_idx, "COMPLETADO",
                   paste("Ganancia:", formatC(ganancia_suavizada_max, format = "f", big.mark = ",", digits = 0),
                         "| Envíos:", nrow(submission)))

    seed_log(paste("SEMILLA COMPLETADA EXITOSAMENTE en", round(duracion_total, 1), "minutos"),
             level = "FINISH")

    log_msg(paste("Semilla", seed_idx, "COMPLETADA - Ganancia:", formatC(ganancia_suavizada_max, format = "f", big.mark = ",", digits = 0),
                  "- Envíos:", nrow(submission)),
            level = "SUCCESS")

    return(list(
      success = TRUE,
      seed_idx = seed_idx,
      semilla = semilla,
      ganancia = ganancia_suavizada_max,
      envios = nrow(submission),
      envios_optimos = envios_optimos,
      duracion_min = duracion_total,
      mejor_auc = mejor_auc,
      mejores_params = mejores_hiperparametros
    ))

  }, error = function(e) {
    error_msg <- paste("ERROR en semilla", seed_idx, ":", e$message)
    log_msg(error_msg, level = "ERROR")

    if (exists("seed_log_file")) {
      cat(paste("[ERROR]", error_msg, "\n"), file = seed_log_file, append = TRUE)
    }

    update_progress(seed_idx, "ERROR", e$message)

    return(list(success = FALSE, seed_idx = seed_idx, semilla = semilla,
                error = e$message))
  })
}

# ============================================================================
# EJECUCIÓN EN PARALELO
# ============================================================================

require("parallel")

log_msg(paste("Creando cluster con", NUM_WORKERS, "workers..."), level = "INFO")
cl <- makeCluster(NUM_WORKERS)

clusterExport(cl, c("PARAM_GLOBAL", "BASE_DIR", "DATASETS_DIR", "EXP_DIR",
                    "LOG_FILE", "PROGRESS_FILE", "THREADS_PER_WORKER",
                    "log_msg", "update_progress", "ejecutar_semilla"))

clusterEvalQ(cl, {
  require("data.table")
  require("lightgbm")
  require("DiceKriging")
  require("mlr")
  require("mlrMBO")
  require("ParamHelpers")
})

log_msg("Cluster creado y configurado", level = "SUCCESS")
health_check("Cluster paralelo creado con 5 workers")

inicio_total <- Sys.time()

log_msg(paste(rep("=", 60), collapse = ""), level = "INFO")
log_msg("EJECUCIÓN PARALELA INICIADA", level = "SUCCESS")
log_msg(paste("Hora inicio:", format(inicio_total)), level = "INFO")
log_msg(paste("Hora estimada fin:", format(inicio_total + as.difftime(1, units = "hours"))), level = "INFO")
log_msg(paste(rep("=", 60), collapse = ""), level = "INFO")

cat("\n⚡ TODAS LAS", length(PARAM_GLOBAL$semillas), "SEMILLAS CORRIENDO EN BASELINE MODE ⚡\n\n")
cat("📊 Monitorea el progreso en tiempo real:\n")
cat(paste("   Log principal:", LOG_FILE, "\n"))
cat(paste("   Progreso:", PROGRESS_FILE, "\n"))
cat(paste("   Health checks:", HEALTH_FILE, "\n\n"))

# Preparar parámetros
seeds_params <- lapply(1:length(PARAM_GLOBAL$semillas), function(i) {
  list(
    seed_idx = i,
    semilla = PARAM_GLOBAL$semillas[i],
    PARAM_GLOBAL = PARAM_GLOBAL,
    BASE_DIR = BASE_DIR,
    DATASETS_DIR = DATASETS_DIR,
    EXP_DIR = EXP_DIR,
    LOG_FILE = LOG_FILE,
    PROGRESS_FILE = PROGRESS_FILE,
    THREADS_PER_WORKER = THREADS_PER_WORKER
  )
})

# EJECUTAR EN PARALELO
resultados <- parLapply(cl, seeds_params, function(params) {
  ejecutar_semilla(
    seed_idx = params$seed_idx,
    semilla = params$semilla,
    PARAM_GLOBAL = params$PARAM_GLOBAL,
    BASE_DIR = params$BASE_DIR,
    DATASETS_DIR = params$DATASETS_DIR,
    EXP_DIR = params$EXP_DIR,
    LOG_FILE = params$LOG_FILE,
    PROGRESS_FILE = params$PROGRESS_FILE,
    THREADS_PER_WORKER = params$THREADS_PER_WORKER
  )
})

stopCluster(cl)

fin_total <- Sys.time()
duracion_total <- as.numeric(difftime(fin_total, inicio_total, units = "hours"))

health_check("Ejecución paralela completada")

# ============================================================================
# CONSOLIDACIÓN DE RESULTADOS
# ============================================================================

log_msg(paste(rep("=", 60), collapse = ""))
log_msg("EJECUCIÓN COMPLETADA", level = "SUCCESS")
log_msg(paste("Duración total:", round(duracion_total, 2), "horas"))
log_msg(paste(rep("=", 60), collapse = ""))

exitosas <- sum(sapply(resultados, function(x) x$success))
fallidas <- length(resultados) - exitosas

cat("\n")
cat("========================================\n")
cat(paste("RESUMEN FINAL\n"))
cat("========================================\n")
cat(paste("Semillas exitosas:", exitosas, "/", length(resultados), "\n"))
cat(paste("Semillas fallidas:", fallidas, "\n"))
cat(paste("Tiempo total:", round(duracion_total, 2), "horas\n"))
cat("========================================\n\n")

# Crear tabla resumen
resumen_df <- do.call(rbind, lapply(resultados, function(x) {
  if (x$success) {
    data.frame(
      seed_idx = x$seed_idx,
      semilla = x$semilla,
      ganancia = x$ganancia,
      envios = x$envios,
      envios_optimos = x$envios_optimos,
      duracion_min = round(x$duracion_min, 1),
      mejor_auc = round(x$mejor_auc, 6),
      status = "OK"
    )
  } else {
    data.frame(
      seed_idx = x$seed_idx,
      semilla = x$semilla,
      ganancia = NA,
      envios = NA,
      envios_optimos = NA,
      duracion_min = NA,
      mejor_auc = NA,
      status = paste("ERROR:", x$error)
    )
  }
}))

# Ordenar por ganancia
if (any(!is.na(resumen_df$ganancia))) {
  resumen_df <- resumen_df[order(-resumen_df$ganancia, na.last = TRUE), ]
  resumen_df$rank <- ifelse(!is.na(resumen_df$ganancia),
                            rank(-resumen_df$ganancia),
                            NA)
}

# Guardar resumen
fwrite(resumen_df, file.path(EXP_DIR, "resumen_baseline_exp6160.txt"), sep = "\t")
saveRDS(resultados, file.path(EXP_DIR, "resultados_baseline_exp6160.rds"))

log_msg("Resumen guardado en: resumen_baseline_exp6160.txt", level = "SUCCESS")

# Mostrar tabla
print(resumen_df)

# Mostrar semillas fallidas
if (fallidas > 0) {
  cat("\n⚠  SEMILLAS FALLIDAS:\n")
  for (r in resultados) {
    if (!r$success) {
      cat(paste("  - Semilla", r$seed_idx, ":", r$error, "\n"))
    }
  }
}

# Mejor semilla
if (any(!is.na(resumen_df$ganancia))) {
  mejor <- resumen_df[which.max(resumen_df$ganancia), ]
  cat("\n🏆 MEJOR SEMILLA:\n")
  cat(paste("  Semilla:", mejor$semilla, "\n"))
  cat(paste("  Ganancia:", formatC(mejor$ganancia, format = "f", big.mark = ",", digits = 0), "\n"))
  cat(paste("  Envíos:", mejor$envios, "\n"))
  cat(paste("  Duración:", mejor$duracion_min, "minutos\n"))
}

log_msg("WORKFLOW BASELINE FINALIZADO EXITOSAMENTE", level = "SUCCESS")

cat("\n✨ LISTO! Revisa los archivos en exp/ ✨\n\n")
