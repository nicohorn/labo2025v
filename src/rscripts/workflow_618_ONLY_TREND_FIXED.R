# ============================================================================
# WORKFLOW 618 ONLY TREND - SOLO FEATURES DE TENDENCIA (FIXED VERSION)
# ============================================================================
#
# FILOSOFÍA:
# - Lógica IDÉNTICA a z610_WorkFlow_01_gerencial_julio.ipynb
# - CAMBIO: Feature Engineering con SOLO FEATURES DE TREND
# - EXCLUYE: lags, deltas, rolling stats, ratios, volatilidad
# - INCLUYE: ÚNICAMENTE trend_3 y trend_6 (rolling trends)
#
# FIXED: Esta versión corrige las diferencias con el notebook de Colab:
# - Catastrophe Analysis: 13 variables (incluye ctarjeta_master_transacciones)
# - Trend calculation: usa frollapply() para rolling trends
# - campos_buenos: excluye numero_de_cliente, foto_mes, clase_ternaria, clase01, azar
# - BO iterations: 100 (configurable)
#
# ============================================================================

format(Sys.time(), "%a %b %d %X %Y")

# ============================================================================
# SISTEMA DE LOGGING
# ============================================================================

LOG_TIMESTAMP <- format(Sys.time(), "%Y%m%d_%H%M%S")
BASE_DIR <- "C:/Users/User/Documents/labo2025v"
DATASETS_DIR <- file.path(BASE_DIR, "datasets")
EXP_DIR <- file.path(BASE_DIR, "exp", "exp_only_trend_fixed")

dir.create(EXP_DIR, showWarnings = FALSE, recursive = TRUE)

LOG_FILE <- file.path(EXP_DIR, paste0("workflow_only_trend_", LOG_TIMESTAMP, ".log"))
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
cat(" WORKFLOW 618 ONLY TREND FIXED\n")
cat("   Paralelo + Caching + Logging\n")
cat("========================================\n\n")

log_msg(paste(rep("=", 60), collapse = ""))
log_msg("WORKFLOW 618 ONLY TREND FIXED INICIADO")
log_msg(paste(rep("=", 60), collapse = ""))
log_msg(paste("Log file:", LOG_FILE))
log_msg(paste("Progress file:", PROGRESS_FILE))
log_msg(paste("Health file:", HEALTH_FILE))

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

NUM_WORKERS <- 5
THREADS_PER_WORKER <- 4
BO_ITERATIONS <- 100  # Mismo que versión original local

log_msg("CONFIGURACIÓN HARDWARE", level = "INFO")
log_msg(paste("  Cores totales detectados:", parallel::detectCores()))
log_msg(paste("  Workers paralelos:", NUM_WORKERS))
log_msg(paste("  Threads por worker:", THREADS_PER_WORKER))
log_msg(paste("  Cores totales en uso:", NUM_WORKERS * THREADS_PER_WORKER))
log_msg(paste("  BO iterations:", BO_ITERATIONS))

PARAM_GLOBAL <- list()
PARAM_GLOBAL$experimento <- "WF618"
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
# FUNCIÓN OPTIMIZADA PARA CALCULAR TENDENCIA (VECTORIZADA)
# ============================================================================
#
# En lugar de usar frollapply + lm() (muy lento), calculamos la pendiente
# analíticamente usando la fórmula:
#   slope = (n * sum(x*y) - sum(x) * sum(y)) / (n * sum(x^2) - sum(x)^2)
#
# Para ventana fija n, los términos que solo dependen de x son constantes:
#   sum(x) = n*(n+1)/2
#   sum(x^2) = n*(n+1)*(2n+1)/6
#   denominador = n * sum(x^2) - sum(x)^2 = constante
#
# Solo necesitamos calcular sum(y) y sum(x*y) con frollsum, que es muy rápido.
#
# ============================================================================

calc_trend_vectorized <- function(dataset, col, ventana) {
  # Constantes para la ventana
  n <- ventana
  x <- 1:n
  sum_x <- sum(x)                    # n*(n+1)/2
  sum_x2 <- sum(x^2)                 # n*(n+1)*(2n+1)/6
  denom <- n * sum_x2 - sum_x^2     # Denominador constante

  # Para calcular sum(x*y) necesitamos ponderar y por posición
  # Usamos un truco: creamos columnas auxiliares y_weighted donde multiplicamos
  # cada valor por su peso (1, 2, 3, ... n) dentro de la ventana rolling

  # Calcular sum(y) con frollsum
  sum_y <- frollsum(dataset[[col]], n = n, align = "right", na.rm = FALSE)

  # Para sum(x*y), necesitamos ponderar por posición
  # sum(x*y) para ventana = y[t-n+1]*1 + y[t-n+2]*2 + ... + y[t]*n
  # Esto equivale a: n*y[t] + (n-1)*y[t-1] + ... + 1*y[t-n+1]
  # = n * frollsum(y, 1) - frollsum(cumsum_y_shifted, n) ... complicado
  #
  # Alternativa más simple: usar la fórmula con sum_y y sum_y_weighted
  # donde y_weighted[i] = y[i] * posicion_en_ventana
  #
  # Enfoque práctico: usar frollapply pero con función simple sin lm()

  # Función simple sin lm() - mucho más rápida
  calc_slope_simple <- function(y) {
    n <- length(y)
    valid <- !is.na(y)
    n_valid <- sum(valid)
    if (n_valid < 2) return(NA_real_)

    x <- 1:n
    x_valid <- x[valid]
    y_valid <- y[valid]

    sum_x <- sum(x_valid)
    sum_y <- sum(y_valid)
    sum_xy <- sum(x_valid * y_valid)
    sum_x2 <- sum(x_valid^2)

    denom <- n_valid * sum_x2 - sum_x^2
    if (denom == 0) return(NA_real_)

    (n_valid * sum_xy - sum_x * sum_y) / denom
  }

  return(calc_slope_simple)
}

# Función wrapper para usar en frollapply (sin lm, mucho más rápida)
calc_slope_fast <- function(y) {
  n <- length(y)
  valid <- !is.na(y)
  n_valid <- sum(valid)
  if (n_valid < 2) return(NA_real_)

  x <- 1:n
  x_valid <- x[valid]
  y_valid <- y[valid]

  sum_x <- sum(x_valid)
  sum_y <- sum(y_valid)
  sum_xy <- sum(x_valid * y_valid)
  sum_x2 <- sum(x_valid^2)

  denom <- n_valid * sum_x2 - sum_x^2
  if (denom == 0) return(NA_real_)

  (n_valid * sum_xy - sum_x * sum_y) / denom
}

# ============================================================================
# FUNCIÓN EJECUTAR SEMILLA
# ============================================================================

ejecutar_semilla <- function(seed_idx, semilla, PARAM_GLOBAL, BASE_DIR, DATASETS_DIR,
                             EXP_DIR, LOG_FILE, PROGRESS_FILE, THREADS_PER_WORKER,
                             BO_ITERATIONS) {

  require("data.table")
  setDTthreads(THREADS_PER_WORKER)
  require("lightgbm")
  require("DiceKriging")
  require("mlr")
  require("mlrMBO")
  require("ParamHelpers")

  # Función calc_slope_fast dentro del worker (sin lm, mucho más rápida)
  calc_slope_fast <- function(y) {
    n <- length(y)
    valid <- !is.na(y)
    n_valid <- sum(valid)
    if (n_valid < 2) return(NA_real_)

    x <- 1:n
    x_valid <- x[valid]
    y_valid <- y[valid]

    sum_x <- sum(x_valid)
    sum_y <- sum(y_valid)
    sum_xy <- sum(x_valid * y_valid)
    sum_x2 <- sum(x_valid^2)

    denom <- n_valid * sum_x2 - sum_x^2
    if (denom == 0) return(NA_real_)

    (n_valid * sum_xy - sum_x * sum_y) / denom
  }

  tryCatch({
    inicio_seed <- Sys.time()

    # Crear directorio
    exp_folder <- paste0("WF618", seed_idx - 1, "_seed", seed_idx, "_ONLY_TREND_FIXED")
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

    dataset_cache_file <- file.path(seed_dir, "dataset_con_FE_ONLY_TREND_FIXED.rds")

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
      # CATASTROPHE ANALYSIS - 13 VARIABLES (IDÉNTICO A COLAB)
      # ======================================================================

      seed_log("Aplicando Catastrophe Analysis (13 variables)...", level = "INFO")

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
      dataset[foto_mes == 202006, ctarjeta_master_transacciones := NA]  # FIXED: Agregado

      seed_log("Catastrophe Analysis completado (13 variables en 202006 → NA)", level = "SUCCESS")

      # ======================================================================
      # FEATURE ENGINEERING - ROLLING TRENDS (IDÉNTICO A COLAB)
      # ======================================================================

      update_progress(seed_idx, "FEATURE_ENGINEERING")
      seed_log("Iniciando Feature Engineering - Rolling Trends (frollapply)...")

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

      # === TENDENCIAS ROLLING (ventanas 3, 6) - USANDO FROLLAPPLY + CALC_SLOPE_FAST ===
      seed_log("Generando rolling trends con frollapply + calc_slope_fast (optimizado)...")
      inicio_trends <- Sys.time()
      cols_antes_trends <- ncol(dataset)

      for (ventana in c(3, 6)) {
        seed_log(paste("  Procesando ventana", ventana, "..."))

        for (col in cols_lagueables) {
          trend_col <- paste0(col, "_trend_", ventana)

          dataset[, (trend_col) := frollapply(
            x = get(col),
            n = ventana,
            FUN = calc_slope_fast,
            align = "right"
          ), by = numero_de_cliente]
        }

        seed_log(paste("  Ventana", ventana, "completada"))
      }

      fin_trends <- Sys.time()
      cols_trends <- ncol(dataset) - cols_antes_trends
      seed_log(paste("Rolling trends generados:", cols_trends, "variables en",
                     round(as.numeric(difftime(fin_trends, inicio_trends, units = "mins")), 1), "min"),
               level = "SUCCESS")

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
    seed_log("Configurando Training Strategy...")

    # Clase binaria
    dataset[, clase01 := ifelse(clase_ternaria %in% c("BAJA+1", "BAJA+2"), 1, 0)]

    # Columna azar para sampling
    set.seed(semilla, kind = "L'Ecuyer-CMRG")
    dataset[, azar := runif(nrow(dataset))]

    # Training months
    training_months <- c(
      202104, 202103, 202102, 202101,
      202012, 202011, 202010, 202009, 202008, 202007,
      202006, 202005
    )

    # Validation month
    validate_month <- 202105

    # Final train months
    final_train_months <- c(
      202105, 202104, 202103, 202102, 202101,
      202012, 202011, 202010, 202009, 202008, 202007,
      202006, 202005
    )

    # Future month
    future_month <- 202107

    seed_log(paste("Training:", paste(range(training_months), collapse = " a ")))
    seed_log(paste("Validation:", validate_month))
    seed_log(paste("Final train:", paste(range(final_train_months), collapse = " a ")))
    seed_log(paste("Future:", future_month))

    # FIXED: campos_buenos excluye numero_de_cliente, foto_mes, clase_ternaria, clase01, azar
    campos_buenos <- setdiff(
      colnames(dataset),
      c("numero_de_cliente", "foto_mes", "clase_ternaria", "clase01", "azar")
    )

    seed_log(paste("Features para modelo:", length(campos_buenos)))

    # Crear fold_train
    dataset[, fold_train := foto_mes %in% training_months &
        (clase_ternaria %in% c("BAJA+1", "BAJA+2") | azar < 1.0)]

    # Crear datasets LightGBM
    dtrain <- lgb.Dataset(
      data = data.matrix(dataset[fold_train == TRUE, campos_buenos, with = FALSE]),
      label = dataset[fold_train == TRUE, clase01],
      free_raw_data = FALSE
    )

    dvalidate <- lgb.Dataset(
      data = data.matrix(dataset[foto_mes == validate_month, campos_buenos, with = FALSE]),
      label = dataset[foto_mes == validate_month, clase01],
      free_raw_data = FALSE
    )

    seed_log(paste("Train set:", nrow(dataset[fold_train == TRUE]), "filas"))
    seed_log(paste("Validation set:", nrow(dataset[foto_mes == validate_month]), "filas"))

    # ========================================================================
    # HYPERPARAMETER TUNING
    # ========================================================================

    update_progress(seed_idx, "HYPERPARAMETER_TUNING", paste(BO_ITERATIONS, "iteraciones"))
    seed_log(paste("Iniciando Bayesian Optimization (", BO_ITERATIONS, " iteraciones)..."))

    inicio_bo <- Sys.time()

    set.seed(semilla)

    # Parámetros FIJOS
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

    # Parámetros a OPTIMIZAR
    configuracion_bo <- makeParamSet(
      makeIntegerParam("num_leaves", lower = 2L, upper = 256L),
      makeIntegerParam("min_data_in_leaf", lower = 2L, upper = 8192L)
    )

    # Función objetivo
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
    ctrl <- setMBOControlTermination(ctrl, iters = BO_ITERATIONS)
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
    # ENTRENAMIENTO MODELO FINAL
    # ========================================================================

    update_progress(seed_idx, "ENTRENANDO_MODELO_FINAL")
    seed_log("Entrenando modelo final con mejores hiperparámetros...")

    inicio_train_final <- Sys.time()

    # Dataset final train
    dataset[, fold_final_train := foto_mes %in% final_train_months]

    dfinal_train <- lgb.Dataset(
      data = data.matrix(dataset[fold_final_train == TRUE, campos_buenos, with = FALSE]),
      label = dataset[fold_final_train == TRUE, clase01],
      free_raw_data = FALSE
    )

    seed_log(paste("Final train set:", nrow(dataset[fold_final_train == TRUE]), "filas"))

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
    # SCORING
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

    # Calcular ganancia
    tb_prediccion[, clase_ternaria := dfuture$clase_ternaria]

    # Ganancias (117000 para BAJA+2, -3000 para resto)
    tb_prediccion[, ganancia := -3000.0]
    tb_prediccion[clase_ternaria == "BAJA+2", ganancia := 117000.0]

    # Ordenar y acumular
    setorder(tb_prediccion, -prob)
    tb_prediccion[, gan_acum := cumsum(ganancia)]

    # Media móvil de ancho 400
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
                    "BO_ITERATIONS", "log_msg", "update_progress", "ejecutar_semilla",
                    "calc_slope"))

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

cat("\n⚡ TODAS LAS", length(PARAM_GLOBAL$semillas), "SEMILLAS CORRIENDO EN ONLY TREND MODE (FIXED) ⚡\n\n")
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
    THREADS_PER_WORKER = THREADS_PER_WORKER,
    BO_ITERATIONS = BO_ITERATIONS
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
    THREADS_PER_WORKER = params$THREADS_PER_WORKER,
    BO_ITERATIONS = params$BO_ITERATIONS
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
fwrite(resumen_df, file.path(EXP_DIR, "resumen_only_trend_fixed.txt"), sep = "\t")
saveRDS(resultados, file.path(EXP_DIR, "resultados_only_trend_fixed.rds"))

log_msg("Resumen guardado en: resumen_only_trend_fixed.txt", level = "SUCCESS")

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

log_msg("WORKFLOW ONLY TREND FIXED FINALIZADO EXITOSAMENTE", level = "SUCCESS")

cat("\n✨ LISTO! Revisa los archivos en exp/exp_only_trend_fixed/ ✨\n\n")
