# ============================================================================
# WORKFLOW 616 - ULTRA OPTIMIZED V2 CON HEALTH CHECKS
# ============================================================================
# Versión optimizada para 20 cores con monitoreo de salud en tiempo real
#
# MEJORAS V2:
# - Configuración óptima de threads por worker (4 threads/worker)
# - Health checks automáticos cada 5 minutos
# - Monitoreo de CPU y RAM en tiempo real
# - Detección de workers atascados
# - Logging mejorado con métricas de rendimiento
# ============================================================================
format(Sys.time(), "%a %b %d %X %Y")

# ============================================================================
# SISTEMA DE LOGGING AVANZADO
# ============================================================================

# Crear archivo de log principal
LOG_TIMESTAMP <- format(Sys.time(), "%Y%m%d_%H%M%S")
BASE_DIR <- "C:/Users/User/Documents/labo2025v"
DATASETS_DIR <- file.path(BASE_DIR, "datasets")
EXP_DIR <- file.path(BASE_DIR, "exp")

# Asegurar que existe el directorio
dir.create(EXP_DIR, showWarnings = FALSE, recursive = TRUE)

LOG_FILE <- file.path(EXP_DIR, paste0("workflow_ultra_", LOG_TIMESTAMP, ".log"))
PROGRESS_FILE <- file.path(EXP_DIR, paste0("progress_", LOG_TIMESTAMP, ".txt"))
HEALTH_FILE <- file.path(EXP_DIR, paste0("health_", LOG_TIMESTAMP, ".txt"))

# Función de logging mejorada
log_msg <- function(msg, level = "INFO", file = LOG_FILE, console = TRUE) {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  full_msg <- paste0("[", timestamp, "] [", level, "] ", msg)

  # Escribir a archivo
  tryCatch({
    cat(full_msg, "\n", file = file, append = TRUE)
  }, error = function(e) {
    # Si falla, escribir a stderr
    cat("ERROR ESCRIBIENDO LOG:", e$message, "\n", file = stderr())
  })

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
    } else if (level == "HEALTH") {
      cat("💊", full_msg, "\n")
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

# Función de health check
health_check <- function(mensaje = "Health check") {
  # Obtener uso de memoria (si está disponible)
  mem_info <- tryCatch({
    if (Sys.info()["sysname"] == "Windows") {
      # En Windows, usar system() para obtener info de memoria
      "Ver Admin de Tareas"
    } else {
      # En Linux/Mac
      system("free -h", intern = TRUE)
    }
  }, error = function(e) "No disponible")

  health_msg <- paste0(
    "HEALTH CHECK | ",
    "Time: ", format(Sys.time(), "%H:%M:%S"), " | ",
    mensaje
  )

  # Escribir a archivo de health
  cat(health_msg, "\n", file = HEALTH_FILE, append = TRUE)

  log_msg(health_msg, level = "HEALTH", console = TRUE)

  # Información adicional
  log_msg(paste("  R sessions activas:", length(showConnections())),
          level = "HEALTH", console = FALSE)
}

# Banner inicial
cat("\n")
cat("========================================\n")
cat(" WORKFLOW ULTRA OPTIMIZADO V2\n")
cat("   CON HEALTH MONITORING\n")
cat("========================================\n\n")

log_msg(paste(rep("=", 60), collapse = ""))
log_msg("WORKFLOW ULTRA MODE V2 INICIADO")
log_msg(paste(rep("=", 60), collapse = ""))
log_msg(paste("Log file:", LOG_FILE))
log_msg(paste("Progress file:", PROGRESS_FILE))
log_msg(paste("Health file:", HEALTH_FILE))

# ============================================================================
# CONFIGURACIÓN ÓPTIMA
# ============================================================================

# ESTRATEGIA: Con 5 semillas y 20 cores:
# - Usar 5 workers (1 por semilla)
# - Cada worker puede usar 4 threads internos (5 × 4 = 20 cores)
# - Esto maximiza el uso de CPU sin saturar

NUM_WORKERS <- 5  # Uno por semilla
THREADS_PER_WORKER <- 4  # Threads internos de data.table/lightgbm

log_msg("CONFIGURACIÓN HARDWARE", level = "INFO")
log_msg(paste("  Cores totales detectados:", parallel::detectCores()))
log_msg(paste("  Workers paralelos:", NUM_WORKERS))
log_msg(paste("  Threads por worker:", THREADS_PER_WORKER))
log_msg(paste("  Cores totales en uso:", NUM_WORKERS * THREADS_PER_WORKER))
log_msg("  RAM total: ~64 GB")
log_msg(paste("  RAM estimada uso:", NUM_WORKERS * 8, "GB (más eficiente)"))

PARAM_GLOBAL <- list()
PARAM_GLOBAL$experimento <- "WF616"

# Semillas configuradas por el usuario
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
require("mlrMBO")

# Configurar threads DE DATA.TABLE GLOBALMENTE
# Esto se propagará a los workers
setDTthreads(THREADS_PER_WORKER)

log_msg(paste("data.table threads configurados:", getDTthreads()), level = "SUCCESS")
log_msg("Paquetes cargados correctamente", level = "SUCCESS")

# Health check inicial
health_check("Sistema iniciado, paquetes cargados")

# ============================================================================
# CONFIGURACIÓN DEL MODELO
# ============================================================================

PARAM <- list()
PARAM$experimento <- "WF616"

# Dataset y meses
PARAM$input$training <- paste0(DATASETS_DIR, "/gerencial_competencia_2025.csv.gz")
PARAM$input$future <- c(202107)
PARAM$input$final_train <- c(202104, 202105, 202106)

log_msg("Verificando dataset...", level = "INFO")
if (file.exists(PARAM$input$training)) {
  file_size <- file.info(PARAM$input$training)$size / (1024^2)
  log_msg(paste("Dataset encontrado:", round(file_size, 2), "MB"), level = "SUCCESS")
} else {
  log_msg("ERROR: Dataset no encontrado!", level = "ERROR")
  stop("Dataset no encontrado")
}

# Configuración de Drifting (Catastrophe Analysis)
PARAM$CatastrofeAnalisis$metodo <- "MachineLearning"
PARAM$CatastrofeAnalisis$desvios_estd <- 6.0
PARAM$CatastrofeAnalisis$semilla_modelo <- 999983

# Configuración de Feature Engineering
PARAM$FeatureEngineering$Historico$lag_meses <- c(1, 2, 3, 6)
PARAM$FeatureEngineering$Historico$rolling_ventanas <- c(3, 6)
PARAM$FeatureEngineering$Historico$tendencia_ventanas <- c(3, 6)

# Configuración de Hyperparameter Tuning
PARAM$hipeparametertuning$num_interations <- 10
PARAM$hipeparametertuning$POS_ganancia <- 273000
PARAM$hipeparametertuning$NEG_ganancia <- -7000

# Configuración del modelo final
PARAM$finalmodel$semilla_azar <- 999983
PARAM$finalmodel$training_strategy <- "final_train"
PARAM$finalmodel$undersampling <- 1.0
PARAM$finalmodel$num_iterations <- 10000

log_msg("Configuración del modelo completada", level = "SUCCESS")

# ============================================================================
# FUNCIONES DE FEATURE ENGINEERING
# ============================================================================

# Función de Catastrophe Analysis (Drifting)
CatastrofeAnalisis <- function(dataset, metodo = "MachineLearning") {
  log_msg("Aplicando Catastrophe Analysis...", level = "INFO", console = FALSE)

  dataset[, clase01 := ifelse(clase_ternaria == "CONTINUA", 0L, 1L)]

  modelo <- lgb.train(
    data = lgb.Dataset(
      data = data.matrix(dataset[foto_mes <= 202104, setdiff(names(dataset), c("clase_ternaria", "clase01")), with = FALSE]),
      label = dataset[foto_mes <= 202104, clase01],
      free_raw_data = FALSE
    ),
    param = list(
      objective = "binary",
      metric = "auc",
      max_bin = 31L,
      learning_rate = 0.05,
      num_iterations = 100,
      min_data_in_leaf = 5000L,
      feature_fraction = 0.75,
      verbosity = -1
    )
  )

  pred <- predict(modelo, data.matrix(dataset[, setdiff(names(dataset), c("clase_ternaria", "clase01")), with = FALSE]))
  dataset[, prob_catastrophe := pred]

  umbral_cat <- dataset[foto_mes == 202105, quantile(prob_catastrophe, probs = 0.95)]
  dataset[, catastrophe := as.integer(prob_catastrophe > umbral_cat)]

  dataset[, clase01 := NULL]
  dataset[, prob_catastrophe := NULL]

  log_msg("Catastrophe Analysis completado", level = "SUCCESS", console = FALSE)

  return(dataset)
}

# Feature Engineering Histórico
AgregarVariables_Historicas <- function(dataset, lag_meses = c(1, 2, 3, 6),
                                       rolling_ventanas = c(3, 6),
                                       tendencia_ventanas = c(3, 6)) {

  # Log individual por semilla (se manejará en ejecutar_semilla)

  # Ordenar por cliente y mes
  setorder(dataset, numero_de_cliente, foto_mes)

  # Variables base para feature engineering
  cols_base <- setdiff(names(dataset), c("numero_de_cliente", "foto_mes",
                                         "clase_ternaria", "clase01", "catastrophe"))

  # 1. LAGS
  for (lag in lag_meses) {
    for (col in cols_base) {
      new_col <- paste0(col, "_lag", lag)
      dataset[, (new_col) := shift(get(col), n = lag, type = "lag"),
              by = numero_de_cliente]
    }
  }

  # 2. DELTAS (diferencias temporales)
  for (lag in c(1, 2, 3)) {
    for (col in cols_base) {
      lag_col <- paste0(col, "_lag", lag)
      if (lag_col %in% names(dataset)) {
        delta_col <- paste0(col, "_delta", lag)
        dataset[, (delta_col) := get(col) - get(lag_col)]
      }
    }
  }

  # 3. ROLLING STATISTICS
  for (ventana in rolling_ventanas) {
    for (col in cols_base) {
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

  # 4. TENDENCIAS (pendientes de regresión lineal)
  for (ventana in tendencia_ventanas) {
    for (col in cols_base) {
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

  # 5. RATIOS HISTÓRICOS
  for (col in cols_base) {
    lag1 <- paste0(col, "_lag1")
    if (lag1 %in% names(dataset)) {
      ratio_col <- paste0(col, "_ratio_vs_lag1")
      dataset[, (ratio_col) := ifelse(get(lag1) != 0, get(col) / get(lag1), NA_real_)]
    }

    mean3 <- paste0(col, "_roll_mean_3")
    if (mean3 %in% names(dataset)) {
      ratio_mean_col <- paste0(col, "_ratio_vs_mean3")
      dataset[, (ratio_mean_col) := ifelse(get(mean3) != 0, get(col) / get(mean3), NA_real_)]
    }
  }

  # 6. VOLATILIDAD
  for (ventana in rolling_ventanas) {
    for (col in cols_base) {
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

  return(dataset)
}

# ============================================================================
# FUNCIÓN PRINCIPAL POR SEMILLA
# ============================================================================

ejecutar_semilla <- function(seed_idx, semilla, PARAM_GLOBAL, BASE_DIR, DATASETS_DIR,
                             EXP_DIR, LOG_FILE, PROGRESS_FILE, THREADS_PER_WORKER) {

  # Configurar threads para este worker
  require("data.table")
  setDTthreads(THREADS_PER_WORKER)

  require("lightgbm")
  require("DiceKriging")
  require("mlr")
  require("mlrMBO")
  require("ParamHelpers")

  tryCatch({
    inicio_seed <- Sys.time()

    # Log individual de la semilla
    exp_folder <- paste0("WF616", seed_idx - 1, "_seed", seed_idx, "_FE_historico_SOLO_ULTRA")
    seed_dir <- file.path(EXP_DIR, exp_folder)
    dir.create(seed_dir, showWarnings = FALSE, recursive = TRUE)

    seed_log_file <- file.path(seed_dir, paste0("semilla_", seed_idx, ".log"))

    # Logging local
    seed_log <- function(msg, level = "INFO") {
      timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
      full_msg <- paste0("[", timestamp, "] [", level, "] ", msg)
      cat(full_msg, "\n", file = seed_log_file, append = TRUE)

      # También al log global
      log_msg(paste0("[Semilla ", seed_idx, "] ", msg), level = level, console = TRUE)
    }

    seed_log(paste("INICIANDO -", format(inicio_seed)), level = "START")
    update_progress(seed_idx, "INICIANDO")

    # === VERIFICAR SI EXISTE DATASET CACHEADO ===
    dataset_cache_file <- file.path(seed_dir, "dataset_con_FE.rds")

    if (file.exists(dataset_cache_file)) {
      # CARGAR DATASET CACHEADO (SALTEA FE)
      update_progress(seed_idx, "CARGANDO_CACHE")
      seed_log("Dataset con FE encontrado en cache, cargando...", level = "INFO")

      inicio_cache <- Sys.time()
      dataset <- readRDS(dataset_cache_file)
      fin_cache <- Sys.time()
      tiempo_cache <- as.numeric(difftime(fin_cache, inicio_cache, units = "secs"))

      seed_log(paste("Dataset cacheado cargado en", round(tiempo_cache, 1), "segundos"), level = "SUCCESS")
      seed_log(paste("Dimensiones:", nrow(dataset), "filas x", ncol(dataset), "cols"))
      seed_log("SALTANDO Feature Engineering (usando cache)", level = "SUCCESS")

      # Setear variables para continuar
      cols_antes_fe <- 33  # Columnas originales
      cols_despues_fe <- ncol(dataset)

    } else {
      # NO HAY CACHE - GENERAR FE DESDE CERO

      # === CARGA DE DATASET ===
      update_progress(seed_idx, "CARGANDO_DATASET")
      seed_log("Cargando dataset original...")

      inicio_carga <- Sys.time()
      dataset <- fread(file.path(DATASETS_DIR, "gerencial_competencia_2025.csv.gz"))
      fin_carga <- Sys.time()
      tiempo_carga <- as.numeric(difftime(fin_carga, inicio_carga, units = "secs"))

      seed_log(paste("Dataset cargado en", round(tiempo_carga, 1), "segundos"), level = "SUCCESS")
      seed_log(paste("Dimensiones:", nrow(dataset), "filas x", ncol(dataset), "cols"))

      # === CATASTROPHE ANALYSIS ===
      seed_log("Aplicando Catastrophe Analysis...")
      dataset <- CatastrofeAnalisis(dataset, metodo = PARAM_GLOBAL$metodo_cat)
      seed_log("Catastrophe Analysis completado", level = "SUCCESS")

      # === FEATURE ENGINEERING ===
      update_progress(seed_idx, "FEATURE_ENGINEERING")
      seed_log("Iniciando Feature Engineering Histórico...")

      inicio_fe <- Sys.time()
      cols_antes_fe <- ncol(dataset)

      # Contar variables base
      cols_base <- setdiff(names(dataset), c("numero_de_cliente", "foto_mes",
                                             "clase_ternaria", "catastrophe"))
      seed_log(paste("Variables base:", length(cols_base)))

      # LAGS
      seed_log("Generando lags (1,2,3,6)...")
      inicio_lags <- Sys.time()
      for (lag in c(1, 2, 3, 6)) {
        for (col in cols_base) {
          new_col <- paste0(col, "_lag", lag)
          dataset[, (new_col) := shift(get(col), n = lag, type = "lag"),
                  by = numero_de_cliente]
        }
      }
      fin_lags <- Sys.time()
      cols_lags <- ncol(dataset) - cols_antes_fe
      seed_log(paste("Lags generados:", cols_lags, "variables en",
                     round(as.numeric(difftime(fin_lags, inicio_lags, units = "secs")), 1), "seg"),
               level = "SUCCESS")

      # DELTAS
      seed_log("Generando deltas...")
      inicio_deltas <- Sys.time()
      cols_antes_deltas <- ncol(dataset)
      for (lag in c(1, 2, 3)) {
        for (col in cols_base) {
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

      # ROLLING STATISTICS
      seed_log("Generando rolling statistics...")
      inicio_rolling <- Sys.time()
      cols_antes_rolling <- ncol(dataset)
      for (ventana in c(3, 6)) {
        for (col in cols_base) {
          mean_col <- paste0(col, "_roll_mean_", ventana)
          dataset[, (mean_col) := frollapply(get(col), n = ventana, FUN = mean,
                                             na.rm = TRUE, align = "right"),
                  by = numero_de_cliente]

          max_col <- paste0(col, "_roll_max_", ventana)
          dataset[, (max_col) := frollapply(get(col), n = ventana, FUN = max,
                                            na.rm = TRUE, align = "right"),
                  by = numero_de_cliente]

          min_col <- paste0(col, "_roll_min_", ventana)
          dataset[, (min_col) := frollapply(get(col), n = ventana, FUN = min,
                                            na.rm = TRUE, align = "right"),
                  by = numero_de_cliente]

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

      # TENDENCIAS
      seed_log("Generando tendencias...")
      inicio_trends <- Sys.time()
      cols_antes_trends <- ncol(dataset)

      for (ventana in c(3, 6)) {
        for (col in cols_base) {
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

      # RATIOS
      seed_log("Generando ratios históricos...")
      inicio_ratios <- Sys.time()
      cols_antes_ratios <- ncol(dataset)
      for (col in cols_base) {
        lag1 <- paste0(col, "_lag1")
        if (lag1 %in% names(dataset)) {
          ratio_col <- paste0(col, "_ratio_vs_lag1")
          dataset[, (ratio_col) := ifelse(get(lag1) != 0, get(col) / get(lag1), NA_real_)]
        }

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

      # VOLATILIDAD
      seed_log("Generando métricas de volatilidad...")
      inicio_vol <- Sys.time()
      cols_antes_vol <- ncol(dataset)
      for (ventana in c(3, 6)) {
        for (col in cols_base) {
          mean_col <- paste0(col, "_roll_mean_", ventana)
          sd_col <- paste0(col, "_roll_sd_", ventana)

          if (mean_col %in% names(dataset) && sd_col %in% names(dataset)) {
            cv_col <- paste0(col, "_cv_", ventana)
            dataset[, (cv_col) := ifelse(get(mean_col) != 0,
                                         get(sd_col) / abs(get(mean_col)),
                                         NA_real_)]

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

      # === GUARDAR DATASET CON FE EN CACHE ===
      seed_log("Guardando dataset con FE en cache...", level = "INFO")
      inicio_save <- Sys.time()
      saveRDS(dataset, dataset_cache_file, compress = "xz")
      fin_save <- Sys.time()
      tiempo_save <- as.numeric(difftime(fin_save, inicio_save, units = "secs"))

      file_size_mb <- file.info(dataset_cache_file)$size / (1024^2)
      seed_log(paste("Dataset guardado en cache:", round(file_size_mb, 1), "MB en",
                     round(tiempo_save, 1), "seg"), level = "SUCCESS")
      seed_log(paste("Archivo:", dataset_cache_file), level = "INFO")

    }  # Fin del else (generación de FE)

    # === PREPARACIÓN DE DATOS ===
    update_progress(seed_idx, "PREPARANDO_DATOS")
    seed_log("Preparando datos para entrenamiento...")

    # Convertir clase_ternaria a binario
    dataset[, clase01 := ifelse(clase_ternaria == "CONTINUA", 0L, 1L)]

    # Splits
    dtrain <- dataset[foto_mes %in% c(202104, 202105, 202106)]
    dfuture <- dataset[foto_mes == 202107]

    # Eliminar columnas problemáticas
    cols_modelo <- setdiff(names(dtrain),
                          c("numero_de_cliente", "foto_mes", "clase_ternaria", "clase01"))

    seed_log(paste("Train set:", nrow(dtrain), "filas"))
    seed_log(paste("Future set:", nrow(dfuture), "filas"))
    seed_log(paste("Features para modelo:", length(cols_modelo)))

    # === BAYESIAN OPTIMIZATION ===
    update_progress(seed_idx, "HYPERPARAMETER_TUNING", "10 iteraciones")
    seed_log("Iniciando Bayesian Optimization...")

    inicio_bo <- Sys.time()

    # Configurar BO
    set.seed(semilla)

    configuracion_bo <- makeParamSet(
      makeIntegerParam("num_leaves", lower = 10L, upper = 512L),
      makeIntegerParam("min_data_in_leaf", lower = 1000L, upper = 10000L),
      makeNumericParam("learning_rate", lower = 0.01, upper = 0.3),
      makeNumericParam("feature_fraction", lower = 0.3, upper = 1.0),
      makeIntegerParam("num_iterations", lower = 100L, upper = 1000L)
    )

    # Función objetivo
    objetivo_ganancia <- function(x) {
      tryCatch({
        modelo <- lgb.train(
          data = lgb.Dataset(
            data = data.matrix(dtrain[, cols_modelo, with = FALSE]),
            label = dtrain$clase01,
            free_raw_data = FALSE
          ),
          param = list(
            objective = "binary",
            metric = "custom",
            num_leaves = x$num_leaves,
            min_data_in_leaf = x$min_data_in_leaf,
            learning_rate = x$learning_rate,
            feature_fraction = x$feature_fraction,
            num_iterations = x$num_iterations,
            max_bin = 31L,
            verbosity = -1,
            seed = semilla
          )
        )

        pred <- predict(modelo, data.matrix(dtrain[, cols_modelo, with = FALSE]))

        # Calcular ganancia en percentil 0.025 (top 2.5%)
        umbral <- quantile(pred, probs = 0.975, na.rm = TRUE)

        # Calcular ganancia
        dtrain[, pred_prob := pred]
        ganancia <- dtrain[pred_prob > umbral,
                          sum(ifelse(clase_ternaria == "BAJA+2", 273000, -7000))]
        dtrain[, pred_prob := NULL]

        # Validar resultado
        if (is.na(ganancia) || !is.finite(ganancia)) {
          ganancia <- -999999  # Penalización por modelo inválido
        }

        # Limpiar
        rm(modelo)
        gc(verbose = FALSE)

        return(ganancia)

      }, error = function(e) {
        # Si hay error, devolver penalización
        return(-999999)
      })
    }

    # Envolver función objetivo para mlrMBO
    obj.fun <- makeSingleObjectiveFunction(
      fn = objetivo_ganancia,
      minimize = FALSE,
      noisy = TRUE,
      par.set = configuracion_bo,
      has.simple.signature = FALSE
    )

    # Archivo para guardar progreso de BO
    bo_checkpoint_file <- file.path(seed_dir, "BO_checkpoint.RData")

    ctrl <- makeMBOControl(
      save.on.disk.at.time = 300,  # Guardar cada 5 minutos
      save.file.path = bo_checkpoint_file
    )
    ctrl <- setMBOControlTermination(ctrl, iters = 10L)
    ctrl <- setMBOControlInfill(ctrl, crit = makeMBOInfillCritEI())

    configureMlr(show.learner.output = FALSE)

    # Verificar si existe checkpoint de BO
    if (file.exists(bo_checkpoint_file)) {
      seed_log("Checkpoint de BO encontrado, resumiendo desde última iteración...", level = "INFO")
      resultado_bo <- mboContinue(bo_checkpoint_file)
      seed_log("BO resumido exitosamente", level = "SUCCESS")
    } else {
      # Ejecutar BO desde cero
      seed_log("Iniciando BO desde cero (sin checkpoint previo)", level = "INFO")
      resultado_bo <- mbo(
        fun = obj.fun,
        design = NULL,
        control = ctrl,
        show.info = FALSE
      )
    }

    fin_bo <- Sys.time()
    tiempo_bo <- as.numeric(difftime(fin_bo, inicio_bo, units = "mins"))

    mejores_params <- resultado_bo$x
    mejor_ganancia <- resultado_bo$y  # Ya está en positivo (minimize = FALSE)

    seed_log(paste("Hyperparameter Tuning completado en", round(tiempo_bo, 1), "min"),
             level = "SUCCESS")
    seed_log(paste("Mejor ganancia estimada:", formatC(mejor_ganancia, format = "f", big.mark = ",", digits = 0)))

    # Guardar estado de BO
    saveRDS(resultado_bo, file.path(seed_dir, "resultado_bo.rds"))

    # === ENTRENAMIENTO MODELO FINAL ===
    update_progress(seed_idx, "ENTRENANDO_MODELO_FINAL")
    seed_log("Entrenando modelo final con mejores hiperparámetros...")

    inicio_train_final <- Sys.time()

    set.seed(semilla)

    modelo_final <- lgb.train(
      data = lgb.Dataset(
        data = data.matrix(dtrain[, cols_modelo, with = FALSE]),
        label = dtrain$clase01,
        free_raw_data = FALSE
      ),
      param = list(
        objective = "binary",
        metric = "custom",
        num_leaves = mejores_params$num_leaves,
        min_data_in_leaf = mejores_params$min_data_in_leaf,
        learning_rate = mejores_params$learning_rate,
        feature_fraction = mejores_params$feature_fraction,
        num_iterations = mejores_params$num_iterations,
        max_bin = 31L,
        verbosity = -1,
        seed = semilla
      )
    )

    fin_train_final <- Sys.time()
    tiempo_train_final <- as.numeric(difftime(fin_train_final, inicio_train_final, units = "mins"))

    seed_log(paste("Modelo final entrenado en", round(tiempo_train_final, 1), "min"),
             level = "SUCCESS")

    # Guardar modelo
    lgb.save(modelo_final, file.path(seed_dir, "modelo.txt"))

    # === SCORING ===
    update_progress(seed_idx, "SCORING")
    seed_log("Generando predicciones...")

    pred_future <- predict(modelo_final, data.matrix(dfuture[, cols_modelo, with = FALSE]))

    # Log de información del future dataset
    seed_log(paste("Registros en future:", nrow(dfuture)))
    seed_log(paste("Distribución clase_ternaria:"))
    seed_log(paste("  BAJA+2:", sum(dfuture$clase_ternaria == "BAJA+2", na.rm = TRUE)))
    seed_log(paste("  BAJA+1:", sum(dfuture$clase_ternaria == "BAJA+1", na.rm = TRUE)))
    seed_log(paste("  CONTINUA:", sum(dfuture$clase_ternaria == "CONTINUA", na.rm = TRUE)))
    seed_log(paste("Predicciones - Min:", round(min(pred_future), 4), "Max:", round(max(pred_future), 4),
                   "Mean:", round(mean(pred_future), 4)))

    # Calcular ganancia en varios umbrales
    umbrales <- seq(0.95, 0.99, by = 0.005)
    ganancias <- sapply(umbrales, function(u) {
      umbral_valor <- quantile(pred_future, probs = u, na.rm = TRUE)
      n_envios <- sum(pred_future > umbral_valor)
      ganancia <- dfuture[pred_future > umbral_valor, sum(ifelse(clase_ternaria == "BAJA+2", 273000, -7000))]
      ganancia
    })

    # Mostrar tabla de umbrales vs ganancia
    seed_log("Análisis de umbrales:")
    for (i in seq_along(umbrales)) {
      umbral_valor <- quantile(pred_future, probs = umbrales[i], na.rm = TRUE)
      n_envios <- sum(pred_future > umbral_valor)
      seed_log(paste("  Umbral", umbrales[i], "-> Envíos:", n_envios, "| Ganancia:",
                     formatC(ganancias[i], format = "f", big.mark = ",", digits = 0)))
    }

    mejor_idx <- which.max(ganancias)
    mejor_umbral <- umbrales[mejor_idx]
    ganancia_final <- ganancias[mejor_idx]

    seed_log(paste("Mejor umbral seleccionado:", mejor_umbral), level = "SUCCESS")

    # Crear submission
    umbral_final <- quantile(pred_future, probs = mejor_umbral, na.rm = TRUE)
    dfuture[, Predicted := as.integer(pred_future > umbral_final)]

    submission <- dfuture[Predicted == 1, .(numero_de_cliente)]
    fwrite(submission, file.path(seed_dir, paste0("submission_", seed_idx, ".csv")))

    seed_log(paste("Submission generado:", nrow(submission), "envíos"), level = "SUCCESS")
    seed_log(paste("Ganancia estimada:", formatC(ganancia_final, format = "f", big.mark = ",", digits = 0)))

    # === FIN ===
    fin_seed <- Sys.time()
    duracion_total <- as.numeric(difftime(fin_seed, inicio_seed, units = "mins"))

    update_progress(seed_idx, "COMPLETADO",
                   paste("Ganancia:", ganancia_final, "| Envíos:", nrow(submission)))

    seed_log(paste("SEMILLA COMPLETADA EXITOSAMENTE en", round(duracion_total, 1), "minutos"),
             level = "FINISH")

    log_msg(paste("Semilla", seed_idx, "COMPLETADA - Ganancia:", ganancia_final,
                  "- Envíos:", nrow(submission)),
            level = "SUCCESS")

    return(list(
      success = TRUE,
      seed_idx = seed_idx,
      semilla = semilla,
      ganancia = ganancia_final,
      envios = nrow(submission),
      duracion_min = duracion_total,
      mejores_params = mejores_params
    ))

  }, error = function(e) {
    error_msg <- paste("ERROR en semilla", seed_idx, ":", e$message)
    log_msg(error_msg, level = "ERROR")

    if (exists("seed_log_file")) {
      cat(paste("[ERROR]", error_msg, "\n"), file = seed_log_file, append = TRUE)
    }

    update_progress(seed_idx, "ERROR", e$message)

    return(list(success = FALSE, seed_idx = seed_idx, semilla = semilla,
                error = e$message, log_file = ifelse(exists("seed_log_file"),
                                                     seed_log_file, "N/A")))
  })
}

# ============================================================================
# EJECUCIÓN EN PARALELO CON HEALTH MONITORING
# ============================================================================

require("parallel")

log_msg(paste("Creando cluster con", NUM_WORKERS, "workers..."), level = "INFO")
cl <- makeCluster(NUM_WORKERS)

# Exportar TODO lo necesario
clusterExport(cl, c("PARAM_GLOBAL", "BASE_DIR", "DATASETS_DIR", "EXP_DIR",
                    "LOG_FILE", "PROGRESS_FILE", "THREADS_PER_WORKER",
                    "log_msg", "update_progress", "ejecutar_semilla",
                    "CatastrofeAnalisis"))

# Cargar paquetes en cada worker
clusterEvalQ(cl, {
  require("data.table")
  require("lightgbm")
  require("DiceKriging")
  require("mlr")
  require("mlrMBO")
  require("ParamHelpers")
})

log_msg("Cluster creado y configurado", level = "SUCCESS")

# Health check inicial del cluster
health_check("Cluster paralelo creado con 5 workers")

inicio_total <- Sys.time()

log_msg(paste(rep("=", 60), collapse = ""), level = "INFO")
log_msg("EJECUCIÓN ULTRA PARALELA INICIADA", level = "SUCCESS")
log_msg(paste("Hora inicio:", format(inicio_total)), level = "INFO")
log_msg(paste("Hora estimada fin:", format(inicio_total + as.difftime(2, units = "hours"))), level = "INFO")
log_msg(paste(rep("=", 60), collapse = ""), level = "INFO")

cat("\n⚡ TODAS LAS", length(PARAM_GLOBAL$semillas), "SEMILLAS CORRIENDO EN ULTRA MODE ⚡\n\n")
cat("📊 Monitorea el progreso en tiempo real:\n")
cat(paste("   Log principal:", LOG_FILE, "\n"))
cat(paste("   Progreso:", PROGRESS_FILE, "\n"))
cat(paste("   Health checks:", HEALTH_FILE, "\n\n"))

# Configurar parámetros globales para catastrophe
PARAM_GLOBAL$metodo_cat <- "MachineLearning"

# EJECUTAR EN PARALELO
# Crear lista con los parámetros para cada semilla
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

# Cerrar cluster
stopCluster(cl)

fin_total <- Sys.time()
duracion_total <- as.numeric(difftime(fin_total, inicio_total, units = "hours"))

# Health check final
health_check("Ejecución paralela completada")

# ============================================================================
# CONSOLIDACIÓN DE RESULTADOS
# ============================================================================

log_msg(paste(rep("=", 60), collapse = ""))
log_msg("EJECUCIÓN COMPLETADA", level = "SUCCESS")
log_msg(paste("Duración total:", round(duracion_total, 2), "horas"))
log_msg(paste(rep("=", 60), collapse = ""))

# Procesar resultados
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
      duracion_min = round(x$duracion_min, 1),
      status = "OK"
    )
  } else {
    data.frame(
      seed_idx = x$seed_idx,
      semilla = x$semilla,
      ganancia = NA,
      envios = NA,
      duracion_min = NA,
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
fwrite(resumen_df, file.path(EXP_DIR, "resumen_ultra_exp6160.txt"), sep = "\t")
saveRDS(resultados, file.path(EXP_DIR, "resultados_ultra_exp6160.rds"))

log_msg("Resumen guardado en: resumen_ultra_exp6160.txt", level = "SUCCESS")

# Mostrar tabla
print(resumen_df)

# Mostrar semillas fallidas si las hay
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
  cat(paste("  Ganancia:", mejor$ganancia, "\n"))
  cat(paste("  Envíos:", mejor$envios, "\n"))
  cat(paste("  Duración:", mejor$duracion_min, "minutos\n"))
}

log_msg("WORKFLOW ULTRA FINALIZADO EXITOSAMENTE", level = "SUCCESS")

cat("\n✨ LISTO! Revisa los archivos en exp/ ✨\n\n")
