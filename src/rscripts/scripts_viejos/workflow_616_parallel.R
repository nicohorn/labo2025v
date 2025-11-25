# ============================================================================
# WORKFLOW 616 - EJECUCIÓN EN PARALELO (MÚLTIPLES SEMILLAS)
# ============================================================================
# Este script ejecuta múltiples semillas EN PARALELO para aprovechar todos
# los núcleos de tu CPU y reducir el tiempo total de ejecución.
#
# VENTAJAS:
# - Reduce tiempo total de 5 semillas de ~10 horas a ~2-3 horas (con 4+ cores)
# - Aprovecha todos los núcleos del CPU
# - Cada semilla corre en su propio proceso independiente
#
# REQUISITOS:
# - CPU con múltiples núcleos (recomendado: 4+ cores)
# - RAM suficiente (recomendado: 16+ GB para 4-5 procesos paralelos)
# - Todos los paquetes instalados
#
# ANTES DE EJECUTAR:
# 1. Ajusta NUM_CORES según tu CPU (ver línea 37)
# 2. Verifica que tengas suficiente RAM
# 3. Cierra otras aplicaciones pesadas
# ============================================================================

format(Sys.time(), "%a %b %d %X %Y")

cat("\n========================================\n")
cat("WORKFLOW PARALELO - MÚLTIPLES SEMILLAS\n")
cat("========================================\n\n")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Configuración de rutas
BASE_DIR <- "C:/Users/User/Documents/labo2025v"
DATASETS_DIR <- file.path(BASE_DIR, "datasets")
EXP_DIR <- file.path(BASE_DIR, "exp")

# Número de cores a usar
# IMPORTANTE: Ajusta según tu CPU
# - Deja al menos 1-2 cores libres para el sistema
# - Ejemplo: Si tienes 8 cores, usa 6
# - Si tienes 4 cores, usa 2-3
NUM_CORES <- parallel::detectCores() - 1

cat("Cores disponibles:", parallel::detectCores(), "\n")
cat("Cores a usar:", NUM_CORES, "\n\n")

# Configuración del experimento
PARAM_GLOBAL <- list()
PARAM_GLOBAL$experimento_base <- 6160
PARAM_GLOBAL$dataset <- "gerencial_competencia_2025.csv.gz"

# Semillas a ejecutar
PARAM_GLOBAL$semillas <- c(102191, 200207, 300313, 400419, 500523)

# Para prueba rápida con menos semillas, comenta la línea anterior y descomenta:
# PARAM_GLOBAL$semillas <- c(102191, 200207)

cat("Experimento base:", PARAM_GLOBAL$experimento_base, "\n")
cat("Dataset:", PARAM_GLOBAL$dataset, "\n")
cat("Semillas a procesar:", length(PARAM_GLOBAL$semillas), "\n")
cat("Semillas:", paste(PARAM_GLOBAL$semillas, collapse = ", "), "\n\n")

# ============================================================================
# VERIFICACIONES PREVIAS
# ============================================================================

cat("Verificando entorno...\n")

# Verificar dataset
dataset_path <- file.path(DATASETS_DIR, PARAM_GLOBAL$dataset)
if (!file.exists(dataset_path)) {
  stop("ERROR: Dataset no encontrado en: ", dataset_path)
}
cat("✓ Dataset encontrado\n")

# Verificar paquetes críticos
required_packages <- c("data.table", "lightgbm", "mlrMBO", "DiceKriging", "parallel", "yaml")
missing_packages <- c()

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    missing_packages <- c(missing_packages, pkg)
  }
}

if (length(missing_packages) > 0) {
  stop("ERROR: Faltan paquetes: ", paste(missing_packages, collapse = ", "),
       "\nInstálalos con: install.packages(c('", paste(missing_packages, collapse = "','"), "'))")
}
cat("✓ Todos los paquetes necesarios están instalados\n")

# Verificar RAM disponible
if (.Platform$OS.type == "windows") {
  # Advertencia sobre RAM
  num_parallel <- min(NUM_CORES, length(PARAM_GLOBAL$semillas))
  ram_recomendada <- num_parallel * 4  # 4 GB por proceso aproximadamente

  cat("\n⚠ IMPORTANTE - Uso de RAM:\n")
  cat("  Procesos paralelos:", num_parallel, "\n")
  cat("  RAM recomendada: ~", ram_recomendada, "GB\n")
  cat("  Si tu sistema tiene menos RAM, reduce NUM_CORES\n\n")

  respuesta <- readline("¿Continuar con la ejecución paralela? (s/n): ")
  if (tolower(respuesta) != "s") {
    cat("Ejecución cancelada.\n")
    quit(save = "no")
  }
}

cat("\n========================================\n")
cat("INICIANDO EJECUCIÓN PARALELA\n")
cat("========================================\n\n")

# ============================================================================
# FUNCIÓN PARA EJECUTAR UNA SEMILLA
# ============================================================================

ejecutar_semilla <- function(seed_idx, semilla, PARAM_GLOBAL, BASE_DIR, DATASETS_DIR, EXP_DIR) {

  # Cargar librerías en el worker
  require("data.table")
  require("R.utils")
  require("lightgbm")
  require("mlrMBO")
  require("DiceKriging")
  require("mlr")
  require("ParamHelpers")
  require("yaml")

  # Inicializar PARAM
  PARAM <- list()
  PARAM$semilla_primigenia <- semilla
  PARAM$experimento <- PARAM_GLOBAL$experimento_base + seed_idx - 1
  PARAM$dataset <- PARAM_GLOBAL$dataset
  PARAM$out <- list()
  PARAM$out$lgbm <- list()

  # Log de inicio
  inicio <- Sys.time()
  cat("\n[Semilla", seed_idx, "] Iniciando a las", format(inicio), "\n")

  tryCatch({

    # ===================================================================
    # Carpeta del Experimento
    # ===================================================================

    experimento_folder <- paste0("WF", PARAM$experimento, "_seed", seed_idx, "_FE_historico_SOLO")
    experimento_path <- file.path(EXP_DIR, experimento_folder)

    dir.create(experimento_path, showWarnings = FALSE, recursive = TRUE)
    setwd(experimento_path)

    cat("[Semilla", seed_idx, "] Carpeta:", experimento_folder, "\n")

    # ===================================================================
    # Cargar y Preparar Dataset
    # ===================================================================

    cat("[Semilla", seed_idx, "] Cargando dataset...\n")
    dataset_path <- file.path(DATASETS_DIR, PARAM$dataset)
    dataset <- fread(dataset_path)

    # Catastrophe Analysis
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
    # Feature Engineering Histórico
    # ===================================================================

    cat("[Semilla", seed_idx, "] Iniciando Feature Engineering Histórico...\n")

    setorder(dataset, numero_de_cliente, foto_mes)

    cols_lagueables <- copy(setdiff(
      colnames(dataset),
      c("numero_de_cliente", "foto_mes", "clase_ternaria")
    ))

    # LAGS
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
      tryCatch({
        coef(lm(y[valid] ~ x[valid]))[2]
      }, error = function(e) NA)
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

    cat("[Semilla", seed_idx, "] FE completado. Total features:", ncol(dataset), "\n")

    # ===================================================================
    # Training Strategy
    # ===================================================================

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

    cat("[Semilla", seed_idx, "] Iniciando Hyperparameter Tuning...\n")

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
      save.file.path = "HT.RDATA"
    )

    ctrl <- setMBOControlTermination(ctrl, iters = PARAM$hipeparametertuning$num_interations)
    ctrl <- setMBOControlInfill(ctrl, crit = makeMBOInfillCritEI())

    surr.km <- makeLearner(
      "regr.km",
      predict.type = "se",
      covtype = "matern3_2",
      control = list(trace = FALSE)
    )

    if (!file.exists("HT.RDATA")) {
      bayesiana_salida <- mbo(obj.fun, learner = surr.km, control = ctrl)
    } else {
      bayesiana_salida <- mboContinue("HT.RDATA")
    }

    tb_bayesiana <- as.data.table(bayesiana_salida$opt.path)
    setorder(tb_bayesiana, -y, -num_iterations)

    fwrite(tb_bayesiana, file = "BO_log.txt", sep = "\t")

    PARAM$out$lgbm$mejores_hiperparametros <- tb_bayesiana[
      1,
      setdiff(colnames(tb_bayesiana),
              c("y", "dob", "eol", "error.message", "exec.time", "ei", "error.model",
                "train.time", "prop.type", "propose.time", "se", "mean", "iter")),
      with = FALSE
    ]

    # ===================================================================
    # Producción
    # ===================================================================

    cat("[Semilla", seed_idx, "] Entrenando modelo final...\n")

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
    fwrite(tb_importancia, file = "impo.txt", sep = "\t")

    # ===================================================================
    # Scoring
    # ===================================================================

    PARAM$trainingstrategy$future <- c(202107)
    dfuture <- dataset[foto_mes %in% PARAM$trainingstrategy$future]

    prediccion <- predict(
      final_model,
      data.matrix(dfuture[, campos_buenos, with = FALSE])
    )

    tb_prediccion <- dfuture[, list(numero_de_cliente)]
    tb_prediccion[, prob := prediccion]

    fwrite(tb_prediccion, file = "prediccion.txt", sep = "\t")

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
    resultado$envios <- which.max(tb_prediccion$gan_suavizada)
    resultado$semilla <- PARAM$semilla_primigenia
    resultado$seed_idx <- seed_idx

    fwrite(tb_prediccion, file = "ganancias.txt", sep = "\t")

    tb_prediccion[, envios := .I]

    pdf("curva_de_ganancia.pdf")
    plot(
      x = tb_prediccion$envios,
      y = tb_prediccion$gan_acum,
      type = "l",
      col = "gray",
      xlim = c(0, 6000),
      ylim = c(0, 8000000),
      main = paste0("Seed ", seed_idx, " - Gan= ", as.integer(resultado$ganancia_suavizada_max)),
      xlab = "Envios",
      ylab = "Ganancia",
      panel.first = grid()
    )
    dev.off()

    PARAM$resultado <- resultado
    write_yaml(PARAM, file = "PARAM.yml")

    # Limpiar memoria
    rm(dataset, dtrain, dvalidate, dfinal_train, final_model, tb_prediccion)
    gc(full = TRUE, verbose = FALSE)

    # Log de finalización
    fin <- Sys.time()
    duracion <- as.numeric(difftime(fin, inicio, units = "mins"))

    cat("\n[Semilla", seed_idx, "] ✓ COMPLETADA\n")
    cat("[Semilla", seed_idx, "] Ganancia:", resultado$ganancia_suavizada_max, "\n")
    cat("[Semilla", seed_idx, "] Envíos:", resultado$envios, "\n")
    cat("[Semilla", seed_idx, "] Duración:", round(duracion, 1), "minutos\n")

    return(list(
      success = TRUE,
      seed_idx = seed_idx,
      semilla = semilla,
      ganancia = resultado$ganancia_suavizada_max,
      envios = resultado$envios,
      duracion_min = duracion
    ))

  }, error = function(e) {
    cat("\n[Semilla", seed_idx, "] ✗ ERROR:", e$message, "\n")

    return(list(
      success = FALSE,
      seed_idx = seed_idx,
      semilla = semilla,
      error = e$message
    ))
  })
}

# ============================================================================
# EJECUTAR EN PARALELO
# ============================================================================

# Cargar librería parallel
require("parallel")

# Crear cluster
cat("Creando cluster con", NUM_CORES, "cores...\n")
cl <- makeCluster(NUM_CORES)

# Exportar variables y funciones necesarias
clusterExport(cl, c("PARAM_GLOBAL", "BASE_DIR", "DATASETS_DIR", "EXP_DIR"))

# Tiempo de inicio
inicio_total <- Sys.time()

cat("\nEjecutando semillas en paralelo...\n")
cat("IMPORTANTE: Esto puede tardar varias horas.\n")
cat("Los mensajes de cada semilla se mostrarán conforme avancen.\n\n")

# Ejecutar en paralelo
resultados <- parLapply(cl, 1:length(PARAM_GLOBAL$semillas), function(i) {
  ejecutar_semilla(
    seed_idx = i,
    semilla = PARAM_GLOBAL$semillas[i],
    PARAM_GLOBAL = PARAM_GLOBAL,
    BASE_DIR = BASE_DIR,
    DATASETS_DIR = DATASETS_DIR,
    EXP_DIR = EXP_DIR
  )
})

# Detener cluster
stopCluster(cl)

# Tiempo de finalización
fin_total <- Sys.time()
duracion_total <- as.numeric(difftime(fin_total, inicio_total, units = "hours"))

# ============================================================================
# RESUMEN FINAL
# ============================================================================

cat("\n\n========================================\n")
cat("RESUMEN FINAL\n")
cat("========================================\n\n")

cat("Tiempo total de ejecución:", round(duracion_total, 2), "horas\n\n")

# Crear tabla de resultados
resultados_exitosos <- Filter(function(x) x$success, resultados)
resultados_fallidos <- Filter(function(x) !x$success, resultados)

if (length(resultados_exitosos) > 0) {

  tb_resumen <- rbindlist(lapply(resultados_exitosos, function(r) {
    data.table(
      seed_idx = r$seed_idx,
      semilla = r$semilla,
      ganancia = r$ganancia,
      envios = r$envios,
      duracion_min = r$duracion_min
    )
  }))

  setorder(tb_resumen, -ganancia)
  tb_resumen[, rank := .I]

  cat("RESULTADOS EXITOSOS:\n\n")
  print(tb_resumen)

  cat("\nESTADÍSTICAS:\n")
  cat("Ganancia promedio:", mean(tb_resumen$ganancia), "\n")
  cat("Ganancia máxima:", max(tb_resumen$ganancia), "\n")
  cat("Ganancia mínima:", min(tb_resumen$ganancia), "\n")
  cat("Desviación estándar:", sd(tb_resumen$ganancia), "\n")
  cat("Mejor semilla:", tb_resumen[rank == 1, semilla], "\n")

  # Guardar resumen
  setwd(EXP_DIR)
  fwrite(tb_resumen,
         file = paste0("resumen_parallel_exp", PARAM_GLOBAL$experimento_base, ".txt"),
         sep = "\t")

  saveRDS(resultados,
          file = paste0("resultados_parallel_exp", PARAM_GLOBAL$experimento_base, ".rds"))

  cat("\nArchivos guardados en:", EXP_DIR, "\n")
}

if (length(resultados_fallidos) > 0) {
  cat("\n⚠ SEMILLAS FALLIDAS:\n")
  for (r in resultados_fallidos) {
    cat("  - Semilla", r$seed_idx, ":", r$error, "\n")
  }
}

cat("\n========================================\n")
cat("EJECUCIÓN PARALELA COMPLETADA\n")
cat("========================================\n")

format(Sys.time(), "%a %b %d %X %Y")
