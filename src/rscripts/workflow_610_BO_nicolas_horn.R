# ============================================================================
# WORKFLOW 610 - BAYESIAN OPTIMIZATION - Nicolas Horn
# ============================================================================
#
# Script local para encontrar hyperparametros optimos usando Bayesian Optimization
# Basado en z610_WorkFlow_01_gerencial_julio.ipynb
#
# Feature Engineering: lags (1, 2) + deltas (1, 2)
# BO iterations: 100 (configurable)
#
# ============================================================================

format(Sys.time(), "%a %b %d %X %Y")

# ============================================================================
# CONFIGURACION
# ============================================================================

BASE_DIR <- "C:/Users/User/Documents/labo2025v"
DATASETS_DIR <- file.path(BASE_DIR, "datasets")
EXP_DIR <- file.path(BASE_DIR, "exp", "exp_610_BO_nicolas_horn")

dir.create(EXP_DIR, showWarnings = FALSE, recursive = TRUE)

# Parametros
SEMILLA <- 153929  # Primera semilla para BO
BO_ITERATIONS <- 100
EXPERIMENTO <- 6100

cat("\n")
cat("========================================\n")
cat(" WORKFLOW 610 - BAYESIAN OPTIMIZATION\n")
cat(" Nicolas Horn\n")
cat("========================================\n\n")

cat("Configuracion:\n")
cat("  Semilla:", SEMILLA, "\n")
cat("  BO iterations:", BO_ITERATIONS, "\n")
cat("  Experimento:", EXPERIMENTO, "\n")
cat("  Output dir:", EXP_DIR, "\n\n")

# ============================================================================
# CARGAR PAQUETES
# ============================================================================

cat("Cargando paquetes...\n")

require("data.table")
require("lightgbm")
require("DiceKriging")
require("mlrMBO")
require("ParamHelpers")

setDTthreads(4)
cat("data.table threads:", getDTthreads(), "\n\n")

# ============================================================================
# CARGAR DATASET
# ============================================================================

cat("Cargando dataset...\n")
dataset_file <- file.path(DATASETS_DIR, "gerencial_competencia_2025.csv.gz")

if (!file.exists(dataset_file)) {
  stop("ERROR: Dataset no encontrado en ", dataset_file)
}

dataset <- fread(dataset_file)
cat("Dataset cargado:", nrow(dataset), "filas x", ncol(dataset), "cols\n\n")

# ============================================================================
# CATASTROPHE ANALYSIS
# ============================================================================

cat("Aplicando Catastrophe Analysis...\n")

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

cat("Catastrophe Analysis completado (12 variables)\n\n")

# ============================================================================
# FEATURE ENGINEERING INTRA-MES
# ============================================================================

cat("Feature Engineering intra-mes...\n")

atributos_presentes <- function(patributos) {
  atributos <- unique(patributos)
  comun <- intersect(atributos, colnames(dataset))
  return(length(atributos) == length(comun))
}

if (atributos_presentes(c("foto_mes")))
  dataset[, kmes := foto_mes %% 100]

if (atributos_presentes(c("mpayroll", "cliente_edad")))
  dataset[, mpayroll_sobre_edad := mpayroll / cliente_edad]

cat("FE intra-mes completado\n\n")

# ============================================================================
# FEATURE ENGINEERING HISTORICO
# ============================================================================

cat("Feature Engineering Historico...\n")
inicio_fe <- Sys.time()

cols_lagueables <- copy(setdiff(
  colnames(dataset),
  c("numero_de_cliente", "foto_mes", "clase_ternaria")
))

cat("  Variables base:", length(cols_lagueables), "\n")

# lags de orden 1
cat("  Generando lags orden 1...\n")
dataset[,
  paste0(cols_lagueables, "_lag1") := shift(.SD, 1, NA, "lag"),
  by = numero_de_cliente,
  .SDcols = cols_lagueables
]

# lags de orden 2
cat("  Generando lags orden 2...\n")
dataset[,
  paste0(cols_lagueables, "_lag2") := shift(.SD, 2, NA, "lag"),
  by = numero_de_cliente,
  .SDcols = cols_lagueables
]

# deltas
cat("  Generando deltas...\n")
for (vcol in cols_lagueables) {
  dataset[, paste0(vcol, "_delta1") := get(vcol) - get(paste0(vcol, "_lag1"))]
  dataset[, paste0(vcol, "_delta2") := get(vcol) - get(paste0(vcol, "_lag2"))]
}

fin_fe <- Sys.time()
tiempo_fe <- as.numeric(difftime(fin_fe, inicio_fe, units = "mins"))

cat("FE Historico completado en", round(tiempo_fe, 1), "min\n")
cat("Dataset final:", nrow(dataset), "filas x", ncol(dataset), "cols\n\n")

# ============================================================================
# TRAINING STRATEGY
# ============================================================================

cat("Configurando Training Strategy...\n")

PARAM <- list()
PARAM$semilla_primigenia <- SEMILLA
PARAM$experimento <- EXPERIMENTO

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

cat("Features para modelo:", length(campos_buenos), "\n")
cat("Filas en training:", sum(dataset$fold_train), "\n")
cat("Filas en validation:", nrow(dataset[foto_mes %in% PARAM$trainingstrategy$validate]), "\n\n")

# ============================================================================
# CREAR DATASETS LIGHTGBM
# ============================================================================

cat("Creando datasets LightGBM...\n")

dtrain <- lgb.Dataset(
  data = data.matrix(dataset[fold_train == TRUE, campos_buenos, with = FALSE]),
  label = dataset[fold_train == TRUE, clase01],
  free_raw_data = FALSE
)

dvalidate <- lgb.Dataset(
  data = data.matrix(dataset[foto_mes %in% PARAM$trainingstrategy$validate, campos_buenos, with = FALSE]),
  label = dataset[foto_mes %in% PARAM$trainingstrategy$validate, clase01],
  free_raw_data = FALSE
)

cat("Datasets creados\n\n")

# ============================================================================
# BAYESIAN OPTIMIZATION
# ============================================================================

cat("========================================\n")
cat("INICIANDO BAYESIAN OPTIMIZATION\n")
cat("Iteraciones:", BO_ITERATIONS, "\n")
cat("========================================\n\n")

setwd(EXP_DIR)

# Parametros fijos
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

# Espacio de busqueda
PARAM$hipeparametertuning <- list()
PARAM$hipeparametertuning$num_iterations <- BO_ITERATIONS

PARAM$hipeparametertuning$hs <- makeParamSet(
  makeIntegerParam("num_leaves", lower = 2L, upper = 256L),
  makeIntegerParam("min_data_in_leaf", lower = 2L, upper = 8192L)
)

# Funcion objetivo
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

  cat(format(Sys.time(), "%H:%M:%S"), "| AUC:", round(AUC, 6),
      "| num_leaves:", x$num_leaves,
      "| min_data_in_leaf:", x$min_data_in_leaf, "\n")

  return(AUC)
}

# Configurar BO
configureMlr(show.learner.output = FALSE)

obj.fun <- makeSingleObjectiveFunction(
  fn = EstimarGanancia_AUC_lightgbm,
  minimize = FALSE,
  noisy = FALSE,
  par.set = PARAM$hipeparametertuning$hs,
  has.simple.signature = FALSE
)

# Checkpoint file
bo_checkpoint_file <- file.path(EXP_DIR, "BO_checkpoint.RData")

ctrl <- makeMBOControl(
  save.on.disk.at.time = 300,
  save.file.path = bo_checkpoint_file
)

ctrl <- setMBOControlTermination(ctrl, iters = PARAM$hipeparametertuning$num_iterations)
ctrl <- setMBOControlInfill(ctrl, crit = makeMBOInfillCritEI())

surr.km <- makeLearner(
  "regr.km",
  predict.type = "se",
  covtype = "matern3_2",
  control = list(trace = FALSE)
)

# Ejecutar BO
inicio_bo <- Sys.time()

if (file.exists(bo_checkpoint_file)) {
  cat("Checkpoint encontrado, resumiendo BO...\n\n")
  bayesiana_salida <- mboContinue(bo_checkpoint_file)
} else {
  cat("Iniciando BO desde cero...\n\n")
  bayesiana_salida <- mbo(obj.fun, learner = surr.km, control = ctrl)
}

fin_bo <- Sys.time()
tiempo_bo <- as.numeric(difftime(fin_bo, inicio_bo, units = "mins"))

# ============================================================================
# RESULTADOS
# ============================================================================

cat("\n========================================\n")
cat("BAYESIAN OPTIMIZATION COMPLETADA\n")
cat("Tiempo:", round(tiempo_bo, 1), "minutos\n")
cat("========================================\n\n")

# Extraer resultados
tb_bayesiana <- as.data.table(bayesiana_salida$opt.path)
setorder(tb_bayesiana, -y, -num_iterations)

# Guardar log completo
fwrite(tb_bayesiana, file.path(EXP_DIR, "BO_log.txt"), sep = "\t")

# Mejores hyperparametros
mejores_hiperparametros <- tb_bayesiana[
  1,
  setdiff(colnames(tb_bayesiana),
    c("y", "dob", "eol", "error.message", "exec.time", "ei", "error.model",
      "train.time", "prop.type", "propose.time", "se", "mean", "iter")),
  with = FALSE
]

mejor_auc <- tb_bayesiana[1, y]

cat("MEJORES HYPERPARAMETROS ENCONTRADOS:\n")
cat("  num_leaves:", mejores_hiperparametros$num_leaves, "\n")
cat("  min_data_in_leaf:", mejores_hiperparametros$min_data_in_leaf, "\n")
cat("  num_iterations:", mejores_hiperparametros$num_iterations, "\n")
cat("  Mejor AUC:", round(mejor_auc, 8), "\n\n")

# Guardar mejores hyperparametros
cat("Guardando resultados...\n")

resultado_final <- list(
  mejores_hiperparametros = mejores_hiperparametros,
  mejor_auc = mejor_auc,
  tiempo_bo_min = tiempo_bo,
  semilla = SEMILLA,
  bo_iterations = BO_ITERATIONS,
  fecha = Sys.time()
)

saveRDS(resultado_final, file.path(EXP_DIR, "mejores_hyperparametros.rds"))

# Guardar como texto para facil lectura
sink(file.path(EXP_DIR, "mejores_hyperparametros.txt"))
cat("MEJORES HYPERPARAMETROS - WORKFLOW 610\n")
cat("======================================\n\n")
cat("Fecha:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("Semilla:", SEMILLA, "\n")
cat("BO iterations:", BO_ITERATIONS, "\n")
cat("Tiempo BO:", round(tiempo_bo, 1), "minutos\n\n")
cat("HYPERPARAMETROS:\n")
cat("  num_leaves =", mejores_hiperparametros$num_leaves, "\n")
cat("  min_data_in_leaf =", mejores_hiperparametros$min_data_in_leaf, "\n")
cat("  num_iterations =", mejores_hiperparametros$num_iterations, "\n\n")
cat("Mejor AUC:", mejor_auc, "\n\n")
cat("TOP 10 CONFIGURACIONES:\n")
print(tb_bayesiana[1:10, .(num_leaves, min_data_in_leaf, num_iterations, y)])
sink()

cat("\nResultados guardados en:", EXP_DIR, "\n")
cat("  - BO_log.txt (log completo)\n")
cat("  - mejores_hyperparametros.rds (objeto R)\n")
cat("  - mejores_hyperparametros.txt (resumen)\n\n")

# ============================================================================
# CODIGO PARA USAR EN NOTEBOOK
# ============================================================================

cat("========================================\n")
cat("CODIGO PARA USAR EN NOTEBOOK:\n")
cat("========================================\n\n")

cat("PARAM$out$lgbm$mejores_hiperparametros <- list(\n")
cat("  num_leaves =", mejores_hiperparametros$num_leaves, ",\n")
cat("  min_data_in_leaf =", mejores_hiperparametros$min_data_in_leaf, ",\n")
cat("  num_iterations =", mejores_hiperparametros$num_iterations, "\n")
cat(")\n\n")

cat("========================================\n")
cat("WORKFLOW COMPLETADO\n")
cat(format(Sys.time(), "%a %b %d %X %Y"), "\n")
cat("========================================\n")
