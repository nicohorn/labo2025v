#!/usr/bin/env Rscript
# ============================================================================
# Feature Engineering Contribution Analysis
# ============================================================================
# Analiza la importancia de las features generadas por FE avanzado
# comparando su contribución al modelo vs features básicas

library(data.table)
library(ggplot2)

# Configuración
EXP_DIR <- "C:/Users/User/Documents/labo2025v/exp/exp_baseline"
OUTPUT_DIR <- file.path(EXP_DIR, "fe_analysis")
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ============================================================================
# 1. Cargar feature importance de todas las semillas
# ============================================================================

cat("==============================================\n")
cat("ANÁLISIS DE CONTRIBUCIÓN DEL FE HISTÓRICO\n")
cat("==============================================\n\n")

# Encontrar archivos de importancia
seed_dirs <- list.dirs(EXP_DIR, recursive = FALSE, full.names = TRUE)
seed_dirs <- seed_dirs[grepl("WF616[0-9]_seed[0-9]_BASELINE", basename(seed_dirs))]

cat("Seeds encontrados:", length(seed_dirs), "\n")
for (d in seed_dirs) cat("  -", basename(d), "\n")
cat("\n")

# Cargar todos los archivos de importancia
lista_importancias <- list()

for (seed_dir in seed_dirs) {
  impo_file <- file.path(seed_dir, "impo.txt")

  if (file.exists(impo_file)) {
    seed_name <- basename(seed_dir)
    cat("Cargando:", seed_name, "... ")

    tb_impo <- fread(impo_file)
    tb_impo[, seed := seed_name]

    lista_importancias[[seed_name]] <- tb_impo
    cat("OK (", nrow(tb_impo), "features )\n")
  }
}

# Combinar todas las importancias
tb_all_impo <- rbindlist(lista_importancias)

cat("\nTotal de registros:", nrow(tb_all_impo), "\n")
cat("Features únicas:", uniqueN(tb_all_impo$Feature), "\n\n")

# ============================================================================
# 2. Categorizar features por tipo de FE
# ============================================================================

cat("Categorizando features por tipo de FE...\n")

categorizar_feature <- function(feature_name) {
  # Orden de prioridad: más específico primero

  # FE Avanzado
  if (grepl("_trend_[0-9]+$", feature_name)) return("Trend")
  if (grepl("_roll_(mean|max|min|sd)_[0-9]+$", feature_name)) return("Rolling Stats")
  if (grepl("_lag[0-9]+$", feature_name)) {
    # Separar lags básicos (1,2) de avanzados (3,6)
    if (grepl("_lag[12]$", feature_name)) return("Lag Basic (1-2)")
    if (grepl("_lag[36]$", feature_name)) return("Lag Advanced (3,6)")
  }
  if (grepl("_delta[0-9]+$", feature_name)) {
    # Separar deltas básicos (1,2) de avanzados (3)
    if (grepl("_delta[12]$", feature_name)) return("Delta Basic (1-2)")
    if (grepl("_delta3$", feature_name)) return("Delta Advanced (3)")
  }
  if (grepl("_cv_[0-9]+$", feature_name)) return("Volatility (CV)")
  if (grepl("_range_norm_[0-9]+$", feature_name)) return("Volatility (Range)")
  if (grepl("_ratio_vs_(lag|mean)[0-9]+$", feature_name)) return("Ratio")

  # FE Básico intrames
  if (grepl("^k", feature_name)) return("Intrames (k)")
  if (grepl("_sobre_", feature_name)) return("Intrames (ratio)")

  # Variables originales
  return("Original")
}

# Aplicar categorización
tb_all_impo[, fe_type := sapply(Feature, categorizar_feature)]

# Resumen por categoría y seed
tb_category_summary <- tb_all_impo[, .(
  num_features = .N,
  total_gain = sum(Gain),
  avg_gain = mean(Gain),
  total_cover = sum(Cover),
  total_freq = sum(Frequency)
), by = .(seed, fe_type)]

setorder(tb_category_summary, seed, -total_gain)

cat("\nResumen por seed y categoría:\n")
print(tb_category_summary)

# ============================================================================
# 3. Análisis agregado entre seeds
# ============================================================================

cat("\n==============================================\n")
cat("ANÁLISIS AGREGADO (todas las seeds)\n")
cat("==============================================\n\n")

# Promediar métricas por feature
tb_avg_importance <- tb_all_impo[, .(
  avg_gain = mean(Gain),
  sd_gain = sd(Gain),
  avg_cover = mean(Cover),
  avg_freq = mean(Frequency),
  num_seeds = .N
), by = .(Feature, fe_type)]

setorder(tb_avg_importance, -avg_gain)

# Guardar top features
fwrite(tb_avg_importance,
       file = file.path(OUTPUT_DIR, "feature_importance_aggregated.txt"),
       sep = "\t")

cat("Top 30 features por ganancia promedio:\n")
print(tb_avg_importance[1:30])

# Análisis por categoría de FE (agregado)
tb_fe_type_summary <- tb_avg_importance[, .(
  num_features = .N,
  total_gain = sum(avg_gain),
  avg_gain_per_feature = mean(avg_gain),
  total_cover = sum(avg_cover),
  total_freq = sum(avg_freq)
), by = fe_type]

setorder(tb_fe_type_summary, -total_gain)

cat("\n==============================================\n")
cat("CONTRIBUCIÓN POR TIPO DE FEATURE ENGINEERING\n")
cat("==============================================\n\n")
print(tb_fe_type_summary)

# Calcular porcentajes
tb_fe_type_summary[, pct_gain := total_gain / sum(total_gain) * 100]
tb_fe_type_summary[, pct_features := num_features / sum(num_features) * 100]

setorder(tb_fe_type_summary, -pct_gain)

cat("\n==============================================\n")
cat("PORCENTAJE DE CONTRIBUCIÓN\n")
cat("==============================================\n\n")
print(tb_fe_type_summary[, .(
  fe_type,
  num_features,
  pct_features = round(pct_features, 2),
  total_gain = round(total_gain, 4),
  pct_gain = round(pct_gain, 2)
)])

# Guardar resumen por tipo
fwrite(tb_fe_type_summary,
       file = file.path(OUTPUT_DIR, "fe_type_contribution.txt"),
       sep = "\t")

# ============================================================================
# 4. Análisis de features AVANZADAS vs BÁSICAS
# ============================================================================

cat("\n==============================================\n")
cat("COMPARACIÓN: FE AVANZADO vs BÁSICO\n")
cat("==============================================\n\n")

# Clasificar como Avanzado o Básico
tb_fe_type_summary[, category := ifelse(
  fe_type %in% c("Trend", "Rolling Stats", "Lag Advanced (3,6)",
                 "Delta Advanced (3)", "Volatility (CV)", "Volatility (Range)", "Ratio"),
  "FE Avanzado",
  ifelse(fe_type %in% c("Lag Basic (1-2)", "Delta Basic (1-2)", "Intrames (k)", "Intrames (ratio)"),
         "FE Básico",
         "Variables Originales")
)]

tb_advanced_vs_basic <- tb_fe_type_summary[, .(
  num_features = sum(num_features),
  total_gain = sum(total_gain),
  avg_gain_per_feature = mean(avg_gain_per_feature),
  total_cover = sum(total_cover)
), by = category]

tb_advanced_vs_basic[, pct_gain := total_gain / sum(total_gain) * 100]
tb_advanced_vs_basic[, pct_features := num_features / sum(num_features) * 100]

setorder(tb_advanced_vs_basic, -pct_gain)

cat("Resumen por categoría:\n")
print(tb_advanced_vs_basic[, .(
  category,
  num_features,
  pct_features = round(pct_features, 2),
  total_gain = round(total_gain, 4),
  pct_gain = round(pct_gain, 2),
  avg_gain_per_feature = round(avg_gain_per_feature, 6)
)])

# ============================================================================
# 5. Top features de cada tipo de FE Avanzado
# ============================================================================

cat("\n==============================================\n")
cat("TOP 10 FEATURES DE CADA TIPO DE FE AVANZADO\n")
cat("==============================================\n\n")

tipos_avanzados <- c("Trend", "Rolling Stats", "Lag Advanced (3,6)",
                     "Delta Advanced (3)", "Ratio", "Volatility (CV)", "Volatility (Range)")

for (tipo in tipos_avanzados) {
  cat("\n---", tipo, "---\n")
  top_features <- tb_avg_importance[fe_type == tipo][order(-avg_gain)][1:10]

  if (nrow(top_features) > 0) {
    print(top_features[, .(Feature, avg_gain = round(avg_gain, 6), avg_cover = round(avg_cover, 6))])
  } else {
    cat("  (No hay features de este tipo)\n")
  }
}

# ============================================================================
# 6. Análisis de variables base más importantes
# ============================================================================

cat("\n==============================================\n")
cat("VARIABLES BASE MÁS BENEFICIADAS POR FE\n")
cat("==============================================\n\n")

# Extraer variable base de cada feature
extraer_variable_base <- function(feature_name) {
  # Quitar sufijos de FE
  var_base <- gsub("_(lag|delta|trend|roll_(mean|max|min|sd)|cv|range_norm|ratio_vs_(lag|mean))[0-9_]+$",
                   "", feature_name)
  return(var_base)
}

tb_avg_importance[, var_base := sapply(Feature, extraer_variable_base)]

# Agregar por variable base (solo FE avanzado)
tb_var_base_summary <- tb_avg_importance[
  fe_type %in% tipos_avanzados,
  .(
    num_derivadas = .N,
    total_gain = sum(avg_gain),
    avg_gain_per_derivada = mean(avg_gain),
    tipos_fe = paste(unique(fe_type), collapse = ", ")
  ),
  by = var_base
]

setorder(tb_var_base_summary, -total_gain)

cat("Top 20 variables base por ganancia total de sus derivadas:\n")
print(tb_var_base_summary[1:20])

fwrite(tb_var_base_summary,
       file = file.path(OUTPUT_DIR, "variables_base_ranking.txt"),
       sep = "\t")

# ============================================================================
# 7. Generar reporte final en texto
# ============================================================================

cat("\n==============================================\n")
cat("Generando reporte final...\n")
cat("==============================================\n\n")

reporte <- c(
  "==============================================================================",
  "                  REPORTE DE ANÁLISIS DE FEATURE ENGINEERING",
  "==============================================================================",
  "",
  paste("Fecha de análisis:", Sys.time()),
  paste("Número de seeds analizados:", length(seed_dirs)),
  paste("Total de features únicas:", uniqueN(tb_all_impo$Feature)),
  "",
  "==============================================================================",
  "1. RESUMEN EJECUTIVO: FE AVANZADO vs BÁSICO",
  "==============================================================================",
  ""
)

# Agregar tabla de comparación
for (i in 1:nrow(tb_advanced_vs_basic)) {
  row <- tb_advanced_vs_basic[i]
  reporte <- c(reporte,
    sprintf("%-25s", row$category),
    sprintf("  - Features: %4d (%5.1f%%)", row$num_features, row$pct_features),
    sprintf("  - Gain Total: %.4f (%5.1f%%)", row$total_gain, row$pct_gain),
    sprintf("  - Gain Promedio por Feature: %.6f", row$avg_gain_per_feature),
    ""
  )
}

reporte <- c(reporte,
  "==============================================================================",
  "2. CONTRIBUCIÓN DETALLADA POR TIPO DE FE",
  "==============================================================================",
  ""
)

for (i in 1:nrow(tb_fe_type_summary)) {
  row <- tb_fe_type_summary[i]
  reporte <- c(reporte,
    sprintf("%-30s", row$fe_type),
    sprintf("  - Features: %4d (%5.1f%%)", row$num_features, row$pct_features),
    sprintf("  - Gain Total: %.4f (%5.1f%%)", row$total_gain, row$pct_gain),
    sprintf("  - Gain Promedio: %.6f", row$avg_gain_per_feature),
    ""
  )
}

reporte <- c(reporte,
  "==============================================================================",
  "3. TOP 30 FEATURES MÁS IMPORTANTES (por ganancia promedio)",
  "==============================================================================",
  "",
  sprintf("%-60s %12s %12s %10s", "Feature", "Avg Gain", "Avg Cover", "FE Type"),
  strrep("-", 100)
)

for (i in 1:min(30, nrow(tb_avg_importance))) {
  row <- tb_avg_importance[i]
  reporte <- c(reporte,
    sprintf("%-60s %12.6f %12.6f %s",
            substr(row$Feature, 1, 60),
            row$avg_gain,
            row$avg_cover,
            row$fe_type)
  )
}

reporte <- c(reporte,
  "",
  "==============================================================================",
  "4. TOP 20 VARIABLES BASE MÁS BENEFICIADAS POR FE AVANZADO",
  "==============================================================================",
  "",
  sprintf("%-40s %12s %12s", "Variable Base", "Total Gain", "# Derivadas"),
  strrep("-", 70)
)

for (i in 1:min(20, nrow(tb_var_base_summary))) {
  row <- tb_var_base_summary[i]
  reporte <- c(reporte,
    sprintf("%-40s %12.6f %12d",
            substr(row$var_base, 1, 40),
            row$total_gain,
            row$num_derivadas)
  )
}

reporte <- c(reporte,
  "",
  "==============================================================================",
  "5. HALLAZGOS CLAVE",
  "==============================================================================",
  ""
)

# Calcular hallazgos
pct_avanzado <- tb_advanced_vs_basic[category == "FE Avanzado", pct_gain]
pct_trend <- tb_fe_type_summary[fe_type == "Trend", pct_gain]
top_tipo <- tb_fe_type_summary[1, fe_type]
top_var <- tb_var_base_summary[1, var_base]

reporte <- c(reporte,
  sprintf("• El FE Avanzado contribuye con %.1f%% de la ganancia total del modelo", pct_avanzado),
  sprintf("• Las features de tipo '%s' son las más importantes (%.1f%% de ganancia)", top_tipo, tb_fe_type_summary[1, pct_gain]),
  sprintf("• La variable '%s' es la más beneficiada por FE avanzado", top_var),
  sprintf("• Las top 3 features individuales son:"),
  sprintf("    1. %s (%.4f gain)", tb_avg_importance[1, Feature], tb_avg_importance[1, avg_gain]),
  sprintf("    2. %s (%.4f gain)", tb_avg_importance[2, Feature], tb_avg_importance[2, avg_gain]),
  sprintf("    3. %s (%.4f gain)", tb_avg_importance[3, Feature], tb_avg_importance[3, avg_gain]),
  "",
  "==============================================================================",
  "FIN DEL REPORTE",
  "=============================================================================="
)

# Guardar reporte
writeLines(reporte, file.path(OUTPUT_DIR, "FE_CONTRIBUTION_REPORT.txt"))

cat("\n")
cat(paste(reporte, collapse = "\n"))
cat("\n\n")

cat("==============================================================================\n")
cat("ARCHIVOS GENERADOS:\n")
cat("==============================================================================\n")
cat("1. FE_CONTRIBUTION_REPORT.txt        - Reporte completo en texto\n")
cat("2. feature_importance_aggregated.txt - Importancia promedio por feature\n")
cat("3. fe_type_contribution.txt          - Contribución por tipo de FE\n")
cat("4. variables_base_ranking.txt        - Ranking de variables base\n")
cat("\nUbicación:", OUTPUT_DIR, "\n")
cat("==============================================================================\n")
