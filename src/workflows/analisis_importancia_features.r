# ============================================================================
# ANÁLISIS DE IMPORTANCIA DE FEATURES HISTÓRICAS
# ============================================================================
# Este script permite identificar qué features del feature engineering
# histórico contribuyen más a la ganancia final del modelo.
#
# Técnicas implementadas:
# 1. Feature Importance de LightGBM (gain, split, cover)
# 2. Análisis por tipo de feature (lags, deltas, rolling, etc.)
# 3. Comparación entre variables originales vs históricas
# 4. Ranking de top features por categoría
# ============================================================================

require("data.table")
require("ggplot2")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Carpeta del experimento a analizar
experimento_folder <- "/content/buckets/b1/exp/WF6150_seed1_FE_historico"
setwd(experimento_folder)

# ============================================================================
# 1. CARGAR Y ANALIZAR FEATURE IMPORTANCE
# ============================================================================

cat("\n========================================\n")
cat("ANÁLISIS DE FEATURE IMPORTANCE\n")
cat("========================================\n\n")

# Leer el archivo de importancia generado por LightGBM
importancia <- fread("impo.txt")

cat("Total de features en el modelo:", nrow(importancia), "\n")
cat("Top 10 features por Gain:\n\n")
print(head(importancia, 10))

# ============================================================================
# 2. CLASIFICAR FEATURES POR TIPO
# ============================================================================

cat("\n\n========================================\n")
cat("CLASIFICACIÓN DE FEATURES POR TIPO\n")
cat("========================================\n\n")

# Función para clasificar el tipo de feature
clasificar_feature <- function(feature_name) {
  if (grepl("_lag[0-9]$", feature_name)) return("LAG")
  if (grepl("_delta[0-9]$", feature_name)) return("DELTA")
  if (grepl("_roll[0-9]_mean$", feature_name)) return("ROLLING_MEAN")
  if (grepl("_roll[0-9]_max$", feature_name)) return("ROLLING_MAX")
  if (grepl("_roll[0-9]_min$", feature_name)) return("ROLLING_MIN")
  if (grepl("_roll[0-9]_sd$", feature_name)) return("ROLLING_SD")
  if (grepl("_trend[0-9]$", feature_name)) return("TREND")
  if (grepl("_ratio_vs_roll[0-9]$", feature_name)) return("RATIO_ROLL")
  if (grepl("_ratio_vs_lag[0-9]$", feature_name)) return("RATIO_LAG")
  if (grepl("_cv[0-9]$", feature_name)) return("VOLATILIDAD_CV")
  if (grepl("_range[0-9]$", feature_name)) return("VOLATILIDAD_RANGE")
  return("ORIGINAL")
}

# Clasificar todas las features
importancia[, tipo_feature := sapply(Feature, clasificar_feature)]

# Calcular importancia agregada por tipo
importancia_por_tipo <- importancia[, .(
  cantidad = .N,
  gain_total = sum(Gain),
  gain_promedio = mean(Gain),
  gain_mediano = median(Gain)
), by = tipo_feature]

setorder(importancia_por_tipo, -gain_total)

cat("Importancia agregada por TIPO de feature:\n\n")
print(importancia_por_tipo)

cat("\n\nInterpretación:\n")
cat("- gain_total: Importancia acumulada de todas las features de ese tipo\n")
cat("- gain_promedio: Importancia promedio por feature de ese tipo\n")
cat("- cantidad: Número de features de ese tipo en el top del modelo\n\n")

# ============================================================================
# 3. TOP FEATURES POR CATEGORÍA
# ============================================================================

cat("\n========================================\n")
cat("TOP 20 FEATURES HISTÓRICAS\n")
cat("========================================\n\n")

# Filtrar solo features históricas (excluir originales)
features_historicas <- importancia[tipo_feature != "ORIGINAL"]

cat("Top 20 features históricas por Gain:\n\n")
print(head(features_historicas[, .(Feature, Gain, tipo_feature)], 20))

# ============================================================================
# 4. ANÁLISIS POR VARIABLE BASE
# ============================================================================

cat("\n\n========================================\n")
cat("ANÁLISIS POR VARIABLE BASE\n")
cat("========================================\n\n")

# Extraer el nombre de la variable base (antes del sufijo _lag, _delta, etc.)
extraer_variable_base <- function(feature_name) {
  # Eliminar sufijos conocidos
  base <- gsub("_lag[0-9]$", "", feature_name)
  base <- gsub("_delta[0-9]$", "", base)
  base <- gsub("_roll[0-9]_(mean|max|min|sd)$", "", base)
  base <- gsub("_trend[0-9]$", "", base)
  base <- gsub("_ratio_vs_roll[0-9]$", "", base)
  base <- gsub("_ratio_vs_lag[0-9]$", "", base)
  base <- gsub("_cv[0-9]$", "", base)
  base <- gsub("_range[0-9]$", "", base)
  return(base)
}

importancia[, variable_base := sapply(Feature, extraer_variable_base)]

# Calcular importancia por variable base
importancia_por_variable <- importancia[, .(
  cantidad_derivadas = .N,
  gain_total = sum(Gain),
  gain_promedio = mean(Gain),
  max_gain_individual = max(Gain),
  tipos_usados = paste(unique(tipo_feature), collapse = ", ")
), by = variable_base]

setorder(importancia_por_variable, -gain_total)

cat("Top 30 variables base (considerando todas sus derivadas históricas):\n\n")
print(head(importancia_por_variable, 30))

# ============================================================================
# 5. COMPARACIÓN: ORIGINALES VS HISTÓRICAS
# ============================================================================

cat("\n\n========================================\n")
cat("COMPARACIÓN: ORIGINALES VS HISTÓRICAS\n")
cat("========================================\n\n")

comparacion <- importancia[, .(
  cantidad = .N,
  gain_total = sum(Gain),
  gain_promedio = mean(Gain)
), by = .(es_historica = ifelse(tipo_feature == "ORIGINAL", "Variable Original", "Feature Histórica"))]

print(comparacion)

cat("\n")
cat("Proporción de importancia:\n")
comparacion[, proporcion := gain_total / sum(gain_total) * 100]
print(comparacion[, .(es_historica, proporcion)])

# ============================================================================
# 6. ANÁLISIS DE HORIZONTE TEMPORAL
# ============================================================================

cat("\n\n========================================\n")
cat("ANÁLISIS DE HORIZONTE TEMPORAL\n")
cat("========================================\n\n")

# Función para extraer el horizonte temporal (lag1, lag2, etc.)
extraer_horizonte <- function(feature_name) {
  if (grepl("_lag1|_delta1|_roll3|_trend3|_cv3|_range3|_ratio_vs_roll3", feature_name)) return("1-3 meses")
  if (grepl("_lag2|_delta2", feature_name)) return("2 meses")
  if (grepl("_lag3|_delta3", feature_name)) return("3 meses")
  if (grepl("_lag6|_roll6|_trend6|_cv6|_ratio_vs_lag6|_ratio_vs_roll6", feature_name)) return("6 meses")
  return("N/A")
}

importancia[, horizonte := sapply(Feature, extraer_horizonte)]

horizonte_stats <- importancia[horizonte != "N/A", .(
  cantidad = .N,
  gain_total = sum(Gain),
  gain_promedio = mean(Gain)
), by = horizonte]

setorder(horizonte_stats, -gain_total)

cat("Importancia por horizonte temporal:\n\n")
print(horizonte_stats)

cat("\n\nInterpretación:\n")
cat("- '1-3 meses': Features de corto plazo (más recientes)\n")
cat("- '6 meses': Features de largo plazo (patrones semestrales)\n")
cat("Esto indica qué horizonte temporal es más predictivo.\n")

# ============================================================================
# 7. GUARDAR ANÁLISIS DETALLADO
# ============================================================================

cat("\n\n========================================\n")
cat("GUARDANDO RESULTADOS\n")
cat("========================================\n\n")

# Guardar tabla completa con clasificaciones
fwrite(importancia,
       file = "importancia_clasificada.txt",
       sep = "\t")

# Guardar resúmenes
fwrite(importancia_por_tipo,
       file = "importancia_por_tipo.txt",
       sep = "\t")

fwrite(importancia_por_variable,
       file = "importancia_por_variable_base.txt",
       sep = "\t")

fwrite(horizonte_stats,
       file = "importancia_por_horizonte.txt",
       sep = "\t")

cat("Archivos generados:\n")
cat("✓ importancia_clasificada.txt - Tabla completa con clasificación de features\n")
cat("✓ importancia_por_tipo.txt - Resumen por tipo de feature\n")
cat("✓ importancia_por_variable_base.txt - Resumen por variable base\n")
cat("✓ importancia_por_horizonte.txt - Resumen por horizonte temporal\n")

# ============================================================================
# 8. VISUALIZACIONES (SI TIENES GGPLOT2)
# ============================================================================

if(require("ggplot2")) {

  cat("\nGenerando visualizaciones...\n")

  # Gráfico 1: Top 20 features
  p1 <- ggplot(head(importancia, 20), aes(x = reorder(Feature, Gain), y = Gain, fill = tipo_feature)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    theme_minimal() +
    labs(title = "Top 20 Features por Importancia (Gain)",
         x = "Feature",
         y = "Gain",
         fill = "Tipo") +
    theme(legend.position = "bottom")

  ggsave("top20_features.pdf", p1, width = 12, height = 8)

  # Gráfico 2: Importancia por tipo
  p2 <- ggplot(importancia_por_tipo, aes(x = reorder(tipo_feature, gain_total), y = gain_total, fill = tipo_feature)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    theme_minimal() +
    labs(title = "Importancia Total por Tipo de Feature",
         x = "Tipo de Feature",
         y = "Gain Total") +
    theme(legend.position = "none")

  ggsave("importancia_por_tipo.pdf", p2, width = 10, height = 6)

  # Gráfico 3: Top 20 variables base
  p3 <- ggplot(head(importancia_por_variable, 20), aes(x = reorder(variable_base, gain_total), y = gain_total)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    theme_minimal() +
    labs(title = "Top 20 Variables Base (con todas sus derivadas)",
         x = "Variable Base",
         y = "Gain Total Acumulado")

  ggsave("top20_variables_base.pdf", p3, width = 12, height = 8)

  cat("✓ top20_features.pdf\n")
  cat("✓ importancia_por_tipo.pdf\n")
  cat("✓ top20_variables_base.pdf\n")
}

# ============================================================================
# 9. RECOMENDACIONES BASADAS EN EL ANÁLISIS
# ============================================================================

cat("\n\n========================================\n")
cat("RECOMENDACIONES\n")
cat("========================================\n\n")

cat("Basado en este análisis, puedes:\n\n")

cat("1. REDUCCIÓN DE FEATURES:\n")
cat("   - Si un tipo de feature tiene bajo gain_total, considera eliminarlo\n")
cat("   - Mantén solo las features con Gain > umbral (ej: percentil 75)\n\n")

cat("2. FEATURE ENGINEERING ADICIONAL:\n")
cat("   - Si un tipo tiene alto gain_promedio, crea más features de ese tipo\n")
cat("   - Si una variable_base tiene alto gain_total, crea más derivadas\n\n")

cat("3. OPTIMIZACIÓN:\n")
cat("   - Si features de corto plazo (1-3 meses) dominan, enfócate en ellas\n")
cat("   - Si features de largo plazo (6 meses) son mejores, expande a 12 meses\n\n")

cat("4. PRÓXIMA ITERACIÓN:\n")
cat("   - Experimenta eliminando features con Gain < percentil 25\n")
cat("   - Crea interacciones entre las top features\n")
cat("   - Prueba transformaciones no lineales de las top variables\n\n")

cat("========================================\n")
cat("ANÁLISIS COMPLETADO\n")
cat("========================================\n")
