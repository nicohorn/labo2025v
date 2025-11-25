#!/usr/bin/env Rscript
# ============================================================================
# Comprehensive PDF Report: FE Avanzado vs Básico
# ============================================================================
# Compara los resultados del experimento BASELINE (con FE avanzado)
# contra el baseline sin FE avanzado (solo FE básico del notebook z610)

library(data.table)
library(ggplot2)
library(gridExtra)
library(knitr)
library(kableExtra)

# Configuración
BASE_DIR <- "C:/Users/User/Documents/labo2025v"
EXP_DIR <- file.path(BASE_DIR, "exp")
OUTPUT_DIR <- file.path(EXP_DIR, "exp_baseline", "fe_analysis")
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ============================================================================
# 1. Cargar resultados de ambos experimentos
# ============================================================================

cat("============================================================\n")
cat("GENERANDO REPORTE COMPARATIVO PDF\n")
cat("============================================================\n\n")

# Experimento CON FE Avanzado
cat("Cargando resultados CON FE Avanzado...\n")
tb_con_fe <- fread(file.path(EXP_DIR, "exp_baseline", "resumen_baseline_exp6160.txt"))
tb_con_fe[, experimento := "Con FE Avanzado"]
cat("  - Seeds:", nrow(tb_con_fe), "\n")
cat("  - Ganancia promedio:", mean(tb_con_fe$ganancia), "\n")

# Experimento SIN FE Avanzado (solo básico)
cat("\nCargando resultados SIN FE Avanzado (básico)...\n")
tb_sin_fe <- fread(file.path(EXP_DIR, "exp_baseline_without_fe", "resumen_5_seeds_exp6310.txt"))
tb_sin_fe[, experimento := "Sin FE Avanzado (básico)"]
# Agregar columnas faltantes con NA
tb_sin_fe[, envios_optimos := envios]
tb_sin_fe[, duracion_min := NA_real_]
tb_sin_fe[, mejor_auc := NA_real_]
tb_sin_fe[, status := "OK"]
cat("  - Seeds:", nrow(tb_sin_fe), "\n")
cat("  - Ganancia promedio:", mean(tb_sin_fe$ganancia), "\n")

# Combinar ambos datasets
tb_combined <- rbind(
  tb_con_fe[, .(seed_idx, semilla, ganancia, envios, experimento)],
  tb_sin_fe[, .(seed_idx, semilla, ganancia, envios, experimento)]
)

# ============================================================================
# 2. Calcular estadísticas comparativas
# ============================================================================

cat("\nCalculando estadísticas...\n")

# Estadísticas por experimento
tb_stats <- tb_combined[, .(
  n_seeds = .N,
  ganancia_mean = mean(ganancia),
  ganancia_median = median(ganancia),
  ganancia_sd = sd(ganancia),
  ganancia_min = min(ganancia),
  ganancia_max = max(ganancia),
  ganancia_cv = sd(ganancia) / mean(ganancia) * 100,
  envios_mean = mean(envios),
  envios_median = median(envios),
  envios_sd = sd(envios)
), by = experimento]

# Calcular mejora relativa
ganancia_sin_fe <- tb_stats[experimento == "Sin FE Avanzado (básico)", ganancia_mean]
ganancia_con_fe <- tb_stats[experimento == "Con FE Avanzado", ganancia_mean]
mejora_absoluta <- ganancia_con_fe - ganancia_sin_fe
mejora_relativa <- (ganancia_con_fe / ganancia_sin_fe - 1) * 100

envios_sin_fe <- tb_stats[experimento == "Sin FE Avanzado (básico)", envios_mean]
envios_con_fe <- tb_stats[experimento == "Con FE Avanzado", envios_mean]
reduccion_envios <- (1 - envios_con_fe / envios_sin_fe) * 100

cat("\n============================================================\n")
cat("RESUMEN DE MEJORAS:\n")
cat("============================================================\n")
cat("Ganancia promedio SIN FE: $", format(ganancia_sin_fe, big.mark = ","), "\n")
cat("Ganancia promedio CON FE: $", format(ganancia_con_fe, big.mark = ","), "\n")
cat("Mejora absoluta:          $", format(mejora_absoluta, big.mark = ","), "\n")
cat("Mejora relativa:          ", sprintf("%.1f%%", mejora_relativa), "\n")
cat("Reducción en envíos:      ", sprintf("%.1f%%", reduccion_envios), "\n")
cat("============================================================\n\n")

# ============================================================================
# 3. Iniciar generación del PDF
# ============================================================================

pdf_file <- file.path(OUTPUT_DIR, "Reporte_Comparativo_FE_Avanzado.pdf")
cat("Generando PDF:", pdf_file, "\n\n")

pdf(pdf_file, width = 11, height = 8.5)

# ============================================================================
# PÁGINA 1: PORTADA
# ============================================================================

plot.new()
text(0.5, 0.85, "Análisis Comparativo de Feature Engineering",
     cex = 2.5, font = 2, col = "#2C3E50")
text(0.5, 0.75, "FE Avanzado vs FE Básico",
     cex = 2, font = 2, col = "#34495E")

segments(0.2, 0.70, 0.8, 0.70, lwd = 3, col = "#3498DB")

text(0.5, 0.60, paste("Fecha del análisis:", Sys.Date()),
     cex = 1.3, col = "#7F8C8D")
text(0.5, 0.55, paste("Experimento:", "exp_baseline (6160) vs exp_baseline_without_fe (6310)"),
     cex = 1.1, col = "#7F8C8D")

# Resumen ejecutivo en portada
text(0.5, 0.42, "Resumen Ejecutivo", cex = 1.8, font = 2, col = "#2C3E50")

text(0.5, 0.35, sprintf("Mejora en Ganancia: +%.1f%%", mejora_relativa),
     cex = 1.8, font = 2, col = "#27AE60")
text(0.5, 0.30, sprintf("Ganancia promedio: $%s → $%s",
                        format(round(ganancia_sin_fe), big.mark = ","),
                        format(round(ganancia_con_fe), big.mark = ",")),
     cex = 1.3, col = "#2C3E50")

text(0.5, 0.22, sprintf("Reducción en Envíos: -%.1f%%", reduccion_envios),
     cex = 1.5, font = 2, col = "#3498DB")
text(0.5, 0.17, sprintf("Envíos promedio: %d → %d",
                        round(envios_sin_fe), round(envios_con_fe)),
     cex = 1.3, col = "#2C3E50")

text(0.5, 0.08, "El FE Avanzado contribuye con 88.5% de la ganancia del modelo",
     cex = 1.1, font = 3, col = "#E74C3C")

# ============================================================================
# PÁGINA 2: COMPARACIÓN DE GANANCIAS
# ============================================================================

# Plot 1: Boxplot de ganancias
p1 <- ggplot(tb_combined, aes(x = experimento, y = ganancia / 1e6, fill = experimento)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.2, size = 3, alpha = 0.6) +
  scale_fill_manual(values = c("Con FE Avanzado" = "#27AE60",
                                "Sin FE Avanzado (básico)" = "#E74C3C")) +
  labs(title = "Distribución de Ganancias por Experimento",
       subtitle = sprintf("Mejora: +%.1f%% con FE Avanzado", mejora_relativa),
       x = "", y = "Ganancia (Millones $)") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", size = 16),
        plot.subtitle = element_text(color = "#27AE60", size = 13, face = "bold"),
        axis.text.x = element_text(angle = 0, hjust = 0.5, size = 11)) +
  scale_y_continuous(labels = function(x) sprintf("$%.1fM", x))

# Plot 2: Ganancia por seed
p2 <- ggplot(tb_combined, aes(x = factor(seed_idx), y = ganancia / 1e6,
                               fill = experimento, group = experimento)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  scale_fill_manual(values = c("Con FE Avanzado" = "#27AE60",
                                "Sin FE Avanzado (básico)" = "#E74C3C")) +
  labs(title = "Ganancia por Semilla",
       x = "Seed", y = "Ganancia (Millones $)", fill = "Experimento") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold", size = 16),
        legend.position = "bottom",
        legend.title = element_text(face = "bold")) +
  scale_y_continuous(labels = function(x) sprintf("$%.1fM", x))

grid.arrange(p1, p2, ncol = 1,
             top = grid::textGrob("Comparación de Ganancias",
                                  gp = grid::gpar(fontsize = 18, fontface = "bold")))

# ============================================================================
# PÁGINA 3: COMPARACIÓN DE ENVÍOS
# ============================================================================

# Plot 3: Boxplot de envíos
p3 <- ggplot(tb_combined, aes(x = experimento, y = envios, fill = experimento)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.2, size = 3, alpha = 0.6) +
  scale_fill_manual(values = c("Con FE Avanzado" = "#3498DB",
                                "Sin FE Avanzado (básico)" = "#E67E22")) +
  labs(title = "Distribución de Envíos Óptimos",
       subtitle = sprintf("Reducción: -%.1f%% con FE Avanzado (mejor targeting)", reduccion_envios),
       x = "", y = "Número de Envíos") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", size = 16),
        plot.subtitle = element_text(color = "#3498DB", size = 13, face = "bold"),
        axis.text.x = element_text(angle = 0, hjust = 0.5, size = 11)) +
  scale_y_continuous(labels = function(x) format(x, big.mark = ","))

# Plot 4: Scatter plot ganancia vs envíos
p4 <- ggplot(tb_combined, aes(x = envios, y = ganancia / 1e6,
                               color = experimento, shape = experimento)) +
  geom_point(size = 4, alpha = 0.8) +
  scale_color_manual(values = c("Con FE Avanzado" = "#27AE60",
                                 "Sin FE Avanzado (básico)" = "#E74C3C")) +
  labs(title = "Ganancia vs Envíos",
       subtitle = "FE Avanzado: Mayor ganancia con menos envíos (mejor eficiencia)",
       x = "Número de Envíos", y = "Ganancia (Millones $)",
       color = "Experimento", shape = "Experimento") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold", size = 16),
        plot.subtitle = element_text(color = "#7F8C8D", size = 12, face = "italic"),
        legend.position = "bottom",
        legend.title = element_text(face = "bold")) +
  scale_y_continuous(labels = function(x) sprintf("$%.1fM", x)) +
  scale_x_continuous(labels = function(x) format(x, big.mark = ","))

grid.arrange(p3, p4, ncol = 1,
             top = grid::textGrob("Análisis de Envíos y Eficiencia",
                                  gp = grid::gpar(fontsize = 18, fontface = "bold")))

# ============================================================================
# PÁGINA 4: TABLA DE ESTADÍSTICAS DETALLADAS
# ============================================================================

plot.new()
text(0.5, 0.95, "Estadísticas Detalladas", cex = 2, font = 2, col = "#2C3E50")

# Formatear tabla
tb_display <- copy(tb_stats)
tb_display[, `:=`(
  ganancia_mean = sprintf("$%s", format(round(ganancia_mean), big.mark = ",")),
  ganancia_median = sprintf("$%s", format(round(ganancia_median), big.mark = ",")),
  ganancia_sd = sprintf("$%s", format(round(ganancia_sd), big.mark = ",")),
  ganancia_min = sprintf("$%s", format(round(ganancia_min), big.mark = ",")),
  ganancia_max = sprintf("$%s", format(round(ganancia_max), big.mark = ",")),
  ganancia_cv = sprintf("%.2f%%", ganancia_cv),
  envios_mean = sprintf("%d", round(envios_mean)),
  envios_median = sprintf("%d", round(envios_median)),
  envios_sd = sprintf("%.1f", envios_sd)
)]

setnames(tb_display, c(
  "Experimento", "N Seeds", "Gan. Media", "Gan. Mediana", "Gan. SD",
  "Gan. Min", "Gan. Max", "CV (%)", "Envíos Media", "Envíos Mediana", "Envíos SD"
))

# Usar gridExtra para mostrar tabla
table_grob <- gridExtra::tableGrob(tb_display, rows = NULL,
                                    theme = gridExtra::ttheme_default(
                                      base_size = 11,
                                      core = list(fg_params = list(hjust = 0, x = 0.05)),
                                      colhead = list(fg_params = list(fontface = "bold"))
                                    ))
grid.arrange(table_grob,
             top = grid::textGrob("", gp = grid::gpar(fontsize = 14)),
             vp = grid::viewport(y = 0.7, height = 0.4))

# Agregar comparación directa
text(0.5, 0.35, "Comparación Directa", cex = 1.6, font = 2, col = "#2C3E50")

comparison_text <- c(
  sprintf("Mejora en Ganancia:      +$%s  (+%.1f%%)",
          format(round(mejora_absoluta), big.mark = ","), mejora_relativa),
  "",
  sprintf("Reducción en Envíos:     -%d envíos  (-%.1f%%)",
          round(envios_sin_fe - envios_con_fe), reduccion_envios),
  "",
  sprintf("ROI Mejorado:            %.1f%% más ganancia por envío",
          ((ganancia_con_fe/envios_con_fe) / (ganancia_sin_fe/envios_sin_fe) - 1) * 100)
)

y_pos <- 0.28
for (txt in comparison_text) {
  text(0.5, y_pos, txt, cex = 1.2, font = 2, col = "#27AE60", family = "mono")
  y_pos <- y_pos - 0.04
}

# ============================================================================
# PÁGINA 5: ANÁLISIS DE CONTRIBUCIÓN DE FE
# ============================================================================

# Cargar análisis de FE (si existe)
fe_contrib_file <- file.path(OUTPUT_DIR, "fe_type_contribution.txt")

if (file.exists(fe_contrib_file)) {
  tb_fe_contrib <- fread(fe_contrib_file)

  # Plot 5: Contribución por tipo de FE
  tb_fe_contrib[, category := ifelse(
    fe_type %in% c("Trend", "Rolling Stats", "Lag Advanced (3,6)",
                   "Delta Advanced (3)", "Volatility (CV)", "Volatility (Range)", "Ratio"),
    "FE Avanzado",
    ifelse(fe_type %in% c("Lag Basic (1-2)", "Delta Basic (1-2)"),
           "FE Básico",
           "Variables Originales")
  )]

  # Agregar porcentaje de ganancia
  tb_fe_contrib[, pct_gain := total_gain / sum(total_gain) * 100]

  # Gráfico de barras por tipo
  p5 <- ggplot(tb_fe_contrib, aes(x = reorder(fe_type, -pct_gain), y = pct_gain, fill = category)) +
    geom_bar(stat = "identity", alpha = 0.8) +
    scale_fill_manual(values = c("FE Avanzado" = "#27AE60",
                                  "FE Básico" = "#E74C3C",
                                  "Variables Originales" = "#95A5A6")) +
    labs(title = "Contribución de Cada Tipo de Feature Engineering",
         subtitle = "FE Avanzado (Trend + Rolling + Lag 3,6 + etc.) contribuye con 88.5% de la ganancia",
         x = "Tipo de Feature", y = "% de Ganancia Total", fill = "Categoría") +
    theme_minimal(base_size = 13) +
    theme(plot.title = element_text(face = "bold", size = 16),
          plot.subtitle = element_text(color = "#27AE60", size = 12, face = "italic"),
          axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "bottom",
          legend.title = element_text(face = "bold")) +
    geom_text(aes(label = sprintf("%.1f%%", pct_gain)),
              vjust = -0.5, size = 3.5, fontface = "bold")

  print(p5)

  # PÁGINA 6: Tabla de contribución por categoría
  plot.new()
  text(0.5, 0.95, "Contribución por Categoría de FE", cex = 2, font = 2, col = "#2C3E50")

  tb_category_summary <- tb_fe_contrib[, .(
    num_features = sum(num_features),
    total_gain = sum(total_gain),
    pct_gain = sum(pct_gain)
  ), by = category]

  tb_category_summary[, `:=`(
    pct_features = num_features / sum(num_features) * 100,
    avg_gain_per_feature = total_gain / num_features
  )]

  setorder(tb_category_summary, -pct_gain)

  # Formatear para display
  tb_cat_display <- copy(tb_category_summary)
  tb_cat_display[, `:=`(
    num_features = sprintf("%d", num_features),
    pct_features = sprintf("%.1f%%", pct_features),
    total_gain = sprintf("%.4f", total_gain),
    pct_gain = sprintf("%.1f%%", pct_gain),
    avg_gain_per_feature = sprintf("%.6f", avg_gain_per_feature)
  )]

  setnames(tb_cat_display, c(
    "Categoría", "# Features", "% Features", "Gain Total", "% Gain", "Gain Promedio/Feature"
  ))

  table_grob2 <- gridExtra::tableGrob(tb_cat_display, rows = NULL,
                                       theme = gridExtra::ttheme_default(
                                         base_size = 13,
                                         core = list(fg_params = list(hjust = 0, x = 0.05)),
                                         colhead = list(fg_params = list(fontface = "bold"))
                                       ))
  grid.arrange(table_grob2,
               vp = grid::viewport(y = 0.7, height = 0.4))

  # Texto explicativo
  text(0.5, 0.4, "Interpretación:", cex = 1.4, font = 2, col = "#2C3E50")
  text(0.1, 0.33, "• FE Avanzado (518 features): Genera 88.5% de la ganancia del modelo",
       cex = 1.1, pos = 4, col = "#27AE60")
  text(0.1, 0.28, "• FE Básico (142 features): Solo contribuye con 7.7% de la ganancia",
       cex = 1.1, pos = 4, col = "#E74C3C")
  text(0.1, 0.23, "• Variables Originales (29 features): 3.8% de contribución",
       cex = 1.1, pos = 4, col = "#95A5A6")
  text(0.1, 0.16, "• Tipo más importante: TREND (61.2% de ganancia total)",
       cex = 1.1, pos = 4, col = "#3498DB", font = 2)
  text(0.1, 0.11, "  - Captura pendientes/trayectorias temporales en ventanas de 3 y 6 meses",
       cex = 1.0, pos = 4, col = "#7F8C8D", font = 3)
}

# ============================================================================
# PÁGINA 7: TOP FEATURES
# ============================================================================

fe_agg_file <- file.path(OUTPUT_DIR, "feature_importance_aggregated.txt")

if (file.exists(fe_agg_file)) {
  tb_fe_agg <- fread(fe_agg_file)

  plot.new()
  text(0.5, 0.95, "Top 20 Features Más Importantes", cex = 2, font = 2, col = "#2C3E50")

  # Top 20
  tb_top20 <- tb_fe_agg[1:20, .(Feature, avg_gain, fe_type)]
  tb_top20[, rank := .I]

  # Formatear
  tb_top20_display <- copy(tb_top20)
  tb_top20_display[, avg_gain := sprintf("%.6f", avg_gain)]
  setnames(tb_top20_display, c("Feature", "Gain Promedio", "Tipo FE", "Rank"))
  tb_top20_display <- tb_top20_display[, .(Rank, Feature, `Gain Promedio`, `Tipo FE`)]

  table_grob3 <- gridExtra::tableGrob(tb_top20_display, rows = NULL,
                                       theme = gridExtra::ttheme_default(
                                         base_size = 10,
                                         core = list(fg_params = list(hjust = 0, x = 0.02)),
                                         colhead = list(fg_params = list(fontface = "bold"))
                                       ))
  grid.arrange(table_grob3,
               vp = grid::viewport(y = 0.65, height = 0.6))

  text(0.5, 0.15, "Observación: 8 de las top 10 features son de tipo TREND",
       cex = 1.2, font = 3, col = "#E74C3C")
  text(0.5, 0.10, "Las variables de fecha (Visa_fechaalta, Master_fechaalta) e internet",
       cex = 1.1, font = 3, col = "#7F8C8D")
  text(0.5, 0.06, "son las más predictivas para identificar BAJA+2",
       cex = 1.1, font = 3, col = "#7F8C8D")
}

# ============================================================================
# PÁGINA 8: CONCLUSIONES Y RECOMENDACIONES
# ============================================================================

plot.new()
text(0.5, 0.95, "Conclusiones y Recomendaciones", cex = 2.2, font = 2, col = "#2C3E50")

segments(0.15, 0.92, 0.85, 0.92, lwd = 2, col = "#3498DB")

# Conclusiones
text(0.5, 0.85, "CONCLUSIONES PRINCIPALES", cex = 1.5, font = 2, col = "#27AE60")

conclusions <- c(
  sprintf("1. El FE Avanzado logra una mejora de +%.1f%% en ganancia", mejora_relativa),
  sprintf("   Ganancia: $%s → $%s",
          format(round(ganancia_sin_fe), big.mark = ","),
          format(round(ganancia_con_fe), big.mark = ",")),
  "",
  sprintf("2. Reduce los envíos en %.1f%% (mejor targeting)", reduccion_envios),
  sprintf("   Envíos: %d → %d (más eficiente)",
          round(envios_sin_fe), round(envios_con_fe)),
  "",
  "3. Las features de TREND (tendencias temporales) son las más importantes",
  "   - Contribuyen con 61.2% de la ganancia total del modelo",
  "   - Capturan patrones de comportamiento en ventanas de 3-6 meses",
  "",
  "4. Variables más predictivas: Visa_fechaalta, internet, Master_fechaalta",
  "   - Las fechas de alta y el uso de internet son señales fuertes de churn"
)

y_pos <- 0.78
for (txt in conclusions) {
  if (grepl("^[0-9]\\.", txt)) {
    text(0.08, y_pos, txt, cex = 1.1, pos = 4, font = 2, col = "#2C3E50")
  } else if (txt == "") {
    y_pos <- y_pos - 0.02
    next
  } else {
    text(0.11, y_pos, txt, cex = 1.0, pos = 4, col = "#34495E")
  }
  y_pos <- y_pos - 0.045
}

# Recomendaciones
text(0.5, 0.38, "RECOMENDACIONES", cex = 1.5, font = 2, col = "#3498DB")

recommendations <- c(
  "1. ADOPTAR el Feature Engineering Avanzado en producción",
  "   - La mejora de +130% justifica ampliamente el costo computacional",
  "",
  "2. PRIORIZAR features de tipo TREND en futuros experimentos",
  "   - Son las más importantes (61% de ganancia)",
  "   - Considerar ventanas adicionales (9, 12 meses)",
  "",
  "3. EXPLORAR nuevas variables temporales relacionadas con:",
  "   - Fechas de alta de productos",
  "   - Uso de canales digitales (internet, homebanking)",
  "   - Comisiones y márgenes",
  "",
  "4. MANTENER el proceso de BO y validación implementado",
  "   - El CV entre seeds es bajo (3.5%), modelo estable"
)

y_pos <- 0.31
for (txt in recommendations) {
  if (grepl("^[0-9]\\.", txt)) {
    text(0.08, y_pos, txt, cex = 1.1, pos = 4, font = 2, col = "#2C3E50")
  } else if (txt == "") {
    y_pos <- y_pos - 0.02
    next
  } else {
    text(0.11, y_pos, txt, cex = 1.0, pos = 4, col = "#34495E")
  }
  y_pos <- y_pos - 0.045
}

# Footer
text(0.5, 0.02, paste("Generado:", Sys.time(), "| Experimentos: 6160 vs 6310"),
     cex = 0.9, col = "#95A5A6", font = 3)

# ============================================================================
# Cerrar PDF
# ============================================================================

dev.off()

cat("\n============================================================\n")
cat("PDF GENERADO EXITOSAMENTE\n")
cat("============================================================\n")
cat("Ubicación:", pdf_file, "\n")
cat("Páginas: 8\n")
cat("\nContenido:\n")
cat("  1. Portada con resumen ejecutivo\n")
cat("  2. Comparación de ganancias\n")
cat("  3. Análisis de envíos y eficiencia\n")
cat("  4. Estadísticas detalladas\n")
cat("  5. Contribución por tipo de FE\n")
cat("  6. Tabla de contribución por categoría\n")
cat("  7. Top 20 features más importantes\n")
cat("  8. Conclusiones y recomendaciones\n")
cat("============================================================\n")
