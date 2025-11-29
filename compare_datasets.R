# ============================================================================
# Script para comparar dataset descargado de URL vs dataset local existente
# ============================================================================

library(data.table)
library(digest)

cat("\n" , rep("=", 80), "\n", sep = "")
cat("COMPARACIÓN DE DATASETS: URL vs LOCAL\n")
cat(rep("=", 80), "\n\n", sep = "")

# ============================================================================
# PASO 1: Descargar dataset desde URL
# ============================================================================

cat("PASO 1: Descargando dataset desde URL...\n")
cat(rep("-", 80), "\n", sep = "")

dataset_url <- "https://storage.googleapis.com/open-courses/austral2025-af91/gerencial_competencia_2025.csv.gz"
dataset_url_file <- "./datasets/gerencial_competencia_2025_FROM_URL.csv.gz"

# Crear directorio si no existe
if(!dir.exists("./datasets")) {
  dir.create("./datasets", recursive = TRUE)
}

download.file(dataset_url, dataset_url_file, mode = "wb")
cat("✓ Dataset descargado exitosamente\n\n")

# ============================================================================
# PASO 2: Cargar dataset local existente
# ============================================================================

cat("PASO 2: Cargando dataset local existente...\n")
cat(rep("-", 80), "\n", sep = "")

dataset_local_file <- "./datasets/gerencial_competencia_2025.csv.gz"

if(!file.exists(dataset_local_file)) {
  cat("ERROR: No se encuentra el archivo local en:", dataset_local_file, "\n")
  cat("Por favor verifica la ruta.\n")
  quit(status = 1)
}

cat("✓ Archivo local encontrado\n\n")

# ============================================================================
# PASO 3: Comparar archivos comprimidos (.csv.gz)
# ============================================================================

cat("PASO 3: Comparando archivos comprimidos (.csv.gz)...\n")
cat(rep("-", 80), "\n", sep = "")

# Tamaños de archivo
size_url <- file.info(dataset_url_file)$size
size_local <- file.info(dataset_local_file)$size

cat(sprintf("Tamaño archivo URL:   %.2f MB\n", size_url / (1024^2)))
cat(sprintf("Tamaño archivo LOCAL: %.2f MB\n", size_local / (1024^2)))

if(size_url == size_local) {
  cat("✓ Tamaños IGUALES\n")
} else {
  cat("✗ Tamaños DIFERENTES\n")
}

# MD5 hashes de archivos comprimidos
cat("\nCalculando MD5 de archivos comprimidos...\n")
md5_url <- digest(dataset_url_file, algo = "md5", file = TRUE)
md5_local <- digest(dataset_local_file, algo = "md5", file = TRUE)

cat(sprintf("MD5 archivo URL:   %s\n", md5_url))
cat(sprintf("MD5 archivo LOCAL: %s\n", md5_local))

if(md5_url == md5_local) {
  cat("\n✓✓✓ ARCHIVOS COMPRIMIDOS SON IDÉNTICOS ✓✓✓\n")
  cat("Los archivos .csv.gz son byte-por-byte idénticos.\n")
  cat("No es necesario continuar con la comparación.\n\n")

  # Limpiar archivo temporal descargado
  file.remove(dataset_url_file)
  cat("Archivo temporal eliminado.\n")

  cat("\n", rep("=", 80), "\n", sep = "")
  cat("CONCLUSIÓN: Los datasets son EXACTAMENTE IGUALES\n")
  cat(rep("=", 80), "\n\n", sep = "")

  quit(status = 0)
} else {
  cat("\n✗✗✗ ARCHIVOS COMPRIMIDOS SON DIFERENTES ✗✗✗\n")
  cat("Continuando con análisis detallado de contenido...\n\n")
}

# ============================================================================
# PASO 4: Cargar y comparar contenido de los datasets
# ============================================================================

cat("PASO 4: Cargando datasets descomprimidos...\n")
cat(rep("-", 80), "\n", sep = "")

dataset_url_data <- fread(dataset_url_file)
cat("✓ Dataset URL cargado\n")

dataset_local_data <- fread(dataset_local_file)
cat("✓ Dataset LOCAL cargado\n\n")

# ============================================================================
# PASO 5: Comparar dimensiones
# ============================================================================

cat("PASO 5: Comparando dimensiones...\n")
cat(rep("-", 80), "\n", sep = "")

cat(sprintf("Filas URL:    %d\n", nrow(dataset_url_data)))
cat(sprintf("Filas LOCAL:  %d\n", nrow(dataset_local_data)))

if(nrow(dataset_url_data) == nrow(dataset_local_data)) {
  cat("✓ Número de filas IGUAL\n")
} else {
  cat("✗ Número de filas DIFERENTE\n")
}

cat(sprintf("\nColumnas URL:   %d\n", ncol(dataset_url_data)))
cat(sprintf("Columnas LOCAL: %d\n", ncol(dataset_local_data)))

if(ncol(dataset_url_data) == ncol(dataset_local_data)) {
  cat("✓ Número de columnas IGUAL\n\n")
} else {
  cat("✗ Número de columnas DIFERENTE\n\n")
}

# ============================================================================
# PASO 6: Comparar nombres de columnas
# ============================================================================

cat("PASO 6: Comparando nombres de columnas...\n")
cat(rep("-", 80), "\n", sep = "")

cols_url <- names(dataset_url_data)
cols_local <- names(dataset_local_data)

if(identical(cols_url, cols_local)) {
  cat("✓ Nombres de columnas IDÉNTICOS\n\n")
} else {
  cat("✗ Nombres de columnas DIFERENTES\n")

  # Mostrar diferencias
  only_url <- setdiff(cols_url, cols_local)
  only_local <- setdiff(cols_local, cols_url)

  if(length(only_url) > 0) {
    cat("\nColumnas solo en URL:\n")
    print(only_url)
  }

  if(length(only_local) > 0) {
    cat("\nColumnas solo en LOCAL:\n")
    print(only_local)
  }
  cat("\n")
}

# ============================================================================
# PASO 7: Comparar distribución de foto_mes
# ============================================================================

cat("PASO 7: Comparando distribución de foto_mes...\n")
cat(rep("-", 80), "\n", sep = "")

foto_mes_url <- dataset_url_data[, .N, by = foto_mes][order(foto_mes)]
foto_mes_local <- dataset_local_data[, .N, by = foto_mes][order(foto_mes)]

cat("\nDistribución URL:\n")
print(foto_mes_url)

cat("\nDistribución LOCAL:\n")
print(foto_mes_local)

if(identical(foto_mes_url, foto_mes_local)) {
  cat("\n✓ Distribución de foto_mes IDÉNTICA\n\n")
} else {
  cat("\n✗ Distribución de foto_mes DIFERENTE\n\n")
}

# ============================================================================
# PASO 8: Comparar distribución de clase_ternaria
# ============================================================================

cat("PASO 8: Comparando distribución de clase_ternaria...\n")
cat(rep("-", 80), "\n", sep = "")

clase_url <- dataset_url_data[, .N, by = clase_ternaria][order(clase_ternaria)]
clase_local <- dataset_local_data[, .N, by = clase_ternaria][order(clase_ternaria)]

cat("\nDistribución URL:\n")
print(clase_url)

cat("\nDistribución LOCAL:\n")
print(clase_local)

if(identical(clase_url, clase_local)) {
  cat("\n✓ Distribución de clase_ternaria IDÉNTICA\n\n")
} else {
  cat("\n✗ Distribución de clase_ternaria DIFERENTE\n\n")
}

# ============================================================================
# PASO 9: Comparar estadísticas de columnas clave
# ============================================================================

cat("PASO 9: Comparando estadísticas de columnas clave...\n")
cat(rep("-", 80), "\n", sep = "")

# numero_de_cliente
cat("\nClientes únicos:\n")
cat(sprintf("  URL:   %d\n", uniqueN(dataset_url_data$numero_de_cliente)))
cat(sprintf("  LOCAL: %d\n", uniqueN(dataset_local_data$numero_de_cliente)))

# mcuentas_saldo
if("mcuentas_saldo" %in% cols_url && "mcuentas_saldo" %in% cols_local) {
  cat("\nmcuentas_saldo:\n")
  cat(sprintf("  URL   - Media: %.2f, SD: %.2f, NAs: %d\n",
              mean(dataset_url_data$mcuentas_saldo, na.rm = TRUE),
              sd(dataset_url_data$mcuentas_saldo, na.rm = TRUE),
              sum(is.na(dataset_url_data$mcuentas_saldo))))
  cat(sprintf("  LOCAL - Media: %.2f, SD: %.2f, NAs: %d\n",
              mean(dataset_local_data$mcuentas_saldo, na.rm = TRUE),
              sd(dataset_local_data$mcuentas_saldo, na.rm = TRUE),
              sum(is.na(dataset_local_data$mcuentas_saldo))))
}

# ============================================================================
# PASO 10: Comparar MD5 de datos
# ============================================================================

cat("\nPASO 10: Comparando MD5 de datos...\n")
cat(rep("-", 80), "\n", sep = "")

# MD5 de primeras 1000 filas
cat("\nCalculando MD5 de primeras 1000 filas...\n")
md5_1000_url <- digest(head(dataset_url_data, 1000), algo = "md5")
md5_1000_local <- digest(head(dataset_local_data, 1000), algo = "md5")

cat(sprintf("MD5 URL:   %s\n", md5_1000_url))
cat(sprintf("MD5 LOCAL: %s\n", md5_1000_local))

if(md5_1000_url == md5_1000_local) {
  cat("✓ Primeras 1000 filas IDÉNTICAS\n")
} else {
  cat("✗ Primeras 1000 filas DIFERENTES\n")
}

# MD5 de clase_ternaria completa
cat("\nCalculando MD5 de clase_ternaria completa...\n")
md5_clase_url <- digest(dataset_url_data$clase_ternaria, algo = "md5")
md5_clase_local <- digest(dataset_local_data$clase_ternaria, algo = "md5")

cat(sprintf("MD5 URL:   %s\n", md5_clase_url))
cat(sprintf("MD5 LOCAL: %s\n", md5_clase_local))

if(md5_clase_url == md5_clase_local) {
  cat("✓ Columna clase_ternaria IDÉNTICA\n\n")
} else {
  cat("✗ Columna clase_ternaria DIFERENTE\n\n")
}

# ============================================================================
# RESUMEN FINAL
# ============================================================================

cat(rep("=", 80), "\n", sep = "")
cat("RESUMEN FINAL DE COMPARACIÓN\n")
cat(rep("=", 80), "\n\n", sep = "")

# Guardar resumen
summary_file <- "./datasets/comparison_summary.txt"
sink(summary_file)

cat("COMPARACIÓN DATASET URL vs LOCAL\n")
cat("================================\n\n")
cat(sprintf("Fecha: %s\n\n", Sys.time()))

cat("ARCHIVOS COMPRIMIDOS:\n")
cat(sprintf("  MD5 URL:   %s\n", md5_url))
cat(sprintf("  MD5 LOCAL: %s\n", md5_local))
cat(sprintf("  Resultado: %s\n\n", ifelse(md5_url == md5_local, "IGUALES ✓", "DIFERENTES ✗")))

cat("DIMENSIONES:\n")
cat(sprintf("  Filas:    URL=%d, LOCAL=%d %s\n",
            nrow(dataset_url_data),
            nrow(dataset_local_data),
            ifelse(nrow(dataset_url_data) == nrow(dataset_local_data), "✓", "✗")))
cat(sprintf("  Columnas: URL=%d, LOCAL=%d %s\n\n",
            ncol(dataset_url_data),
            ncol(dataset_local_data),
            ifelse(ncol(dataset_url_data) == ncol(dataset_local_data), "✓", "✗")))

cat("DISTRIBUCIONES:\n")
cat(sprintf("  foto_mes:       %s\n", ifelse(identical(foto_mes_url, foto_mes_local), "IGUALES ✓", "DIFERENTES ✗")))
cat(sprintf("  clase_ternaria: %s\n\n", ifelse(identical(clase_url, clase_local), "IGUALES ✓", "DIFERENTES ✗")))

cat("MD5 DE DATOS:\n")
cat(sprintf("  Primeras 1000 filas: %s\n", ifelse(md5_1000_url == md5_1000_local, "IGUALES ✓", "DIFERENTES ✗")))
cat(sprintf("  clase_ternaria:      %s\n\n", ifelse(md5_clase_url == md5_clase_local, "IGUALES ✓", "DIFERENTES ✗")))

if(md5_url == md5_local) {
  cat("CONCLUSIÓN: Los datasets son EXACTAMENTE IGUALES (archivos .gz idénticos)\n")
} else if(md5_1000_url == md5_1000_local && md5_clase_url == md5_clase_local) {
  cat("CONCLUSIÓN: Los DATOS parecen iguales, pero los archivos comprimidos difieren\n")
  cat("            (posiblemente diferente nivel de compresión)\n")
} else {
  cat("CONCLUSIÓN: Los datasets son DIFERENTES\n")
  cat("            El problema de resultados distintos puede ser por el dataset\n")
}

sink()

# Mostrar en consola también
readLines(summary_file) |> cat(sep = "\n")

cat("\n", rep("=", 80), "\n", sep = "")
cat(sprintf("Resumen guardado en: %s\n", summary_file))
cat(rep("=", 80), "\n\n", sep = "")

# Limpiar archivo temporal
file.remove(dataset_url_file)
cat("Archivo temporal de URL eliminado.\n\n")
