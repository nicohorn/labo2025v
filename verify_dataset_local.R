# ============================================================================
# Script para verificar dataset LOCAL y comparar con Google Colab
# ============================================================================

library(data.table)
library(digest)

# Ajusta esta ruta si tu dataset está en otro lugar
dataset_file <- "C:/Users/User/Documents/labo2025v/datasets/gerencial_competencia_2025.csv.gz"

# Verificar que el archivo existe
if(!file.exists(dataset_file)) {
  cat("ERROR: No se encuentra el archivo en:", dataset_file, "\n")
  cat("Por favor ajusta la ruta en el script.\n")
  quit(status = 1)
}

cat("\n=== VERIFICACIÓN DE DATASET LOCAL ===\n")
cat("Archivo:", dataset_file, "\n\n")

# Verificar tamaño del archivo
file_size <- file.info(dataset_file)$size
cat(sprintf("Tamaño del archivo: %.2f MB\n", file_size / (1024^2)))

# Calcular MD5 hash del archivo comprimido
cat("\n=== HASH DEL ARCHIVO COMPRIMIDO ===\n")
md5_gz <- digest(dataset_file, algo = "md5", file = TRUE)
cat(sprintf("MD5 (.csv.gz): %s\n", md5_gz))

# Cargar dataset
cat("\n=== CARGANDO DATASET ===\n")
dataset <- fread(dataset_file)
cat("Dataset cargado exitosamente\n")

# Estadísticas básicas
cat("\n=== ESTADÍSTICAS BÁSICAS ===\n")
cat(sprintf("Número de filas: %d\n", nrow(dataset)))
cat(sprintf("Número de columnas: %d\n", ncol(dataset)))
cat(sprintf("Memoria utilizada: %.2f MB\n", object.size(dataset) / (1024^2)))

# Información de foto_mes
cat("\n=== DISTRIBUCIÓN DE FOTO_MES ===\n")
foto_mes_dist <- dataset[, .N, by = foto_mes][order(foto_mes)]
print(foto_mes_dist)

# Información de clase_ternaria
cat("\n=== DISTRIBUCIÓN DE CLASE_TERNARIA ===\n")
clase_dist <- dataset[, .N, by = clase_ternaria]
print(clase_dist)

# Primeras y últimas columnas
cat("\n=== NOMBRES DE COLUMNAS ===\n")
cat("Primeras 10 columnas:\n")
print(head(names(dataset), 10))
cat("\nÚltimas 10 columnas:\n")
print(tail(names(dataset), 10))

# Estadísticas de algunas columnas clave
cat("\n=== ESTADÍSTICAS DE COLUMNAS CLAVE ===\n")
if("numero_de_cliente" %in% names(dataset)) {
  cat(sprintf("Clientes únicos: %d\n", uniqueN(dataset$numero_de_cliente)))
}

if("mcuentas_saldo" %in% names(dataset)) {
  cat(sprintf("mcuentas_saldo - Media: %.2f, SD: %.2f, NAs: %d\n",
              mean(dataset$mcuentas_saldo, na.rm = TRUE),
              sd(dataset$mcuentas_saldo, na.rm = TRUE),
              sum(is.na(dataset$mcuentas_saldo))))
}

if("mtarjeta_visa_consumo" %in% names(dataset)) {
  cat(sprintf("mtarjeta_visa_consumo - Media: %.2f, SD: %.2f, NAs: %d\n",
              mean(dataset$mtarjeta_visa_consumo, na.rm = TRUE),
              sd(dataset$mtarjeta_visa_consumo, na.rm = TRUE),
              sum(is.na(dataset$mtarjeta_visa_consumo))))
}

# Calcular hash de los datos (primeras 1000 filas)
cat("\n=== HASH DE DATOS (primeras 1000 filas) ===\n")
data_sample <- head(dataset, 1000)
data_hash <- digest(data_sample, algo = "md5")
cat(sprintf("MD5 (primeras 1000 filas): %s\n", data_hash))

# Hash de toda la columna clase_ternaria
cat("\n=== HASH DE COLUMNA CLASE_TERNARIA ===\n")
clase_hash <- digest(dataset$clase_ternaria, algo = "md5")
cat(sprintf("MD5 (clase_ternaria completa): %s\n", clase_hash))

# Guardar resumen en archivo
cat("\n=== GUARDANDO RESUMEN ===\n")
summary_file <- "C:/Users/User/Documents/labo2025v/dataset_verification_local.txt"
sink(summary_file)
cat("VERIFICACIÓN DE DATASET - LOCAL\n")
cat("===============================\n\n")
cat(sprintf("Fecha: %s\n\n", Sys.time()))
cat(sprintf("Archivo: %s\n", dataset_file))
cat(sprintf("Tamaño archivo: %.2f MB\n", file_size / (1024^2)))
cat(sprintf("MD5 archivo .gz: %s\n\n", md5_gz))
cat(sprintf("Filas: %d\n", nrow(dataset)))
cat(sprintf("Columnas: %d\n\n", ncol(dataset)))
cat("Distribución foto_mes:\n")
print(foto_mes_dist)
cat("\nDistribución clase_ternaria:\n")
print(clase_dist)
cat(sprintf("\nMD5 (primeras 1000 filas): %s\n", data_hash))
cat(sprintf("MD5 (clase_ternaria): %s\n", clase_hash))
sink()

cat(sprintf("Resumen guardado en: %s\n", summary_file))

cat("\n=== VERIFICACIÓN COMPLETA ===\n")
cat("\nAhora compara estos valores con los de Google Colab:\n")
cat("- Si los MD5 del .csv.gz son iguales → archivos idénticos\n")
cat("- Si los MD5 de las 1000 filas son iguales → datos iguales\n")
cat("- Si los MD5 de clase_ternaria son iguales → target igual\n")
cat("\nSi algún hash difiere, los datasets NO son los mismos.\n")
