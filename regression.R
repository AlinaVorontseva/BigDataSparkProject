library(SparkR)
library(data.table)
library(Metrics)

sparkR.session(master = "local[*]", sparkConfig = list(spark.driver.memory = "16g", spark.executor.memory="16g"))

#read data
sgem <- fread("sgemm_product.csv", header = TRUE)
msd <- fread("YearPredictionMSD.txt", header = FALSE)

set.seed(123)
train_ind_sgem <- base::sample(seq_len(nrow(sgem)), size = floor(0.8 * nrow(sgem)))
train_ind_msd <- base::sample(seq_len(nrow(msd)), size = floor(0.8 * nrow(msd)))

col_sgem <- colnames(sgem)
col_msd <- colnames(msd)

col_sgem_test <- as.vector(col_sgem[!col_sgem %in% "average"])
col_msd_test <- as.vector(col_msd[!col_msd %in% "V1"])

training_sgem <- sgem[train_ind_sgem, ]
training_msd <- msd[train_ind_msd, ]

test_sgem <- sgem[-train_ind_sgem, col_sgem_test, with = FALSE]
test_msd <- msd[-train_ind_msd, col_msd_test, with = FALSE]

test_label_sgem <- sgem[-train_ind_sgem, "average"]
test_label_msd <- msd[-train_ind_msd, "V1"]

print(paste0("number of rows in test_label sgem: ", nrow(test_label_sgem)))
print(paste0("number of rows in test_label msd: ", nrow(test_label_msd)))

training_sgem <- as.DataFrame(training_sgem)
training_msd <- as.DataFrame(training_msd)

test_sgem <- as.DataFrame(test_sgem)
test_msd <- as.DataFrame(test_msd)

start_time <- Sys.time()
model <- spark.glm(training_sgem, average ~ ., family = "gaussian")
predictions <- predict(model, test_sgem)
end_time <- Sys.time()
elapsed_time <- end_time - start_time
rmse <- rmse(as.data.frame(predictions)$prediction, test_label_sgem)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(paste0("dataset: SGEM with rmse = ", rmse, ". Time spent: ", elapsed_time))

start_time <- Sys.time()
model <- spark.glm(training_msd, V1 ~ ., family = "gaussian")
predictions <- predict(model, test_msd)
end_time <- Sys.time()
elapsed_time <- end_time - start_time
rmse <- rmse(as.data.frame(predictions)$prediction,test_label_msd)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(paste0("dataset: MSD with rmse = ", rmse, ". Time spent: ", elapsed_time))

