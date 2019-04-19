library(SparkR)
install.packages("data.table")
library(data.table)
install.packages("Metrics")
library(Metrics)

sparkR.session(master = "local[*]", sparkConfig = list(spark.driver.memory = "16g", spark.executor.memory="16g"))

#read data
sgem <- fread("Datasets/SGEMM/sgemm_product_dataset/sgemm_product.csv", header = TRUE)
msd <- fread("Datasets/Year/YearPredictionMSD.txt", header = FALSE)
sgem$average <- rowMeans(sgem[, 15:18])
sgem[, 15:18] <- NULL

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
r2 <- cor(as.data.frame(predictions)$prediction, test_label_sgem)**2
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(paste0("dataset: SGEM (linear regr) with rmse = ", rmse, " and R^2 ", r2, ". Time spent: ", elapsed_time))

start_time <- Sys.time()
model <- spark.glm(training_msd, V1 ~ ., family = "gaussian")
predictions <- predict(model, test_msd)
end_time <- Sys.time()
elapsed_time <- end_time - start_time
rmse <- rmse(as.data.frame(predictions)$prediction,test_label_msd)
r2 <- cor(as.data.frame(predictions)$prediction, test_label_msd)**2
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(paste0("dataset: MSD (linear regr) with rmse = ", rmse, " and R^2 ", r2,". Time spent: ", elapsed_time))

######### decision tree
start_time <- Sys.time()
model <- spark.decisionTree(training_sgem, average ~ ., "regression")
predictions <- predict(model, test_sgem)
end_time <- Sys.time()
elapsed_time <- end_time - start_time
rmse <- rmse(as.data.frame(predictions)$prediction, test_label_sgem)
r2 <- cor(as.data.frame(predictions)$prediction, test_label_sgem)**2
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(paste0("dataset: SGEM (with decision tree) with rmse = ", rmse," and R^2 ", r2, ". Time spent: ", elapsed_time))

start_time <- Sys.time()
model <- spark.decisionTree(training_msd, V1 ~ ., "regression")
predictions <- predict(model, test_msd)
end_time <- Sys.time()
elapsed_time <- end_time - start_time
rmse <- rmse(as.data.frame(predictions)$prediction,test_label_msd)
r2 <- cor(as.data.frame(predictions)$prediction, test_label_msd)**2
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(paste0("dataset: MSD (with decision tree) with rmse = ", rmse, " and R^2 ", r2, ". Time spent: ", elapsed_time))

