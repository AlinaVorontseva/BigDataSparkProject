library(SparkR)
library(data.table)
sparkR.session(master = "local[*]", sparkConfig = list(spark.driver.memory = "16g", spark.executor.memory="16g"))

#read data
higgs <- fread("gs://bigdata_data/Datasets/HIGGS/HIGGS.csv", header = FALSE)
hepmas <- fread("gs://bigdata_data/Datasets/HEPMASS/all_train.csv", header = TRUE)

set.seed(123)
train_ind_higgs <- base::sample(seq_len(nrow(higgs)), size = floor(0.8 * nrow(higgs)))
#train_ind_hepmas <- base::sample(seq_len(nrow(hepmas)), size = floor(0.8 * nrow(hepmas)))

col_higgs <- colnames(higgs)
col_hepmas <- colnames(hepmas)

col_higgs_test <- as.vector(col_higgs[!col_higgs %in% "V1"])
col_hepmas_test <- as.vector(col_hepmas[!col_hepmas %in% "V1"])

training_higgs <- higgs[train_ind_higgs, ]
#training_hepmas <- hepmas[train_ind_hepmas, ]
training_hepmas <- hepmas

test_higgs <- higgs[-train_ind_higgs, col_higgs_test, with = FALSE]
#test_hepmas <- hepmas[-train_ind_hepmas, col_hepmas_test, with = FALSE]
test_hepmas <- fread("gs://bigdata_data/Datasets/HEPMASS/all_test.csv", header = TRUE)
test_hepmas <- test_hepmas[, -1]

test_label_higgs <- higgs[-train_ind_higgs, "V1"]
#test_label_hepmas <- hepmas[-train_ind_hepmas, "label"]
test_label_hepmas <- test_label_hepmas[, 1]

print(paste0("number of rows in test_label higgs: ", nrow(test_label_higgs)))
print(paste0("number of rows in test_label hepmas: ", nrow(test_label_hepmas)))

training_higgs <- as.DataFrame(training_higgs)
training_hepmas <- as.DataFrame(training_hepmas)

test_higgs <- as.DataFrame(test_higgs)
test_hepmas <- as.DataFrame(test_hepmas)

###### logistic regressin
start_time <- Sys.time()

model <- spark.logit(training_higgs, V1 ~ ., regParam = 0.3, elasticNetParam = 0.8, maxIter = 15)
predictions <- predict(model, test_higgs)
end_time <- Sys.time()

elapsed_time <- end_time - start_time
acc <- sum(as.data.frame(predictions)$prediction == test_label_higgs)/nrow(test_label_higgs)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(paste0("dataset: HIGGS (logistic regression) with accuracy = ", acc, ". Time spent: ", elapsed_time))

start_time <- Sys.time()
model <- spark.logit(training_hepmas, V1 ~ ., regParam = 0.3, elasticNetParam = 0.8, maxIter = 15)
# Prediction
predictions <- predict(model, test_hepmas)
end_time <- Sys.time()
elapsed_time <- end_time - start_time
acc <- sum(as.data.frame(predictions)$prediction == test_label_hepmas)/nrow(test_label_hepmas)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(paste0("dataset: Hepmass (logistic regression) with accuracy = ", acc, ". Time spent: ", elapsed_time))


###### SVM with linear kernel
start_time <- Sys.time()

model <- spark.svmLinear(training_higgs, V1 ~ ., regParam = 0.3,  maxIter = 15)
predictions <- predict(model, test_higgs)
end_time <- Sys.time()

elapsed_time <- end_time - start_time
acc <- sum(as.data.frame(predictions)$prediction == test_label_higgs)/nrow(test_label_higgs)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(paste0("dataset: HIGGS (linear svm) with accuracy = ", acc, ". Time spent: ", elapsed_time))

start_time <- Sys.time()
model <- spark.svmLinear(training_hepmas, V1 ~ ., regParam = 0.3, maxIter = 15)
# Prediction
predictions <- predict(model, test_hepmas)
end_time <- Sys.time()
elapsed_time <- end_time - start_time
acc <- sum(as.data.frame(predictions)$prediction == test_label_hepmas)/nrow(test_label_hepmas)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(paste0("dataset: Hepmass (linear svm) with accuracy = ", acc, ". Time spent: ", elapsed_time))
