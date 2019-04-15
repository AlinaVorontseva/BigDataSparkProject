library(SparkR)
library(data.table)
sparkR.session(master = "local[*]", sparkConfig = list(spark.driver.memory = "16g", spark.executor.memory="16g"))

#read data
higgs <- fread("HIGGS.csv.gz", header = FALSE)
hepmas <- fread("all_train.csv.gz", header = TRUE)

colnames(hepmas)[1] <- "label"

set.seed(123)
train_ind_higgs <- base::sample(seq_len(nrow(higgs)), size = floor(0.8 * nrow(higgs)))
train_ind_hepmas <- base::sample(seq_len(nrow(hepmas)), size = floor(0.8 * nrow(hepmas)))

col_higgs <- colnames(higgs)
col_hepmas <- colnames(hepmas)

col_higgs_test <- as.vector(col_higgs[!col_higgs %in% "V1"])
col_hepmas_test <- as.vector(col_hepmas[!col_hepmas %in% "label"])

training_higgs <- higgs[train_ind_higgs, ]
training_hepmas <- hepmas[train_ind_hepmas, ]

test_higgs <- higgs[-train_ind_higgs, col_higgs_test, with = FALSE]
test_hepmas <- hepmas[-train_ind_hepmas, col_hepmas_test, with = FALSE]
print(colnames(higgs))
test_label_higgs <- higgs[-train_ind_higgs, "V1"]
test_label_hepmas <- hepmas[-train_ind_hepmas, "label"]

print(paste0("number of rows in test_label higgs: ", nrow(test_label_higgs)))
print(paste0("number of rows in test_label hepmas: ", nrow(test_label_hepmas)))

training_higgs <- as.DataFrame(training_higgs)
training_hepmas <- as.DataFrame(training_hepmas)

test_higgs <- as.DataFrame(test_higgs)
test_hepmas <- as.DataFrame(test_hepmas)

start_time <- Sys.time()

model <- spark.logit(training_higgs, V1 ~ ., regParam = 0.3, elasticNetParam = 0.8, maxIter = 15)
predictions <- predict(model, test_higgs)
end_time <- Sys.time()

elapsed_time <- end_time - start_time
acc <- sum(as.data.frame(predictions)$prediction == test_label_higgs)/nrow(test_label_higgs)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(paste0("dataset: HIGGS with accuracy = ", acc, ". Time spent: ", elapsed_time))

start_time <- Sys.time()
model <- spark.logit(training_hepmas, label ~ ., regParam = 0.3, elasticNetParam = 0.8, maxIter = 15)
# Prediction
predictions <- predict(model, test_hepmas)
end_time <- Sys.time()
elapsed_time <- end_time - start_time
acc <- sum(as.data.frame(predictions)$prediction == test_label_hepmas)/nrow(test_label_hepmas)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(paste0("dataset: Hepmass with accuracy = ", acc, ". Time spent: ", elapsed_time))

