library(SparkR)
library(data.table)
sparkR.session(master = "local[*]", sparkConfig = list(spark.driver.memory = "16g", spark.executor.memory="16g"))

### covtype
df <- read.csv("covtype.data", header = FALSE) 

set.seed(123)
train_ind <- base::sample(seq_len(nrow(df)), size = floor(0.8 * nrow(df)))

for (col in colnames(df)) {
	if(sd(df[, col]) == 0) df[, col] <- NULL
}

x <- colnames(df)
x2 <- as.vector(x[!x %in% "V55"])
training <- df[train_ind, ]
test <- df[-train_ind, x2]

test_label <- df[-train_ind, label]

print(paste0("number of rows in test_label: ", nrow(test_label)))

training <- as.DataFrame(training)
test <- as.DataFrame(test)

start_time <- Sys.time()
# Fit a random forest classification model with spark.randomForest
model_rf <- spark.randomForest(training, V55 ~ ., "classification", numTrees = 10)

# Prediction
predictions <- predict(model_rf, test)
end_time <- Sys.time()
elapsed_time <- end_time - start_time
acc <- sum(as.data.frame(predictions)$prediction == test_label)/nrow(test_label)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(paste0("dataset: covtype, randomforest"," with accuracy = ", acc, ". Time spent: ", elapsed_time))

start_time <- Sys.time()
# multi-layer perceptron
model_mlp <- spark.mlp(training, V55 ~ ., 2)

#Prediction
predictions <- predict(model_mlp, test)
end_time <- Sys.time()
elapsed_time <- end_time - start_time
acc <- sum(as.data.frame(predictions)$prediction == test_label)/nrow(test_label)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(paste0("dataset: covtype, MLP"," with accuracy = ", acc, ". Time spent: ", elapsed_time))

######################################### wearable #############################################################
df <- read.csv("dataset-har-PUC-Rio-ugulino.csv", header = TRUE) 

for (col in colnames(df)) {
  if(sd(df[, col]) == 0) df[, col] <- NULL
}

x <- colnames(df)
x2 <- as.vector(x[!x %in% "class"])
training <- df[train_ind, ]
test <- df[-train_ind, x2]

test_label <- df[-train_ind, label]

print(paste0("number of rows in test_label: ", nrow(test_label)))

training <- as.DataFrame(training)
test <- as.DataFrame(test)

start_time <- Sys.time()
# Fit a random forest classification model with spark.randomForest
model_rf <- spark.randomForest(training, class ~ ., "classification", numTrees = 10)

# Prediction
predictions <- predict(model_rf, test)
end_time <- Sys.time()
elapsed_time <- end_time - start_time
acc <- sum(as.data.frame(predictions)$prediction == test_label)/nrow(test_label)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(paste0("dataset: wearable dataset, randomforest"," with accuracy = ", acc, ". Time spent: ", elapsed_time))

start_time <- Sys.time()
# multi-layer perceptron
model_mlp <- spark.mlp(training, V55 ~ ., 2)

#Prediction
predictions <- predict(model_mlp, test)
end_time <- Sys.time()
elapsed_time <- end_time - start_time
acc <- sum(as.data.frame(predictions)$prediction == test_label)/nrow(test_label)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(paste0("dataset: wearable, MLP"," with accuracy = ", acc, ". Time spent: ", elapsed_time))
