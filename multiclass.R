require("SparkR")
sparkR.session()

### covtype

df <- read.csv("https://storage.googleapis.com/bigdata_data/Datasets/Covertype/covtype.data", header = FALSE) 

set.seed(123)
train_ind <- base::sample(seq_len(nrow(df)), size = floor(0.8 * nrow(df)))

for (col in colnames(df)) {
    if(sd(df[, col]) == 0) df[, col] <- NULL
}

x <- colnames(df)
x2 <- as.vector(x[!x %in% "V55"])
training <- df[train_ind, ]
test <- df[-train_ind, x2]

test_label <- df[-train_ind, "V55"]

print(paste0("number of rows in test_label: ", length(test_label)))

training <- as.DataFrame(training)
test <- as.DataFrame(test)

start_time <- Sys.time()
# Fit a random forest classification model with spark.randomForest
model_rf <- spark.randomForest(training, V55 ~ ., "classification", numTrees = 10)

# Prediction
predictions <- predict(model_rf, test)
predictions_train <- predict(model_rf, training)
end_time <- Sys.time()
elapsed_time <- end_time - start_time
acc <- sum(as.data.frame(predictions)$prediction == test_label)/length(test_label)
acc_train <- sum(as.data.frame(predictions_train)$prediction == training$V55)/nrow(training)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(paste0("train acc: ", acc_train))
print(paste0("dataset: covtype, randomforest"," with accuracy = ", acc, ". Time spent: ", elapsed_time))

start_time <- Sys.time()
# multi-layer perceptron
model_mlp <- spark.mlp(training, V55 ~ ., 2)
end_time <- Sys.time()
#Prediction
predictions <- predict(model_mlp, test)
predictions_train <- predict(model_mlp, training)

elapsed_time <- end_time - start_time
acc <- sum(as.data.frame(predictions)$prediction == test_label)/length(test_label)
acc_train <- sum(as.data.frame(predictions_train)$prediction == training$V55)/nrow(training)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(paste0("train acc: ", acc_train))
print(paste0("dataset: covtype, MLP"," with accuracy = ", acc, ". Time spent: ", elapsed_time))


######################################### wearable #############################################################
df <- read.csv('https://storage.googleapis.com/bigdata_data/Datasets/Wearable/dataset-har-PUC-Rio-ugulino.csv', header = TRUE) 

for (col in colnames(df)) {
  if(sd(df[, col]) == 0) df[, col] <- NULL
}

x <- colnames(df)
x2 <- as.vector(x[!x %in% 'class'])
training <- df[train_ind, ]
test <- df[-train_ind, x2]

test_label <- df[-train_ind, 'class']

print(paste0('number of rows in test_label: ', nrow(test_label)))

training <- as.DataFrame(training)
test <- as.DataFrame(test)

start_time <- Sys.time()
# Fit a random forest classification model with spark.randomForest
model_rf <- spark.randomForest(training, class ~ ., 'classification', numTrees = 10)
end_time <- Sys.time()
# Prediction
predictions <- predict(model_rf, test)
predictions_train <- predict(model_rf, training)

elapsed_time <- end_time - start_time

acc <- sum(as.data.frame(predictions)$prediction == test_label)/length(test_label)
acc_train <- sum(as.data.frame(predictions_train)$prediction == training$class)/nrow(training)
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print(paste0('train acc: ', acc_train))
print(paste0('dataset: wearable dataset, randomforest',' with accuracy = ', acc, '. Time spent: ', elapsed_time))

start_time <- Sys.time()
# multi-layer perceptron
model_mlp <- spark.mlp(training, class ~ ., 2)
end_time <- Sys.time()
#Prediction
predictions <- predict(model_mlp, test)
predictions_train <- predict(model_mlp, training)

elapsed_time <- end_time - start_time
acc <- sum(as.data.frame(predictions)$prediction == test_label)/length(test_label)
acc_train <- sum(as.data.frame(predictions_train)$prediction == training$class)/nrow(training)
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print(paste0('train acc: ', acc_train))
print(paste0('dataset: wearable, MLP ' with accuracy = ', acc, '. Time spent: ', elapsed_time))