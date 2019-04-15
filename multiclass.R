library(SparkR)
library(data.table)
sparkR.session(master = "local[*]", sparkConfig = list(spark.driver.memory = "16g", spark.executor.memory="16g"))

results <- list()
list_of_names <- c(#"covtype.data", 
		   "WearableComputing_weight_lifting_exercises_biceps_curl_variations.csv")

for(name in list_of_names){
### read data
#if(name == "covtype.data") {
df <- read.csv(name, header = FALSE)
#else {df <- fread(name, header = TRUE)

set.seed(123)
train_ind <- base::sample(seq_len(nrow(df)), size = floor(0.8 * nrow(df)))

if(name == "covtype.data"){label = "V55"}
else { label = "V153" } 

for (col in colnames(df)) {
	if(sd(df[, col]) == 0) df[, col] <- NULL
}

x <- colnames(df)
x2 <- as.vector(x[!x %in% label])
training <- df[train_ind, ]
test <- df[-train_ind, x2]
#if(name == "WearableComputing_weight_lifting_exercises_biceps_curl_variations.csv"){
#test_label <- df[-train_ind, class]
#} else {
test_label <- df[-train_ind, label]
#}
print(paste0("number of rows in test_label: ", nrow(test_label)))
print(head(training))
print(head(test))
training <- as.DataFrame(training)
test <- as.DataFrame(test)

start_time <- Sys.time()
print(colnames(training))
# Fit a random forest classification model with spark.randomForest
if(name == "covtype.data") {model <- spark.randomForest(training, V55 ~ ., "classification", numTrees = 10)}
else {model <- spark.randomForest(training, V153 ~ ., "classification", numTrees = 10)}
# Prediction
predictions <- predict(model, test)
end_time <- Sys.time()
elapsed_time <- end_time - start_time
acc <- sum(as.data.frame(predictions)$prediction == test_label)/nrow(test_label)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(paste0("dataset: ", name," with accuracy = ", acc, ". Time spent: ", elapsed_time))
}

