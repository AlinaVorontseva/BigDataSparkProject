from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LinearSVC
import time


def TrainSVM(trainingData,testData):    
    # create the trainer and set its parameters
    lsvc = LinearSVC(maxIter=10, regParam=0.1)

    # train the model
    start = time.time()
    model = lsvc.fit(trainingData)
    end = time.time()
    print('Training LR model took',end-start)
    
    # Make predictions train.
    predictions = model.transform(trainData)

    # Select (prediction, true label) and compute test error
    evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    print("AUC train = %g" % (auc,))

    # Make predictions test.
    predictions = model.transform(testData)

    # Select (prediction, true label) and compute test error
    evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    print("AUC test = %g" % (auc,))
    
    return model


def TrainLogReg(trainingData,testData):
    # Train a  model.
    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    start = time.time()
    model = lr.fit(trainingData)
    end = time.time()
    print('Training LR model took',end-start)
    
    # Make predictions train.
    predictions = model.transform(trainingData)
    
    # Select (prediction, true label) and compute test error
    evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    print("AUC train = %g" % (auc,))
    
    # Make predictions test.
    predictions = model.transform(testData)
    
    # Select (prediction, true label) and compute test error
    evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    print("AUC test = %g" % (auc,))
    
    return model


def TrainDT(trainingData,testData):
    # Train a DecisionTree model.
    dt = DecisionTreeRegressor()

    # Train model.  This also runs the indexer.
    start = time.time()
    model = dt.fit(trainingData)
    end = time.time()
    print('Training DT model took',end-start)

    # Make predictions.
    predictions = model.transform(testData)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
    
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="r2")
    r2 = evaluator.evaluate(predictions)
    print("R2 on test data = %g" % r2)
    
    # Make predictions for train
    predictions = model.transform(trainingData)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on train data = %g" % rmse)
    
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="r2")
    r2 = evaluator.evaluate(predictions)
    print("R2 on train data = %g" % r2)
    
    return model


def TrainLinReg(trainingData,testData):
    
    lr = LinearRegression(maxIter=15, regParam=0.3, elasticNetParam=0.8)

    # Fit the model
    start = time.time()
    lrModel = lr.fit(trainingData)
    end = time.time()
    print('Training LR model took',end-start)

    # Print the coefficients and intercept for linear regression
    #print("Coefficients: %s" % str(lrModel.coefficients))
    #print("Intercept: %s" % str(lrModel.intercept))

    # Summarize the model over the training set and print out some metrics
    trainingSummary = lrModel.summary
    print("numIterations: %d" % trainingSummary.totalIterations)
    #print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
    #trainingSummary.residuals.show()
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)
    
    # Evaluate on test dataset
    #lr_predictions = lrModel.transform(testData)
    #lr_predictions.select("prediction","MV","features").show(5)
    
    test_result = lrModel.evaluate(testData)
    print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)
    print("R2 on test data = %g" % test_result.r2)
    
    #lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
    #                 labelCol='label',metricName="r2")
    #print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))
    return lrModel


def TrainMLP(trainingData,testData,layers):    
    # specify layers for the neural network:
    # input layer of size (features), two intermediate layers
    # and output of size (classes)

    # create the trainer and set its parameters
    mlp = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128)

    # train the model
    start = time.time()
    model = mlp.fit(trainingData)
    end = time.time()
    print('Training MLP model took',end-start)

    # Make predictions.
    predictions = model.transform(testData)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g, accuracy = %g" % (1.0 - accuracy,accuracy))
    
    return model


def TrainRF(trainingData,testData,numTrees=10):
    # Train a RandomForest model.
    rf = RandomForestClassifier(numTrees=numTrees)
    start = time.time()
    model = rf.fit(trainingData)
    end = time.time()
    print('Training RF model took',end-start)
    # Make predictions.
    predictions = model.transform(testData)
    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g, accuracy = %g" % (1.0 - accuracy,accuracy))
    
    return model
