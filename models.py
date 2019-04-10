from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def TrainLR(trainingData,testData):
    # Train a  model.
    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    model = lr.fit(trainingData)
    # Make predictions.
    predictions = model.transform(testData)
    
    # Select (prediction, true label) and compute test error
    evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    print("AUC = %g" % (auc,))
    
    return model


def TrainDT(trainingData,testData):
    # Train a DecisionTree model.
    dt = DecisionTreeRegressor()

    # Train model.  This also runs the indexer.
    model = dt.fit(trainingData)

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
    
    return model


def TrainLR(trainingData,testData):
    
    lr = LinearRegression(maxIter=15, regParam=0.3, elasticNetParam=0.8)

    # Fit the model
    lrModel = lr.fit(trainingData)

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
    model = mlp.fit(trainingData)

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
    model = rf.fit(trainingData)
    # Make predictions.
    predictions = model.transform(testData)
    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g, accuracy = %g" % (1.0 - accuracy,accuracy))
    
    return model
