from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler

def Prepare(dataset,labelcol,readycols=[],categoricalColumns=[]):
    stages = [] # stages in Pipeline

    # Convert label into label indices using the StringIndexer
    label_stringIdx = StringIndexer(inputCol=labelcol, outputCol="label")
    stages += [label_stringIdx]

    if len(categoricalColumns)>0:
        for categoricalCol in categoricalColumns:
            # Category Indexing with StringIndexer
            stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
            # Use OneHotEncoder to convert categorical variables into binary SparseVectors
            # encoder = OneHotEncoderEstimator(inputCol=categoricalCol + "Index", outputCol=categoricalCol + "classVec")
            encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
            # Add stages.  These are not run here, but will run all at once later on.
            stages += [stringIndexer, encoder]
        inputcols = readycols + [c + "classVec" for c in categoricalColumns]
    else:
        inputcols = readycols

    # Transform all features into a vector using VectorAssembler
    assembler = VectorAssembler(inputCols=inputcols, outputCol="features")
    stages += [assembler]

    partialPipeline = Pipeline().setStages(stages)
    pipelineModel = partialPipeline.fit(dataset)
    data = pipelineModel.transform(dataset)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])
    
    return trainingData, testData