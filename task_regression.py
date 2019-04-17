import time
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import *
from utils import *
from models import TrainLinReg, TrainDT

# First setup
conf = SparkConf().setAppName("BDProjectRegression").setMaster("local")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

# Load and parse the data file, converting it to a DataFrame.
start = time. time()
sgemm = sqlContext.read.load('gs://bigdata_data/Datasets/SGEMM/sgemm_product.csv', 
                          format='com.databricks.spark.csv', 
                          header='true', 
                          delimiter=',',
                          inferSchema='true')
end = time. time()
print('SGEMM file reading time:', end - start)

# Prepocessing

start = time. time()
# Averaging runtimes
sgemm = sgemm.withColumn('Run', (sgemm['Run1 (ms)']+sgemm['Run2 (ms)']+sgemm['Run3 (ms)']+sgemm['Run4 (ms)'])/4)

labelcol = "Run"
readycols = sgemm.columns[:-5]
categoricalColumns = []
dataset = sgemm
dataset_name = 'sgemm'

trainingData,testData = Prepare(dataset,labelcol,readycols,categoricalColumns)
end = time. time()
print('SGEMM preprocessing time:', end - start)

# Training

start = time. time()
model = TrainDT(trainingData,testData)
end = time. time()
print('SGEMM DT training time:', end - start)
#model.save('Models/'+dataset_name+'_DT')

start = time. time()
model = TrainLinReg(trainingData,testData)
end = time. time()
print('SGEMM LR training time:', end - start)
#model.save('Models/'+dataset_name+'_LR')

# Load and parse the data file, converting it to a DataFrame.
start = time. time()
year = sqlContext.read.load('gs://bigdata_data/Datasets/Year/YearPredictionMSD.txt', 
                          format='com.databricks.spark.csv', 
                          header='false', 
                          delimiter=',',
                          inferSchema='true')
end = time. time()
print('Year file reading time:', end - start)

# Prepocessing

start = time. time()
labelcol = "_c0"
readycols = year.columns[1:]
categoricalColumns = []
dataset = year
dataset_name = 'year'

trainingData,testData = Prepare(dataset,labelcol,readycols,categoricalColumns)
end = time. time()
print('Year preprocessing time:', end - start)

# Training

start = time. time()
model = TrainLinReg(trainingData,testData)
end = time. time()
print('Year LR training time:', end - start)
#model.save('Models/'+dataset_name+'_LR')

start = time. time()
model = TrainDT(trainingData,testData)
end = time. time()
print('Year DT training time:', end - start)
#model.save('Models/'+dataset_name+'_DT')