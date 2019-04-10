import time
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import *
from utils import *
from models import TrainLR, TrainSVM

# First setup
conf = SparkConf().setAppName("BDProject").setMaster("local")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

# Load and parse the data file, converting it to a DataFrame.
start = time. time()
higgs = sqlContext.read.load('Datasets/HIGGS/HIGGS.csv', 
                          format='com.databricks.spark.csv', 
                          header='false', 
                          inferSchema='true')
end = time. time()
print('HIGGS file reading time:', end - start)

# Preprocessing

labelcol = '_c0'
readycols = higgs.columns[1:]
dataset = higgs
nclasses = 2
dataset_name = 'higgs'

trainingData,testData = Prepare(dataset,labelcol,readycols,[])
end = time. time()
print('HIGGS preprocessing time:', end - start)

# Training

start = time. time()
model = TrainLR(trainingData,testData)
end = time. time()
print('HIGGS LR training time:', end - start)
#model.save('Models/'+dataset_name+'_RF')

start = time. time()
model = TrainSVM(trainingData,testData,layers)
end = time. time()
print('HIGGS SVM training time:', end - start)
#model.save('Models/'+dataset_name+'_MLP')

# Load and parse the data file, converting it to a DataFrame.
start = time. time()
hepmass = sqlContext.read.load('Datasets/HEPMASS/all_train.csv', 
                          format='com.databricks.spark.csv', 
                          header='true', 
                          delimiter=',',
                          inferSchema='true')
end = time. time()
print('HEPMASS file reading time:', end - start)

# Preprocessing

start = time. time()
labelcol = '# label'
readycols = hepmass.columns[1:]
dataset = hepmass
nclasses = 2
dataset_name = 'hepmass'

trainingData,testData = Prepare(dataset,labelcol,readycols,[])
end = time. time()
print('HEPMASS preprocessing time:', end - start)

# Training

start = time. time()
model = TrainLR(trainingData,testData)
end = time. time()
print('HEPMASS RF training time:', end - start)
#model.save('Models/'+dataset_name+'_RF')

start = time. time()
model = TrainSVM(trainingData,testData)
end = time. time()
print('HEPMASS MLP training time:', end - start)
#model.save('Models/'+dataset_name+'_MLP')