from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import *
from utils import *
from models import TrainLR, TrainDT

# First setup
conf = SparkConf().setAppName("BDProjectRegression").setMaster("local")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

# Load and parse the data file, converting it to a DataFrame.
sgemm = sqlContext.read.load('Datasets/SGEMM/sgemm_product_dataset/sgemm_product.csv', 
                          format='com.databricks.spark.csv', 
                          header='true', 
                          delimiter=',',
                          inferSchema='true')

# Averaging runtimes
sgemm = sgemm.withColumn('Run', (sgemm['Run1 (ms)']+sgemm['Run2 (ms)']+sgemm['Run3 (ms)']+sgemm['Run4 (ms)'])/4)

labelcol = "Run"
readycols = sgemm.columns[:-5]
categoricalColumns = []
dataset = sgemm
dataset_name = 'sgemm'

trainingData,testData = Prepare(dataset,labelcol,readycols,categoricalColumns)

model = TrainDT(trainingData,testData)
#model.save('Models/'+dataset_name+'_DT')

model = TrainLR(trainingData,testData)
#model.save('Models/'+dataset_name+'_LR')

# Load and parse the data file, converting it to a DataFrame.
year = sqlContext.read.load('Datasets/Year/YearPredictionMSD.txt', 
                          format='com.databricks.spark.csv', 
                          header='false', 
                          delimiter=',',
                          inferSchema='true')

labelcol = "_c0"
readycols = year.columns[1:]
categoricalColumns = []
dataset = year
dataset_name = 'year'

trainingData,testData = Prepare(dataset,labelcol,readycols,categoricalColumns)

model = TrainLR(trainingData,testData)
#model.save('Models/'+dataset_name+'_LR')

model = TrainDT(trainingData,testData)
#model.save('Models/'+dataset_name+'_DT')