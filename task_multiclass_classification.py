import time
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import *
from utils import *
from models import TrainRF, TrainMLP

# First setup
conf = SparkConf().setAppName("BDProject").setMaster("local")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
'''
# Load and parse the data file, converting it to a DataFrame.
start = time. time()
covertype = sqlContext.read.load('gs://bigdata_data/Datasets/Covertype/covtype.data', 
                          format='com.databricks.spark.csv', 
                          header='false', 
                          inferSchema='true')
end = time. time()
print('Covertype file reading time:', end - start)

# Preprocessing

start = time. time()
labelcol = "_c54"
readycols = covertype.columns[0:-1]
dataset = covertype
nclasses = 7
dataset_name = 'covertype'

trainingData,testData = Prepare(dataset,labelcol,readycols,[])
end = time. time()
print('Covertype preprocessing time:', end - start)

# Training

start = time. time()
model = TrainRF(trainingData,testData)
end = time. time()
print('Covertype RF training time:', end - start)
#model.save('Models/'+dataset_name+'_RF')

layers = [len(trainingData.select('features').take(1)[0][0]), 100, 100, nclasses]
start = time. time()
model = TrainMLP(trainingData,testData,layers)
end = time. time()
print('Covertype MLP training time:', end - start)
#model.save('Models/'+dataset_name+'_MLP')
'''
# Load and parse the data file, converting it to a DataFrame.
start = time. time()
wearable = sqlContext.read.load('gs://bigdata_data/Datasets/Wearable/dataset-har-PUC-Rio-ugulino.csv', 
                          format='com.databricks.spark.csv', 
                          header='true', 
                          delimiter=';',
                          inferSchema='true')
end = time. time()
print('Wearable file reading time:', end - start)

# Preprocessing

start = time. time()
strtodouble = ['how_tall_in_meters','body_mass_index','z4']
for col in strtodouble:
    wearable = wearable.withColumn(col, regexp_replace(col, ',', '.'))
    wearable = wearable.withColumn(col, wearable[col].cast("double"))

wearable = wearable.na.drop(subset=["z4"])

labelcol = "class"
readycols = [k for k,v in wearable.dtypes if v in ['int','double']]
categoricalColumns = [k for k,v in wearable.dtypes if v == 'string' and k!= labelcol]
dataset = wearable
nclasses = 5
dataset_name = 'wearable'

trainingData,testData = Prepare(dataset,labelcol,readycols,categoricalColumns)
end = time. time()
print('Wearable preprocessing time:', end - start)

# Training

start = time. time()
model = TrainRF(trainingData,testData)
end = time. time()
print('Wearable RF training time:', end - start)
#model.save('Models/'+dataset_name+'_RF')

layers = [len(trainingData.select('features').take(1)[0][0]), 100, 100, nclasses]
start = time. time()
model = TrainMLP(trainingData,testData,layers)
end = time. time()
print('Wearable MLP training time:', end - start)
#model.save('Models/'+dataset_name+'_MLP')