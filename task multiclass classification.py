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

# Load and parse the data file, converting it to a DataFrame.
covertype = sqlContext.read.load('Datasets/Covertype/covtype.data', 
                          format='com.databricks.spark.csv', 
                          header='false', 
                          inferSchema='true')
labelcol = "_c54"
readycols = covertype.columns[0:-1]
dataset = covertype
nclasses = 7
dataset_name = 'covertype'

trainingData,testData = Prepare(dataset,labelcol,readycols,[])

model = TrainRF(trainingData,testData)
#model.save('Models/'+dataset_name+'_RF')

layers = [len(trainingData.select('features').take(1)[0][0]), 100, 100, nclasses]
model = TrainMLP(trainingData,testData,layers)
#model.save('Models/'+dataset_name+'_MLP')

# Load and parse the data file, converting it to a DataFrame.
wearable = sqlContext.read.load('Datasets/Wearable/dataset-har-PUC-Rio-ugulino.csv', 
                          format='com.databricks.spark.csv', 
                          header='true', 
                          delimiter=';',
                          inferSchema='true')

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

accuracy, model = TrainRF(trainingData,testData)
#model.save('Models/'+dataset_name+'_RF')

layers = [len(trainingData.select('features').take(1)[0][0]), 100, 100, nclasses]
accuracy, model = TrainMLP(trainingData,testData,layers)
#model.save('Models/'+dataset_name+'_MLP')