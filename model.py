from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.tree import RandomForest

# Initialize SparkSession and SparkContext
conf = SparkConf().setAppName("RandomForestModel").setMaster("yarn")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# Read data from S3
inputDF = spark.read.csv('s3://winepredictiontest/TrainingDataset.csv', header=True, inferSchema=True, sep=';')

# Feature columns (exclude 'quality')
featureColumns = [c for c in inputDF.columns if c != 'quality']

# Transform DataFrame to RDD of LabeledPoint
transformed_rdd = inputDF.rdd.map(lambda row: LabeledPoint(row['quality'], Vectors.dense([row[col] for col in featureColumns])))

# Train Random Forest model
model = RandomForest.trainClassifier(
    data=transformed_rdd,
    numClasses=10,  # Based on the number of unique values in 'quality'
    categoricalFeaturesInfo={},  # Assuming no categorical features
    numTrees=50,  # Number of trees in the forest
    featureSubsetStrategy="auto",  # Auto selection of feature subset strategy
    impurity="gini",  # Gini impurity for classification
    maxDepth=20,  # Maximum depth of trees
    maxBins=64,  # Maximum number of bins for splitting features
    seed=33  # Random seed
)

# Save the trained model to S3
model.save(sc, "s3a://winepredictiontest/model_created.model")

# Stop SparkContext
sc.stop()
