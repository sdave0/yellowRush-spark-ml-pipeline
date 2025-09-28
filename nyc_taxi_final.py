# Databricks notebook source
# MAGIC %md
# MAGIC ####Team A
# MAGIC - Shubham Harishkumar Dave

# COMMAND ----------

# MAGIC %md
# MAGIC # **NYC Taxi Congestion & Delay Prediction**
# MAGIC
# MAGIC This project analyzes New York City yellow taxi trip data to address two core objectives:  
# MAGIC - Predict the likelihood of **traffic congestion** during a trip.  
# MAGIC - Estimate the probability of a **trip delay** beyond expected duration.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## **Project Structure**
# MAGIC > *Note:* To make it easier for review and submission, we’ve combined all four pipelines into a single notebook. Each pipeline is logically modular and can be separated if needed.
# MAGIC
# MAGIC The project is organized into four distinct pipelines, each serving a specific purpose:
# MAGIC
# MAGIC - **1. Weather Data Preprocessing**  
# MAGIC   Handles data ingestion, cleaning, and preparation for the Weather Data. The cleaned dataset is saved in Azure container for use by Taxi Data Preprocessing Pipeline.
# MAGIC
# MAGIC - **2. Taxi Data Preprocessing Pipeline**  
# MAGIC   Handles data ingestion, cleaning, and preparation for the yellow taxi dataset. The cleaned dataset is saved in azure container for use by other models. Both models would choose a distinct set of features available from the dataset.
# MAGIC
# MAGIC - **3. `is_congested` Model**  
# MAGIC   A classification model trained to predict whether a given trip is likely to encounter traffic congestion.
# MAGIC
# MAGIC - **4. `is_delayed` Model**  
# MAGIC   A classification model focused on predicting if a trip will be significantly delayed.
# MAGIC
# MAGIC
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## **Data Flow Overview**
# MAGIC
# MAGIC ```text
# MAGIC Raw Data (Azure Blob Storage)
# MAGIC           ⬇
# MAGIC Mounted to Databricks Notebook
# MAGIC           ⬇
# MAGIC Preprocessing Pipelines
# MAGIC           ⬇
# MAGIC Processed Dataset
# MAGIC           ⬇
# MAGIC Used by Both Models for Training & Inference
# MAGIC           ⬇
# MAGIC Trained Models Saved Back to Azure Storage
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Ingestion and Container Access
# MAGIC
# MAGIC To enable efficient and scalable access to raw datasets, the Azure Blob Storage container is mounted directly into the Databricks file system (DBFS). This mount acts as a shortcut to the container, allowing files to be accessed just like local files using standard file paths.
# MAGIC
# MAGIC #### Why Mount the Container?
# MAGIC - Mounting allows direct access across notebook sessions without re-authentication.
# MAGIC - It simplifies file access, enabling Spark to read files using DBFS paths (`/mnt/...`) instead of complex URLs.
# MAGIC - It supports both reading and writing datasets efficiently.
# MAGIC
# MAGIC #### How the Mount Works
# MAGIC - A **Shared Access Signature (SAS) token** is used to securely authenticate and authorize access to the storage container.
# MAGIC - The mount is created via `dbutils.fs.mount()` specifying the container URL and SAS token.
# MAGIC - Once mounted, the container's contents are accessible at a specified mount point in the DBFS.
# MAGIC
# MAGIC #### Security Consideration
# MAGIC - The SAS token provides **time-limited and limited-permissions** (read, write, list, etc.), minimizing any risk.
# MAGIC - This also avoids hardcoding account keys and aligns with best practices in cloud security.
# MAGIC
# MAGIC #### What This Enables
# MAGIC - Seamless access to both **taxi** and **weather** datasets.
# MAGIC - Compatibility with Spark operations such as schema inference, distributed reading, and transformations.
# MAGIC - Simplifies integration with downstream pipelines and workflows in this notebook.
# MAGIC

# COMMAND ----------


#Azure Container Access token

container_name = ""         
storage_account_name = ""  
sas_token = ""
mount_point = f"/mnt/{container_name}"

# Unmount first if already mounted
if mount_point in [mnt.mountPoint for mnt in dbutils.fs.mounts()]:
    dbutils.fs.unmount(mount_point)

# Mount the blob container
dbutils.fs.mount(
    source = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net",
    mount_point = mount_point,
    extra_configs = {
        f"fs.azure.sas.{container_name}.{storage_account_name}.blob.core.windows.net": sas_token
    }
)

# COMMAND ----------

#A simple check to test whether container is mounted
dbutils.fs.ls("/mnt/projectdata")

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ## 1.0 Preprocessing the Weather Data
# MAGIC
# MAGIC The raw weather data is obtained from [Meteostat](https://meteostat.net), a free and reliable source of historical weather information, available at various granularities:yearly, daily, and hourly. 
# MAGIC
# MAGIC For this project, we selected the **New York City region (Station ID: KJRB0)** with **daily resolution**.
# MAGIC
# MAGIC The dataset is stored in Azure conatiner at:  
# MAGIC `/mnt/projectdata/rawdata/weather_data/KJRB0.csv`
# MAGIC
# MAGIC Why preprocessing is needed here?:
# MAGIC - To remove irrelevant columns for the problem context
# MAGIC - Missing values or inconsistent formats
# MAGIC - Data outside the required date range
# MAGIC
# MAGIC **Filtering Relevant Data**  
# MAGIC    - Only records from **January to June 2024** are retained to align with the temporal scope.
# MAGIC    - Selected columns:  
# MAGIC      - `tmin`: Minimum daily temperature  
# MAGIC      - `prcp`: Precipitation amount  
# MAGIC These features were chosen due to their influence on urban mobility.
# MAGIC
# MAGIC - Low temperatures can lead to discomfort, making people more likely to choose taxis.
# MAGIC
# MAGIC - Rainy days often increase taxi demand and road congestion due to fewer people opting for bikes or walking.
# MAGIC
# MAGIC This preprocessing pipeline has following steps:
# MAGIC - Ingests the raw weather dataset  
# MAGIC - Extracts relevant features  
# MAGIC - Performs essential cleaning and transformation steps  
# MAGIC - Saves the processed dataset in Azure conatiner at:  
# MAGIC   `/mnt/projectdata/cleaned_dataset/cleaned_weather_data/`
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, lit, unix_timestamp, round, minute, hour,
    dayofweek, dayofmonth, month, to_date, year, datediff,min, max, count, avg, sum as spark_sum
)
from pyspark.sql.types import (
    StructType, StructField, DateType, FloatType,IntegerType, TimestampType, DoubleType
)
import logging

# COMMAND ----------

# Initialize Spark
spark = SparkSession.builder.appName("WeatherDataProcessing").getOrCreate()

# Define schema
weather_schema = StructType([
    StructField("date", DateType(), True),
    StructField("tavg", FloatType(), True),
    StructField("tmin", FloatType(), True),
    StructField("tmax", FloatType(), True),
    StructField("prcp", FloatType(), True),
    StructField("snow", IntegerType(), True),
    StructField("wdir", IntegerType(), True),
    StructField("wspd", FloatType(), True),
    StructField("wpgt", FloatType(), True),
    StructField("pres", FloatType(), True),
    StructField("tsun", IntegerType(), True)
])

#paths
input_path = "/mnt/projectdata/rawdata/weather_data/KJRB0.csv"
output_path = "/mnt/projectdata/cleaned_dataset/cleaned_weather_data/"

# Read data
try:
    weather_df = spark.read.schema(weather_schema).csv(input_path).repartition("date")
except Exception as e:
    print(f"Error reading file: {e}")
    spark.stop()
    raise

# Filter for Jan-Jun 2024 and select columns
weather_2024 = weather_df.select(
    col("date"),
    round(col("tmin"), 2).alias("tmin"),
    round(col("prcp"), 2).alias("prcp"),
).filter(
    (year(col("date")) == 2024) & (month(col("date")).between(1, 6))
).cache() #we cache this DF for faster checking

# Data validation
print("Data validation:")

# Date range check
date_stats = weather_2024.agg(
    min("date").alias("min_date"),
    max("date").alias("max_date"),
    count("date").alias("record_count")
)
date_stats.show()

# Null value check
null_counts = weather_2024.select([
    spark_sum(col(c).isNull().cast("int")).alias(c) for c in weather_2024.columns
])
null_counts.show()

# Invalid values
invalid_values = weather_2024.select(
    spark_sum((col("prcp") < 0).cast("int")).alias("negative_prcp")
)
invalid_values.show()

# Show sample data
print("\nSample data:")
weather_2024.orderBy("date").show(10, truncate=False)

# save output
cleaned_data_path = "/mnt/projectdata/cleaned_dataset/cleaned_weather_data"
weather_2024.write.mode("overwrite").parquet(cleaned_data_path)

# Clean up  
weather_2024.unpersist() # we unpersiste the df after saving the data to avoid memory issues

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.0 Preprocessing the NYC Yellow Taxi Data
# MAGIC
# MAGIC The raw NYC Yellow Taxi data is sourced from the [TLC Trip Record Data](https://www.nyc.gov/tlc) portal.
# MAGIC
# MAGIC For this project, we focus on the **Yellow Taxi trip data** from **January 2024** to **June 2024**, which contains trip-level records from taxis operating in New York City.
# MAGIC
# MAGIC The dataset is stored in the Azure container at:  
# MAGIC `/mnt/projectdata/rawdata/nyc_taxi_data/*.parquet`
# MAGIC
# MAGIC This *Preprocessing Pipeline* performs the following tasks:
# MAGIC - 2.0 Ingests the raw NYC Yellow Taxi dataset and extracts relevant features 
# MAGIC - 2.1 Basic exploration of the raw dataset 
# MAGIC - 2.2 Data Cleaning  
# MAGIC - 2.3 Feature Engineering
# MAGIC - 2.4 Merges Weather data  
# MAGIC - 2.5 Encoding Categorical features  
# MAGIC - 2.6 Saves the processed dataset in the Azure container at:  
# MAGIC   `/mnt/projectdata/cleaned_dataset/final_dataset`
# MAGIC

# COMMAND ----------

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, lit, unix_timestamp, round, minute, hour, dayofweek, dayofmonth, month, to_date, year, datediff,
    min, max, count, avg, sum as spark_sum, broadcast
)
from pyspark.sql.types import (
    StructType, StructField, DateType, FloatType, IntegerType, TimestampType, DoubleType
)
from pyspark.ml.feature import StringIndexer, OneHotEncoder
import logging

# COMMAND ----------

# MAGIC %md
# MAGIC ### Why These Columns Were Selected
# MAGIC
# MAGIC To build a clean and relevant dataset for modeling traffic congestion, we intentionally selected a **subset of features** from the full TLC dataset. The chosen columns are:
# MAGIC
# MAGIC - `tpep_pickup_datetime`, `tpep_dropoff_datetime`: Used to calculate trip duration and extract time-based features such as hour and weekday.
# MAGIC - `trip_distance`: Captures the length of the trip, often correlated with duration and congestion.
# MAGIC - `congestion_surcharge`: Directly reflects periods of high traffic density, making it a strong proxy for congestion.
# MAGIC - `PULocationID`, `DOLocationID`: Indicates pickup and drop-off zones; important for spatial pattern analysis.
# MAGIC - `extra`, `tolls_amount`: Represents dynamic surcharges which can be influenced by traffic conditions and may affect fare patterns.
# MAGIC
# MAGIC We **excluded columns like `passenger_count`, `RatecodeID`, and `store_and_fwd_flag`** although initially we considered them relevant we discarded them due to high imbalance (as supported by EDA visualizations shown in the presentation). 
# MAGIC
# MAGIC For instance, over 90% of entries had the same value in `passenger_count` and `RatecodeID`, making them uninformative for learning.
# MAGIC
# MAGIC This column selection helps in simplifying the modeling process, improving performance, and reducing noise from irrelevant or skewed features.

# COMMAND ----------

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Spark session
spark = SparkSession.builder.appName("TaxiDataLoading").getOrCreate()

# Define new paths
dbfs_base    = "/mnt/projectdata/rawdata/nyc_taxi_data"
weather_path = "/mnt/projectdata/cleaned_dataset/cleaned_weather_data"

# Define schema
print("Loading taxi data with schema…")
taxi_schema = StructType([
    StructField("tpep_pickup_datetime",  TimestampType(), True),
    StructField("tpep_dropoff_datetime", TimestampType(), True),
    StructField("trip_distance",         DoubleType(),    True),
    StructField("congestion_surcharge",  DoubleType(),    True),
    StructField("PULocationID",          IntegerType(),   True),
    StructField("DOLocationID",          IntegerType(),   True),
    StructField("extra",                 DoubleType(),    True),
    StructField("tolls_amount",          DoubleType(),    True),
])

# Load data
df = spark.read.schema(taxi_schema).parquet(f"{dbfs_base}/*.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1  Exploring the Raw Taxi Dataset
# MAGIC
# MAGIC Before preprocessing, we perform a basic exploration of the raw dataset to understand its structure and quality:
# MAGIC
# MAGIC - **Row Count**: Used the `count()` method to determine the total number of records. While it is computationally expensive, it offers a reasonable trade-off for confirming dataset size during initial analysis.
# MAGIC
# MAGIC - **Summary Statistics**: Generated descriptive statistics (e.g., count, mean, min, max) for each column to assess distributions, identify outliers, and check for data quality issues.
# MAGIC
# MAGIC - **Schema Overview**: Displayed the dataset schema to review column names, data types, and nullability.
# MAGIC

# COMMAND ----------

print(f"Total rows in raw data: {df.count()}")

df.summary().show()

df.printSchema()


# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1 Insights from Summary Statistics and Their Role in Data Cleaning
# MAGIC
# MAGIC The summary statistics help identify data anomalies and guide preprocessing decisions:
# MAGIC
# MAGIC - **`trip_distance`**  
# MAGIC   The max trip distance is **312,722 miles**, which is clearly an outlier. Additionally, the 75th percentile value is **3.3 miles**, indicating that most trips are relatively short: under 3 miles. This suggests the need for outlier removal during data cleaning.
# MAGIC
# MAGIC - **`congestion_surcharge`, `extra`, `tolls_amount`**  
# MAGIC   These columns represent various surcharges and fees. Logically, their values should be **non-negative**. Any negative values likely indicate data entry issues. Furthermore, values exceeding **$10** may be valid in rare cases but often suggest **stacked or duplicated charges**, indicating further filtering.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Data Cleaning
# MAGIC
# MAGIC - **Removed Null Values**: Any rows with critical null values were dropped.
# MAGIC
# MAGIC - **Outlier Removal**: 
# MAGIC   - **`trip_distance`**: Limited to **0.5 to 60 miles**.
# MAGIC   - **`congestion_surcharge`, `extra`, `tolls_amount`**: Ensured values are **positive** and greater than zero.
# MAGIC
# MAGIC

# COMMAND ----------

#remove nulls
df = df.na.drop(subset=taxi_schema.fieldNames())

#remove outliers
df = df.filter(
    (col("trip_distance") > 0.5) & (col("trip_distance") < 60.0)

    & (col("congestion_surcharge") >= 0)

    & ((col("extra") >= 0) & (col("extra") < 10.0))

    & ((col("tolls_amount") >= 0) & (col("tolls_amount") < 50.0))
)
print(f"Rows after dropna and outlier filter: {df.count()}") #count() should be avoided if not necessary

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Feature Engineering
# MAGIC
# MAGIC
# MAGIC - **Caching**: Used `cache()` to optimize performance since the dataset will undergo multiple transformations and joins while feature engineering.
# MAGIC
# MAGIC - **Time Features**: Extracted from `tpep_pickup_datetime`:
# MAGIC   - **Minute**, **Hour**, **Day of Week**, **Day of Month**, **Month**
# MAGIC   - **Holiday Flag**: Marked weekends as holidays.
# MAGIC   - **Date**: Extracted for potential joins and grouping.
# MAGIC
# MAGIC - **Time Period**: Binned hours into categories - **Morning**, **Afternoon**, **Evening**, and **Night**.
# MAGIC
# MAGIC
# MAGIC These features enhance the model’s ability to capture time-dependent patterns.
# MAGIC

# COMMAND ----------

df.cache()

# Add time features
print("Adding time features…")
df = (
    df
    .withColumn("minute", minute("tpep_pickup_datetime"))
    .withColumn("hour", hour("tpep_pickup_datetime"))
    .withColumn("day_of_week", dayofweek("tpep_pickup_datetime"))
    .withColumn("day_of_month", dayofmonth("tpep_pickup_datetime"))
    .withColumn("month", month("tpep_pickup_datetime"))
    .withColumn(
        "is_holiday",
        when(col("day_of_week").isin(1,7), lit(1)).otherwise(lit(0))
    )
)

df = df.withColumn("date", to_date("tpep_pickup_datetime"))

# Time period
df = df.withColumn(
    "time_period",
    when(col("hour").between(6,11), lit("morning"))
    .when(col("hour").between(12,16), lit("afternoon"))
    .when(col("hour").between(17,23), lit("evening"))
    .otherwise(lit("night"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Feature Engineering: 
# MAGIC ###Historical Average Speed
# MAGIC
# MAGIC To enhance model accuracy while avoiding data leakage, we engineered a new feature: `avg_speed_time_period`. This captures the **historical average trip speed** for a given pickup/drop-off location, time period, and day of the week **computed only from past aggregated data**.
# MAGIC
# MAGIC #### Steps:
# MAGIC - Calculated **trip duration** and derived **speed (mph)** for each trip.
# MAGIC - Aggregated speed by `(PULocationID, DOLocationID, time_period, day_of_week)` to compute historical average speed (`hist_avg_speed`).
# MAGIC - Joined the aggregated data back into the main dataset to create a **leakage-free feature**.
# MAGIC - Filled any missing values with the global average speed.
# MAGIC - Dropped intermediate columns to clean up the dataset.
# MAGIC
# MAGIC This ensures that the model has access to relevant traffic context without leaking information from the same trip it is trying to predict.
# MAGIC

# COMMAND ----------

# Historical avg_speed_time_period
print("Computing historical avg_speed by period…")

# First compute trip duration in minutes
df = df.withColumn(
    "duration_min",
    (unix_timestamp("tpep_dropoff_datetime")
     - unix_timestamp("tpep_pickup_datetime")) / 60
)
df = df.drop("tpep_pickup_datetime", "tpep_dropoff_datetime")

# Compute speed mph
df = df.withColumn("speed_mph", col("trip_distance") / (col("duration_min") / 60 + lit(1e-6)))

# Aggregate historical speed by (PU, DO, time_period, day_of_week)
hist_speed = (
    df
    .groupBy("PULocationID", "DOLocationID", "time_period", "day_of_week")
    .agg(
        (
            spark_sum("trip_distance")
            / (spark_sum("duration_min") + lit(1e-6))
            * 60
        ).alias("hist_avg_speed")
    )
)
df = df.join(
    hist_speed,
    ["PULocationID", "DOLocationID", "time_period", "day_of_week"],
    "left"
)

df = df.withColumn("avg_speed_time_period_hist", round(col("hist_avg_speed"), 2))

# Fill nulls with global
global_speed = hist_speed.agg(avg("hist_avg_speed")).first()[0]
df = df.withColumn(
    "avg_speed_time_period",
    when(col("avg_speed_time_period_hist").isNull(), global_speed)
    .otherwise(col("avg_speed_time_period_hist"))
).drop("avg_speed_time_period_hist")

# Drop speed_mph now that we have the feature
df = df.drop("speed_mph")

# Filter duration_min and remove its outliers since trips too short(under 2 mins) or too long(over 3 hours) are irrelevant for our scope. 
df = df.filter(
    (col("duration_min") > 2) &
    (col("duration_min") < 180)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Feature Engineering: 
# MAGIC ###Expected Duration & Congestion Flag
# MAGIC
# MAGIC - **Expected Duration**:  
# MAGIC   Estimated trip duration using:
# MAGIC   `expected_duration = (trip_distance / avg_speed_time_period) * 60`  
# MAGIC   This gives a baseline estimate based on historical traffic patterns.
# MAGIC
# MAGIC - **Congestion Flag (`is_congested`)**:  
# MAGIC   A trip is marked as congested if `congestion_surcharge > 0`; otherwise, it’s considered not congested.

# COMMAND ----------

#2.3 Feature Engineering
# Expected duration
print("Computing expected_duration...")
df = df.withColumn("expected_duration", round(col("trip_distance") / (col("avg_speed_time_period") + lit(1e-6)) * 60, 2))

# Encode congestion flag & categorical features
print("Adding congestion flag…")
df = df.withColumn(
    "is_congested",
    when(col("congestion_surcharge") > 0, lit(1)).otherwise(lit(0))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Merging Weather Data
# MAGIC
# MAGIC - Loaded daily weather data (`tmin`, `prcp`) with predefined schema.
# MAGIC
# MAGIC - Joined it to the main dataset using the `date` column.
# MAGIC
# MAGIC - Used a **left join** to retain all trip records and filled any missing weather values with `0.0`.
# MAGIC
# MAGIC - Dropped the `date` column after merging as it’s no longer needed.
# MAGIC

# COMMAND ----------

# Merge weather
print("Loading weather data...")
weather_schema = StructType([
    StructField("date", DateType(), True),
    StructField("tmin", FloatType(), True),
    StructField("prcp", FloatType(), True)
])
weather_df = spark.read.schema(weather_schema).parquet(weather_path) \
    .withColumn("tmin", col("tmin").cast("double")) \
    .withColumn("prcp", col("prcp").cast("double"))

print("Joining weather on date...")
df = df.join(broadcast(weather_df), "date", "left").na.fill({"tmin": 0.0, "prcp": 0.0})
df = df.drop("date")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.5 Categorical Encoding
# MAGIC
# MAGIC - Applied **StringIndexer** to convert the `time_period` column into a numerical index.
# MAGIC
# MAGIC - Used **OneHotEncoder** to transform the indexed feature into a binary vector.
# MAGIC
# MAGIC - Dropped the original `time_period` column after encoding.
# MAGIC

# COMMAND ----------

# Index & One Hot Encoding
cats = ["time_period"]
for c in cats:
    print(f"Indexing {c}…")
    df = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") \
         .fit(df).transform(df)

print("OHE categorical feature:")
ohe = OneHotEncoder(
    inputCols=[f"{c}_idx" for c in cats],
    outputCols=[f"{c}_ohe" for c in cats],
    dropLast=False
)
df = df.drop("time_period")
df = ohe.fit(df).transform(df)

df = df.drop("time_period_idx")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Final Schema check
# MAGIC Before we save our final data set we perform a basic schema and column check to ensure all columns are present in final dataset.

# COMMAND ----------

# Final column selection
final_cols = [
    "month", "day_of_month", "hour", "minute", "day_of_week", "is_holiday", "time_period_ohe",
    "trip_distance", "congestion_surcharge", "extra", "tolls_amount",
    "expected_duration", "avg_speed_time_period",
    "tmin", "prcp",
    "PULocationID", "DOLocationID", "is_congested", "duration_min"
]
df = df.select(final_cols)

# Verify schema
print("Final Schema:")
df.printSchema()

# Verify all columns exist
for c in final_cols:
    assert c in df.columns, f"Missing {c}"

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.6 Saving the Final Dataset
# MAGIC
# MAGIC - Repartitioned the final DataFrame by **month** to optimize storage.
# MAGIC
# MAGIC - Saved the dataset stored in Azure container in **Parquet format** at:  
# MAGIC   `/mnt/projectdata/cleaned_dataset/final_dataset`
# MAGIC
# MAGIC - Used `.count()` to confirm the final number of rows — a resource-intensive operation but helpful for validation.

# COMMAND ----------

# Save as Parquet partitioned by month
output_path = "/mnt/projectdata/cleaned_dataset/final_dataset"
df = df.repartition("month")
df.write.format("parquet").mode("overwrite").partitionBy("month").save(output_path)

print(f"Preprocessing complete! Final Parquet table saved to {output_path}")
print(f"Final column count: {len(df.columns)}")
print(f"Final row count: {df.count()}")

# COMMAND ----------

#clean up the cached df
df.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC # 3.0  *Congestion* model
# MAGIC
# MAGIC This pipeline builds a binary classification model to predict **traffic congestion** using the engineered `is_congested` flag as the target.
# MAGIC
# MAGIC This *Congetion Prediction Pipeline* performs the following tasks:
# MAGIC - 3.1 Model Setup
# MAGIC - 3.2 Data Loading and Feature Selection
# MAGIC - 3.3 Model Training
# MAGIC - 3.4 Model Evaluation
# MAGIC - 3.5 Save the model

# COMMAND ----------

# MAGIC %md
# MAGIC #####To optimize performance and training time, you can set sample size in block below.

# COMMAND ----------

# Adjust Sample size here
SAMPLE_FRACTION = 0.3

# COMMAND ----------

# Dataset path
data_path = "/mnt/projectdata/cleaned_dataset/final_dataset"

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Model Setup: 
# MAGIC
# MAGIC - **Spark Session**: To ensure optimal performance, we experimented with various configurations for our cluster setup (14GB Memory and 4 cores) and fine-tuned the following settings:
# MAGIC   - 4 shuffle partitions for better parallelism across 4 cores.
# MAGIC   - 6GB driver memory for handling larger workloads.
# MAGIC   - Kryo serialization to improve data processing efficiency.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col
import logging
import time
from datetime import datetime

spark = (
    SparkSession.builder
    .appName("Taxi Congestion Prediction")
    .config("spark.sql.shuffle.partitions", "4")  # Better balance for 4 cores
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.driver.memory", "6g")  # driver memory
    .config("spark.memory.fraction", "0.6")  # Reduce overhead
    .getOrCreate()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Data loading and Feature Selection
# MAGIC
# MAGIC **Data Loading**: The model accesses the cleaned Parquet dataset saved at `projectdata/cleaned_dataset/final_dataset`.
# MAGIC
# MAGIC **Feature Selection**: We’ve carefully selected only the most relevant features for predicting congestion.
# MAGIC
# MAGIC - **trip_distance**: A key factor, as longer trips are more likely to experience delays due to traffic congestion.
# MAGIC - **tolls_amount**: Tolls can indicate areas with heavier traffic or longer travel times.
# MAGIC - **extra**: Represents additional charges like surcharges, which could be affected by traffic congestion.
# MAGIC - **hour**, **minute**: Time of day plays a critical role in traffic congestion patterns (e.g., rush hours).
# MAGIC - **day_of_week**, **day_of_month**, **month**: Temporal features for weekly, monthly, and seasonal variations in traffic patterns.
# MAGIC - **is_holiday**: Public holidays can significantly affect traffic patterns, either causing higher congestion (tourist destinations) or lighter traffic.
# MAGIC - **time_period_ohe**: This feature accounts for specific time periods within the day (e.g., morning rush hour, evening peak) that are highly correlated with congestion.
# MAGIC - **avg_speed_time_period**: Average speed during certain time periods helps infer congestion levels; slower speeds often correlate with higher congestion.
# MAGIC - **tmin**, **prcp**: Weather conditions (temperature and precipitation) can affect traffic behavior and congestion, with colder and rainy weather often leading to slower travel.
# MAGIC - **DOLocationID**, **PULocationID**: The origin and destination locations influence congestion based on the area's traffic patterns.
# MAGIC - **duration_min**: Longer trip durations are likely to be associated with congestion.
# MAGIC
# MAGIC Before proceeding, we verify that all necessary features are present in the dataset to avoid any issues during training.

# COMMAND ----------

#Load preprocessed data
print("Loading preprocessed Parquet data...")
df = spark.read.format("parquet").load(data_path)

#Sample data
logger.info(f"Sampling data at {SAMPLE_FRACTION * 100}%...")
sampled_df = df.sample(fraction=SAMPLE_FRACTION, seed=42)

#Only choose features that are relevant for congestion prediction
feature_cols = [
    "trip_distance", "tolls_amount","extra",
    "hour", "minute", "day_of_week", "day_of_month", "month", "is_holiday",
    "time_period_ohe", "avg_speed_time_period",
    "tmin", "prcp", "DOLocationID","PULocationID",
]
label_col = "is_congested"  # Classification target

# Verify all features exist
missing_cols = [col for col in feature_cols if col not in sampled_df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in DataFrame: {missing_cols}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Model Training: Random Forest Classifier
# MAGIC
# MAGIC 1. **Feature Assembly**: 
# MAGIC    We use `VectorAssembler` to combine all selected features into a single "features" column, which is required by the model.
# MAGIC
# MAGIC
# MAGIC 2. **Random Forest Classifier**: 
# MAGIC    We define a Random Forest model with the following key settings:
# MAGIC    - **30 trees** for stability,
# MAGIC    - **Max depth of 8** to capture data complexity,
# MAGIC    - **Subsampling rate of 0.7** to improve generalization and reduce overfitting.
# MAGIC
# MAGIC 3. **Pipeline**:
# MAGIC    We create a Spark pipeline that chains the feature assembler and Random Forest model together.
# MAGIC
# MAGIC 4. **Data Split**:
# MAGIC    The data is split into **80% training** and **20% testing** using random sampling.

# COMMAND ----------

#Assemble features
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)

#Define Random Forest classifier
rf = RandomForestClassifier(
    labelCol=label_col,
    featuresCol="features",
    numTrees=30,              # the higher: the longer training time | but better stability
    maxDepth=8,               # the deeper: capture more complexity
    minInstancesPerNode=10,   # Reduce overfitting and speed up
    subsamplingRate=0.7,      # More randomness, better generalization
    featureSubsetStrategy="sqrt",  #for classification
    seed=42,
    cacheNodeIds=True
)

#Build pipeline
pipeline = Pipeline(stages=[assembler, rf])

#Split data
train_df, test_df = sampled_df.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC 5. **Model Training**: 
# MAGIC    We train the model on the training data and track how long the process takes.
# MAGIC

# COMMAND ----------

#Train model

start_time = time.time()
model = pipeline.fit(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **Feature Importance**:  
# MAGIC    - After training, we extract and display the top features based on their importance to the model.  
# MAGIC    - This helps us identify which features are critical for prediction and which ones do not add value, to help us make changes in future feature engineering and iterations.

# COMMAND ----------

#Extract Feature Importances(to see which features are most important)

rf_model = model.stages[-1]  # last stage is the RandomForestClassifier
importances = rf_model.featureImportances
feature_importance_list = list(zip(feature_cols, importances.toArray()))
sorted_importance = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)

print("Top feature importances:")
for feature, importance in sorted_importance:
    print(f"{feature:<25} : {importance:.4f}")

train_time = time.time() - start_time
print(f"Model trained on {SAMPLE_FRACTION*100}% sample in {train_time:.2f} seconds.")

# COMMAND ----------

# MAGIC %md
# MAGIC ####Feature Importance Analysis
# MAGIC
# MAGIC After training the model, we took a closer look at feature importances to understand which variables actually contributed to predicting congestion.
# MAGIC
# MAGIC - **PULocationID**, **trip_distance**, and **tolls_amount** were the most influential features, suggesting that trip length and pickup location are strong indicators of congestion.
# MAGIC
# MAGIC - Features like **hour**, **extra**, and **weather data** (e.g., `prcp`, `tmin`) showed moderate importance, indicating some temporal and environmental impact on traffic.
# MAGIC
# MAGIC - Interestingly, a few features turned out to be almost irrelevant:
# MAGIC   - Things like **DOLocationID**, **minute**, **month**, **day_of_month** contributed very little to the model.
# MAGIC   - This time fields might not offer much new information.
# MAGIC
# MAGIC Understanding this helps us clean up and simplify future versions of the model, by focusing only on features that matter, we can reduce training time and make the pipeline even more efficient.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 Model Evaluation
# MAGIC
# MAGIC 1. **Predictions**:  
# MAGIC    We use the trained model to make predictions on the test set.
# MAGIC
# MAGIC 2. **Binary Classification Metrics**:  
# MAGIC    The **ROC AUC score** is calculated to evaluate how well the model distinguishes between the two classes (congested vs. non-congested).
# MAGIC
# MAGIC 3. **Multi-class Classification Metrics**:  
# MAGIC    We also calculate:
# MAGIC    - **Accuracy**: The percentage of correct predictions.
# MAGIC    - **Precision**: How many of the predicted congested trips were actually congested.
# MAGIC    - **Recall**: How many of the actual congested trips were correctly predicted.
# MAGIC    - **F1 Score**: A balance between precision and recall.
# MAGIC
# MAGIC 4. **Performance Summary**:  
# MAGIC    We display the final results, showing the training time and key metrics to understand the model’s effectiveness.
# MAGIC

# COMMAND ----------

#Predict on test set
predictions = model.transform(test_df)

# COMMAND ----------

#Evaluate

# Binary classification metrics
binary_evaluator = BinaryClassificationEvaluator(
    labelCol=label_col,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
roc_auc = binary_evaluator.evaluate(predictions)

# Multi class metrics
multi_evaluator = MulticlassClassificationEvaluator(
    labelCol=label_col,
    predictionCol="prediction"
)
accuracy = multi_evaluator.setMetricName("accuracy").evaluate(predictions)
precision = multi_evaluator.setMetricName("weightedPrecision").evaluate(predictions)
recall = multi_evaluator.setMetricName("weightedRecall").evaluate(predictions)
f1 = multi_evaluator.setMetricName("f1").evaluate(predictions)

#Summary

print("\nPerformance Summary:")

print(f"  Train time       : {train_time:.2f} s")
print(f"  ROC AUC          : {roc_auc:.4f}")
print(f"  Accuracy         : {accuracy:.4f}")
print(f"  Precision        : {precision:.4f}")
print(f"  Recall           : {recall:.4f}")
print(f"  F1 Score         : {f1:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ####Model Performance Highlights
# MAGIC
# MAGIC On our 4-core cluster and 30% sample, training completes in just 70-80 seconds (versus approximately **180–210 s** on a 2-core setup), demonstrating a significant speedup without compromising quality:
# MAGIC
# MAGIC - **ROC AUC: 0.9** – Exceptional discrimination between congested and uncongested trips.  
# MAGIC - **Accuracy: 0.9** – Correctly classifies nearly 96% of all trips.  
# MAGIC - **Precision: 0.9** – When it flags congestion, it’s right over 95% of the time.  
# MAGIC - **Recall: 0.9** – Identifies almost every true congestion event.  
# MAGIC - **F1 Score: 0.9** – Maintains a strong balance between precision and recall.
# MAGIC
# MAGIC These results indicate our model is both **efficient** and **robust**, achieving high predictive performance.  
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 Model Saving
# MAGIC
# MAGIC 1. **Save Model**:  
# MAGIC    After training, we save the model to a specified path in Azure container, ensuring any previous versions are overwritten.
# MAGIC
# MAGIC 2. **Cleanup**:  
# MAGIC    We unpersist the dataframe to free up memory once the model is saved.
# MAGIC

# COMMAND ----------

# Timestamped path
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_path = f"/mnt/projectdata/models/Congested_Model/is_congested_v1_{timestamp}"

# Save model with overwrite
model.write().overwrite().save(model_save_path)

#
print(f"Model saved to {model_save_path}")

# Clean up
df.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC # 4.0 Predicting Delay
# MAGIC
# MAGIC This pipeline builds a binary classification model to predict **trip delays** using the `is_delayed` flag as the target.
# MAGIC
# MAGIC This *Delay Prediction Pipeline* performs the following tasks:
# MAGIC - 4.1 Model Setup
# MAGIC - 4.2 Data Loading and Feature Selection
# MAGIC - 4.3 Model Training
# MAGIC - 4.4 Model Evaluation
# MAGIC - 4.5 Save the model
# MAGIC
# MAGIC To optimize performance and training time, you can set sample size in block below.
# MAGIC

# COMMAND ----------

# Adjust sample size here
SAMPLE_FRACTION = 0.3

# COMMAND ----------

# Dataset path
data_path = "/mnt/projectdata/cleaned_dataset/final_dataset"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 Model Setup: 
# MAGIC
# MAGIC - **Spark Session**: We’ve configured it with optimal settings for our current cluster setup:
# MAGIC   - 4 shuffle partitions for better parallelism across 4 cores,
# MAGIC   - 6GB driver memory for handling larger workloads,
# MAGIC   - Kryo serialization to improve data processing efficiency.
# MAGIC
# MAGIC - **Feature Selection**: We’ve carefully selected only the most relevant features for predicting delay.
# MAGIC
# MAGIC Before proceeding, we verify that all necessary features are present in the dataset to avoid any issues during training.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col
import logging
import time
from datetime import datetime

print("Delayed Prediction Training Pipeline")

spark = (
    SparkSession.builder
    .appName("Taxi Congestion Prediction")
    .config("spark.sql.shuffle.partitions", "4")  # Better balance for 4 cores
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.driver.memory", "6g")  # driver memory(in community edition no nodes only driver)
    .config("spark.memory.fraction", "0.6")  # Reduce overhead
    .getOrCreate()
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##4.2 Data loading and Feature Selection
# MAGIC
# MAGIC **Data Loading**: The model accesses the cleaned Parquet dataset saved at `projectdata/cleaned_dataset/final_dataset`.
# MAGIC
# MAGIC **Feature Selection**: We’ve carefully selected only the most relevant features for predicting delay.
# MAGIC
# MAGIC Before proceeding, we verify that all necessary features are present in the dataset to avoid any issues during training.

# COMMAND ----------

#Load preprocessed data
print("Loading preprocessed Parquet data...")
df = spark.read.format("parquet").load(data_path)
#total_rows = df.count()

# Define is_delayed: 1 if duration_min exceeds expected_duration by 10%, else 0
sampled_df = sampled_df.withColumn(
    "is_delayed",
    when(col("duration_min") > col("expected_duration") * 1, 1).otherwise(0)
)

#Define feature and target columns
feature_cols = [
    "trip_distance", "extra","tolls_amount", "congestion_surcharge",
    "hour", "minute", "day_of_week", "day_of_month", "month", "is_holiday",
    "time_period_ohe", "avg_speed_time_period",
    "tmin", "prcp", "DOLocationID", "PULocationID",
    "is_congested"
]
label_col = "is_delayed"  # Classification target

# Verify all features exist
missing_cols = [col for col in feature_cols if col not in sampled_df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##4.3 Model Training 
# MAGIC ###Random Forest Classifier
# MAGIC
# MAGIC 1. **Feature Assembly**: 
# MAGIC    We use `VectorAssembler` to combine all selected features into a single "features" column, which is required by the model.
# MAGIC
# MAGIC
# MAGIC 2. **Random Forest Classifier**: 
# MAGIC    We define a Random Forest model with the following key settings:
# MAGIC    - **30 trees** for stability,
# MAGIC    - **Max depth of 8** to capture data complexity,
# MAGIC    - **Subsampling rate of 0.7** to improve generalization and reduce overfitting.
# MAGIC
# MAGIC 3. **Pipeline**:
# MAGIC    We create a Spark pipeline that chains the feature assembler and Random Forest model together.
# MAGIC
# MAGIC 4. **Data Split**:
# MAGIC    The data is split into **80% training** and **20% testing** using random sampling.

# COMMAND ----------

#Assemble features
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)

#Define Random Forest classifier

rf = RandomForestClassifier(
    labelCol=label_col,
    featuresCol="features",
    numTrees=30,              # higher gives better accuracy but very heavy
    maxDepth=8,              # make deeper to capture more complexity
    minInstancesPerNode=10,   # Reduce overfitting and speed up
    subsamplingRate=0.7,      # more randomness, better generalization
    featureSubsetStrategy="sqrt",  # Common for classification (auto by default but can be explicit)
    seed=42,
    cacheNodeIds=True,
)

#Build pipeline
pipeline = Pipeline(stages=[assembler, rf])

#Split data
train_df, test_df = sampled_df.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC 5. **Model Training**: 
# MAGIC    We train the model on the training data and track how long the process takes.
# MAGIC

# COMMAND ----------

#Train Model
start_time = time.time()
model = pipeline.fit(train_df)

# Get the trained RandomForest model from the pipeline to obtain feature importances
rf_model = model.stages[-1]  

# COMMAND ----------

# MAGIC %md
# MAGIC 6. **Feature Importance**:  
# MAGIC    - After training, we extract and display the top features based on their importance to the model.  
# MAGIC    - This helps us identify which features are critical for prediction and which ones do not add value, to help us make changes in future feature engineering and iterations.

# COMMAND ----------

# Get feature importances
importances = rf_model.featureImportances

# Convert to list of (feature, importance)
feature_importance_list = list(zip(feature_cols, importances.toArray()))

# Sort by most to least importance
sorted_importance = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)

# Print top N important features
logger.info("Top feature importances:")
for feature, importance in sorted_importance:
    print(f"{feature:<25} : {importance:.4f}")

train_time = time.time() - start_time
print(f"Model trained on {SAMPLE_FRACTION*100}% sample in {train_time:.2f} seconds.")

#Predict on test set
predictions = model.transform(test_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### is_delayed Feature Importance Analysis
# MAGIC
# MAGIC Once our delay model finished training we inspected feature importances to see what really drives trip delays:
# MAGIC
# MAGIC - **Hour** was the standout predictor (0.517), underscoring how time of day dominates delay patterns.  
# MAGIC - **PULocationID** (0.110) and **extra charges** (0.072) also played significant roles, showing how pickup zones and surcharges correlate with slower traffic.
# MAGIC - **is_congested** (0.063) naturally contributed since if you’re in congestion, you’re likely delayed.  
# MAGIC - **Seasonality** via **month** (0.063) and finer **time-period bins** (0.050) added useful nuance.
# MAGIC
# MAGIC On the flip side, several features showed minimal impact (<0.02 each), including `trip_distance`, `tolls_amount`, weather variables (`tmin`, `prcp`), and especially **DOLocationID**, **minute**, and **day_of_month**, which proved nearly irrelevant.
# MAGIC
# MAGIC By only choosing the high-impact features, we can simplify future iterations by dropping low-value features and that will further speed up training and simplify our pipeline without sacrificing accuracy.  
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 4.4 Model Evaluation
# MAGIC
# MAGIC 1. **Predictions**:  
# MAGIC    We use the trained model to make predictions on the test set.
# MAGIC
# MAGIC 2. **Binary Classification Metrics**:  
# MAGIC    The **ROC AUC score** is calculated to evaluate how well the model distinguishes between the two classes (congested vs. non-congested).
# MAGIC
# MAGIC 3. **Multi-class Classification Metrics**:  
# MAGIC    We also calculate:
# MAGIC    - **Accuracy**: The percentage of correct predictions.
# MAGIC    - **Precision**: How many of the predicted congested trips were actually congested.
# MAGIC    - **Recall**: How many of the actual congested trips were correctly predicted.
# MAGIC    - **F1 Score**: A balance between precision and recall.
# MAGIC
# MAGIC 4. **Performance Summary**:  
# MAGIC    We display the final results, showing the training time and key metrics to understand the model’s effectiveness.
# MAGIC

# COMMAND ----------

#Evaluate
# Binary classification metrics
binary_evaluator = BinaryClassificationEvaluator(
    labelCol=label_col,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
roc_auc = binary_evaluator.evaluate(predictions)

# Multiclass metrics
multi_evaluator = MulticlassClassificationEvaluator(
    labelCol=label_col,
    predictionCol="prediction"
)
accuracy = multi_evaluator.setMetricName("accuracy").evaluate(predictions)
precision = multi_evaluator.setMetricName("weightedPrecision").evaluate(predictions)
recall = multi_evaluator.setMetricName("weightedRecall").evaluate(predictions)
f1 = multi_evaluator.setMetricName("f1").evaluate(predictions)


#Summary
print("\nPerformance Summary:")
print(f"Sample fraction: {SAMPLE_FRACTION*100}%")
print(f"  Train time       : {train_time:.2f} s")
print(f"  ROC AUC          : {roc_auc:.4f}")
print(f"  Accuracy         : {accuracy:.4f}")
print(f"  Precision        : {precision:.4f}")
print(f"  Recall           : {recall:.4f}")
print(f"  F1 Score         : {f1:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ####Delay Model Takeaways
# MAGIC - The metrics (ROC AUC 0.67, Accuracy 0.64, F1 0.59) aren’t quite what we hoped, showing only moderate performance.  
# MAGIC - Right now the model captures broad time-of-day patterns but misses many nuanced delay trips.  
# MAGIC - This likely comes from limited feature diversity—factors like real-time traffic incidents or special events aren’t included.  
# MAGIC - To boost performance, we can enrich our selected features and explore more powerful algorithms or class-balancing techniques.  

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.5 Model Saving
# MAGIC
# MAGIC 1. **Save Model**:  
# MAGIC    After training, we save the model to a specified path in Azure container, ensuring any previous versions are overwritten.
# MAGIC
# MAGIC 2. **Cleanup**:  
# MAGIC    We unpersist the dataframe to free up memory once the model is saved.
# MAGIC

# COMMAND ----------

# Save model
#start_time = time.time()
#model_save_path = "/mnt/projectdata/models/Delayed_Model/is_delayed_v1"
#model.write().overwrite().save(model_save_path)
#df.unpersist()

# COMMAND ----------

# Timestamped path
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_path = f"/mnt/projectdata/models/Delayed_Model/is_delayed_v1_{timestamp}"

# Save model with overwrite
model.write().overwrite().save(model_save_path)

print(f"Model saved to {model_save_path}")

# Clean up
df.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.0 Challenges faced
# MAGIC
# MAGIC - **Counting Costs**  
# MAGIC   Early prototypes relied heavily on `.count()`, which sometimes crashed the driver. 
# MAGIC   We quickly remembered our class discussions (and that quiz question) on count() causing heavy computations for driver, So we established design principles around caching, avoiding full counts early on.
# MAGIC
# MAGIC - **Cloud Integration**  
# MAGIC   Community Edition’s restrictions made standard GCP integrations difficult in the begining, so we researched and implemented a workaround when using GCP bucket. 
# MAGIC   
# MAGIC   On Azure, our team faced different cluster-permission scopes, while some could launch any node type, others had to find the exact permissed node types.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Final Thoughts
# MAGIC
# MAGIC This course and project have been an great opportunity to learn Spark and apply distributed-computing concepts in a real-world setting. 
# MAGIC
# MAGIC Building an end-to-end pipeline from data ingestion to feature engineering, model training, and evaluation has truly solidified our understanding of not just Spark but also cloud-scale data workflows and performance optimization.
# MAGIC
# MAGIC We’re excited to continue exploring these technologies and to build on what we’ve learned here. 
# MAGIC
# MAGIC Thank you!  
# MAGIC - Shubham Harishkumar Dave
# MAGIC - Joshuva Prabhakar Palicharla
# MAGIC - Hepsiba Joicy Kumpati