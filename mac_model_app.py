from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
import os
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import shutil
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Household Power Consumption Machine Learning") \
    .getOrCreate()

spark.sparkContext.setLogLevel("INFO")

# Function to train models and evaluate performance
def train_model(**kwargs):
    file_path = '/home/john-thomas-a/projects/airflow/household_power_consumption.txt'
    if not os.path.isfile(file_path):
        logging.error(f"File not found: {file_path}")
        return

    # Initialize Spark DataFrame
    dataset = spark.read.csv(file_path, header=True, sep=';', inferSchema=True)
    
    # Data preprocessing: selecting necessary columns and casting to DoubleType
    dataset = dataset.select(
        col("Global_active_power").cast(DoubleType()),
        col("Global_reactive_power").cast(DoubleType()),
        col("Global_intensity").cast(DoubleType()),
        col("Voltage").cast(DoubleType())
    ).na.drop()  # Dropping rows with missing values

    # Combine features into a single vector
    assembler = VectorAssembler(inputCols=["Global_active_power", "Global_reactive_power", "Global_intensity"], outputCol="features")
    data = assembler.transform(dataset).select("features", "Voltage")

    # Split data into training and testing sets
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
    
    # Initialize a dictionary to store the models and their results
    model_results = {}
    
    # Define paths for saving models
    lr_model_path = "/home/john-thomas-a/projects/airflow/model/lr_model"
    rf_model_path = "/home/john-thomas-a/projects/airflow/model/rf_model"
    gbt_model_path = "/home/john-thomas-a/projects/airflow/model/gbt_model"
    results_file = "/home/john-thomas-a/projects/airflow/model/model_results.txt"

    # Helper function to remove directory if it exists
    def remove_dir_if_exists(path):
        if os.path.exists(path):
            shutil.rmtree(path)

    # Remove directories if they exist before saving the models
    remove_dir_if_exists(lr_model_path)
    remove_dir_if_exists(rf_model_path)
    remove_dir_if_exists(gbt_model_path)

    # Function to evaluate model performance
    def evaluate_model(predictions):
        evaluator_rmse = RegressionEvaluator(labelCol='Voltage', predictionCol='prediction', metricName='rmse')
        evaluator_mae = RegressionEvaluator(labelCol='Voltage', predictionCol='prediction', metricName='mae')
        evaluator_r2 = RegressionEvaluator(labelCol='Voltage', predictionCol='prediction', metricName='r2')

        rmse = evaluator_rmse.evaluate(predictions)
        mae = evaluator_mae.evaluate(predictions)
        r2 = evaluator_r2.evaluate(predictions)

        return rmse, mae, r2

    # 1. Linear Regression Model
    lr = LinearRegression(featuresCol='features', labelCol='Voltage')
    lr_model = lr.fit(train_data)
    lr_predictions = lr_model.transform(test_data)

    # Evaluate the Linear Regression model
    lr_rmse, lr_mae, lr_r2 = evaluate_model(lr_predictions)
    model_results['LinearRegression'] = (lr_rmse, lr_mae, lr_r2)

    # Save the Linear Regression model using Spark's save method
    lr_model.save(lr_model_path)

    # 2. Random Forest Regression Model
    rf = RandomForestRegressor(featuresCol='features', labelCol='Voltage')
    rf_model = rf.fit(train_data)
    rf_predictions = rf_model.transform(test_data)

    # Evaluate the Random Forest model
    rf_rmse, rf_mae, rf_r2 = evaluate_model(rf_predictions)
    model_results['RandomForestRegressor'] = (rf_rmse, rf_mae, rf_r2)

    # Save the Random Forest model using Spark's save method
    rf_model.save(rf_model_path)

    # 3. Gradient-Boosted Tree Regression Model (GBTRegressor)
    gbt = GBTRegressor(featuresCol="features", labelCol="Voltage")
    gbt_model = gbt.fit(train_data)
    gbt_predictions = gbt_model.transform(test_data)

    # Evaluate the GBT model
    gbt_rmse, gbt_mae, gbt_r2 = evaluate_model(gbt_predictions)
    model_results['GBTRegressor'] = (gbt_rmse, gbt_mae, gbt_r2)

    # Save the GBT model using Spark's save method
    gbt_model.save(gbt_model_path)

    # Log and print the evaluation results
    for model_name, metrics in model_results.items():
        rmse, mae, r2 = metrics
        logging.info(f"{model_name} - RMSE: {rmse}, MAE: {mae}, R2: {r2}")
        print(f"{model_name} - RMSE: {rmse}, MAE: {mae}, R2: {r2}")

    # Save results to a text file (open in write mode to overwrite if file exists)
    with open(results_file, "w") as f:
        for model_name, metrics in model_results.items():
            rmse, mae, r2 = metrics
            f.write(f"{model_name} - RMSE: {rmse}, MAE: {mae}, R2: {r2}\n")

# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 17),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'household_power_model_training',
    default_args=default_args,
    description='A DAG to perform analytics on household power consumption data',
    schedule='0 */2 * * *',  # Run every 2 hours
    catchup=False,  # Ensure that only the latest schedule is run
)

perform_analytics_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)
