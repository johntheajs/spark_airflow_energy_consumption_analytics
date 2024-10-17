from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Household Power Consumption Prediction") \
    .getOrCreate()

spark.sparkContext.setLogLevel("INFO")

# Function to load models, predict voltage, and save results
def predict_voltage(**kwargs):
    # Paths for loading models and saving predictions
    lr_model_path = "/home/john-thomas-a/projects/airflow/model/lr_model"
    rf_model_path = "/home/john-thomas-a/projects/airflow/model/rf_model"
    gbt_model_path = "/home/john-thomas-a/projects/airflow/model/gbt_model"
    predictions_dir = "/home/john-thomas-a/projects/airflow/predictions"
    predictions_file = os.path.join(predictions_dir, "voltage_predictions.csv")
    
    # Create predictions directory if it doesn't exist
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    # Load the dataset
    file_path = '/home/john-thomas-a/projects/airflow/household_power_consumption.txt'
    dataset = spark.read.csv(file_path, header=True, sep=';', inferSchema=True)

    # Select necessary columns and drop rows with missing values
    dataset = dataset.select(
        col("Global_active_power").cast("double"),
        col("Global_reactive_power").cast("double"),
        col("Global_intensity").cast("double"),
        col("Voltage").cast("double")
    ).na.drop()

    # Sample 500 rows
    sampled_data = dataset.sample(False, 500 / dataset.count(), seed=42)

    # Prepare features using VectorAssembler
    from pyspark.ml.feature import VectorAssembler
    assembler = VectorAssembler(
        inputCols=["Global_active_power", "Global_reactive_power", "Global_intensity"],
        outputCol="features"
    )
    assembled_data = assembler.transform(sampled_data)

    # Load models as individual regression models
    from pyspark.ml.regression import LinearRegressionModel, RandomForestRegressionModel, GBTRegressionModel
    lr_model = LinearRegressionModel.load(lr_model_path)
    rf_model = RandomForestRegressionModel.load(rf_model_path)
    gbt_model = GBTRegressionModel.load(gbt_model_path)

    # Make predictions using each model
    lr_predictions = lr_model.transform(assembled_data)
    rf_predictions = rf_model.transform(assembled_data)
    gbt_predictions = gbt_model.transform(assembled_data)

    # Select the actual and predicted voltage for each model
    lr_results = lr_predictions.select(col("Voltage").alias("actual_voltage"), col("prediction").alias("predicted_voltage_lr"))
    rf_results = rf_predictions.select(col("Voltage").alias("actual_voltage"), col("prediction").alias("predicted_voltage_rf"))
    gbt_results = gbt_predictions.select(col("Voltage").alias("actual_voltage"), col("prediction").alias("predicted_voltage_gbt"))

    # Join results into a single DataFrame
    final_results = lr_results.join(rf_results, on="actual_voltage").join(gbt_results, on="actual_voltage")

    # Convert to Pandas DataFrame for saving
    final_results_pd = final_results.toPandas()

    # Save predictions to CSV
    final_results_pd.to_csv(predictions_file, index=False)
    logging.info(f"Predictions saved to {predictions_file}")


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
    'household_power_voltage_prediction',
    default_args=default_args,
    description='A DAG to predict voltage from household power consumption data',
    schedule='0 */2 * * *',  # Run every 2 hours
    catchup=False,  # Ensure that only the latest schedule is run
)

predict_voltage_task = PythonOperator(
    task_id='predict_voltage',
    python_callable=predict_voltage,
    dag=dag,
)

