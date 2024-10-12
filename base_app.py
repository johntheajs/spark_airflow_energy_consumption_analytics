from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean
from pyspark.sql.types import StringType, DoubleType
import os
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
import shutil
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Household Power Consumption Analysis") \
    .getOrCreate()

spark.sparkContext.setLogLevel("INFO")


def perform_analytics():
    file_path = '/home/john-thomas-a/projects/airflow/household_power_consumption.txt'
    if not os.path.isfile(file_path):
        logging.error(f"File not found: {file_path}")
        return

    # Initialize Spark DataFrame
    df = spark.read.csv(file_path, header=True, sep=';', inferSchema=True)
    
    if df.count() == 0:
        logging.error("DataFrame is empty. Check the data source.")
        return
    
    # Convert columns to appropriate data types
    df = df.withColumn('Global_active_power', col('Global_active_power').cast(DoubleType())) \
           .withColumn('Voltage', col('Voltage').cast(DoubleType())) \
           .withColumn('Global_intensity', col('Global_intensity').cast(DoubleType())) \
           .withColumn('Sub_metering_1', col('Sub_metering_1').cast(DoubleType())) \
           .withColumn('Sub_metering_2', col('Sub_metering_2').cast(DoubleType())) \
           .withColumn('Sub_metering_3', col('Sub_metering_3').cast(DoubleType()))
    
    df.show()

    # Ensure the directory exists
    os.makedirs('/home/john-thomas-a/projects/airflow/analytics', exist_ok=True)

    # Analytics and file writing
    power_by_hour_path = '/home/john-thomas-a/projects/airflow/analytics/power_by_hour'
    avg_voltage_path = '/home/john-thomas-a/projects/airflow/analytics/avg_voltage_by_day'
    submetering_stats_path = '/home/john-thomas-a/projects/airflow/analytics/submetering_stats'

    logging.info("Saving power_by_hour DataFrame")
    power_by_hour = df.groupBy('Time').agg(mean('Global_active_power').alias('avg_power'))
    power_by_hour.write.mode('overwrite').csv(power_by_hour_path, header=True)

    logging.info("Saving avg_voltage_by_day DataFrame")
    avg_voltage_by_day = df.groupBy('Date').agg(mean('Voltage').alias('avg_voltage'))
    avg_voltage_by_day.write.mode('overwrite').csv(avg_voltage_path, header=True)

    df.na.fill(0)

    logging.info("Saving submetering_stats DataFrame")
    submetering_stats = df.agg(mean('Sub_metering_1').alias('avg_sub_metering_1'),
                               mean('Sub_metering_2').alias('avg_sub_metering_2'),
                               mean('Sub_metering_3').alias('avg_sub_metering_3'))
    submetering_stats.write.mode('overwrite').csv(submetering_stats_path, header=True)

    logging.info("Analytics have been completed and saved.")

    # Function to consolidate part files into a single CSV
    def consolidate_csv(directory, output_file):
        with open(output_file, 'wb') as outfile:
            for filename in os.listdir(directory):
                if filename.endswith('.csv'):
                    file_path = os.path.join(directory, filename)
                    with open(file_path, 'rb') as f:
                        shutil.copyfileobj(f, outfile)
        logging.info(f"Consolidated files in {directory} into {output_file}")

    # Consolidate files
    consolidate_csv(power_by_hour_path, '/home/john-thomas-a/projects/airflow/analytics/power_by_hour_final.csv')
    consolidate_csv(avg_voltage_path, '/home/john-thomas-a/projects/airflow/analytics/avg_voltage_by_day_final.csv')
    consolidate_csv(submetering_stats_path, '/home/john-thomas-a/projects/airflow/analytics/submetering_stats_final.csv')

    # Google Drive integration
    SCOPES = ['https://www.googleapis.com/auth/drive']
    creds = Credentials.from_authorized_user_file('/home/john-thomas-a/projects/airflow/token.json', SCOPES)
    service = build('drive', 'v3', credentials=creds)

    folder_id = os.getenv('FOLDER_ID')

    def delete_old_files():
        try:
            results = service.files().list(q=f"'{folder_id}' in parents", fields="files(id, name)").execute()
            items = results.get('files', [])
            if not items:
                logging.info("No old files found to delete.")
            else:
                for item in items:
                    service.files().delete(fileId=item['id']).execute()
                    logging.info(f"Deleted file: {item['name']} with ID: {item['id']}")
        except Exception as e:
            logging.error(f"Error deleting files: {e}")

    delete_old_files()

    # Upload consolidated CSV files to Google Drive
    final_csv_files = [
        '/home/john-thomas-a/projects/airflow/analytics/power_by_hour_final.csv',
        '/home/john-thomas-a/projects/airflow/analytics/avg_voltage_by_day_final.csv',
        '/home/john-thomas-a/projects/airflow/analytics/submetering_stats_final.csv'
    ]
    for file_path in final_csv_files:
        if os.path.isfile(file_path) and file_path.endswith(".csv"):  # Ensure it's a file and is a CSV
            file_name = os.path.basename(file_path)
            file_metadata = {
                'name': file_name,
                'parents': [folder_id]
            }
            media = MediaFileUpload(file_path, mimetype='text/csv')
            try:
                service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                logging.info(f"Uploaded {file_name} to Google Drive.")
            except Exception as e:
                logging.error(f"Error uploading {file_name}: {e}")

# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 7, 20),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'household_power_analysis',
    default_args=default_args,
    description='A DAG to perform analytics on household power consumption data',
    schedule='0 */2 * * *',  # Run every 2 hours
    catchup=False,  # Ensure that only the latest schedule is run
)

perform_analytics_task = PythonOperator(
    task_id='perform_analytics',
    python_callable=perform_analytics,
    dag=dag,
)
