from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess
import pandas as pd
import random
from airflow.models import Variable
import os


default_args = {
    'dag_id':"1001",
    'owner': 'airflow',
    'retries': 3,
    'start_date': datetime.today(),  # Adjust this to the date you want to start the DAG
    'retry_delay': timedelta(minutes=5),
    'sla': timedelta(hours=1)  # Model training should finish within 1 hour
   
}


plot_args= {
    'dag_id':"1002",
    'owner': 'airflow',
    'retries': 3,
    'start_date': datetime.today(),  # Adjust this to the date you want to start the DAG
    'retry_delay': timedelta(minutes=5),
    'sla': timedelta(hours=1)  # Model training should finish within 1 hour
   
}

#Training ML Model
def train_model(**kwargs):
    # Call your model training script here
    from model_generation import train_random_forest
    from model_generation import save_model_to_s3
    from model_generation import download_s3_file
    import boto3

    s3_client=boto3.client("s3",region_name=os.getenv("AWS_DEFAULT_REGION"),aws_access_key_id= os.getenv("AWS_ACCESS_KEY_ID"),aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))

    #Downloading data from s3 bucket
    test=download_s3_file(s3_client,bucket_name=os.getenv("Data_S3_Bucket_name"),s3_object_key=os.getenv("Data_filename"),local_filename="data.csv")
 
    df = pd.read_csv('data.csv')
    hyperparameters = {
        'n_estimators': random.randint(1, 2),  # Number of trees in the forest
        'max_depth': random.randint(1, 5),
        # Maximum depth of the tree (None means nodes are expanded until all leaves are pure)
        'min_samples_split': random.randint(2, 3),  # Minimum number of samples required to split an internal node
        'min_samples_leaf': random.randint(1, 2),  # Minimum number of samples required to be at a leaf node
        'random_state': 42,  # Controls the randomness of the estimator for reproducibility
    }
    model, accuracy = train_random_forest(df, 'variety', hyperparameters)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

   
    save_model_to_s3(model,hyperparameters,accuracy,os.getenv("Model_S3_Bucket_name"),s3_client)
    return accuracy
    
#Plotting Weekly plots
def weekly_plots():
     from model_generation import load_json_files_into_dataframe
     from model_generation import plot_accuracy
     from model_generation import get_s3_json_files
     import boto3
     import pandas as pd


     s3_client=boto3.client("s3",region_name=os.getenv("AWS_DEFAULT_REGION"),aws_access_key_id= os.getenv("AWS_ACCESS_KEY_ID"),aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
     
     prefix=os.getenv("Data_prefix")
     json_files=[]
     json_files=get_s3_json_files(s3_client,os.getenv("Model_S3_Bucket_name"),prefix)
     df=pd.DataFrame()
     df=load_json_files_into_dataframe(s3_client,os.getenv("Model_S3_Bucket_name"),json_files)
     plot_accuracy(df)    

def alerts(**kwargs):
    import boto3
    from model_generation import send_sns_alert
    ti = kwargs['ti']
    accuracy = ti.xcom_pull(task_ids='train_model')
    thresold=0.95
    sns_client = boto3.client('sns', region_name=os.getenv("AWS_DEFAULT_REGION"),aws_access_key_id= os.getenv("AWS_ACCESS_KEY_ID"),aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
    s3_client=boto3.client("s3",region_name=os.getenv("AWS_DEFAULT_REGION"),aws_access_key_id= os.getenv("AWS_ACCESS_KEY_ID"),aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
    # Define the SNS topic ARN (replace with your actual SNS topic ARN)
    sns_topic_arn = os.getenv("sns_topic_arn")
    send_sns_alert(s3_client,sns_client,sns_topic_arn,accuracy,thresold)

      

dag = DAG('daily_model_training',default_args=default_args, schedule_interval='@daily')
dag_2=DAG("weekly_plots",default_args=plot_args,schedule_interval='@weekly')


task_1 = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)
task_2 = PythonOperator(
    task_id='weekly_plots',
    python_callable=weekly_plots,
    dag=dag_2
)
task_3 = PythonOperator(
    task_id='alerts',
    python_callable=alerts,
    dag=dag
)
task_1 >> task_3
