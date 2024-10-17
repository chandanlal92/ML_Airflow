from fastapi import FastAPI
import uvicorn
import joblib
import os
import pandas as pd
import numpy as np
import boto3
from datetime import datetime
import subprocess
import json
from pydantic import BaseModel

app = FastAPI()
bucket_name = os.getenv("Model_S3_Bucket_name")
prefix = os.getenv("Data_prefix")  # e.g., '/files/ftp_upload/'
local_file_name="latest_model.pkl"

s3_client=boto3.client("s3",region_name=os.getenv("AWS_DEFAULT_REGION"),aws_access_key_id= os.getenv("AWS_ACCESS_KEY_ID"),aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))


# Define the input data schema using Pydantic for validation
class IrisInput(BaseModel):
    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float

# Function to get list of all JSON files 
def get_s3_json_files(s3_client,bucket_name, prefix):
    json_files = []
    result = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    
    # Check if there are files in the result
    if 'Contents' in result:
        for obj in result['Contents']:
            if obj['Key'].endswith('.json'):  # Only include JSON files
                json_files.append(obj['Key'])
    
    return json_files


# Function to read the content of each JSON file and store it into a DataFrame
def load_json_files_into_dataframe(s3_client,bucket_name, json_files):
    data_list = []
    
    for file in json_files:
        obj = s3_client.get_object(Bucket=bucket_name, Key=file)
        file_content = obj['Body'].read().decode('utf-8')
        json_content = json.loads(file_content)  # Parse the JSON file
        
        # Assuming each JSON file contains a list of dictionaries (for multiple rows) or a single dictionary
        if isinstance(json_content, list):
            data_list.extend(json_content)  # Add multiple rows if the JSON contains a list
        else:
            data_list.append(json_content)  # Add a single row if it's just one dictionary
    
    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(data_list)
    return df

def download_s3_file(s3_client,bucket_name, s3_object_key, local_filename):
 

    try:
        # Download the file from S3 and save it locally with the specified filename
        s3_client.download_file(bucket_name, s3_object_key, local_filename)
        print(f"File downloaded successfully as {local_filename}")
    except Exception as e:
        print(f"Error downloading the file: {str(e)}")

# Function to load the latest model from a predefined location
def load_latest_model():
    json_files=[]
    json_files=get_s3_json_files(s3_client=s3_client,bucket_name=bucket_name,prefix=prefix)
    print(json_files)
    df=load_json_files_into_dataframe(s3_client=s3_client,bucket_name=bucket_name,json_files=json_files)
    
    # Get the latest timestamp row
    latest_row = df.loc[df['timestamp'].idxmax()]
    dt_timestamp=(latest_row['timestamp'])
    dt = datetime.strptime(dt_timestamp,"%m-%d-%Y-%H:%M:%S")
    # Convert to the desired format YY-MM-DD-HHMMSS
    formatted_timestamp = dt.strftime("%Y-%m-%d-%H%M%S")
    s3_object_key="ML_Models/"+formatted_timestamp+"/"+"model_"+formatted_timestamp+".pkl"
    
    print(s3_object_key)
    download_s3_file(s3_client,bucket_name=bucket_name,s3_object_key=s3_object_key,local_filename=local_file_name)
    return local_file_name

@app.post("/predict")
def predict(iris_data: IrisInput):
    # Convert input data to a DataFrame (assuming a tabular structure)
    print(iris_data)
    local_file_name=load_latest_model()
    model = joblib.load(local_file_name)
    input_df = pd.DataFrame([{
        'sepal length (cm)': iris_data.sepal_length_cm,
        'sepal width (cm)': iris_data.sepal_width_cm,
        'petal length (cm)': iris_data.petal_length_cm,
        'petal width (cm)': iris_data.petal_width_cm
    }])
    
    # Make prediction using the loaded model
    prediction = model.predict(input_df)
    
    return {"prediction": prediction[0]}




