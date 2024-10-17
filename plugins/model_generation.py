import random
from typing import Dict, Tuple
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import json
from datetime import datetime
import boto3
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime,timedelta
import plotly.express as px
#Training Random_forest_model
def train_random_forest(df: pd.DataFrame,
                        target_column: str,
                        hyperparameters: Dict[str, int]):
    """
    Train a Random Forest Classifier on the given dataset.

    Parameters:
        df (pd.DataFrame): The input dataframe containing features and target.
        target_column (str): The name of the target column in the dataframe.
        hyperparameters (dict): A dictionary of hyperparameters for the RandomForestClassifier.

    Returns:
        model: The trained Random Forest model.
        accuracy: The accuracy score of the model on the test set.
    """

    # Separate features and target from the DataFrame
    X = df.drop(target_column, axis=1)  # Features
    y = df[target_column]  # Target

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features (optional but recommended for some models, not strictly necessary for Random Forest)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and train the Random Forest Classifier model with hyperparameters
    model = RandomForestClassifier(**hyperparameters)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy


def save_model_to_s3(model, hyperparameters, accuracy, bucket_name, s3_client):
    # Serialize model
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    iso_timestamp=datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
    model_filename = f"model_{timestamp}.pkl"
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

    # Save hyperparameters and accuracy
    metadata = {
        'timestamp': iso_timestamp,
        'hyperparameters': hyperparameters,
        'accuracy': accuracy
    }
    metadata_filename = f"model_metadata_{timestamp}.json"
    with open(metadata_filename, 'w') as file:
        json.dump(metadata, file)
    Object_name="ML_Models/"+timestamp+"/"
     # Upload to S3
    s3_client.upload_file(model_filename, bucket_name, Object_name+model_filename)
    s3_client.upload_file(metadata_filename, bucket_name, Object_name+metadata_filename)

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

#Download files from s3 bucket
def download_s3_file(s3_client,bucket_name, s3_object_key, local_filename):
 
    try:
        # Download the file from S3 and save it locally with the specified filename
        s3_client.download_file(bucket_name, s3_object_key, local_filename)
        print(f"File downloaded successfully as {local_filename}")
    except Exception as e:
        print(f"Error downloading the file: {str(e)}")


def get_latest_model_time(s3_client,bucket_name,prefix):
    json_files=[]
    json_files=get_s3_json_files(s3_client=s3_client,bucket_name=bucket_name,prefix=prefix)
    df=load_json_files_into_dataframe(s3_client=s3_client,bucket_name=bucket_name,json_files=json_files)
    
    # Get the latest timestamp row
    latest_row = df.loc[df['timestamp'].idxmax()]
    dt_timestamp=(latest_row['timestamp'])
    return dt_timestamp
    
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



#Plot Seven Days Average Accuracy
def plot_accuracy(data):
    df = pd.DataFrame(data)
    df['rolling_avg'] = df['accuracy'].rolling(window=7).mean()
    plt.figure(figsize=(24,16))
    plt.plot(df['timestamp'], df['accuracy'], label='Daily Accuracy')
    plt.plot(df['timestamp'], df['rolling_avg'], label='7-day Average Accuracy', linestyle='--')
    plt.xlabel('timestamp')
    plt.ylabel('Accuracy')
    # Save the plot as a file (you can save it to a location accessible by Airflow)
    time_stamp=datetime.now().strftime("%Y-%m-%d-%H%M")
    plot_path = "accuracy_plot_"+time_stamp+".png"

    # Rotate x-axis labels vertically (90 degrees)
    plt.xticks(rotation=90)
    plt.legend()
    plt.savefig(plot_path)
    plt.show()
    s3_client=boto3.client("s3",region_name=os.getenv("AWS_DEFAULT_REGION"),aws_access_key_id= os.getenv("AWS_ACCESS_KEY_ID"),aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
    s3_client.upload_file(plot_path, os.getenv("Plot_S3_Bucket_name"),plot_path)

def send_sns_alert(s3_client,sns_client,sns_topic_arn,accuracy,threshold):
    
    # Alert for Reaching greater Threshold
    if accuracy > threshold:
        print(f"ALERT: Accuracy is {accuracy:.2f}, which is greater than 0.95!")
        
        # # Define the SNS topic ARN (replace with your actual SNS topic ARN)
        # sns_topic_arn = "arn:aws:sns:us-east-1:913799593672:model_alerts"


        # Create the message for the SNS alert
        message = f"ALERT: Accuracy reached {accuracy:.2f}, which is above the threshold of {threshold:.2f}."

        # Send the SNS alert
        response = sns_client.publish(
            TopicArn=sns_topic_arn,
            Message=message,
            Subject='Accuracy Alert',
        )
        print(f"SNS Alert Sent: {response}")

    #Alert for latest model not generated for last 3 days
    latest_model_time=get_latest_model_time(s3_client=s3_client,bucket_name=os.getenv("Model_S3_Bucket_name"),prefix=os.getenv("Data_prefix"))
    timestamp_format = "%m-%d-%Y-%H:%M:%S"  # "MM-DD-YYYY-HH:MM:SS"

    # Convert the string timestamp to a datetime object
    latest_model_timestamp = datetime.strptime(latest_model_time, timestamp_format)
    current_time = datetime.now()
    print(latest_model_timestamp)
    time_difference=current_time - latest_model_timestamp
    print(time_difference)
    if(time_difference > timedelta(hours=36)):
             
        message = f"ALERT: No model Generated for last 36 Hours!!!"

        # Send the SNS alert
        response = sns_client.publish(
            TopicArn=sns_topic_arn,
            Message=message,
            Subject='Model Alert',
        )
        print(f"SNS Alert Sent: {response}")


