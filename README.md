# ML Model Docker with Airflow and FastAPI

This project sets up an ML model serving application using Docker, Apache Airflow, and FastAPI. It includes scripts to manage ML model deployment, Analysis, Trigger and predictions using data stored in AWS S3.

## Table of Contents


- [Requirements](#requirements)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Usage](#usage)

## Folder Structure

Here’s the folder structure of the project:


## Requirements

- Docker
- Docker Compose
- AWS account 
- WSL (Incase of Window OS)
- visual studio code (optional)

## Installation

1. **Clone the repository**:

   ```bash
   git clone github.com/chandanlal92/ML_Airflow
   cd ML_Airflow

2. Create a .env file:

Create a .env file in the root directory of your project with the following content:

   ```bash
    AWS_ACCESS_KEY_ID=your_access_key
    AWS_SECRET_ACCESS_KEY=your_secret_key
    Model_S3_Bucket_name=your_bucket_name_with_models
    AWS_DEFAULT_REGION=your_region
    Data_prefix=/files/ftp_upload/
    sns_topic_arn=your_sns_topic_arn
    Data_S3_Bucket_name=your_bucket_name_with_dataset
    Data_filename=data_object_name
    Plot_S3_Bucket_name=your_bucket_name_to_store_plots
    
3. Build the Docker image:

    Run the following command to build the Docker image for Airflow :

    docker-compose build

Environment Variables
The following environment variables should be defined in your .env file:

- AWS_ACCESS_KEY_ID: Your AWS Access Key ID.
- AWS_SECRET_ACCESS_KEY: Your AWS Secret Access Key.
- Model_S3_Bucket_name: The name of your S3 bucket where models are stored.
- AWS_DEFAULT_REGION: The AWS region where your resources are located.
- Data_prefix: The prefix for the S3 bucket to locate files.
- sns_topic_arn: The ARN of the SNS topic for notifications.
- Data_S3_Bucket_name: The name of your S3 bucket where data is stored.
- Data_file_name: The object name of your data file that is used for training.
- Plot_S3_Bucket_name: The name of the S3 bucket to plot files

Usage

1. Start the services:

    To run the application, execute the following command:

    ```bash
    docker-compose up

2. Access Airflow:

    Airflow will be available at http://localhost:8080. You can log in with the default username: admin and password can be found in a file airflow/standlone_admin_password.txt(generated after docker is up).
     

3. Access FastAPI:

    Precition FastAPI will be available at http://localhost:8000/predict. Example for request json data for prediction
    
    ```bash
    curl --location 'http://localhost:8000/predict' \
    --header 'Content-Type: application/json' \
    --data '{
        "sepal_length_cm": 5.4,
        "sepal_width_cm": 2.5,
        "petal_length_cm": 1.2,
        "petal_width_cm": 4.2
    }'

