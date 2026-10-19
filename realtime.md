Cell 0: Package Installation
!pip install -q xgboost pandas scikit-learn sagemaker boto3


AWS SageMaker MLOps Pipeline: End-to-End Production Guide
This document contains the complete, sequential set of Python code cells that successfully built, trained, registered, deployed, tested, and cleaned up the XGBoost Bank Marketing Conversion Prediction Pipeline on AWS SageMaker.


Cell 1: Imports, Environment Setup, and S3 Bucket Initialization
import boto3

import sagemaker

import pandas as pd

import numpy as np

import xgboost as xgb

import tarfile

import csv

import time

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sagemaker import image_uris

from sagemaker.model import Model

# Initialize SageMaker session and default region/bucket

sagemaker_session = sagemaker.Session()

role = sagemaker.get_execution_role()

region = boto3.Session().region_name

s3_client = boto3.client('s3', region_name=region)

sagemaker_client = boto3.client('sagemaker', region_name=region)

my_bucket = sagemaker_session.default_bucket()

print(f"SageMaker Role: {role}")

print(f"Region: {region}")

print(f"Default S3 Bucket: {my_bucket}")


Cell 2: Dataset Preparation and Local Training
import os

# Download/prepare train.csv and test.csv if they do not exist locally
if not os.path.exists('train.csv') or not os.path.exists('test.csv'):
    print("Local train.csv or test.csv not found. Preparing dataset...")
    try:
        # Fetch public sample file using boto3 unsigned client (bypasses region auth errors)
        from botocore import UNSIGNED
        from botocore.config import Config
        
        s3_anon = boto3.client('s3', region_name='us-east-1', config=Config(signature_version=UNSIGNED))
        s3_anon.download_file('sagemaker-sample-files', 'datasets/tabular/bank_marketing/bank-additional-full.csv', 'bank-additional-full.csv')
        raw_df = pd.read_csv('bank-additional-full.csv', sep=';')
    except Exception as e:
        print(f"Could not download directly from S3 ({e}). Generating fallback processed tabular dataset...")
        # Fallback: create synthetic/sample tabular data matching bank marketing features
        np.random.seed(42)
        n_samples = 5000
        n_features = 20
        X_dummy = np.random.randn(n_samples, n_features)
        y_dummy = (X_dummy[:, 0] + X_dummy[:, 1] > 0).astype(int)
        raw_df = pd.DataFrame(X_dummy, columns=[f'feature_{i}' for i in range(n_features)])
        raw_df['y'] = y_dummy

    # Preprocess categorical features and target
    if 'y' in raw_df.columns:
        if raw_df['y'].dtype == object:
            raw_df['y'] = (raw_df['y'] == 'yes').astype(int)
        raw_df = pd.get_dummies(raw_df)
        cols = ['y'] + [col for col in raw_df.columns if col != 'y']
        raw_df = raw_df[cols]
    
    # Train / Test split (80/20)
    train_data, test_data = train_test_split(raw_df, test_size=0.2, random_state=42)
    train_data.to_csv('train.csv', index=False, header=False)
    test_data.to_csv('test.csv', index=False, header=False)
    print("Created 'train.csv' and 'test.csv' successfully.")

# Load local processed datasets
train_df = pd.read_csv('train.csv', header=None)
test_df = pd.read_csv('test.csv', header=None)

# Separate features and target (target column is at index 0)
X_train = train_df.iloc[:, 1:]
y_train = train_df.iloc[:, 0]
X_test = test_df.iloc[:, 1:]
y_test = test_df.iloc[:, 0]

# Convert to XGBoost DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set model parameters
params = {
    'max_depth': 5,
    'eta': 0.2,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

# Train the model locally
print("Training XGBoost model locally...")
bst = xgb.train(params, dtrain, num_boost_round=100)

# Evaluate model performance
preds = bst.predict(dtest)
predictions = [1 if p > 0.5 else 0 for p in preds]
accuracy = accuracy_score(y_test, predictions)
print(f"Local Model Test Accuracy: {accuracy * 100:.2f}%")




Cell 3: Package and Upload Model Artifacts to S3
SageMaker inference containers require model artifacts to be saved, packaged into a compressed .tar.gz file, and stored in Amazon S3.

# Save model artifact

model_filename = "xgboost-model"

bst.save_model(model_filename)

# Package artifact into model.tar.gz

tar_filename = "model.tar.gz"

with tarfile.open(tar_filename, "w:gz") as tar:

    tar.add(model_filename)

# Upload packaged artifact to S3

s3_key = "models/output/model.tar.gz"

s3_client.upload_file(tar_filename, my_bucket, s3_key)

model_url = f"s3://{my_bucket}/{s3_key}"

print(f"Successfully uploaded model artifact to: {model_url}")


Cell 4: Register Model Package in SageMaker Model Registry
Organize and version your model artifacts using SageMaker Model Registry with metadata lineage tracking.

model_package_group_name = "bank-fraud-model-group-v2"

# 1. Create or verify the Model Package Group

print(f"Creating Model Package Group: {model_package_group_name}...")

try:

    sagemaker_client.create_model_package_group(

        ModelPackageGroupName=model_package_group_name,

        ModelPackageGroupDescription="XGBoost models for bank marketing conversion prediction"

    )

    print("Successfully created Model Package Group.")

except Exception as e:

    print(f"Model Package Group status note: {e}")

time.sleep(3)

# 2. Retrieve container image URI

container_uri = image_uris.retrieve('xgboost', region, version='1.7-1')

# 3. Register the Model Package version

model_package_input = {

    "ModelPackageGroupName": model_package_group_name,

    "ModelPackageDescription": "Version 1 - Local training run with 89.54 accuracy",

    "InferenceSpecification": {

        "Containers": [{

            "Image": container_uri,

            "ModelDataUrl": model_url

        }],

        "SupportedContentTypes": ["text/csv"],

        "SupportedResponseMIMETypes": ["text/csv"],

        "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large", "ml.m5.xlarge"],

        "SupportedTransformInstanceTypes": ["ml.m5.large"]

    },

    "ModelApprovalStatus": "Approved",

    "CustomerMetadataProperties": {

        "test_accuracy": f"{accuracy * 100:.2f}",

        "framework": "xgboost"

    }

}

response = sagemaker_client.create_model_package(**model_package_input)

print(f"\nSuccessfully registered model version!")

print(f"Model Package ARN: {response['ModelPackageArn']}")


Cell 5: Deploy Real-Time SageMaker Endpoint (XGBoost Container 1.7-1)
Deploy the model to an online HTTPS endpoint using ml.m5.large and modern XGBoost container 1.7-1. (Note: Complete cleanup of endpoints, endpoint configurations, and models handles any existing naming or resource collisions).

endpoint_name = "bank-fraud-endpoint-v3"

# Comprehensive cleanup to avoid ValidationExceptions (endpoint configuration & endpoint conflicts)

try:

    print(f"Cleaning up any existing endpoint or config named: {endpoint_name}")

    sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

except Exception as e:

    pass

try:

    sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)

except Exception as e:

    pass

try:

    sagemaker_client.delete_model(ModelName=endpoint_name)

except Exception as e:

    pass

time.sleep(3)

# Retrieve modern container URI (version 1.7-1 supports JSON and modern artifacts)

container_uri = image_uris.retrieve('xgboost', region, version='1.7-1')

# Create SageMaker Model resource

model = Model(

    image_uri=container_uri,

    model_data=model_url,

    role=role,

    sagemaker_session=sagemaker_session

)

print(f"Deploying model to endpoint '{endpoint_name}' using ml.m5.large...")

predictor = model.deploy(

    initial_instance_count=1,

    instance_type='ml.m5.large',

    endpoint_name=endpoint_name

)

print(f"Deployment initiated for endpoint: {endpoint_name}")


Cell 6: Poll Endpoint Status and Run Real-Time Test Invocation
Wait until the endpoint transitions to InService and send a live test payload (text/csv string format) using the sagemaker-runtime client.

runtime_client = boto3.client('sagemaker-runtime', region_name=region)

# Poll until endpoint status is InService

print(f"Waiting for endpoint '{endpoint_name}' to reach InService status...")

while True:

    status_response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)

    status = status_response['EndpointStatus']

    

    if status == 'InService':

        print("\nEndpoint is InService and ready for inference!")

        break

    elif status == 'Failed':

        failure_reason = status_response.get('FailureReason', 'Unknown')

        raise Exception(f"Endpoint deployment failed: {failure_reason}")

        

    print(f"Current status: {status}... checking again in 15 seconds.")

    time.sleep(15)

# Read sample row from local test dataset

with open('test.csv', 'r') as f:

    reader = csv.reader(f)

    sample_row = next(reader)

# Extract features (drop target label at index 0)

features = [float(x) for x in sample_row[1:]]

payload = ",".join([str(x) for x in features])

# Invoke live real-time endpoint

print("Sending real-time inference payload to SageMaker endpoint...")

response = runtime_client.invoke_endpoint(

    EndpointName=endpoint_name,

    ContentType='text/csv',

    Body=payload

)

result = response['Body'].read().decode('utf-8')

print(f"Live Prediction Probability: {result.strip()}")


Cell 7: Resource Cleanup (Deleting Endpoint to Prevent Ongoing Charges)
Delete the active endpoint and endpoint configuration to avoid incurring recurring charges in sandbox or production environments.

print(f"Cleaning up resources for endpoint: {endpoint_name}...")

try:

    sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

    print("Deleted endpoint successfully.")

except Exception as e:

    print(f"Error deleting endpoint: {e}")

try:

    sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)

    print("Deleted endpoint configuration successfully.")

except Exception as e:

    print(f"Error deleting endpoint config: {e}")

print("Cleanup complete!")
