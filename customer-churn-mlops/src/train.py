import pandas as pd
import joblib
import mlflow
import subprocess
import yaml
import logging
import os

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)


# logging configuration
logger = logging.getLogger('training_data')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'training_data.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

mlflow.set_experiment('customer-churn')

n_estimators =200

# Load dataseet
try:
    df = pd.read_csv('../data/raw/customer.csv')
    logger.debug('Data file loaded successfully')
except FileNotFoundError as e:
    logger.error('File not found %s', e)

# seaparate features and target
x = df.drop('churned', axis = 1)
y = df['churned']

# Indentify columns
categorical_features = ['city']
numerical_features = [
    'age',
    'salary',
    'monthly_spend',
    'tenure',
    'support_tickets'
    ]

# Preprocessing
try:
    logger.debug('Starting preprocessing...')
    preprocessor = ColumnTransformer(
        transformers = [
            (
                'city',
                OneHotEncoder(handle_unknown='ignore'),
                categorical_features
            )
        ],
        remainder = 'passthrough'
    )
    logger.debug('Completed preprocessing...')
except Exception as e:
    logger.error('Error occured: %s', e )

# Create complete ML pipeline
try:
    logger.debug('Initializing pipeline...')
    pipeline = Pipeline(
        steps=[
            ('Preprocessor',preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators = n_estimators,
                    random_state=42
                )

            )
        ]
    )
    logger.debug('Pipeline completed...')

except Exception as e:
    logger.error('Error occured during pipeline: %s', e)

# Train / test split
try:
    logger.debug('Starting train test split process...')
    x_train,x_test,y_train,y_test = train_test_split(
        x,
        y,
        test_size = 0.2,
        random_state=42,
        stratify=y
    )
    logger.debug('Completed train test split process...')

except Exception as e:
    logger.error('Error occured during train test split: %s', e)

# Checks if the dvc status
# result = subprocess.run(
#     ['dvc','status'],
#     capture_output=True,
#     text=True
#                         )
# print(result.stdout)


# Get DVC hash from the .dvc file for logging into MLFLOW.
try: 
    logger.debug('Extracting DVC hash for MLFLOW...')
    with open('../data/raw/customer.csv.dvc','r') as file:
        dvc_data = yaml.safe_load(file)
    dataset_version = dvc_data['outs'][0]['md5']
    print('dataset version:' ,dataset_version)
    logger.debug('Extraction completed...')

except Exception as e:
    logger.error('Error occured during DVC hash extraction: %s', e)

try:
    with mlflow.start_run():

        # Train complete pipeline
        pipeline.fit(x_train,y_train)

        #  Make predicitons
        y_pred = pipeline.predict(x_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Save complete pipeline
        joblib.dump(pipeline,'../models/customer_churned_pipeline_V2.pkl')
        logger.debug('Model saved to the path successfully!!')

        # Log paramters for MLFOW
        mlflow.log_param("n_estimators",n_estimators)
        mlflow.log_param("random_state",42)
        mlflow.log_param("Dataset version", dataset_version)

        # Log Metrics
        mlflow.log_metric("accuracy",accuracy)

        # Log model artifacts
        mlflow.log_artifact('../models/customer_churned_pipeline_V2.pkl') 

except Exception as e:
    logger.error('Error occured during Evaluation: %s', e)


print('Predictions')
print(y_pred)

print('\n Acutal values: ')
print(y_test.values)

print('Accuracy: ',accuracy)