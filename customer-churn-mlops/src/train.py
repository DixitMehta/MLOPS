import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# Load dataseet
df = pd.read_csv('../data/raw/Customer.csv')

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

# Train / test split
x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size = 0.2,
    random_state=42,
    stratify=y
)

# Create Random Forest Model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Transform training data
x_train_processed = preprocessor.fit_transform(x_train)

# Transform test data
x_test_processed = preprocessor.transform(x_test)

# Train model
model.fit(x_train_processed, y_train)
joblib.dump(model,'../models/random_forest_model.pkl')
print('Model saved to the path successfully!!')


# Make predictions
y_pred = model.predict(x_test_processed)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)

print('Predictions')
print(y_pred)

print('\n Acutal values: ')
print(y_test.values)

print('Accuracy: ',accuracy)
