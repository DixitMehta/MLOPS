import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

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


# Create complete ML pipeline
pipeline = Pipeline(
    steps=[
        ('Preprocessor',preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators =100,
                random_state=42
            )

        )
    ]
)

# Train / test split
x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size = 0.2,
    random_state=42,
    stratify=y
)


# Train complete pipeline
pipeline.fit(x_train,y_train)

#  Make predicitons
y_pred = pipeline.predict(x_test)


# Save complete pipeline
joblib.dump(pipeline,'../models/customer_churned_pipeline.pkl')
print('Model saved to the path successfully!!')


# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)

print('Predictions')
print(y_pred)

print('\n Acutal values: ')
print(y_test.values)

print('Accuracy: ',accuracy)
