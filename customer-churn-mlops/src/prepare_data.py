import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Loading dataset
df = pd.read_csv('../data/raw/Customer.csv')

# Separate Features and Targets
x = df.drop('churned', axis=1)
y = df['churned']

print('\n Features')
print(x)

print('\n Targets')
print(y)

# Encode categorical column (converting string data of city column in numerical values or bianry values)

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
city_encoded = encoder.fit_transform(x[['city']])

city_columns = encoder.get_feature_names_out(['city'])

city_df = pd.DataFrame(
    city_encoded,
    columns = city_columns,
    index=x.index
)

# Remove original city column
x = x.drop('city', axis=1)

#  Add encoded city column
x = pd.concat([x,city_df], axis=1)


# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#  Display result
print('\nEncoded features: ')
print(x)

print("\n X train shape: ", x_train.shape)
print("\n X test shape: ", x_test.shape)

print("\n Y train")
print(y_train)

print("\n Y test")
print(y_test)