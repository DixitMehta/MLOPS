import pandas as pd

#Load dataset
df = pd.read_csv("../data/raw/Customer.csv")

#Display first 5 rows
print("First 5 rows")
print(df.head())

#Display dataset information
print("\nDataset information:")
print(df.info())

#Check for missing values
print("\nMissing values:b")
print(df.isnull().sum())

#Display basic statistics
print('\n Statistical Summary:')
print(df.describe())

# Display number of records
print('\n Total records: ', len(df))

# Display columns
print('Colums:')
print(df.columns.tolist())
