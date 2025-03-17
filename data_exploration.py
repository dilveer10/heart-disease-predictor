import pandas as pd

# Load the data
df = pd.read_csv('heart.csv')

# Display the first 5 rows of the data
print(df.head())

#This will che the dataset shape
print("\nDatatset Shape: ", df.shape)

#Check for missing values
print(df.isnull())

#Display column names and data types
print(df.info())

#display the summary statistics
print(df.describe())
