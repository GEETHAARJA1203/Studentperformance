import pandas as pd

# Load dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

print("Dataset loaded successfully!")
print()

print("First 5 rows:")
print(df.head())

print()
print("Dataset shape:")
print(df.shape)

print()
print("Column names:")
print(df.columns.tolist())

print()
print("Dataset information:")
print(df.info())

print()
print("Missing values:")
print(df.isnull().sum())

print()
print("Duplicate rows:")
print(df.duplicated().sum())