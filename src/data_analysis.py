import pandas as pd

# Load dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

print("=" * 50)
print("DATASET BASIC INFORMATION")
print("=" * 50)

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())


print("\n" + "=" * 50)
print("DATA TYPES")
print("=" * 50)

print(df.dtypes)


print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)

print(df.isnull().sum())


print("\n" + "=" * 50)
print("DUPLICATE ROWS")
print("=" * 50)

print("Number of duplicate rows:", df.duplicated().sum())


print("\n" + "=" * 50)
print("NUMERICAL SUMMARY")
print("=" * 50)

print(df.describe())


print("\n" + "=" * 50)
print("TARGET VARIABLE - G3")
print("=" * 50)

print(df["G3"].describe())


print("\n" + "=" * 50)
print("G3 VALUE COUNTS")
print("=" * 50)

print(df["G3"].value_counts().sort_index())