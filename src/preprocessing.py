import pandas as pd

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("data/student-mat.csv", sep=";")

print("=" * 60)
print("DATA CLEANING AND PREPROCESSING")
print("=" * 60)

print("\nOriginal dataset shape:")
print(df.shape)


# ============================================================
# 2. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)

if missing_values.sum() == 0:
    print("\nNo missing values found.")
else:
    print("\nMissing values found. Further treatment is required.")


# ============================================================
# 3. CHECK DUPLICATES
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE RECORDS")
print("=" * 60)

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)

if duplicates == 0:
    print("No duplicate records found.")
else:
    print("Duplicate records found.")


# ============================================================
# 4. CHECK CATEGORICAL VALUES
# ============================================================

print("\n" + "=" * 60)
print("CATEGORICAL VALUE CHECK")
print("=" * 60)

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].unique())


# ============================================================
# 5. CHECK NUMERICAL COLUMNS
# ============================================================

print("\n" + "=" * 60)
print("NUMERICAL COLUMNS")
print("=" * 60)

numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns

print(numerical_columns.tolist())


# ============================================================
# 6. CHECK OUTLIERS USING IQR
# ============================================================

print("\n" + "=" * 60)
print("OUTLIER CHECK")
print("=" * 60)

for column in numerical_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print(f"{column}: {len(outliers)} possible outliers")


# ============================================================
# 7. CHECK TARGET VARIABLE
# ============================================================

print("\n" + "=" * 60)
print("TARGET VARIABLE CHECK")
print("=" * 60)

print("Target variable: G3")
print("Minimum G3:", df["G3"].min())
print("Maximum G3:", df["G3"].max())
print("Mean G3:", df["G3"].mean())


# ============================================================
# 8. SAVE CLEANED DATA
# ============================================================

df.to_csv("data/cleaned_data.csv", index=False)

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print("Cleaned dataset saved as:")
print("data/cleaned_data.csv")