import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os


# ============================================================
# 1. LOAD FEATURE-ENGINEERED DATA
# ============================================================

df = pd.read_csv("data/feature_engineered_data.csv")

print("=" * 60)
print("MACHINE LEARNING DATA PREPARATION")
print("=" * 60)

print("Dataset shape:", df.shape)


# ============================================================
# 2. DEFINE TARGET
# ============================================================

# Regression target
y_regression = df["G3"]


# Classification target
y_classification = df["support_category"]


# ============================================================
# 3. REMOVE TARGET / LEAKAGE COLUMNS
# ============================================================

columns_to_remove = [
    "G3",
    "support_category",
    "G1",
    "G2",
    "previous_grade_average",
    "grade_trend"
]

X = df.drop(columns=columns_to_remove)


print("\nInput features:")
print(X.columns.tolist())


# ============================================================
# 4. IDENTIFY COLUMN TYPES
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)


# ============================================================
# 5. PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ]
)


# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X,
    y_regression,
    test_size=0.20,
    random_state=42
)


# Classification split
X_train_cls, X_test_cls, y_cls_train, y_cls_test = train_test_split(
    X,
    y_classification,
    test_size=0.20,
    random_state=42,
    stratify=y_classification
)


# ============================================================
# 7. FIT PREPROCESSOR USING TRAINING DATA ONLY
# ============================================================

preprocessor.fit(X_train)


X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)


print("\nTraining data shape:", X_train_processed.shape)
print("Testing data shape:", X_test_processed.shape)


# ============================================================
# 8. SAVE PREPROCESSOR
# ============================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)


# ============================================================
# 9. SAVE PROCESSED DATA
# ============================================================

pd.DataFrame(
    X_train_processed
).to_csv(
    "data/X_train_processed.csv",
    index=False
)

pd.DataFrame(
    X_test_processed
).to_csv(
    "data/X_test_processed.csv",
    index=False
)


y_reg_train.to_csv(
    "data/y_reg_train.csv",
    index=False
)

y_reg_test.to_csv(
    "data/y_reg_test.csv",
    index=False
)


y_cls_train.to_csv(
    "data/y_cls_train.csv",
    index=False
)

y_cls_test.to_csv(
    "data/y_cls_test.csv",
    index=False
)


# ============================================================
# 10. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("ML PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Preprocessor saved:")
print("models/preprocessor.pkl")

print("\nProcessed training and testing data saved.")