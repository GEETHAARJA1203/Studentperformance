
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASET_PATH = os.path.join(
    BASE_DIR, "data", "student_academic_data.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR, "risk_prediction_model.pkl"
)

RESULTS_PATH = os.path.join(
    BASE_DIR, "risk_model_results.txt"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("\n" + "=" * 60)
print("TREKVERSE - STUDENT RISK PREDICTION")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

required_columns = [
    "Risk_Level",
    "Department",
    "Semester",
    "Attendance",
    "Internal_Marks",
    "Assignment_Marks",
    "Quiz_Marks",
    "Study_Hours",
    "Previous_CGPA",
    "Backlogs",
    "Assignment_Submission_Rate",
    "Learning_Activity",
]

missing = [
    col for col in required_columns if col not in df.columns
]

if missing:
    raise ValueError(f"Missing dataset columns: {missing}")

# Keep rows with a known risk label.
df = df.dropna(subset=["Risk_Level"]).copy()
df["Risk_Level"] = df["Risk_Level"].astype(str).str.strip()

# Use only information that can be available before the
# risk prediction is made. Do not include IDs, names,
# current/final results, or the target label as inputs.
FEATURES = [
    "Department",
    "Semester",
    "Attendance",
    "Internal_Marks",
    "Assignment_Marks",
    "Quiz_Marks",
    "Study_Hours",
    "Previous_CGPA",
    "Backlogs",
    "Assignment_Submission_Rate",
    "Learning_Activity"
]


X = df[FEATURES].copy()
y = df["Risk_Level"]

print("\nDataset size:", len(df))
print("\nRisk category counts:")
print(y.value_counts().to_string())

# Each category needs at least two records for stratified
# splitting. Stop with a clear error if the data cannot
# support this split.
if y.value_counts().min() < 2:
    raise ValueError(
        "Each risk category needs at least 2 records."
    )

# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    ))
])

print("\nTraining risk prediction model...")
model.fit(X_train, y_train)

# --------------------------------------------------
# EVALUATE MODEL
# --------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
balanced_accuracy = balanced_accuracy_score(
    y_test, predictions
)

report = classification_report(
    y_test,
    predictions,
    zero_division=0
)

labels = sorted(y.unique())

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

print("\nAccuracy:", round(accuracy, 4))
print("Balanced accuracy:", round(balanced_accuracy, 4))

print("\nClassification report:")
print(report)

print("\nConfusion matrix (rows = actual, columns = predicted):")
print("Label order:", labels)
print(matrix)

# --------------------------------------------------
# SAVE MODEL AND RESULTS
# --------------------------------------------------

joblib.dump(
    {
        "model": model,
        "preprocessor": preprocessor,
        "features": FEATURES
    },
    os.path.join(
        BASE_DIR,
        "models",
        "risk_prediction_model.pkl"
    )
)

with open(RESULTS_PATH, "w", encoding="utf-8") as file:
    file.write("TREKVERSE STUDENT RISK PREDICTION RESULTS\n")
    file.write("=" * 45 + "\n")
    file.write(f"Records used: {len(df)}\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Balanced accuracy: {balanced_accuracy:.4f}\n\n")
    file.write("Classification report:\n")
    file.write(report)
    file.write("\n\nLabel order for confusion matrix:\n")
    file.write(str(labels))
    file.write("\n\nConfusion matrix:\n")
    file.write(str(matrix))

print("\nModel saved to:", MODEL_PATH)
print("Evaluation saved to:", RESULTS_PATH)
print("\nRisk model training completed.")
