
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASET_PATH = os.path.join(
    BASE_DIR, "data", "student_academic_data.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("TREKVERSE - ACADEMIC PERFORMANCE MODEL TRAINING")
print("=" * 65)

df = pd.read_csv(DATASET_PATH)

if "Final_Marks" not in df.columns:
    raise ValueError("Dataset must contain a Final_Marks column.")

df = df.dropna(subset=["Final_Marks"]).copy()

# Use only information intended to be available before final marks.
feature_columns = [
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

missing_columns = [
    col for col in feature_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

X = df[feature_columns].copy()
y = pd.to_numeric(df["Final_Marks"], errors="coerce")

valid_rows = y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows]

print(f"Dataset: {X.shape[0]} student records")
print(f"Input features: {len(feature_columns)}")
print("Prediction target: Final_Marks")


# ------------------------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ------------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------------

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ------------------------------------------------------------
# MODELS
# ------------------------------------------------------------

models = {
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(max_iter=10000),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=200, random_state=42
    ),
    "Support Vector Regression": SVR(),
    "XGBoost": XGBRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=150,
        learning_rate=0.05,
        verbosity=-1,
        random_state=42
    ),
    "CatBoost": CatBoostRegressor(
        iterations=150,
        learning_rate=0.05,
        depth=4,
        verbose=0,
        random_seed=42
    ),
}


# ------------------------------------------------------------
# TRAIN AND EVALUATE
# ------------------------------------------------------------

results = []
trained_models = {}

for name, estimator in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    try:
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(
            y_test, predictions
        ) ** 0.5
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4)
        })

        trained_models[name] = pipeline

        print(
            f"MAE: {mae:.3f} | "
            f"RMSE: {rmse:.3f} | "
            f"R2: {r2:.3f}"
        )

    except Exception as error:
        print(f"Could not train {name}: {error}")


# ------------------------------------------------------------
# SAVE RESULTS AND BEST MODEL
# ------------------------------------------------------------

if not results:
    raise RuntimeError(
        "No models trained successfully. Check the errors above."
    )

results_df = pd.DataFrame(results).sort_values(
    by="RMSE", ascending=True
)

results_path = os.path.join(
    BASE_DIR, "new_model_results.csv"
)
results_df.to_csv(results_path, index=False)

best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

model_path = os.path.join(
    MODEL_DIR, "academic_performance_model.pkl"
)
joblib.dump(best_model, model_path)

print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)
print(results_df.to_string(index=False))

print(f"\nBest model on this test split: {best_name}")
print(f"Results saved to: {results_path}")
print(f"Model saved to: {model_path}")
print("\nTraining completed.")
