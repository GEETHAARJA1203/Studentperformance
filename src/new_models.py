import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "student-mat.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("       TREKVERSE - NEW ML MODELS")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    sep=";"
)

print(f"Dataset loaded successfully: {df.shape}")


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop(
    "G3",
    axis=1
)

y = df["G3"]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# LOAD EXISTING PREPROCESSOR
# ============================================================

print("\nLoading preprocessor...")

preprocessor_path = os.path.join(
    MODEL_DIR,
    "preprocessor.pkl"
)

preprocessor = joblib.load(
    preprocessor_path
)

print("Preprocessor loaded successfully.")


# ============================================================
# PREPROCESS DATA
# ============================================================

print("\nPreprocessing data...")

X_train_processed = preprocessor.transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)

print(
    "Processed training shape:",
    X_train_processed.shape
)

print(
    "Processed testing shape:",
    X_test_processed.shape
)


# ============================================================
# NEW MODELS
# ============================================================

models = {

    "Ridge Regression": Ridge(
        alpha=1.0
    ),

    "Lasso Regression": Lasso(
        alpha=0.01,
        max_iter=10000
    ),

    "Extra Trees Regressor": ExtraTreesRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Support Vector Regression": SVR(
        kernel="rbf",
        C=10,
        gamma="scale"
    )
}


# ============================================================
# TRAIN MODELS
# ============================================================

results = []


for model_name, model in models.items():

    print("\n" + "-" * 70)
    print(f"Training: {model_name}")
    print("-" * 70)

    model.fit(
        X_train_processed,
        y_train
    )

    predictions = model.predict(
        X_test_processed
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # Save model name
    filename = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    model_path = os.path.join(
        MODEL_DIR,
        f"{filename}.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"Model saved: {model_path}"
    )

    results.append({
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = os.path.join(
    BASE_DIR,
    "new_model_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("              NEW MODEL RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("\nResults saved to:")
print(results_path)

print("\nNew models completed successfully!")

print("=" * 70)