import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import joblib
import os


# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================

X_train = pd.read_csv("data/X_train_processed.csv")
X_test = pd.read_csv("data/X_test_processed.csv")

y_train = pd.read_csv("data/y_reg_train.csv").squeeze()
y_test = pd.read_csv("data/y_reg_test.csv").squeeze()


print("=" * 60)
print("MULTIPLE REGRESSION MODELS")
print("=" * 60)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================================
# 2. CREATE MODELS
# ============================================================

models = {

    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        max_depth=5
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    ),

    "KNN": KNeighborsRegressor(
        n_neighbors=5
    )
}


# ============================================================
# 3. TRAIN AND EVALUATE
# ============================================================

results = []

os.makedirs("models", exist_ok=True)

for name, model in models.items():

    print("\n" + "-" * 60)
    print("Training:", name)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = mse ** 0.5

    r2 = r2_score(y_test, y_pred)

    # Store results
    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2  :", round(r2, 4))

    # Save model
    filename = name.lower().replace(" ", "_") + "_regression.pkl"

    joblib.dump(
        model,
        "models/" + filename
    )

    print("Saved:", "models/" + filename)


# ============================================================
# 4. CREATE COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# ============================================================
# 5. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "data/regression_model_comparison.csv",
    index=False
)

print("\nComparison saved:")
print("data/regression_model_comparison.csv")


print("\n" + "=" * 60)
print("MULTIPLE REGRESSION COMPLETED")
print("=" * 60)