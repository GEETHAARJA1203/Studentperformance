import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load processed training and testing data
# --------------------------------------------------

X_train = pd.read_csv("data/X_train_processed.csv")
X_test = pd.read_csv("data/X_test_processed.csv")

y_train = pd.read_csv("data/y_reg_train.csv").squeeze()
y_test = pd.read_csv("data/y_reg_test.csv").squeeze()

print("=" * 60)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 60)

print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)


# --------------------------------------------------
# 2. Define Random Forest model
# --------------------------------------------------

rf = RandomForestRegressor(
    random_state=42
)


# --------------------------------------------------
# 3. Define parameters to test
# --------------------------------------------------

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}


# --------------------------------------------------
# 4. Grid Search with 5-Fold Cross Validation
# --------------------------------------------------

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

print()
print("Starting Grid Search...")
print("Please wait...")

grid_search.fit(X_train, y_train)


# --------------------------------------------------
# 5. Display best parameters
# --------------------------------------------------

print()
print("=" * 60)
print("BEST PARAMETERS")
print("=" * 60)

print(grid_search.best_params_)

print()
print("Best CV RMSE:")
print(-grid_search.best_score_)


# --------------------------------------------------
# 6. Evaluate tuned model on test data
# --------------------------------------------------

best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)


print()
print("=" * 60)
print("TUNED RANDOM FOREST TEST RESULTS")
print("=" * 60)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# --------------------------------------------------
# 7. Save tuned model
# --------------------------------------------------

joblib.dump(
    best_rf,
    "models/tuned_random_forest_regression.pkl"
)

print()
print("Tuned model saved successfully!")


# --------------------------------------------------
# 8. Save Grid Search results
# --------------------------------------------------

results = pd.DataFrame(grid_search.cv_results_)

results.to_csv(
    "data/rf_tuning_results.csv",
    index=False
)

print("Tuning results saved successfully!")

print()
print("=" * 60)
print("STEP 14 COMPLETED")
print("=" * 60)