import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# LOAD PROCESSED DATA
# ============================================================

X_train = pd.read_csv("data/X_train_processed.csv")
X_test = pd.read_csv("data/X_test_processed.csv")

y_train = pd.read_csv("data/y_reg_train.csv").squeeze()
y_test = pd.read_csv("data/y_reg_test.csv").squeeze()


print("=" * 60)
print("BASELINE REGRESSION MODEL")
print("=" * 60)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================================
# CREATE LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Linear Regression...")

model.fit(X_train, y_train)


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)


print("\n" + "=" * 60)
print("REGRESSION RESULTS")
print("=" * 60)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

results = pd.DataFrame({
    "Actual_G3": y_test.values,
    "Predicted_G3": y_pred
})

print(results.head(10))


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/linear_regression.pkl"
)

print("\nModel saved:")
print("models/linear_regression.pkl")


print("\n" + "=" * 60)
print("BASELINE REGRESSION COMPLETED")
print("=" * 60)