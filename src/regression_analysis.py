import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import joblib


# ============================================================
# 1. LOAD DATA
# ============================================================

X_test = pd.read_csv("data/X_test_processed.csv")
y_test = pd.read_csv("data/y_reg_test.csv").squeeze()

comparison = pd.read_csv(
    "data/regression_model_comparison.csv"
)

os.makedirs("eda_graphs", exist_ok=True)

print("=" * 60)
print("REGRESSION MODEL ANALYSIS")
print("=" * 60)


# ============================================================
# 2. DISPLAY MODEL COMPARISON
# ============================================================

print("\nModel comparison:")

print(comparison.to_string(index=False))


# ============================================================
# 3. MAE COMPARISON
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    comparison["Model"],
    comparison["MAE"]
)

plt.xlabel("Model")
plt.ylabel("MAE")
plt.title("Regression Models - MAE Comparison")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "eda_graphs/11_regression_mae_comparison.png"
)

plt.close()


# ============================================================
# 4. RMSE COMPARISON
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    comparison["Model"],
    comparison["RMSE"]
)

plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("Regression Models - RMSE Comparison")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "eda_graphs/12_regression_rmse_comparison.png"
)

plt.close()


# ============================================================
# 5. R2 COMPARISON
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    comparison["Model"],
    comparison["R2"]
)

plt.xlabel("Model")
plt.ylabel("R²")
plt.title("Regression Models - R² Comparison")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "eda_graphs/13_regression_r2_comparison.png"
)

plt.close()


# ============================================================
# 6. SELECT BEST MODEL BY RMSE
# ============================================================

best_model_name = comparison.loc[
    comparison["RMSE"].idxmin(),
    "Model"
]

print("\nModel with lowest RMSE:")
print(best_model_name)


# ============================================================
# 7. LOAD BEST MODEL
# ============================================================

filename = (
    best_model_name.lower()
    .replace(" ", "_")
    + "_regression.pkl"
)

best_model = joblib.load(
    "models/" + filename
)

y_pred = best_model.predict(X_test)


# ============================================================
# 8. RESIDUAL ANALYSIS
# ============================================================

residuals = y_test - y_pred


print("\nResidual statistics:")

print(
    pd.Series(residuals).describe()
)


# ============================================================
# 9. RESIDUAL PLOT
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    y_pred,
    residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted G3")
plt.ylabel("Residual")
plt.title(
    "Residual Analysis - " + best_model_name
)

plt.tight_layout()

plt.savefig(
    "eda_graphs/14_residual_analysis.png"
)

plt.close()


# ============================================================
# 10. ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title(
    "Actual vs Predicted - " + best_model_name
)

plt.tight_layout()

plt.savefig(
    "eda_graphs/15_actual_vs_predicted.png"
)

plt.close()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("REGRESSION ANALYSIS COMPLETED")
print("=" * 60)

print("New graphs saved in:")
print("eda_graphs/")