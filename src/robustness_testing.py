import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load original test data
# --------------------------------------------------

X_test = pd.read_csv("data/X_test_processed.csv")
y_test = pd.read_csv("data/y_reg_test.csv").squeeze()

print("=" * 60)
print("MODEL ROBUSTNESS TESTING")
print("=" * 60)

print("Test data shape:", X_test.shape)


# --------------------------------------------------
# 2. Load tuned model
# --------------------------------------------------

model = joblib.load(
    "models/tuned_random_forest_regression.pkl"
)

print("Tuned Random Forest model loaded successfully!")


# --------------------------------------------------
# 3. Normal prediction
# --------------------------------------------------

normal_predictions = model.predict(X_test)

normal_mae = mean_absolute_error(
    y_test,
    normal_predictions
)

normal_rmse = mean_squared_error(
    y_test,
    normal_predictions
) ** 0.5

normal_r2 = r2_score(
    y_test,
    normal_predictions
)


print()
print("=" * 60)
print("1. NORMAL INPUT")
print("=" * 60)

print(f"MAE  : {normal_mae:.4f}")
print(f"RMSE : {normal_rmse:.4f}")
print(f"R²   : {normal_r2:.4f}")


# --------------------------------------------------
# 4. Perturbed input
# --------------------------------------------------

X_perturbed = X_test.copy()

# Change one numerical processed feature slightly
feature_to_change = X_perturbed.columns[0]

X_perturbed[feature_to_change] = (
    X_perturbed[feature_to_change] + 0.5
)

perturbed_predictions = model.predict(
    X_perturbed
)

perturbed_mae = mean_absolute_error(
    y_test,
    perturbed_predictions
)

perturbed_rmse = mean_squared_error(
    y_test,
    perturbed_predictions
) ** 0.5

perturbed_r2 = r2_score(
    y_test,
    perturbed_predictions
)


print()
print("=" * 60)
print("2. PERTURBED INPUT")
print("=" * 60)

print("Changed feature:", feature_to_change)

print(f"MAE  : {perturbed_mae:.4f}")
print(f"RMSE : {perturbed_rmse:.4f}")
print(f"R²   : {perturbed_r2:.4f}")


# --------------------------------------------------
# 5. Missing input
# --------------------------------------------------

X_missing = X_test.copy()

missing_feature = X_missing.columns[0]

X_missing[missing_feature] = np.nan


print()
print("=" * 60)
print("3. MISSING INPUT")
print("=" * 60)

print("Missing feature:", missing_feature)

try:

    missing_predictions = model.predict(
        X_missing
    )

    missing_mae = mean_absolute_error(
        y_test,
        missing_predictions
    )

    missing_rmse = mean_squared_error(
        y_test,
        missing_predictions
    ) ** 0.5

    missing_r2 = r2_score(
        y_test,
        missing_predictions
    )

    print("Model accepted missing input.")

    print(f"MAE  : {missing_mae:.4f}")
    print(f"RMSE : {missing_rmse:.4f}")
    print(f"R²   : {missing_r2:.4f}")

except Exception as error:

    print(
        "Model rejected missing input directly."
    )

    print("Reason:", error)

    print()
    print(
        "This is expected because the saved model "
        "expects already-processed data."
    )


# --------------------------------------------------
# 6. Compare predictions
# --------------------------------------------------

comparison = pd.DataFrame({
    "Actual": y_test,
    "Normal_Prediction": normal_predictions,
    "Perturbed_Prediction": perturbed_predictions
})

if "missing_predictions" in locals():

    comparison["Missing_Prediction"] = (
        missing_predictions
    )


comparison.to_csv(
    "data/robustness_prediction_comparison.csv",
    index=False
)


# --------------------------------------------------
# 7. Calculate prediction changes
# --------------------------------------------------

prediction_difference = (
    np.abs(
        perturbed_predictions -
        normal_predictions
    )
)

print()
print("=" * 60)
print("PREDICTION SENSITIVITY")
print("=" * 60)

print(
    "Average prediction change after perturbation:",
    f"{prediction_difference.mean():.4f}"
)

print(
    "Maximum prediction change:",
    f"{prediction_difference.max():.4f}"
)


# --------------------------------------------------
# 8. Save robustness results
# --------------------------------------------------

results = {
    "Normal_MAE": normal_mae,
    "Normal_RMSE": normal_rmse,
    "Normal_R2": normal_r2,
    "Perturbed_MAE": perturbed_mae,
    "Perturbed_RMSE": perturbed_rmse,
    "Perturbed_R2": perturbed_r2,
    "Average_Prediction_Change":
        prediction_difference.mean(),
    "Maximum_Prediction_Change":
        prediction_difference.max()
}


if "missing_predictions" in locals():

    results["Missing_MAE"] = missing_mae
    results["Missing_RMSE"] = missing_rmse
    results["Missing_R2"] = missing_r2


robustness_results = pd.DataFrame(
    [results]
)

robustness_results.to_csv(
    "data/robustness_results.csv",
    index=False
)


# --------------------------------------------------
# 9. Final message
# --------------------------------------------------

print()
print("=" * 60)
print("ROBUSTNESS RESULTS SAVED")
print("=" * 60)

print("data/robustness_results.csv")
print("data/robustness_prediction_comparison.csv")

print()
print("=" * 60)
print("STEP 19 COMPLETED")
print("=" * 60)