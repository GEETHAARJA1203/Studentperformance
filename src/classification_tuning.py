import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# --------------------------------------------------
# 1. Load processed classification data
# --------------------------------------------------

X_train = pd.read_csv("data/X_train_processed.csv")
X_test = pd.read_csv("data/X_test_processed.csv")

y_train = pd.read_csv("data/y_cls_train.csv").squeeze()
y_test = pd.read_csv("data/y_cls_test.csv").squeeze()

print("=" * 60)
print("RANDOM FOREST CLASSIFICATION HYPERPARAMETER TUNING")
print("=" * 60)

print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)


# --------------------------------------------------
# 2. Define Random Forest Classifier
# --------------------------------------------------

rf = RandomForestClassifier(
    random_state=42
)


# --------------------------------------------------
# 3. Define parameters
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
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=1
)

print()
print("Starting Grid Search...")
print("Please wait...")

grid_search.fit(X_train, y_train)


# --------------------------------------------------
# 5. Best parameters
# --------------------------------------------------

print()
print("=" * 60)
print("BEST PARAMETERS")
print("=" * 60)

print(grid_search.best_params_)

print()
print("Best Cross-Validation F1 Score:")
print(grid_search.best_score_)


# --------------------------------------------------
# 6. Evaluate tuned model
# --------------------------------------------------

best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print()
print("=" * 60)
print("TUNED RANDOM FOREST CLASSIFICATION RESULTS")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# --------------------------------------------------
# 7. Detailed classification report
# --------------------------------------------------

print()
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))


# --------------------------------------------------
# 8. Save tuned model
# --------------------------------------------------

joblib.dump(
    best_rf,
    "models/tuned_random_forest_classification.pkl"
)

print("Tuned classification model saved successfully!")


# --------------------------------------------------
# 9. Save tuning results
# --------------------------------------------------

results = pd.DataFrame(
    grid_search.cv_results_
)

results.to_csv(
    "data/rf_classification_tuning_results.csv",
    index=False
)

print("Classification tuning results saved successfully!")

print()
print("=" * 60)
print("STEP 15 COMPLETED")
print("=" * 60)