import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/feature_engineered_data.csv"
)

X = df.drop(
    columns=[
        "G3",
        "support_category",
        "G1",
        "G2",
        "previous_grade_average",
        "grade_trend"
    ]
)

y = df["support_category"]


# ============================================================
# 2. TRAIN TEST SPLIT
# ============================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 3. LOAD PREPROCESSOR
# ============================================================

preprocessor = joblib.load(
    "models/classification_preprocessor.pkl"
)

X_test_processed = preprocessor.transform(
    X_test
)


# ============================================================
# 4. CREATE GRAPH FOLDER
# ============================================================

os.makedirs(
    "eda_graphs",
    exist_ok=True
)


# ============================================================
# 5. MODEL NAMES
# ============================================================

models = {

    "Logistic Regression":
        "models/logistic_regression_classification.pkl",

    "Decision Tree":
        "models/decision_tree_classification.pkl",

    "Random Forest":
        "models/random_forest_classification.pkl",

    "Gradient Boosting":
        "models/gradient_boosting_classification.pkl",

    "KNN":
        "models/knn_classification.pkl"
}


# ============================================================
# 6. LOAD RESULTS
# ============================================================

comparison = pd.read_csv(
    "data/classification_model_comparison.csv"
)

print("=" * 60)
print("CLASSIFICATION ERROR ANALYSIS")
print("=" * 60)

print("\nModel comparison:")
print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# 7. SELECT MODEL WITH HIGHEST F1
# ============================================================

best_model_name = comparison.loc[
    comparison["F1"].idxmax(),
    "Model"
]

print(
    "\nModel with highest F1 Score:"
)

print(
    best_model_name
)


# ============================================================
# 8. LOAD BEST MODEL
# ============================================================

best_model = joblib.load(
    models[best_model_name]
)

y_pred = best_model.predict(
    X_test_processed
)


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

labels = [
    "High Support",
    "Medium Support",
    "Low Support"
]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)


print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 11. CONFUSION MATRIX GRAPH
# ============================================================

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel(
    "Predicted Category"
)

plt.ylabel(
    "Actual Category"
)

plt.title(
    "Confusion Matrix - " +
    best_model_name
)

plt.tight_layout()

plt.savefig(
    "eda_graphs/16_classification_confusion_matrix.png"
)

plt.close()


# ============================================================
# 12. F1 COMPARISON
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.bar(
    comparison["Model"],
    comparison["F1"]
)

plt.xlabel("Model")
plt.ylabel("F1 Score")

plt.title(
    "Classification Models - F1 Score"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.savefig(
    "eda_graphs/17_classification_f1_comparison.png"
)

plt.close()


# ============================================================
# 13. ACCURACY COMPARISON
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.bar(
    comparison["Model"],
    comparison["Accuracy"]
)

plt.xlabel("Model")
plt.ylabel("Accuracy")

plt.title(
    "Classification Models - Accuracy"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.savefig(
    "eda_graphs/18_classification_accuracy_comparison.png"
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)

print(
    "CLASSIFICATION ERROR ANALYSIS COMPLETED"
)

print("=" * 60)

print(
    "\nGraphs saved in:"
)

print(
    "eda_graphs/"
)