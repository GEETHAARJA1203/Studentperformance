import pandas as pd
import joblib
import shap
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os


# --------------------------------------------------
# 1. Load original feature-engineered data
# --------------------------------------------------

df = pd.read_csv("data/feature_engineered_data.csv")


# --------------------------------------------------
# 2. Load processed test data
# --------------------------------------------------

X_test = pd.read_csv("data/X_test_processed.csv")
y_test = pd.read_csv("data/y_reg_test.csv").squeeze()

print("=" * 60)
print("EXPLAINABLE AI ANALYSIS")
print("=" * 60)

print("Test data shape:", X_test.shape)


# --------------------------------------------------
# 3. Load preprocessor
# --------------------------------------------------

preprocessor = joblib.load(
    "models/preprocessor.pkl"
)

print("Preprocessor loaded successfully!")


# --------------------------------------------------
# 4. Load tuned Random Forest model
# --------------------------------------------------

model = joblib.load(
    "models/tuned_random_forest_regression.pkl"
)

print("Tuned Random Forest model loaded successfully!")


# --------------------------------------------------
# 5. Get actual processed feature names
# --------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

feature_names = [
    name.replace("num__", "")
         .replace("cat__", "")
    for name in feature_names
]

print()
print("Number of processed features:", len(feature_names))


# --------------------------------------------------
# 6. Assign meaningful feature names
# --------------------------------------------------

X_test_named = X_test.copy()

X_test_named.columns = feature_names


# --------------------------------------------------
# 7. Create SHAP Explainer
# --------------------------------------------------

print()
print("Creating SHAP explanations...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(
    X_test_named
)

print("SHAP values calculated successfully!")


# --------------------------------------------------
# 8. Create output folder
# --------------------------------------------------

os.makedirs(
    "eda_graphs",
    exist_ok=True
)


# --------------------------------------------------
# 9. Global Feature Importance
# --------------------------------------------------

print()
print("Creating global feature importance plot...")

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_named,
    plot_type="bar",
    show=False
)

plt.title(
    "Global Feature Importance - SHAP"
)

plt.tight_layout()

plt.savefig(
    "eda_graphs/20_shap_global_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Global importance plot saved!"
)


# --------------------------------------------------
# 10. SHAP Summary Plot
# --------------------------------------------------

print()
print("Creating SHAP summary plot...")

shap.summary_plot(
    shap_values,
    X_test_named,
    show=False
)

plt.title(
    "SHAP Summary Plot"
)

plt.tight_layout()

plt.savefig(
    "eda_graphs/21_shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "SHAP summary plot saved!"
)


# --------------------------------------------------
# 11. Individual Student Explanation
# --------------------------------------------------

student_index = 0

student_features = X_test_named.iloc[
    [student_index]
]

student_shap_values = shap_values[
    student_index
]


print()
print("=" * 60)
print("INDIVIDUAL STUDENT EXPLANATION")
print("=" * 60)

print(
    "Student test index:",
    student_index
)


prediction = model.predict(
    X_test.iloc[[student_index]]
)[0]

print(
    f"Predicted Final Grade: {prediction:.2f}"
)


# --------------------------------------------------
# 12. Create individual explanation table
# --------------------------------------------------

individual_explanation = pd.DataFrame({
    "Feature": feature_names,
    "SHAP_Value": student_shap_values,
    "Feature_Value": student_features.iloc[0].values
})

individual_explanation[
    "Absolute_SHAP"
] = individual_explanation[
    "SHAP_Value"
].abs()


individual_explanation = (
    individual_explanation
    .sort_values(
        "Absolute_SHAP",
        ascending=False
    )
)


# --------------------------------------------------
# 13. Display top factors
# --------------------------------------------------

print()
print(
    "Top factors influencing this prediction:"
)

print(
    individual_explanation[
        [
            "Feature",
            "SHAP_Value",
            "Feature_Value"
        ]
    ].head(10)
)


# --------------------------------------------------
# 14. Save explanation
# --------------------------------------------------

individual_explanation.to_csv(
    "data/individual_student_explanation.csv",
    index=False
)

print()
print(
    "Individual explanation saved!"
)


# --------------------------------------------------
# 15. Individual explanation graph
# --------------------------------------------------

top_features = (
    individual_explanation
    .head(10)
    .sort_values("SHAP_Value")
)


plt.figure(
    figsize=(9, 6)
)

plt.barh(
    top_features["Feature"],
    top_features["SHAP_Value"]
)

plt.xlabel(
    "SHAP Value"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Individual Student Prediction Explanation"
)

plt.tight_layout()

plt.savefig(
    "eda_graphs/22_individual_student_explanation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Individual explanation graph saved!"
)


# --------------------------------------------------
# 16. Final message
# --------------------------------------------------

print()
print("=" * 60)
print("STEP 18 COMPLETED")
print("=" * 60)