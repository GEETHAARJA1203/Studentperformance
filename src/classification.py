import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/feature_engineered_data.csv"
)

print("=" * 60)
print("CLASSIFICATION MODEL")
print("=" * 60)

print("Dataset shape:", df.shape)


# ============================================================
# 2. DEFINE TARGET
# ============================================================

y = df["support_category"]


# ============================================================
# 3. DEFINE FEATURES
# ============================================================

columns_to_remove = [
    "G3",
    "support_category",
    "G1",
    "G2",
    "previous_grade_average",
    "grade_trend"
]

X = df.drop(
    columns=columns_to_remove
)


# ============================================================
# 4. DISPLAY CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

print(
    y.value_counts()
)


# ============================================================
# 5. IDENTIFY COLUMN TYPES
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


# ============================================================
# 6. PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# ============================================================
# 7. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 8. PREPROCESS DATA
# ============================================================

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


print("\nTraining shape:", X_train_processed.shape)
print("Testing shape:", X_test_processed.shape)


# ============================================================
# 9. CREATE CLASSIFICATION MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        )
}


# ============================================================
# 10. TRAIN AND EVALUATE
# ============================================================

results = []

os.makedirs(
    "models",
    exist_ok=True
)

for name, model in models.items():

    print("\n" + "-" * 60)

    print(
        "Training:",
        name
    )

    model.fit(
        X_train_processed,
        y_train
    )

    y_pred = model.predict(
        X_test_processed
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

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

    print(
        "Accuracy :",
        round(accuracy, 4)
    )

    print(
        "Precision:",
        round(precision, 4)
    )

    print(
        "Recall   :",
        round(recall, 4)
    )

    print(
        "F1 Score :",
        round(f1, 4)
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

    filename = (
        name.lower()
        .replace(" ", "_")
        + "_classification.pkl"
    )

    joblib.dump(
        model,
        "models/" + filename
    )


# ============================================================
# 11. COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n" + "=" * 60)
print("CLASSIFICATION MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "data/classification_model_comparison.csv",
    index=False
)

print("\nResults saved to:")

print(
    "data/classification_model_comparison.csv"
)


# ============================================================
# 13. SAVE PREPROCESSOR
# ============================================================

joblib.dump(
    preprocessor,
    "models/classification_preprocessor.pkl"
)

print(
    "\nClassification preprocessor saved:"
)

print(
    "models/classification_preprocessor.pkl"
)


print("\n" + "=" * 60)
print("CLASSIFICATION COMPLETED")
print("=" * 60)