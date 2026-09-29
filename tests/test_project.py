import os
import pandas as pd
import joblib


# ============================================================
# TEST 1: DATASET EXISTS
# ============================================================

def test_dataset_exists():

    file_path = "data/student-mat.csv"

    assert os.path.exists(file_path), (
        "Dataset file does not exist."
    )


# ============================================================
# TEST 2: DATASET LOADS CORRECTLY
# ============================================================

def test_dataset_loads():

    file_path = "data/student-mat.csv"

    df = pd.read_csv(
        file_path,
        sep=";"
    )

    assert not df.empty, (
        "Dataset is empty."
    )

    assert len(df) > 0, (
        "Dataset contains no rows."
    )


# ============================================================
# TEST 3: REQUIRED COLUMNS EXIST
# ============================================================

def test_required_columns():

    file_path = "data/student-mat.csv"

    df = pd.read_csv(
        file_path,
        sep=";"
    )

    required_columns = [
        "school",
        "sex",
        "age",
        "studytime",
        "failures",
        "absences",
        "G3"
    ]

    for column in required_columns:

        assert column in df.columns, (
            f"Required column missing: {column}"
        )


# ============================================================
# TEST 4: NO MISSING VALUES
# ============================================================

def test_no_missing_values():

    file_path = "data/student-mat.csv"

    df = pd.read_csv(
        file_path,
        sep=";"
    )

    assert df.isnull().sum().sum() == 0, (
        "Dataset contains missing values."
    )


# ============================================================
# TEST 5: FINAL GRADE RANGE
# ============================================================

def test_final_grade_range():

    file_path = "data/student-mat.csv"

    df = pd.read_csv(
        file_path,
        sep=";"
    )

    assert df["G3"].min() >= 0, (
        "Final grade contains values below 0."
    )

    assert df["G3"].max() <= 20, (
        "Final grade contains values above 20."
    )


# ============================================================
# TEST 6: SUPPORT CATEGORY VALUES
# ============================================================

def test_support_categories():

    valid_categories = [
        "High Support",
        "Medium Support",
        "Low Support"
    ]

    grades = [5, 12, 18]

    categories = []

    for grade in grades:

        if grade < 10:

            categories.append(
                "High Support"
            )

        elif grade < 15:

            categories.append(
                "Medium Support"
            )

        else:

            categories.append(
                "Low Support"
            )

    for category in categories:

        assert category in valid_categories


# ============================================================
# TEST 7: REGRESSION MODEL EXISTS
# ============================================================

def test_regression_model_exists():

    model_path = (
        "models/tuned_random_forest_regression.pkl"
    )

    assert os.path.exists(model_path), (
        "Regression model file does not exist."
    )


# ============================================================
# TEST 8: CLASSIFICATION MODEL EXISTS
# ============================================================

def test_classification_model_exists():

    model_path = (
        "models/tuned_random_forest_classification.pkl"
    )

    assert os.path.exists(model_path), (
        "Classification model file does not exist."
    )


# ============================================================
# TEST 9: REGRESSION MODEL CAN LOAD
# ============================================================

def test_regression_model_loads():

    model_path = (
        "models/tuned_random_forest_regression.pkl"
    )

    model = joblib.load(
        model_path
    )

    assert model is not None


# ============================================================
# TEST 10: CLASSIFICATION MODEL CAN LOAD
# ============================================================

def test_classification_model_loads():

    model_path = (
        "models/tuned_random_forest_classification.pkl"
    )

    model = joblib.load(
        model_path
    )

    assert model is not None