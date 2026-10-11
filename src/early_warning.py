import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# Load trained model and preprocessor
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

preprocessor_path = BASE_DIR / "models" / "preprocessor.pkl"
model_path = BASE_DIR / "models" / "tuned_random_forest_regression.pkl"

preprocessor = joblib.load(preprocessor_path)
model = joblib.load(model_path)


# --------------------------------------------------
# Risk classification
# --------------------------------------------------

def get_risk_level(predicted_grade):

    if predicted_grade < 8:
        return "High Risk"

    elif predicted_grade <= 10:
        return "Medium Risk"

    else:
        return "Low Risk"


# --------------------------------------------------
# Predict student performance
# --------------------------------------------------

def predict_student(student_data):

    # Convert input into DataFrame
    student_df = pd.DataFrame([student_data])

    # Apply same preprocessing used during training
    processed_data = preprocessor.transform(student_df)

    # Predict final grade
    prediction = model.predict(processed_data)[0]

    # Keep grade between 0 and 20
    predicted_grade = max(0, min(20, prediction))

    # Determine risk
    risk_level = get_risk_level(predicted_grade)

    return predicted_grade, risk_level


# --------------------------------------------------
# Example student
# --------------------------------------------------

if __name__ == "__main__":

    student = {
        "school": "GP",
        "sex": "F",
        "age": 17,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "A",
        "Medu": 2,
        "Fedu": 2,
        "Mjob": "other",
        "Fjob": "other",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 2,
        "studytime": 2,
        "failures": 1,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 3,
        "Dalc": 1,
        "Walc": 1,
        "health": 3,
        "absences": 5
    }

    grade, risk = predict_student(student)

    print("\n====================================")
    print("      EARLY WARNING SYSTEM")
    print("====================================")
    print(f"Predicted Final Grade : {grade:.2f}")
    print(f"Risk Level            : {risk}")
    print("====================================")