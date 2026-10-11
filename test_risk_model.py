
import joblib
import pandas as pd

bundle = joblib.load("models/risk_prediction_model.pkl")

model = bundle["model"]
features = bundle["features"]

sample = {
    "Department": ["CSE"],
    "Semester": [3],
    "Attendance": [80],
    "Internal_Marks": [70],
    "Assignment_Marks": [75],
    "Quiz_Marks": [70],
    "Study_Hours": [3],
    "Previous_CGPA": [7.5],
    "Backlogs": [0],
    "Assignment_Submission_Rate": [90],
    "Learning_Activity": [1],
}

X = pd.DataFrame(sample).reindex(columns=features)

print("Input shape:", X.shape)

# The model pipeline handles preprocessing internally
prediction = model.predict(X)

print("Risk prediction:", prediction[0])
