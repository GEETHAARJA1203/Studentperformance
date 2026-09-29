import pandas as pd
import os


# --------------------------------------------------
# 1. Load student data
# --------------------------------------------------

df = pd.read_csv("data/feature_engineered_data.csv")

print("=" * 60)
print("PERSONALIZED LEARNING RECOMMENDATION SYSTEM")
print("=" * 60)


# --------------------------------------------------
# 2. Recommendation function
# --------------------------------------------------

def generate_recommendations(student):

    recommendations = []

    # Study time
    if student["studytime"] <= 1:
        recommendations.append(
            "Increase regular study time and follow a daily study schedule."
        )
    elif student["studytime"] == 2:
        recommendations.append(
            "Maintain a regular study routine and gradually increase focused study time."
        )

    # Previous failures
    if student["failures"] > 0:
        recommendations.append(
            "Review previously difficult subjects and practice basic concepts regularly."
        )

    # Absences
    if student["absences"] > 10:
        recommendations.append(
            "Reduce unnecessary absences and review missed classroom topics."
        )

    # School support
    if student["schoolsup"] == "yes":
        recommendations.append(
            "Make effective use of available school support and ask teachers for clarification."
        )

    # Family support
    if student["famsup"] == "yes":
        recommendations.append(
            "Continue using family support to maintain a consistent learning routine."
        )

    # Free time
    if student["freetime"] >= 4:
        recommendations.append(
            "Balance free-time activities with academic study and assignment completion."
        )

    # Going out
    if student["goout"] >= 4:
        recommendations.append(
            "Maintain a balanced schedule between social activities and study time."
        )

    # Alcohol-related variables
    if student["Dalc"] >= 3 or student["Walc"] >= 3:
        recommendations.append(
            "Maintain healthy daily habits and avoid activities that may interfere with study."
        )

    # Activities
    if student["activities"] == "no":
        recommendations.append(
            "Consider participating in suitable extracurricular activities for balanced development."
        )

    # Higher education goal
    if student["higher"] == "yes":
        recommendations.append(
            "Set clear academic goals and prepare consistently for higher education."
        )

    # Internet access
    if student["internet"] == "yes":
        recommendations.append(
            "Use reliable online educational resources for revision and practice."
        )

    # Default recommendation
    if len(recommendations) == 0:
        recommendations.append(
            "Continue the current learning routine and regularly monitor academic progress."
        )

    return recommendations


# --------------------------------------------------
# 3. Generate recommendations for all students
# --------------------------------------------------

all_recommendations = []

for index, student in df.iterrows():

    recommendations = generate_recommendations(student)

    all_recommendations.append(
        " | ".join(recommendations)
    )


df["learning_recommendations"] = all_recommendations


# --------------------------------------------------
# 4. Save recommendation data
# --------------------------------------------------

os.makedirs("data", exist_ok=True)

df.to_csv(
    "data/student_recommendations.csv",
    index=False
)


# --------------------------------------------------
# 5. Display one example
# --------------------------------------------------

student_index = 0

print()
print("=" * 60)
print("EXAMPLE STUDENT")
print("=" * 60)

print("Student index:", student_index)

print()
print("Support Category:")
print(df.loc[student_index, "support_category"])

print()
print("Personalized Recommendations:")

recommendations = generate_recommendations(
    df.loc[student_index]
)

for number, recommendation in enumerate(
    recommendations,
    start=1
):
    print(f"{number}. {recommendation}")


# --------------------------------------------------
# 6. Final message
# --------------------------------------------------

print()
print("=" * 60)
print("RECOMMENDATION DATA SAVED")
print("=" * 60)

print("File:")
print("data/student_recommendations.csv")

print()
print("STEP 17 COMPLETED")
print("=" * 60)