import pandas as pd

# Data dictionary for Student Performance Dataset

data_dictionary = [
    ["school", "Categorical", "Student's school", "GP or MS", "Feature"],
    ["sex", "Categorical", "Student's sex", "F or M", "Feature"],
    ["age", "Numerical", "Student's age", "15-22", "Feature"],
    ["address", "Categorical", "Home address type", "U=Urban, R=Rural", "Feature"],
    ["famsize", "Categorical", "Family size", "LE3 or GT3", "Feature"],
    ["Pstatus", "Categorical", "Parents' cohabitation status", "T=Together, A=Apart", "Feature"],
    ["Medu", "Numerical", "Mother's education level", "0-4", "Feature"],
    ["Fedu", "Numerical", "Father's education level", "0-4", "Feature"],
    ["Mjob", "Categorical", "Mother's job", "teacher, health, services, at_home, other", "Feature"],
    ["Fjob", "Categorical", "Father's job", "teacher, health, services, at_home, other", "Feature"],
    ["reason", "Categorical", "Reason for choosing the school", "home, reputation, course, other", "Feature"],
    ["guardian", "Categorical", "Student's guardian", "mother, father, other", "Feature"],
    ["traveltime", "Numerical", "Travel time from home to school", "1-4", "Feature"],
    ["studytime", "Numerical", "Weekly study time", "1-4", "Feature"],
    ["failures", "Numerical", "Number of past class failures", "0, 1, 2, 3+", "Feature"],
    ["schoolsup", "Categorical", "Extra educational support from school", "yes/no", "Feature"],
    ["famsup", "Categorical", "Extra educational support from family", "yes/no", "Feature"],
    ["paid", "Categorical", "Extra paid classes in the subject", "yes/no", "Feature"],
    ["activities", "Categorical", "Participation in extracurricular activities", "yes/no", "Feature"],
    ["nursery", "Categorical", "Attended nursery school", "yes/no", "Feature"],
    ["higher", "Categorical", "Wants to take higher education", "yes/no", "Feature"],
    ["internet", "Categorical", "Internet access at home", "yes/no", "Feature"],
    ["romantic", "Categorical", "Has a romantic relationship", "yes/no", "Feature"],
    ["famrel", "Numerical", "Quality of family relationships", "1-5", "Feature"],
    ["freetime", "Numerical", "Free time after school", "1-5", "Feature"],
    ["goout", "Numerical", "Going out with friends", "1-5", "Feature"],
    ["Dalc", "Numerical", "Workday alcohol consumption", "1-5", "Feature"],
    ["Walc", "Numerical", "Weekend alcohol consumption", "1-5", "Feature"],
    ["health", "Numerical", "Current health status", "1-5", "Feature"],
    ["absences", "Numerical", "Number of school absences", "0-93", "Feature"],
    ["G1", "Numerical", "First period grade", "0-20", "Intermediate Grade"],
    ["G2", "Numerical", "Second period grade", "0-20", "Intermediate Grade"],
    ["G3", "Numerical", "Final grade", "0-20", "Target"],
]

df_dictionary = pd.DataFrame(
    data_dictionary,
    columns=["Feature", "Type", "Meaning", "Allowed / Range", "Role"]
)

print("=" * 70)
print("DATA DICTIONARY")
print("=" * 70)

print(df_dictionary.to_string(index=False))

# Save the data dictionary
df_dictionary.to_csv("data/data_dictionary.csv", index=False)

print("\nData dictionary saved successfully!")
print("File: data/data_dictionary.csv")