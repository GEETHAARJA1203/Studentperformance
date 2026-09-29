import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/cleaned_data.csv")

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("Dataset shape:", df.shape)


# Create folder for EDA graphs
os.makedirs("eda_graphs", exist_ok=True)


# ============================================================
# GRAPH 1 — FINAL GRADE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))
sns.histplot(df["G3"], bins=21, kde=True)
plt.title("Distribution of Final Grades (G3)")
plt.xlabel("Final Grade")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("eda_graphs/01_final_grade_distribution.png")
plt.close()

print("Graph 1 saved.")


# ============================================================
# GRAPH 2 — STUDY TIME VS FINAL GRADE
# ============================================================

plt.figure(figsize=(8, 5))
sns.boxplot(x="studytime", y="G3", data=df)
plt.title("Study Time vs Final Grade")
plt.xlabel("Weekly Study Time")
plt.ylabel("Final Grade")
plt.tight_layout()
plt.savefig("eda_graphs/02_studytime_vs_grade.png")
plt.close()

print("Graph 2 saved.")


# ============================================================
# GRAPH 3 — ABSENCES VS FINAL GRADE
# ============================================================

plt.figure(figsize=(8, 5))
sns.scatterplot(x="absences", y="G3", data=df)
plt.title("Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade")
plt.tight_layout()
plt.savefig("eda_graphs/03_absences_vs_grade.png")
plt.close()

print("Graph 3 saved.")


# ============================================================
# GRAPH 4 — FAILURES VS FINAL GRADE
# ============================================================

plt.figure(figsize=(8, 5))
sns.boxplot(x="failures", y="G3", data=df)
plt.title("Past Failures vs Final Grade")
plt.xlabel("Number of Past Failures")
plt.ylabel("Final Grade")
plt.tight_layout()
plt.savefig("eda_graphs/04_failures_vs_grade.png")
plt.close()

print("Graph 4 saved.")


# ============================================================
# GRAPH 5 — AGE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))
sns.countplot(x="age", data=df)
plt.title("Age Distribution of Students")
plt.xlabel("Age")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("eda_graphs/05_age_distribution.png")
plt.close()

print("Graph 5 saved.")


# ============================================================
# GRAPH 6 — GENDER VS FINAL GRADE
# ============================================================

plt.figure(figsize=(8, 5))
sns.boxplot(x="sex", y="G3", data=df)
plt.title("Gender vs Final Grade")
plt.xlabel("Sex")
plt.ylabel("Final Grade")
plt.tight_layout()
plt.savefig("eda_graphs/06_gender_vs_grade.png")
plt.close()

print("Graph 6 saved.")


# ============================================================
# GRAPH 7 — MOTHER'S EDUCATION VS FINAL GRADE
# ============================================================

plt.figure(figsize=(8, 5))
sns.boxplot(x="Medu", y="G3", data=df)
plt.title("Mother's Education vs Final Grade")
plt.xlabel("Mother's Education Level")
plt.ylabel("Final Grade")
plt.tight_layout()
plt.savefig("eda_graphs/07_mother_education_vs_grade.png")
plt.close()

print("Graph 7 saved.")


# ============================================================
# GRAPH 8 — CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(12, 8))

numeric_df = df.select_dtypes(include=["int64", "float64"])
correlation = numeric_df.corr()

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_graphs/08_correlation_heatmap.png")
plt.close()

print("Graph 8 saved.")


# ============================================================
# GRAPH 9 — SCHOOL VS FINAL GRADE
# ============================================================

plt.figure(figsize=(8, 5))
sns.boxplot(x="school", y="G3", data=df)
plt.title("School vs Final Grade")
plt.xlabel("School")
plt.ylabel("Final Grade")
plt.tight_layout()
plt.savefig("eda_graphs/09_school_vs_grade.png")
plt.close()

print("Graph 9 saved.")


# ============================================================
# GRAPH 10 — WEEKEND ALCOHOL VS FINAL GRADE
# ============================================================

plt.figure(figsize=(8, 5))
sns.boxplot(x="Walc", y="G3", data=df)
plt.title("Weekend Alcohol Consumption vs Final Grade")
plt.xlabel("Weekend Alcohol Consumption")
plt.ylabel("Final Grade")
plt.tight_layout()
plt.savefig("eda_graphs/10_weekend_alcohol_vs_grade.png")
plt.close()

print("Graph 10 saved.")


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("10 graphs have been saved in:")
print("eda_graphs/")