import pandas as pd

# ============================================================
# LOAD CLEANED DATA
# ============================================================

df = pd.read_csv("data/cleaned_data.csv")

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

print("Original shape:", df.shape)


# ============================================================
# 1. PREVIOUS GRADE AVERAGE
# ============================================================

df["previous_grade_average"] = (df["G1"] + df["G2"]) / 2


# ============================================================
# 2. GRADE TREND
# ============================================================

df["grade_trend"] = df["G2"] - df["G1"]


# ============================================================
# 3. SUPPORT CATEGORY
# ============================================================

def classify_support(score):

    if score < 10:
        return "High Support"

    elif score < 15:
        return "Medium Support"

    else:
        return "Low Support"


df["support_category"] = df["G3"].apply(classify_support)


# ============================================================
# 4. DISPLAY NEW FEATURES
# ============================================================

print("\nNew features created:")

print(
    df[
        [
            "G1",
            "G2",
            "G3",
            "previous_grade_average",
            "grade_trend",
            "support_category"
        ]
    ].head(10)
)


# ============================================================
# 5. SUPPORT CATEGORY DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("SUPPORT CATEGORY DISTRIBUTION")
print("=" * 60)

print(df["support_category"].value_counts())


# ============================================================
# 6. SAVE FEATURE-ENGINEERED DATA
# ============================================================

df.to_csv(
    "data/feature_engineered_data.csv",
    index=False
)

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print("Saved file:")
print("data/feature_engineered_data.csv")