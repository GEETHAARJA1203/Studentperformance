import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Admin Dashboard",
    page_icon="👨‍💼",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("👨‍💼 Admin Dashboard")

st.write(
    "This dashboard provides an overview of the student dataset, "
    "support categories, machine-learning models, and saved model files."
)

st.divider()


# =========================================================
# LOAD DATASET
# =========================================================

try:

    df = pd.read_csv(
        "data/feature_engineered_data.csv"
    )

except Exception as e:

    st.error("Unable to load the student dataset.")

    st.write(e)

    st.stop()


# =========================================================
# DATASET OVERVIEW
# =========================================================

st.header("📊 Dataset Overview")


total_students = len(df)

total_columns = len(df.columns)

missing_values = int(
    df.isnull().sum().sum()
)

duplicate_rows = int(
    df.duplicated().sum()
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Students",
        total_students
    )


with col2:

    st.metric(
        "Total Columns",
        total_columns
    )


with col3:

    st.metric(
        "Missing Values",
        missing_values
    )


with col4:

    st.metric(
        "Duplicate Rows",
        duplicate_rows
    )


st.divider()


# =========================================================
# STUDENT SUPPORT DISTRIBUTION
# =========================================================

st.header("🎯 Student Support Distribution")


if "support_category" in df.columns:

    support_counts = (
        df["support_category"]
        .value_counts()
        .reset_index()
    )

    support_counts.columns = [
        "Support Category",
        "Number of Students"
    ]


    # -----------------------------------------------------
    # TABLE
    # -----------------------------------------------------

    st.subheader("Support Category Summary")

    st.dataframe(
        support_counts,
        use_container_width=True,
        hide_index=True
    )


    # -----------------------------------------------------
    # MATPLOTLIB BAR CHART
    # -----------------------------------------------------

    st.subheader("Support Category Chart")

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.bar(
        support_counts["Support Category"],
        support_counts["Number of Students"]
    )

    ax.set_xlabel(
        "Support Category"
    )

    ax.set_ylabel(
        "Number of Students"
    )

    ax.set_title(
        "Student Support Distribution"
    )

    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


else:

    st.warning(
        "The support_category column was not found."
    )


st.divider()


# =========================================================
# DATASET DETAILS
# =========================================================

st.header("📚 Dataset Details")


col1, col2 = st.columns(2)


with col1:

    st.subheader("Dataset Shape")

    st.write(
        f"Rows: **{df.shape[0]}**"
    )

    st.write(
        f"Columns: **{df.shape[1]}**"
    )


with col2:

    st.subheader("Data Quality")

    st.write(
        f"Missing values: **{missing_values}**"
    )

    st.write(
        f"Duplicate rows: **{duplicate_rows}**"
    )


st.divider()


# =========================================================
# MACHINE LEARNING MODELS
# =========================================================

st.header("🤖 Machine Learning Models")


# ---------------------------------------------------------
# REGRESSION MODELS
# ---------------------------------------------------------

st.subheader("📈 Regression Models")


regression_models = [
    "Linear Regression",
    "Decision Tree Regressor",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
    "KNN Regressor"
]


for model in regression_models:

    st.write(
        "• " + model
    )


# ---------------------------------------------------------
# CLASSIFICATION MODELS
# ---------------------------------------------------------

st.subheader("🎯 Classification Models")


classification_models = [
    "Logistic Regression",
    "Decision Tree Classifier",
    "Random Forest Classifier",
    "Gradient Boosting Classifier",
    "KNN Classifier"
]


for model in classification_models:

    st.write(
        "• " + model
    )


st.divider()


# =========================================================
# MODEL ARTIFACT STATUS
# =========================================================

st.header("💾 Saved Model Status")


model_files = [

    "models/tuned_random_forest_regression.pkl",

    "models/tuned_random_forest_classification.pkl",

    "models/random_forest_regression.pkl",

    "models/random_forest_classification.pkl",

    "models/preprocessor.pkl",

    "models/classification_preprocessor.pkl",

    "models/kmeans.pkl"

]


for file_path in model_files:

    if os.path.exists(file_path):

        st.success(
            f"✅ Available: {file_path}"
        )

    else:

        st.warning(
            f"⚠️ Not found: {file_path}"
        )


st.divider()


# =========================================================
# CLUSTER INFORMATION
# =========================================================

st.header("🔬 Student Cluster Information")


cluster_file = "data/cluster_profiles.csv"


if os.path.exists(cluster_file):

    try:

        cluster_data = pd.read_csv(
            cluster_file
        )

        st.write(
            "The project groups students into behavioral/academic profiles "
            "using unsupervised learning."
        )

        st.dataframe(
            cluster_data,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:

        st.warning(
            "Cluster profile file could not be read."
        )

        st.write(e)

else:

    st.info(
        "Cluster profile data is not available."
    )


st.divider()


# =========================================================
# PROJECT SUMMARY
# =========================================================

st.header("📝 Project Summary")


st.info(
    """
    The Intelligent Student Performance Prediction and
    Personalized Learning Recommendation System uses
    machine-learning techniques to predict student academic
    performance, identify the level of support required,
    provide personalized learning recommendations, and
    explain predictions using Explainable AI (SHAP).
    """
)


st.divider()


# =========================================================
# ADMIN NOTE
# =========================================================

st.caption(
    "Admin Dashboard: Used for project monitoring, "
    "model overview, dataset analysis, and academic "
    "decision-support."
)