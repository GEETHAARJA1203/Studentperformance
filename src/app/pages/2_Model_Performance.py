import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Model Performance Dashboard",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📊 Model Performance Dashboard")

st.write(
    "This dashboard compares the machine-learning models "
    "used for student performance prediction and support "
    "classification."
)

st.divider()


# =========================================================
# LOAD REGRESSION RESULTS
# =========================================================

try:

    regression_results = pd.read_csv(
        "data/regression_model_comparison.csv"
    )

except Exception as e:

    regression_results = None

    st.error("Regression results file could not be loaded.")

    st.write(e)


# =========================================================
# LOAD CLASSIFICATION RESULTS
# =========================================================

try:

    classification_results = pd.read_csv(
        "data/classification_model_comparison.csv"
    )

except Exception as e:

    classification_results = None

    st.error(
        "Classification results file could not be loaded."
    )

    st.write(e)


# =========================================================
# REGRESSION SECTION
# =========================================================

st.header("📈 Regression Model Performance")

st.write(
    "Regression predicts the student's final numerical grade."
)


if regression_results is not None:

    # -----------------------------------------------------
    # DISPLAY DATA
    # -----------------------------------------------------

    st.subheader("Regression Model Comparison")

    st.dataframe(
        regression_results,
        use_container_width=True,
        hide_index=True
    )


    # -----------------------------------------------------
    # METRIC CARDS
    # -----------------------------------------------------

    metric_columns = regression_results.columns.tolist()


    if "RMSE" in metric_columns:

        best_rmse_row = regression_results.loc[
            regression_results["RMSE"].idxmin()
        ]

        best_rmse_model = best_rmse_row["Model"]

        best_rmse = best_rmse_row["RMSE"]

    else:

        best_rmse_model = "Not available"
        best_rmse = 0


    if "MAE" in metric_columns:

        best_mae_row = regression_results.loc[
            regression_results["MAE"].idxmin()
        ]

        best_mae_model = best_mae_row["Model"]

        best_mae = best_mae_row["MAE"]

    else:

        best_mae_model = "Not available"
        best_mae = 0


    if "R2" in metric_columns:

        best_r2_row = regression_results.loc[
            regression_results["R2"].idxmax()
        ]

        best_r2_model = best_r2_row["Model"]

        best_r2 = best_r2_row["R2"]

    elif "R²" in metric_columns:

        best_r2_row = regression_results.loc[
            regression_results["R²"].idxmax()
        ]

        best_r2_model = best_r2_row["Model"]

        best_r2 = best_r2_row["R²"]

    else:

        best_r2_model = "Not available"
        best_r2 = 0


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Lowest MAE",
            f"{best_mae:.3f}"
        )

        st.caption(
            f"Model: {best_mae_model}"
        )


    with col2:

        st.metric(
            "Lowest RMSE",
            f"{best_rmse:.3f}"
        )

        st.caption(
            f"Model: {best_rmse_model}"
        )


    with col3:

        st.metric(
            "Highest R²",
            f"{best_r2:.3f}"
        )

        st.caption(
            f"Model: {best_r2_model}"
        )


    # -----------------------------------------------------
    # MAE CHART
    # -----------------------------------------------------

    if "MAE" in regression_results.columns:

        st.subheader("MAE Comparison")

        fig, ax = plt.subplots()

        ax.bar(
            regression_results["Model"],
            regression_results["MAE"]
        )

        ax.set_xlabel("Model")

        ax.set_ylabel("MAE")

        ax.set_title(
            "Regression Models - MAE"
        )

        plt.xticks(rotation=30)

        plt.tight_layout()

        st.pyplot(fig)


    # -----------------------------------------------------
    # RMSE CHART
    # -----------------------------------------------------

    if "RMSE" in regression_results.columns:

        st.subheader("RMSE Comparison")

        fig, ax = plt.subplots()

        ax.bar(
            regression_results["Model"],
            regression_results["RMSE"]
        )

        ax.set_xlabel("Model")

        ax.set_ylabel("RMSE")

        ax.set_title(
            "Regression Models - RMSE"
        )

        plt.xticks(rotation=30)

        plt.tight_layout()

        st.pyplot(fig)


    # -----------------------------------------------------
    # R2 CHART
    # -----------------------------------------------------

    r2_column = None

    if "R2" in regression_results.columns:

        r2_column = "R2"

    elif "R²" in regression_results.columns:

        r2_column = "R²"


    if r2_column is not None:

        st.subheader("R² Comparison")

        fig, ax = plt.subplots()

        ax.bar(
            regression_results["Model"],
            regression_results[r2_column]
        )

        ax.set_xlabel("Model")

        ax.set_ylabel("R²")

        ax.set_title(
            "Regression Models - R²"
        )

        plt.xticks(rotation=30)

        plt.tight_layout()

        st.pyplot(fig)


else:

    st.warning(
        "Regression model comparison data is not available."
    )


# =========================================================
# CLASSIFICATION SECTION
# =========================================================

st.divider()

st.header("🎯 Classification Model Performance")

st.write(
    "Classification identifies the student's support category."
)


if classification_results is not None:

    # -----------------------------------------------------
    # DISPLAY DATA
    # -----------------------------------------------------

    st.subheader("Classification Model Comparison")

    st.dataframe(
        classification_results,
        use_container_width=True,
        hide_index=True
    )


    # -----------------------------------------------------
    # FIND F1 COLUMN
    # -----------------------------------------------------

    f1_column = None

    for column in classification_results.columns:

        if column.lower() == "f1":

            f1_column = column

        elif "f1" in column.lower():

            f1_column = column


    accuracy_column = None

    for column in classification_results.columns:

        if column.lower() == "accuracy":

            accuracy_column = column


    precision_column = None

    for column in classification_results.columns:

        if "precision" in column.lower():

            precision_column = column


    recall_column = None

    for column in classification_results.columns:

        if "recall" in column.lower():

            recall_column = column


    # -----------------------------------------------------
    # METRIC CARDS
    # -----------------------------------------------------

    if f1_column is not None:

        best_f1_row = classification_results.loc[
            classification_results[f1_column].idxmax()
        ]

        best_f1_model = best_f1_row["Model"]

        best_f1 = best_f1_row[f1_column]

    else:

        best_f1_model = "Not available"
        best_f1 = 0


    if accuracy_column is not None:

        best_accuracy_row = classification_results.loc[
            classification_results[accuracy_column].idxmax()
        ]

        best_accuracy_model = best_accuracy_row["Model"]

        best_accuracy = best_accuracy_row[accuracy_column]

    else:

        best_accuracy_model = "Not available"
        best_accuracy = 0


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Highest Accuracy",
            f"{best_accuracy:.3f}"
        )

        st.caption(
            f"Model: {best_accuracy_model}"
        )


    with col2:

        st.metric(
            "Highest F1 Score",
            f"{best_f1:.3f}"
        )

        st.caption(
            f"Model: {best_f1_model}"
        )


    # -----------------------------------------------------
    # F1 CHART
    # -----------------------------------------------------

    if f1_column is not None:

        st.subheader("F1 Score Comparison")

        fig, ax = plt.subplots()

        ax.bar(
            classification_results["Model"],
            classification_results[f1_column]
        )

        ax.set_xlabel("Model")

        ax.set_ylabel("F1 Score")

        ax.set_title(
            "Classification Models - F1 Score"
        )

        plt.xticks(rotation=30)

        plt.tight_layout()

        st.pyplot(fig)


    # -----------------------------------------------------
    # ACCURACY CHART
    # -----------------------------------------------------

    if accuracy_column is not None:

        st.subheader("Accuracy Comparison")

        fig, ax = plt.subplots()

        ax.bar(
            classification_results["Model"],
            classification_results[accuracy_column]
        )

        ax.set_xlabel("Model")

        ax.set_ylabel("Accuracy")

        ax.set_title(
            "Classification Models - Accuracy"
        )

        plt.xticks(rotation=30)

        plt.tight_layout()

        st.pyplot(fig)


    # -----------------------------------------------------
    # PRECISION / RECALL
    # -----------------------------------------------------

    if (
        precision_column is not None
        and recall_column is not None
    ):

        st.subheader(
            "Precision and Recall Comparison"
        )

        fig, ax = plt.subplots()

        x = range(len(classification_results))

        width = 0.35

        ax.bar(
            [
                i - width / 2
                for i in x
            ],
            classification_results[precision_column],
            width=width,
            label="Precision"
        )

        ax.bar(
            [
                i + width / 2
                for i in x
            ],
            classification_results[recall_column],
            width=width,
            label="Recall"
        )

        ax.set_xticks(list(x))

        ax.set_xticklabels(
            classification_results["Model"],
            rotation=30
        )

        ax.set_ylabel("Score")

        ax.set_title(
            "Classification Models - Precision vs Recall"
        )

        ax.legend()

        plt.tight_layout()

        st.pyplot(fig)


else:

    st.warning(
        "Classification model comparison data is not available."
    )


# =========================================================
# EXPLANATION
# =========================================================

st.divider()

st.header("ℹ️ How to Interpret the Metrics")

st.write(
    """
    **MAE:** Average absolute difference between the actual
    and predicted grades. Lower is better.

    **RMSE:** Measures prediction error and gives more weight
    to larger errors. Lower is better.

    **R²:** Indicates how much variation in the target is
    explained by the regression model. Higher is generally
    better.

    **Accuracy:** Percentage of correctly classified students.

    **Precision:** Measures how accurately the model identifies
    a predicted class.

    **Recall:** Measures how many relevant students were
    correctly identified.

    **F1 Score:** Combines precision and recall into one metric.
    Higher values indicate better balance between them.
    """
)