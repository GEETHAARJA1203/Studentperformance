import streamlit as st
import pandas as pd
import joblib
import shap


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# LOAD MODELS AND PREPROCESSORS
# ============================================================

@st.cache_resource
def load_models():

    regression_model = joblib.load(
        "models/tuned_random_forest_regression.pkl"
    )

    classification_model = joblib.load(
        "models/tuned_random_forest_classification.pkl"
    )

    regression_preprocessor = joblib.load(
        "models/preprocessor.pkl"
    )

    classification_preprocessor = joblib.load(
        "models/classification_preprocessor.pkl"
    )

    return (
        regression_model,
        classification_model,
        regression_preprocessor,
        classification_preprocessor
    )


# Try loading models
try:

    (
        regression_model,
        classification_model,
        regression_preprocessor,
        classification_preprocessor
    ) = load_models()

except Exception as e:

    st.error("❌ Error loading the trained models.")

    st.write("Please make sure the following files exist:")

    st.write("""
    - models/tuned_random_forest_regression.pkl
    - models/tuned_random_forest_classification.pkl
    - models/preprocessor.pkl
    - models/classification_preprocessor.pkl
    """)

    st.error(f"Technical error: {e}")

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🎓 Intelligent Student Performance Prediction")
st.subheader(
    "AI/ML Based Student Performance Prediction and Personalized Learning Recommendation System"
)

st.write(
    "Enter the student's academic, demographic, family, and lifestyle information "
    "to predict the final grade and identify the level of support required."
)

st.divider()


# ============================================================
# INPUT FORM
# ============================================================

st.header("📝 Student Information")

with st.form("student_prediction_form"):

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # COLUMN 1
    # --------------------------------------------------------

    with col1:

        st.subheader("🏫 Basic Information")

        school = st.selectbox(
            "School",
            ["GP", "MS"]
        )

        sex = st.selectbox(
            "Gender",
            ["F", "M"]
        )

        age = st.number_input(
            "Age",
            min_value=15,
            max_value=25,
            value=17,
            step=1
        )

        address = st.selectbox(
            "Address",
            ["U", "R"]
        )

        famsize = st.selectbox(
            "Family Size",
            ["LE3", "GT3"]
        )

        Pstatus = st.selectbox(
            "Parent Cohabitation Status",
            ["T", "A"]
        )

        Medu = st.selectbox(
            "Mother's Education",
            [0, 1, 2, 3, 4]
        )

        Fedu = st.selectbox(
            "Father's Education",
            [0, 1, 2, 3, 4]
        )

        Mjob = st.selectbox(
            "Mother's Job",
            ["teacher", "health", "services", "at_home", "other"]
        )

        Fjob = st.selectbox(
            "Father's Job",
            ["teacher", "health", "services", "at_home", "other"]
        )


    # --------------------------------------------------------
    # COLUMN 2
    # --------------------------------------------------------

    with col2:

        st.subheader("📚 Academic Information")

        reason = st.selectbox(
            "Reason for Choosing School",
            ["home", "reputation", "course", "other"]
        )

        guardian = st.selectbox(
            "Guardian",
            ["mother", "father", "other"]
        )

        traveltime = st.selectbox(
            "Travel Time",
            [1, 2, 3, 4]
        )

        studytime = st.selectbox(
            "Weekly Study Time",
            [1, 2, 3, 4]
        )

        failures = st.selectbox(
            "Past Class Failures",
            [0, 1, 2, 3]
        )

        famrel = st.selectbox(
            "Family Relationship Quality",
            [1, 2, 3, 4, 5]
        )

        freetime = st.selectbox(
            "Free Time After School",
            [1, 2, 3, 4, 5]
        )

        goout = st.selectbox(
            "Going Out With Friends",
            [1, 2, 3, 4, 5]
        )

        Dalc = st.selectbox(
            "Weekday Alcohol Consumption",
            [1, 2, 3, 4, 5]
        )

        Walc = st.selectbox(
            "Weekend Alcohol Consumption",
            [1, 2, 3, 4, 5]
        )


    # --------------------------------------------------------
    # COLUMN 3
    # --------------------------------------------------------

    with col3:

        st.subheader("🌐 Lifestyle & Support")

        health = st.selectbox(
            "Current Health Status",
            [1, 2, 3, 4, 5]
        )

        absences = st.number_input(
            "Number of Absences",
            min_value=0,
            max_value=100,
            value=5,
            step=1
        )

        schoolsup = st.selectbox(
            "Extra Educational Support",
            ["yes", "no"]
        )

        famsup = st.selectbox(
            "Family Educational Support",
            ["yes", "no"]
        )

        paid = st.selectbox(
            "Extra Paid Classes",
            ["yes", "no"]
        )

        activities = st.selectbox(
            "Extra-curricular Activities",
            ["yes", "no"]
        )

        nursery = st.selectbox(
            "Attended Nursery School",
            ["yes", "no"]
        )

        higher = st.selectbox(
            "Wants Higher Education",
            ["yes", "no"]
        )

        internet = st.selectbox(
            "Internet Access at Home",
            ["yes", "no"]
        )

        romantic = st.selectbox(
            "In a Romantic Relationship",
            ["yes", "no"]
        )


    # ========================================================
    # SUBMIT BUTTON
    # ========================================================

    predict_button = st.form_submit_button(
        "🔮 Predict Student Performance",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # ========================================================
    # STEP 25 - INPUT VALIDATION
    # ========================================================

    validation_errors = []

    # Age validation
    if age < 15 or age > 25:
        validation_errors.append(
            "Age must be between 15 and 25."
        )

    # Absence validation
    if absences < 0:
        validation_errors.append(
            "Absences cannot be negative."
        )

    if absences > 100:
        validation_errors.append(
            "Absences cannot be greater than 100."
        )

    # Failure validation
    if failures < 0 or failures > 3:
        validation_errors.append(
            "Past class failures must be between 0 and 3."
        )

    # Study time validation
    if studytime < 1 or studytime > 4:
        validation_errors.append(
            "Study time must be between 1 and 4."
        )


    # ========================================================
    # SHOW VALIDATION ERRORS
    # ========================================================

    if validation_errors:

        st.error(
            "❌ Please correct the following input errors:"
        )

        for error in validation_errors:

            st.write(
                "• " + error
            )

        st.stop()


    # ========================================================
    # VALIDATION SUCCESS
    # ========================================================

    st.success(
        "✅ Input validation successful."
    )


    # Warning for high absences
    if absences > 20:

        st.warning(
            "⚠️ This student has a high number of absences. "
            "The prediction should be interpreted carefully."
        )


    # ========================================================
    # CREATE INPUT DATAFRAME
    # ========================================================

    input_data = pd.DataFrame({

        "school": [school],
        "sex": [sex],
        "age": [age],
        "address": [address],
        "famsize": [famsize],
        "Pstatus": [Pstatus],
        "Medu": [Medu],
        "Fedu": [Fedu],
        "Mjob": [Mjob],
        "Fjob": [Fjob],
        "reason": [reason],
        "guardian": [guardian],
        "traveltime": [traveltime],
        "studytime": [studytime],
        "failures": [failures],
        "famrel": [famrel],
        "freetime": [freetime],
        "goout": [goout],
        "Dalc": [Dalc],
        "Walc": [Walc],
        "health": [health],
        "absences": [absences],
        "schoolsup": [schoolsup],
        "famsup": [famsup],
        "paid": [paid],
        "activities": [activities],
        "nursery": [nursery],
        "higher": [higher],
        "internet": [internet],
        "romantic": [romantic]

    })


    # ========================================================
    # TRANSFORM INPUT
    # ========================================================

    try:

        transformed_input = regression_preprocessor.transform(
            input_data
        )

    except Exception as e:

        st.error(
            "❌ Error while preprocessing the student input."
        )

        st.error(
            f"Technical error: {e}"
        )

        st.stop()


    # ========================================================
    # REGRESSION PREDICTION
    # ========================================================

    try:

        predicted_grade = regression_model.predict(
            transformed_input
        )[0]

        # Keep grade between 0 and 20
        predicted_grade = max(
            0,
            min(20, predicted_grade)
        )

    except Exception as e:

        st.error(
            "❌ Error while predicting the final grade."
        )

        st.error(
            f"Technical error: {e}"
        )

        st.stop()


    # ========================================================
    # CLASSIFICATION PREDICTION
    # ========================================================

    try:

        classification_input = classification_preprocessor.transform(
            input_data
        )

        support_category = classification_model.predict(
            classification_input
        )[0]

    except Exception as e:

        st.error(
            "❌ Error while predicting the support category."
        )

        st.error(
            f"Technical error: {e}"
        )

        st.stop()


    # ========================================================
    # DISPLAY PREDICTION
    # ========================================================

    st.divider()

    st.header("🎯 Prediction Result")

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Predicted Final Grade",
            f"{predicted_grade:.2f} / 20"
        )


    with result_col2:

        st.metric(
            "Support Category",
            support_category
        )


    # ========================================================
    # SUPPORT CATEGORY EXPLANATION
    # ========================================================

    if support_category == "High Support":

        st.error(
            "🔴 High Support Required"
        )

        st.write(
            "The student may require additional academic support, "
            "regular monitoring, and personalized learning resources."
        )

    elif support_category == "Medium Support":

        st.warning(
            "🟡 Medium Support Required"
        )

        st.write(
            "The student may benefit from additional practice "
            "and regular academic monitoring."
        )

    else:

        st.success(
            "🟢 Low Support Required"
        )

        st.write(
            "The student is predicted to require relatively less "
            "additional academic support."
        )


    # ========================================================
    # PERSONALIZED RECOMMENDATIONS
    # ========================================================

    st.divider()

    st.header("📚 Personalized Learning Recommendations")

    recommendations = []


    # Study time
    if studytime <= 2:

        recommendations.append(
            "Increase weekly study time and follow a fixed study schedule."
        )


    # Failures
    if failures > 0:

        recommendations.append(
            "Focus on previously failed subjects and practice basic concepts regularly."
        )


    # Absences
    if absences > 10:

        recommendations.append(
            "Reduce unnecessary absences and attend classes regularly."
        )


    # School support
    if schoolsup == "no":

        recommendations.append(
            "Consider using additional academic support or tutoring."
        )


    # Family support
    if famsup == "no":

        recommendations.append(
            "Discuss academic progress with family members or mentors."
        )


    # Free time
    if freetime >= 4:

        recommendations.append(
            "Use some free time for revision, assignments, and practice."
        )


    # Going out
    if goout >= 4:

        recommendations.append(
            "Maintain a balance between social activities and study time."
        )


    # Alcohol consumption
    if Dalc >= 3 or Walc >= 3:

        recommendations.append(
            "Maintain healthy lifestyle habits and avoid activities that may affect study performance."
        )


    # Activities
    if activities == "no":

        recommendations.append(
            "Participating in suitable extracurricular activities may help develop useful skills."
        )


    # Higher education
    if higher == "yes":

        recommendations.append(
            "Set clear higher-education goals and maintain consistent academic preparation."
        )


    # Internet
    if internet == "yes":

        recommendations.append(
            "Use reliable online educational resources for additional practice."
        )

    else:

        recommendations.append(
            "Use available offline learning materials such as textbooks and notes."
        )


    # Default recommendation
    if len(recommendations) == 0:

        recommendations.append(
            "Continue regular study, revision, attendance, and healthy learning habits."
        )


    # Remove duplicate recommendations
    recommendations = list(
        dict.fromkeys(recommendations)
    )


    for i, recommendation in enumerate(
        recommendations,
        start=1
    ):

        st.write(
            f"**{i}.** {recommendation}"
        )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.divider()

    st.header("🔍 Explainable AI - Prediction Explanation")

    st.write(
        "The following factors show how the trained Random Forest "
        "model influenced the predicted grade."
    )


    try:

        # Create SHAP explainer
        explainer = shap.TreeExplainer(
            regression_model
        )

        shap_values = explainer.shap_values(
            transformed_input
        )


        # Handle different SHAP output formats
        if isinstance(shap_values, list):

            shap_values = shap_values[0]


        shap_values = shap_values.flatten()


        # Get feature names
        try:

            feature_names = (
                regression_preprocessor
                .get_feature_names_out()
            )

        except Exception:

            feature_names = [
                f"Feature {i + 1}"
                for i in range(
                    len(shap_values)
                )
            ]


        # Make sure lengths match
        min_length = min(
            len(feature_names),
            len(shap_values)
        )

        feature_names = feature_names[
            :min_length
        ]

        shap_values = shap_values[
            :min_length
        ]


        # Create SHAP dataframe
        shap_df = pd.DataFrame({

            "Feature": feature_names,

            "SHAP Impact": shap_values,

            "Absolute Impact": abs(
                shap_values
            )

        })


        # Sort by importance
        shap_df = shap_df.sort_values(
            by="Absolute Impact",
            ascending=False
        )


        # Top 10 features
        top_features = shap_df.head(10)


        st.subheader(
            "Top Factors Influencing the Prediction"
        )


        # Display table
        display_df = top_features[
            ["Feature", "SHAP Impact"]
        ].copy()


        display_df["SHAP Impact"] = (
            display_df["SHAP Impact"]
            .round(4)
        )


        st.dataframe(
            display_df,
            use_container_width=True
        )


        # Explain positive / negative
        st.subheader(
            "📌 Interpretation"
        )

        for _, row in top_features.iterrows():

            feature = row["Feature"]

            impact = row["SHAP Impact"]


            if impact > 0:

                st.write(
                    f"🟢 **{feature}** increased the predicted grade "
                    f"by approximately **{abs(impact):.2f}**."
                )

            elif impact < 0:

                st.write(
                    f"🔴 **{feature}** decreased the predicted grade "
                    f"by approximately **{abs(impact):.2f}**."
                )

            else:

                st.write(
                    f"⚪ **{feature}** had very little effect "
                    f"on this prediction."
                )


    except Exception as e:

        st.warning(
            "⚠️ SHAP explanation could not be generated for this input."
        )

        st.write(
            f"Technical information: {e}"
        )

        top_features = pd.DataFrame(
            columns=[
                "Feature",
                "SHAP Impact"
            ]
        )


    # ========================================================
    # DOWNLOAD PREDICTION REPORT
    # ========================================================

    st.divider()

    st.header("📥 Download Prediction Report")


    # Prepare SHAP information
    if not top_features.empty:

        shap_report = top_features[
            ["Feature", "SHAP Impact"]
        ].copy()

        shap_report["SHAP Impact"] = (
            shap_report["SHAP Impact"]
            .round(4)
        )

        shap_text = "; ".join(

            [
                f"{row['Feature']}: {row['SHAP Impact']}"
                for _, row in shap_report.iterrows()
            ]

        )

    else:

        shap_text = "SHAP explanation unavailable"


    # Convert recommendations to text
    recommendations_text = " | ".join(
        recommendations
    )


    # Create report
    report_data = pd.DataFrame({

        "Predicted Final Grade": [
            round(
                predicted_grade,
                2
            )
        ],

        "Support Category": [
            support_category
        ],

        "Recommendations": [
            recommendations_text
        ],

        "Top SHAP Factors": [
            shap_text
        ]

    })


    # Convert report to CSV
    report_csv = report_data.to_csv(
        index=False
    )


    st.download_button(

        label="📥 Download Prediction Report",

        data=report_csv,

        file_name="student_prediction_report.csv",

        mime="text/csv",

        use_container_width=True

    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 Intelligent Student Performance Prediction and Personalized Learning Recommendation System"
)

st.caption(
    "This system provides AI-based decision support. "
    "Predictions should not be treated as guaranteed academic outcomes."
)