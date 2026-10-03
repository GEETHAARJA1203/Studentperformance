import os
import sys
import io
from datetime import datetime

import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PATH CONFIGURATION
# ============================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(APP_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_FILE = os.path.join(BASE_DIR, "student-mat.csv")

# Make sure local imports work
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from auth import authenticate_user


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 30px;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed
        );
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(79,70,229,0.20);
    }

    .hero h1 {
        font-size: 38px;
        margin-bottom: 8px;
    }

    .hero p {
        font-size: 17px;
        margin: 0;
    }

    .card {
        background: white;
        padding: 22px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(0,0,0,0.05);
        margin-bottom: 18px;
    }

    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        text-align: center;
        box-shadow: 0 5px 18px rgba(0,0,0,0.05);
    }

    .metric-title {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        color: #111827;
        font-size: 30px;
        font-weight: 700;
        margin-top: 5px;
    }

    .section-title {
        color: #1e293b;
        font-size: 25px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .success-box {
        padding: 18px;
        border-radius: 15px;
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
    }

    .warning-box {
        padding: 18px;
        border-radius: 15px;
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #92400e;
    }

    .danger-box {
        padding: 18px;
        border-radius: 15px;
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #991b1b;
    }

    .info-box {
        padding: 18px;
        border-radius: 15px;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e40af;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "predicted_grade" not in st.session_state:
    st.session_state.predicted_grade = None

if "risk_level" not in st.session_state:
    st.session_state.risk_level = None

if "support_category" not in st.session_state:
    st.session_state.support_category = None

if "input_data" not in st.session_state:
    st.session_state.input_data = None

if "recommendations" not in st.session_state:
    st.session_state.recommendations = []


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():

    regression_model_path = os.path.join(
        MODELS_DIR,
        "tuned_random_forest_regression.pkl"
    )

    classification_model_path = os.path.join(
        MODELS_DIR,
        "tuned_random_forest_classification.pkl"
    )

    regression_preprocessor_path = os.path.join(
        MODELS_DIR,
        "preprocessor.pkl"
    )

    classification_preprocessor_path = os.path.join(
        MODELS_DIR,
        "classification_preprocessor.pkl"
    )

    regression_model = None
    classification_model = None
    regression_preprocessor = None
    classification_preprocessor = None

    errors = []

    try:
        regression_model = joblib.load(
            regression_model_path
        )
    except Exception as e:
        errors.append(
            f"Regression model: {e}"
        )

    try:
        classification_model = joblib.load(
            classification_model_path
        )
    except Exception as e:
        errors.append(
            f"Classification model: {e}"
        )

    try:
        regression_preprocessor = joblib.load(
            regression_preprocessor_path
        )
    except Exception as e:
        errors.append(
            f"Regression preprocessor: {e}"
        )

    try:
        classification_preprocessor = joblib.load(
            classification_preprocessor_path
        )
    except Exception as e:
        errors.append(
            f"Classification preprocessor: {e}"
        )

    return (
        regression_model,
        classification_model,
        regression_preprocessor,
        classification_preprocessor,
        errors
    )


(
    regression_model,
    classification_model,
    regression_preprocessor,
    classification_preprocessor,
    model_errors
) = load_models()


# ============================================================
# LOGIN PAGE
# ============================================================

def show_login():

    st.markdown(
        """
        <div class="hero">
            <h1>🎓 Student Performance AI</h1>
            <p>
                Intelligent Student Performance Prediction,
                Early Warning and Explainable AI Platform
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    left, center, right = st.columns(
        [1, 2, 1]
    )

    with center:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "## 🔐 Secure Login"
        )

        st.write(
            "Login to access the Student Performance AI dashboard."
        )

        username = st.text_input(
            "Username",
            placeholder="Enter username"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter password"
        )

        login_button = st.button(
            "🔓 Login",
            use_container_width=True
        )

        if login_button:

            if not username or not password:

                st.error(
                    "Please enter both username and password."
                )

            else:

                user = authenticate_user(
                    username,
                    password
                )

                if user:

                    st.session_state.logged_in = True
                    st.session_state.user = user

                    st.success(
                        "Login successful!"
                    )

                    st.rerun()

                else:

                    st.error(
                        "Invalid username or password."
                    )

        st.markdown(
            """
            <br>
            <b>Demo Accounts</b><br>
            Admin: admin / admin123<br>
            Faculty: faculty / faculty123<br>
            Student: student / student123
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_risk_details(predicted_grade):

    if predicted_grade < 8:

        return (
            "High Risk",
            "Intensive Academic Support"
        )

    elif predicted_grade < 12:

        return (
            "Medium Risk",
            "Moderate Academic Support"
        )

    else:

        return (
            "Low Risk",
            "Regular Monitoring"
        )


def generate_recommendations(data):

    recommendations = []

    absences = int(data.get("absences", 0))
    studytime = int(data.get("studytime", 2))
    failures = int(data.get("failures", 0))
    goout = int(data.get("goout", 2))
    higher = data.get("higher", "yes")

    if absences >= 10:

        recommendations.append(
            "Reduce unnecessary absences and maintain regular attendance."
        )

    elif absences >= 5:

        recommendations.append(
            "Try to improve attendance and avoid missing important classes."
        )

    else:

        recommendations.append(
            "Maintain the current good attendance pattern."
        )

    if studytime <= 2:

        recommendations.append(
            "Increase daily study time and follow a structured study schedule."
        )

    else:

        recommendations.append(
            "Continue following a consistent study routine."
        )

    if failures > 0:

        recommendations.append(
            "Focus on previously failed subjects and seek faculty guidance."
        )

    if goout >= 4:

        recommendations.append(
            "Balance social activities with academic responsibilities."
        )

    if str(higher).lower() == "no":

        recommendations.append(
            "Consider discussing higher education opportunities with a mentor."
        )

    recommendations.append(
        "Review academic progress regularly with faculty or mentors."
    )

    return recommendations


def create_prediction_dataframe(data):

    return pd.DataFrame(
        [data]
    )


# ============================================================
# SIDEBAR
# ============================================================

def show_sidebar():

    user = st.session_state.user

    st.sidebar.markdown(
        """
        <div style="
            padding:18px;
            border-radius:15px;
            background:linear-gradient(
                135deg,#4f46e5,#7c3aed
            );
            color:white;
            margin-bottom:20px;
        ">
            <h2 style="margin:0;">🎓 Student AI</h2>
            <p style="margin:5px 0 0;">
                Performance Intelligence
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.write(
        f"👤 **{user['username']}**"
    )

    st.sidebar.write(
        f"🔑 **Role:** {user['role']}"
    )

    if user.get("student_id"):
        st.sidebar.write(
            f"🆔 **Student ID:** {user['student_id']}"
        )

    st.sidebar.divider()

    pages = [
        "🏠 Home",
        "🔮 What-If Prediction",
        "📊 Analytics",
        "🤖 XAI / SHAP"
    ]

    selected_page = st.sidebar.radio(
        "Navigation",
        pages
    )

    st.sidebar.divider()

    if st.sidebar.button(
        "🚪 Logout",
        use_container_width=True
    ):

        st.session_state.logged_in = False
        st.session_state.user = None
        st.session_state.prediction_done = False

        st.rerun()

    return selected_page


# ============================================================
# HOME PAGE
# ============================================================

def show_home():

    user = st.session_state.user

    st.markdown(
        """
        <div class="hero">
            <h1>Welcome to Student Performance AI 🎓</h1>
            <p>
                Predict academic performance, identify students
                at risk and understand the factors behind predictions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="card">
            <h3>👋 Welcome, {user['username']}</h3>
            <p>
                You are logged in as <b>{user['role']}</b>.
                Use the navigation menu to explore the system.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Dataset Records</div>
                <div class="metric-value">395</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Input Features</div>
                <div class="metric-value">30</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">ML Model</div>
                <div class="metric-value">RF</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Explainability</div>
                <div class="metric-value">SHAP</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">✨ System Features</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="card">
                <h3>🔮 Prediction</h3>
                <p>
                    Enter student information and predict
                    the expected final academic grade.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="card">
                <h3>⚠️ Early Warning</h3>
                <p>
                    Identify High, Medium and Low academic
                    risk levels using the prediction.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="card">
                <h3>🤖 Explainable AI</h3>
                <p>
                    Understand which student factors
                    influence the prediction.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">📌 How the System Works</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

        <b>Step 1:</b> Student or faculty member logs into the system.<br><br>

        <b>Step 2:</b> Student academic and demographic information
        is entered.<br><br>

        <b>Step 3:</b> The trained Random Forest model processes
        the information.<br><br>

        <b>Step 4:</b> The system predicts the expected final grade.<br><br>

        <b>Step 5:</b> The prediction is converted into a risk level.<br><br>

        <b>Step 6:</b> SHAP explains the important factors behind
        the prediction.<br><br>

        <b>Step 7:</b> Personalized recommendations are generated.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# WHAT-IF PREDICTION PAGE
# ============================================================

def show_prediction():

    st.markdown(
        """
        <div class="hero">
            <h1>🔮 What-If Student Prediction</h1>
            <p>
                Modify student characteristics and observe
                the predicted academic outcome.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if model_errors:

        st.warning(
            "Some model files could not be loaded."
        )

        with st.expander(
            "View model loading details"
        ):

            for error in model_errors:
                st.write(error)

    if regression_model is None:

        st.error(
            "Regression model is unavailable."
        )

        return

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">👤 Basic Information</div>',
        unsafe_allow_html=True
    )

    with st.container():

        c1, c2, c3 = st.columns(3)

        with c1:

            school = st.selectbox(
                "School",
                ["GP", "MS"]
            )

        with c2:

            sex = st.selectbox(
                "Gender",
                ["F", "M"]
            )

        with c3:

            age = st.slider(
                "Age",
                15,
                22,
                17
            )

        c1, c2, c3 = st.columns(3)

        with c1:

            address = st.selectbox(
                "Address",
                ["U", "R"]
            )

        with c2:

            famsize = st.selectbox(
                "Family Size",
                ["LE3", "GT3"]
            )

        with c3:

            pstatus = st.selectbox(
                "Parent Status",
                ["A", "T"]
            )

    # --------------------------------------------------------
    # FAMILY AND EDUCATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🏠 Family & Education</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        medu = st.selectbox(
            "Mother Education",
            [0, 1, 2, 3, 4]
        )

    with c2:

        fedu = st.selectbox(
            "Father Education",
            [0, 1, 2, 3, 4]
        )

    with c3:

        guardian = st.selectbox(
            "Guardian",
            ["mother", "father", "other"]
        )

    c1, c2, c3 = st.columns(3)

    with c1:

        mjob = st.selectbox(
            "Mother Job",
            [
                "teacher",
                "health",
                "services",
                "at_home",
                "other"
            ]
        )

    with c2:

        fjob = st.selectbox(
            "Father Job",
            [
                "teacher",
                "health",
                "services",
                "at_home",
                "other"
            ]
        )

    with c3:

        reason = st.selectbox(
            "Reason for School Choice",
            [
                "home",
                "reputation",
                "course",
                "other"
            ]
        )

    # --------------------------------------------------------
    # ACADEMIC INFORMATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">📚 Academic Information</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        traveltime = st.selectbox(
            "Travel Time",
            [1, 2, 3, 4]
        )

    with c2:

        studytime = st.selectbox(
            "Weekly Study Time",
            [1, 2, 3, 4]
        )

    with c3:

        failures = st.selectbox(
            "Past Class Failures",
            [0, 1, 2, 3]
        )

    c1, c2, c3 = st.columns(3)

    with c1:

        absences = st.number_input(
            "Absences",
            min_value=0,
            max_value=93,
            value=4
        )

    with c2:

        famrel = st.selectbox(
            "Family Relationship Quality",
            [1, 2, 3, 4, 5]
        )

    with c3:

        freetime = st.selectbox(
            "Free Time",
            [1, 2, 3, 4, 5]
        )

    # --------------------------------------------------------
    # SOCIAL & HEALTH
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">❤️ Social & Health</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        goout = st.selectbox(
            "Going Out Frequency",
            [1, 2, 3, 4, 5]
        )

    with c2:

        dalc = st.selectbox(
            "Weekday Alcohol Consumption",
            [1, 2, 3, 4, 5]
        )

    with c3:

        walc = st.selectbox(
            "Weekend Alcohol Consumption",
            [1, 2, 3, 4, 5]
        )

    health = st.slider(
        "Current Health Status",
        1,
        5,
        3
    )

    # --------------------------------------------------------
    # SUPPORT & ACTIVITIES
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🎯 Support & Activities</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        schoolsup = st.selectbox(
            "School Support",
            ["yes", "no"]
        )

    with c2:

        famsup = st.selectbox(
            "Family Support",
            ["yes", "no"]
        )

    with c3:

        paid = st.selectbox(
            "Paid Extra Classes",
            ["yes", "no"]
        )

    c1, c2, c3 = st.columns(3)

    with c1:

        activities = st.selectbox(
            "Extra Activities",
            ["yes", "no"]
        )

    with c2:

        nursery = st.selectbox(
            "Attended Nursery",
            ["yes", "no"]
        )

    with c3:

        higher = st.selectbox(
            "Wants Higher Education",
            ["yes", "no"]
        )

    internet = st.selectbox(
        "Internet Access",
        ["yes", "no"]
    )

    romantic = st.selectbox(
        "Romantic Relationship",
        ["yes", "no"]
    )

    # --------------------------------------------------------
    # INPUT DATA
    # --------------------------------------------------------

    input_data = {
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": pstatus,
        "Medu": medu,
        "Fedu": fedu,
        "Mjob": mjob,
        "Fjob": fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": dalc,
        "Walc": walc,
        "health": health,
        "absences": absences,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic
    }

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    predict_button = st.button(
        "🚀 Generate Prediction",
        use_container_width=True
    )

    if predict_button:

        try:

            input_df = create_prediction_dataframe(
                input_data
            )

            # ------------------------------------------------
            # PREPROCESSING
            # ------------------------------------------------

            if regression_preprocessor is not None:

                transformed_data = (
                    regression_preprocessor.transform(
                        input_df
                    )
                )

            else:

                transformed_data = input_df

            # ------------------------------------------------
            # REGRESSION PREDICTION
            # ------------------------------------------------

            prediction = regression_model.predict(
                transformed_data
            )

            predicted_grade = float(
                prediction[0]
            )

            # Keep prediction between 0 and 20
            predicted_grade = max(
                0,
                min(
                    20,
                    predicted_grade
                )
            )

            risk_level, support_category = (
                get_risk_details(
                    predicted_grade
                )
            )

            recommendations = (
                generate_recommendations(
                    input_data
                )
            )

            st.session_state.prediction_done = True
            st.session_state.predicted_grade = predicted_grade
            st.session_state.risk_level = risk_level
            st.session_state.support_category = support_category
            st.session_state.input_data = input_data
            st.session_state.recommendations = recommendations

            st.success(
                "Prediction generated successfully!"
            )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.code(
                str(e)
            )

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    if st.session_state.prediction_done:

        predicted_grade = (
            st.session_state.predicted_grade
        )

        risk_level = (
            st.session_state.risk_level
        )

        support_category = (
            st.session_state.support_category
        )

        recommendations = (
            st.session_state.recommendations
        )

        st.markdown(
            '<div class="section-title">📈 Prediction Result</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                        Predicted Final Grade
                    </div>
                    <div class="metric-value">
                        {predicted_grade:.2f}
                    </div>
                    <p>out of 20</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                        Risk Level
                    </div>
                    <div class="metric-value">
                        {risk_level}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                        Support
                    </div>
                    <div class="metric-value">
                        {support_category}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Risk message

        if risk_level == "High Risk":

            st.markdown(
                """
                <div class="danger-box">
                    <b>⚠️ Early Warning:</b>
                    This student may require intensive
                    academic support and monitoring.
                </div>
                """,
                unsafe_allow_html=True
            )

        elif risk_level == "Medium Risk":

            st.markdown(
                """
                <div class="warning-box">
                    <b>⚠️ Attention Required:</b>
                    This student may benefit from
                    additional academic support.
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="success-box">
                    <b>✅ Positive Status:</b>
                    Continue regular academic monitoring
                    and maintain good study habits.
                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # RECOMMENDATIONS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">💡 Personalized Recommendations</div>',
            unsafe_allow_html=True
        )

        for recommendation in recommendations:

            st.info(
                "💡 " + recommendation
            )

        # ----------------------------------------------------
        # CSV REPORT
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">📊 Download Reports</div>',
            unsafe_allow_html=True
        )

        report_data = {
            "Student ID": [
                st.session_state.user.get(
                    "student_id",
                    "N/A"
                )
            ],
            "Username": [
                st.session_state.user["username"]
            ],
            "Predicted Grade": [
                predicted_grade
            ],
            "Risk Level": [
                risk_level
            ],
            "Support Category": [
                support_category
            ],
            "Report Date": [
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            ]
        }

        report_df = pd.DataFrame(
            report_data
        )

        csv_data = report_df.to_csv(
            index=False
        )

        st.download_button(
            label="📥 Download CSV Report",
            data=csv_data,
            file_name="student_performance_report.csv",
            mime="text/csv",
            use_container_width=True
        )

        # ====================================================
        # PDF REPORT
        # ====================================================

        st.markdown(
            '<div class="section-title">📄 PDF Report</div>',
            unsafe_allow_html=True
        )

        try:

            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.styles import (
                getSampleStyleSheet,
                ParagraphStyle
            )
            from reportlab.lib.enums import TA_CENTER
            from reportlab.platypus import (
                SimpleDocTemplate,
                Paragraph,
                Spacer,
                Table,
                TableStyle
            )

            pdf_buffer = io.BytesIO()

            doc = SimpleDocTemplate(
                pdf_buffer,
                pagesize=A4,
                rightMargin=40,
                leftMargin=40,
                topMargin=40,
                bottomMargin=40
            )

            styles = getSampleStyleSheet()

            title_style = ParagraphStyle(
                "ReportTitle",
                parent=styles["Title"],
                alignment=TA_CENTER,
                fontSize=22,
                leading=28,
                spaceAfter=15
            )

            heading_style = ParagraphStyle(
                "ReportHeading",
                parent=styles["Heading2"],
                fontSize=15,
                leading=20,
                spaceBefore=12,
                spaceAfter=8
            )

            normal_style = ParagraphStyle(
                "ReportNormal",
                parent=styles["BodyText"],
                fontSize=10,
                leading=15
            )

            story = []

            story.append(
                Paragraph(
                    "Student Performance AI",
                    title_style
                )
            )

            story.append(
                Paragraph(
                    "Student Performance Prediction Report",
                    styles["Heading2"]
                )
            )

            story.append(
                Spacer(1, 15)
            )

            # Student information

            story.append(
                Paragraph(
                    "1. Student Information",
                    heading_style
                )
            )

            student_id = (
                st.session_state.user.get(
                    "student_id"
                )
                or "N/A"
            )

            student_table = Table(
                [
                    [
                        "Student ID",
                        str(student_id)
                    ],
                    [
                        "Username",
                        str(
                            st.session_state.user[
                                "username"
                            ]
                        )
                    ],
                    [
                        "Role",
                        str(
                            st.session_state.user[
                                "role"
                            ]
                        )
                    ],
                    [
                        "Report Date",
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    ]
                ],
                colWidths=[
                    150,
                    330
                ]
            )

            student_table.setStyle(
                TableStyle(
                    [
                        (
                            "BACKGROUND",
                            (0, 0),
                            (0, -1),
                            colors.HexColor(
                                "#eef2ff"
                            )
                        ),
                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.5,
                            colors.grey
                        ),
                        (
                            "FONTNAME",
                            (0, 0),
                            (0, -1),
                            "Helvetica-Bold"
                        ),
                        (
                            "VALIGN",
                            (0, 0),
                            (-1, -1),
                            "TOP"
                        ),
                        (
                            "PADDING",
                            (0, 0),
                            (-1, -1),
                            8
                        )
                    ]
                )
            )

            story.append(
                student_table
            )

            story.append(
                Spacer(1, 15)
            )

            # Prediction

            story.append(
                Paragraph(
                    "2. Prediction Result",
                    heading_style
                )
            )

            prediction_table = Table(
                [
                    [
                        "Predicted Final Grade",
                        f"{predicted_grade:.2f} / 20"
                    ],
                    [
                        "Risk Level",
                        risk_level
                    ],
                    [
                        "Support Category",
                        support_category
                    ]
                ],
                colWidths=[
                    200,
                    280
                ]
            )

            prediction_table.setStyle(
                TableStyle(
                    [
                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.5,
                            colors.grey
                        ),
                        (
                            "BACKGROUND",
                            (0, 0),
                            (0, -1),
                            colors.HexColor(
                                "#f1f5f9"
                            )
                        ),
                        (
                            "FONTNAME",
                            (0, 0),
                            (0, -1),
                            "Helvetica-Bold"
                        ),
                        (
                            "PADDING",
                            (0, 0),
                            (-1, -1),
                            8
                        )
                    ]
                )
            )

            story.append(
                prediction_table
            )

            # Recommendations

            story.append(
                Paragraph(
                    "3. Personalized Recommendations",
                    heading_style
                )
            )

            for recommendation in recommendations:

                story.append(
                    Paragraph(
                        "• " + recommendation,
                        normal_style
                    )
                )

                story.append(
                    Spacer(1, 5)
                )

            # Disclaimer

            story.append(
                Paragraph(
                    "4. Disclaimer",
                    heading_style
                )
            )

            story.append(
                Paragraph(
                    "This system provides AI-based decision "
                    "support. The prediction is an estimate "
                    "based on the trained machine learning "
                    "model and should not be treated as a "
                    "guaranteed academic outcome.",
                    normal_style
                )
            )

            doc.build(
                story
            )

            pdf_buffer.seek(0)

            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_buffer.getvalue(),
                file_name="student_performance_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        except Exception as pdf_error:

            st.error(
                "PDF generation failed."
            )

            st.code(
                str(pdf_error)
            )


# ============================================================
# ANALYTICS PAGE
# ============================================================

def show_analytics():

    st.markdown(
        """
        <div class="hero">
            <h1>📊 Student Analytics</h1>
            <p>
                Explore the dataset and understand
                student performance patterns.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not os.path.exists(DATA_FILE):

        st.error(
            "student-mat.csv was not found."
        )

        return

    try:

        df = pd.read_csv(
            DATA_FILE,
            sep=";"
        )

    except Exception as e:

        st.error(
            "Unable to load dataset."
        )

        st.code(
            str(e)
        )

        return

    st.markdown(
        '<div class="section-title">📌 Dataset Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Total Students",
            len(df)
        )

    with c2:

        st.metric(
            "Total Features",
            len(df.columns)
        )

    with c3:

        if "G3" in df.columns:

            st.metric(
                "Average Final Grade",
                f"{df['G3'].mean():.2f}"
            )

    with c4:

        if "G3" in df.columns:

            st.metric(
                "Maximum Grade",
                f"{df['G3'].max():.0f}"
            )

    st.markdown(
        '<div class="section-title">📈 Grade Distribution</div>',
        unsafe_allow_html=True
    )

    if "G3" in df.columns:

        grade_counts = (
            df["G3"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(
            grade_counts
        )

    st.markdown(
        '<div class="section-title">📚 Study Time vs Final Grade</div>',
        unsafe_allow_html=True
    )

    if (
        "studytime" in df.columns
        and "G3" in df.columns
    ):

        study_data = (
            df.groupby("studytime")["G3"]
            .mean()
            .reset_index()
        )

        study_data = study_data.set_index(
            "studytime"
        )

        st.line_chart(
            study_data
        )

    st.markdown(
        '<div class="section-title">📅 Absences vs Final Grade</div>',
        unsafe_allow_html=True
    )

    if (
        "absences" in df.columns
        and "G3" in df.columns
    ):

        absence_data = (
            df.groupby("absences")["G3"]
            .mean()
            .reset_index()
        )

        absence_data = absence_data.set_index(
            "absences"
        )

        st.line_chart(
            absence_data
        )

    st.markdown(
        '<div class="section-title">📋 Dataset Preview</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# ============================================================
# XAI / SHAP PAGE
# ============================================================

def show_xai():

    st.markdown(
        """
        <div class="hero">
            <h1>🤖 Explainable AI</h1>
            <p>
                Understand which factors influence
                the student performance prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.prediction_done:

        st.info(
            "First generate a prediction from the "
            "What-If Prediction page."
        )

        return

    if regression_model is None:

        st.error(
            "Regression model is unavailable."
        )

        return

    input_data = (
        st.session_state.input_data
    )

    if input_data is None:

        st.warning(
            "No prediction input is available."
        )

        return

    try:

        import shap

        input_df = pd.DataFrame(
            [input_data]
        )

        if regression_preprocessor is not None:

            transformed_data = (
                regression_preprocessor.transform(
                    input_df
                )
            )

        else:

            transformed_data = input_df

        st.markdown(
            '<div class="section-title">🔍 SHAP Explanation</div>',
            unsafe_allow_html=True
        )

        st.write(
            """
            SHAP explains how different student features
            contribute to the model prediction.

            Positive SHAP values push the prediction higher,
            while negative SHAP values push it lower.
            """
        )

        try:

            explainer = shap.TreeExplainer(
                regression_model
            )

            shap_values = explainer.shap_values(
                transformed_data
            )

            if isinstance(
                shap_values,
                list
            ):

                shap_values = shap_values[0]

            shap_values = shap_values[0]

            # Feature names

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

            shap_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "SHAP Impact": shap_values
                }
            )

            shap_df["Absolute Impact"] = (
                shap_df["SHAP Impact"]
                .abs()
            )

            shap_df = (
                shap_df
                .sort_values(
                    "Absolute Impact",
                    ascending=False
                )
                .head(10)
            )

            st.markdown(
                "### Top Factors Influencing Prediction"
            )

            display_df = shap_df[
                [
                    "Feature",
                    "SHAP Impact"
                ]
            ].copy()

            st.dataframe(
                display_df,
                use_container_width=True
            )

            st.markdown(
                "### 📊 Feature Impact"
            )

            chart_df = display_df.set_index(
                "Feature"
            )

            st.bar_chart(
                chart_df[
                    "SHAP Impact"
                ]
            )

            st.markdown(
                """
                <div class="info-box">
                    <b>How to interpret:</b><br><br>
                    Positive SHAP values indicate that the
                    feature pushed the prediction upward.<br><br>
                    Negative SHAP values indicate that the
                    feature pushed the prediction downward.
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as shap_error:

            st.warning(
                "SHAP explanation could not be generated "
                "for this prediction."
            )

            st.code(
                str(shap_error)
            )

    except ImportError:

        st.error(
            "SHAP is not installed."
        )

        st.code(
            "pip install shap"
        )


# ============================================================
# FOOTER
# ============================================================

def show_footer():

    st.markdown(
        """
        <hr>

        <div style="
            text-align:center;
            color:#64748b;
            padding:15px;
        ">

        <b>Student Performance AI</b><br>

        Machine Learning • Explainable AI •
        Early Warning System<br>

        Built using Python, Streamlit,
        Scikit-learn and SHAP

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

if not st.session_state.logged_in:

    show_login()

else:

    selected_page = show_sidebar()

    if selected_page == "🏠 Home":

        show_home()

    elif selected_page == "🔮 What-If Prediction":

        show_prediction()

    elif selected_page == "📊 Analytics":

        show_analytics()

    elif selected_page == "🤖 XAI / SHAP":

        show_xai()

    show_footer()