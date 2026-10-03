import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import os
import sys

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from access_control import require_role, logout_button

user = require_role(["Student", "Admin"])

logout_button()

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "tuned_random_forest_regression.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR, "models", "preprocessor.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR, "student-mat.csv"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Remove default Streamlit top spacing */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #f8fbff 0%,
            #eef5ff 50%,
            #f8fbff 100%
        );
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #172554 0%,
            #1e3a8a 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Hero */
    .hero {
        background: linear-gradient(
            135deg,
            #1e3a8a,
            #2563eb,
            #3b82f6
        );
        padding: 32px;
        border-radius: 22px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.20);
    }

    .hero h1 {
        color: white;
        font-size: 38px;
        margin: 0;
        font-weight: 750;
    }

    .hero p {
        color: #dbeafe;
        font-size: 17px;
        margin-top: 8px;
        margin-bottom: 0;
    }

    /* Section heading */
    .section-header {
        background: white;
        padding: 20px 24px;
        border-radius: 16px;
        margin-top: 18px;
        margin-bottom: 18px;
        border-left: 6px solid #2563eb;
        box-shadow: 0 5px 20px rgba(15, 23, 42, 0.07);
    }

    .section-header h2 {
        margin: 0;
        color: #172554;
        font-size: 25px;
    }

    .section-header p {
        margin: 5px 0 0 0;
        color: #64748b;
        font-size: 14px;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        padding: 22px;
        border-radius: 18px;
        box-shadow: 0 7px 25px rgba(15, 23, 42, 0.08);
        border: 1px solid #e2e8f0;
        min-height: 135px;
    }

    .metric-icon {
        font-size: 27px;
    }

    .metric-title {
        color: #64748b;
        font-size: 14px;
        margin-top: 5px;
    }

    .metric-value {
        color: #172554;
        font-size: 29px;
        font-weight: 750;
        margin-top: 5px;
    }

    .metric-subtitle {
        color: #94a3b8;
        font-size: 12px;
        margin-top: 4px;
    }

    /* Info cards */
    .info-card {
        background: white;
        padding: 22px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
        height: 100%;
    }

    .info-card h3 {
        color: #172554;
        margin-top: 0;
    }

    .info-card p {
        color: #64748b;
        line-height: 1.6;
    }

    /* Warning */
    .warning-card {
        background: #fff7ed;
        border-left: 6px solid #f97316;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
    }

    .warning-card h3 {
        color: #9a3412;
        margin-top: 0;
    }

    .warning-card p {
        color: #7c2d12;
        margin-bottom: 0;
    }

    /* Success */
    .success-card {
        background: #f0fdf4;
        border-left: 6px solid #22c55e;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
    }

    .success-card h3 {
        color: #166534;
        margin-top: 0;
    }

    .success-card p {
        color: #166534;
        margin-bottom: 0;
    }

    /* Suggestion cards */
    .suggestion-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 17px;
        border-radius: 14px;
        margin-bottom: 12px;
    }

    .suggestion-card strong {
        color: #1e3a8a;
    }

    .suggestion-card span {
        color: #475569;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 25px;
        color: #64748b;
        font-size: 13px;
        margin-top: 35px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: none;
        padding: 10px 18px;
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH, sep=";")


try:
    model = load_model()
    preprocessor = load_preprocessor()
except Exception as e:
    st.error("Unable to load the trained model or preprocessor.")
    st.code(str(e))
    st.stop()

# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🎓 Student Performance Dashboard</h1>
        <p>AI-powered academic performance monitoring and personalized insights</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎓 Student Portal")
    st.markdown("---")

    st.markdown("### 👤 Student Profile")

    student_name = st.text_input(
        "Student Name",
        value="Geetha"
    )

    student_id = st.text_input(
        "Student ID",
        value="STU001"
    )

    department = st.selectbox(
        "Department",
        [
            "Information Technology",
            "Computer Science",
            "Artificial Intelligence",
            "Data Science",
            "Electronics"
        ]
    )

    st.markdown("---")

    st.markdown("### 📚 Academic Inputs")

    study_time = st.slider(
        "Study Time",
        min_value=1,
        max_value=4,
        value=2,
        help="1 = very low, 4 = very high"
    )

    absences = st.slider(
        "Absences",
        min_value=0,
        max_value=75,
        value=5
    )

    failures = st.slider(
        "Past Failures",
        min_value=0,
        max_value=4,
        value=0
    )

    goout = st.slider(
        "Going Out",
        min_value=1,
        max_value=5,
        value=3
    )

    freetime = st.slider(
        "Free Time",
        min_value=1,
        max_value=5,
        value=3
    )

    health = st.slider(
        "Health",
        min_value=1,
        max_value=5,
        value=3
    )

    famrel = st.slider(
        "Family Relationship",
        min_value=1,
        max_value=5,
        value=4
    )

    traveltime = st.slider(
        "Travel Time",
        min_value=1,
        max_value=4,
        value=1
    )

    st.markdown("---")

    predict_button = st.button(
        "🔮 Predict Performance",
        use_container_width=True
    )

# ============================================================
# CREATE MODEL INPUT
# ============================================================

def create_input():

    data = {
        "school": "GP",
        "sex": "F",
        "age": 17,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 3,
        "Fedu": 2,
        "Mjob": "other",
        "Fjob": "other",
        "reason": "course",
        "guardian": "mother",
        "traveltime": traveltime,
        "studytime": study_time,
        "failures": failures,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "no",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": 1,
        "Walc": 1,
        "health": health,
        "absences": absences
    }

    return pd.DataFrame([data])

# ============================================================
# PREDICTION
# ============================================================

input_df = create_input()

try:

    transformed_data = preprocessor.transform(input_df)

    prediction = float(
        model.predict(transformed_data)[0]
    )

    prediction = max(0, min(20, prediction))

except Exception as e:

    st.error("Prediction error occurred.")
    st.code(str(e))
    st.stop()

# ============================================================
# PERFORMANCE LEVEL
# ============================================================

if prediction >= 16:
    performance_level = "Excellent"
    performance_icon = "🌟"

elif prediction >= 12:
    performance_level = "Good"
    performance_icon = "👍"

elif prediction >= 10:
    performance_level = "Average"
    performance_icon = "📘"

else:
    performance_level = "Needs Attention"
    performance_icon = "⚠️"

# ============================================================
# EARLY WARNING
# ============================================================

risk_reasons = []

if prediction < 10:
    risk_reasons.append("Predicted score is below 10")

if failures >= 2:
    risk_reasons.append("Multiple past failures")

if absences >= 20:
    risk_reasons.append("High number of absences")

if len(risk_reasons) > 0:
    warning_status = "Attention Required"
    warning_icon = "⚠️"
else:
    warning_status = "On Track"
    warning_icon = "✅"

# ============================================================
# PERFORMANCE OVERVIEW
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>📊 Performance Overview</h2>
        <p>Your academic performance summary based on the machine learning model</p>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-title">Predicted Score</div>
            <div class="metric-value">{prediction:.2f} / 20</div>
            <div class="metric-subtitle">AI model prediction</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{performance_icon}</div>
            <div class="metric-title">Performance Level</div>
            <div class="metric-value">{performance_level}</div>
            <div class="metric-subtitle">Based on predicted score</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{warning_icon}</div>
            <div class="metric-title">Early Warning</div>
            <div class="metric-value">{warning_status}</div>
            <div class="metric-subtitle">Risk monitoring</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">📚</div>
            <div class="metric-title">Study Time</div>
            <div class="metric-value">{study_time} / 4</div>
            <div class="metric-subtitle">Weekly study level</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# SCORE PROGRESS
# ============================================================

st.markdown("### 🎯 Score Progress")

progress_value = prediction / 20

st.progress(progress_value)

st.caption(
    f"Current predicted score: {prediction:.2f} out of 20"
)

# ============================================================
# EARLY WARNING SECTION
# ============================================================

if len(risk_reasons) > 0:

    st.markdown(
        """
        <div class="warning-card">
            <h3>⚠️ Early Warning Alert</h3>
            <p>
                Some factors indicate that the student may need
                additional academic support.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    for reason in risk_reasons:
        st.warning(reason)

else:

    st.markdown(
        """
        <div class="success-card">
            <h3>✅ Student is On Track</h3>
            <p>
                The current academic indicators do not trigger
                an early-warning condition.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# STUDENT INFORMATION
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>👤 Student Information</h2>
        <p>Current student profile and academic inputs</p>
    </div>
    """,
    unsafe_allow_html=True
)

info1, info2, info3, info4 = st.columns(4)

with info1:
    st.metric("Student Name", student_name)

with info2:
    st.metric("Student ID", student_id)

with info3:
    st.metric("Department", department)

with info4:
    st.metric("Absences", absences)

# ============================================================
# FACTORS ANALYSIS
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>📈 Student Factors</h2>
        <p>Important academic and lifestyle factors entered for prediction</p>
    </div>
    """,
    unsafe_allow_html=True
)

factor_data = pd.DataFrame(
    {
        "Factor": [
            "Study Time",
            "Family Relationship",
            "Free Time",
            "Going Out",
            "Health",
            "Travel Time"
        ],
        "Value": [
            study_time,
            famrel,
            freetime,
            goout,
            health,
            traveltime
        ]
    }
)

fig, ax = plt.subplots(figsize=(10, 4))

ax.barh(
    factor_data["Factor"],
    factor_data["Value"]
)

ax.set_xlabel("Value")
ax.set_title("Student Academic & Lifestyle Factors")

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ============================================================
# ABSENCE & FAILURE ANALYSIS
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>📌 Risk Indicators</h2>
        <p>Factors that can influence academic performance</p>
    </div>
    """,
    unsafe_allow_html=True
)

risk1, risk2, risk3 = st.columns(3)

with risk1:

    if absences >= 20:
        status = "High"
    elif absences >= 10:
        status = "Moderate"
    else:
        status = "Low"

    st.metric(
        "Absence Risk",
        status,
        f"{absences} absences"
    )

with risk2:

    if failures >= 2:
        status = "High"
    elif failures == 1:
        status = "Moderate"
    else:
        status = "Low"

    st.metric(
        "Failure Risk",
        status,
        f"{failures} failures"
    )

with risk3:

    if study_time >= 3:
        status = "Good"
    elif study_time == 2:
        status = "Average"
    else:
        status = "Low"

    st.metric(
        "Study Level",
        status,
        f"{study_time}/4"
    )

# ============================================================
# PERSONALIZED SUGGESTIONS
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>💡 Personalized Suggestions</h2>
        <p>Simple recommendations based on the current student inputs</p>
    </div>
    """,
    unsafe_allow_html=True
)

suggestions = []

if study_time <= 2:

    suggestions.append(
        (
            "📚 Increase Study Time",
            "Try to create a daily study schedule and gradually increase focused study time."
        )
    )

if absences >= 10:

    suggestions.append(
        (
            "🏫 Improve Attendance",
            "Regular attendance can help you follow lessons consistently and avoid missing important topics."
        )
    )

if failures >= 1:

    suggestions.append(
        (
            "📝 Focus on Difficult Subjects",
            "Identify subjects where you are struggling and spend additional practice time on them."
        )
    )

if goout >= 4:

    suggestions.append(
        (
            "⏰ Manage Social Time",
            "Balance social activities with study time and maintain a consistent academic routine."
        )
    )

if freetime >= 4:

    suggestions.append(
        (
            "📅 Plan Your Free Time",
            "Use some free time for revision, coding practice, assignments, or skill development."
        )
    )

if not suggestions:

    suggestions.append(
        (
            "🌟 Keep Going",
            "Your current inputs look reasonably balanced. Continue maintaining consistent study habits."
        )
    )

for title, description in suggestions:

    st.markdown(
        f"""
        <div class="suggestion-card">
            <strong>{title}</strong><br>
            <span>{description}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# QUICK ACTIONS
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>🚀 Quick Actions</h2>
        <p>Explore other modules of the student performance system</p>
    </div>
    """,
    unsafe_allow_html=True
)

action1, action2, action3 = st.columns(3)

with action1:

    st.markdown(
        """
        <div class="info-card">
            <h3>🔮 What-If Prediction</h3>
            <p>
                Change academic factors and observe how the
                predicted performance changes.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with action2:

    st.markdown(
        """
        <div class="info-card">
            <h3>📊 Model Performance</h3>
            <p>
                Compare machine learning models and understand
                their evaluation metrics.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with action3:

    st.markdown(
        """
        <div class="info-card">
            <h3>🤖 AI Study Planner</h3>
            <p>
                Generate personalized study recommendations
                based on student performance.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# ML INFORMATION
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>🤖 Machine Learning Information</h2>
        <p>Technology used behind this student dashboard</p>
    </div>
    """,
    unsafe_allow_html=True
)

ml1, ml2, ml3 = st.columns(3)

with ml1:

    st.markdown(
        """
        <div class="info-card">
            <h3>🌲 Random Forest</h3>
            <p>
                The dashboard uses a tuned Random Forest regression
                model to estimate student performance.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with ml2:

    st.markdown(
        """
        <div class="info-card">
            <h3>⚙️ Preprocessing</h3>
            <p>
                Numerical and categorical student information is
                processed before sending it to the trained model.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with ml3:

    st.markdown(
        """
        <div class="info-card">
            <h3>🔍 Explainable AI</h3>
            <p>
                SHAP analysis is used in the project to understand
                which factors influence model predictions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# DATASET INFORMATION
# ============================================================

try:

    df = load_dataset()

    st.markdown(
        """
        <div class="section-header">
            <h2>🗃️ Dataset Information</h2>
            <p>Information from the student performance dataset</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.metric(
            "Total Records",
            len(df)
        )

    with d2:
        st.metric(
            "Total Features",
            len(df.columns)
        )

    with d3:
        st.metric(
            "Average G3",
            f"{df['G3'].mean():.2f}"
        )

    with d4:
        st.metric(
            "Maximum G3",
            f"{df['G3'].max():.0f}"
        )

except Exception:
    pass

# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="info-card" style="margin-top:25px;">
        <h3>ℹ️ Important Note</h3>
        <p>
            This dashboard provides machine-learning-based estimates
            for educational analysis. Predictions are not guaranteed
            results and should be considered as supportive information
            rather than a final academic decision.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🎓 Student Performance Prediction System
        <br>
        Powered by Machine Learning • Streamlit • Python
        <br><br>
        Trekverse Internship Project
    </div>
    """,
    unsafe_allow_html=True
)

