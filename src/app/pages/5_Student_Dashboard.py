import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================


st.set_page_config(
    page_title="Student Academic Performance Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

# app.py is inside:
# Trekverse_Student_Performance/src/app/app.py
#
# Therefore:
# app.py -> app -> src -> project root

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)

RISK_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "risk_prediction_model.pkl"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "academic_performance_model.pkl"
)

PREPROCESSOR_PATH = None

DATA_PATH = os.path.join(
    BASE_DIR, "data", "student_academic_data.csv"
)
RISK_MODEL = None

try:
    risk_bundle = joblib.load(RISK_MODEL_PATH)
    RISK_MODEL = risk_bundle["model"]
    RISK_PREPROCESSOR = risk_bundle["preprocessor"]
    RISK_FEATURES = risk_bundle["features"]
except Exception as e:
    RISK_MODEL = None
    RISK_PREPROCESSOR = None
    RISK_FEATURES = []


# ============================================================
# HTML RENDERING FUNCTION
# ============================================================

def render_html(content):
    """
    Safely render HTML in Streamlit.

    Streamlit can treat HTML that starts with spaces
    as a code block. We therefore remove leading and
    trailing spaces from every line before rendering.
    """
    if content is None:
        return

    cleaned_html = "\n".join(
        line.strip()
        for line in str(content).strip().splitlines()
    )

    st.markdown(
        cleaned_html,
        unsafe_allow_html=True
    )


# ============================================================
# CUSTOM CSS
# ============================================================

render_html("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
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

/* Header */

.dashboard-header {
    background: linear-gradient(
        135deg,
        #1e3a8a,
        #2563eb
    );
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
}

.dashboard-header h1 {
    margin: 0;
    font-size: 34px;
    font-weight: 700;
}

.dashboard-header p {
    margin-top: 8px;
    font-size: 16px;
    opacity: 0.9;
}

/* Cards */

.metric-card {
    background: white;
    border-radius: 16px;
    padding: 22px;
    min-height: 145px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
}

.metric-icon {
    font-size: 28px;
    margin-bottom: 8px;
}

.metric-title {
    color: #64748b;
    font-size: 14px;
    font-weight: 600;
}

.metric-value {
    color: #0f172a;
    font-size: 27px;
    font-weight: 700;
    margin-top: 5px;
}

.metric-description {
    color: #94a3b8;
    font-size: 12px;
    margin-top: 6px;
}

/* Section */

.section-title {
    font-size: 23px;
    font-weight: 700;
    color: #0f172a;
    margin-top: 30px;
    margin-bottom: 15px;
}

/* Info card */

.info-card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    margin-bottom: 15px;
}

/* Status */

.status-good {
    background: #dcfce7;
    color: #166534;
    padding: 12px 18px;
    border-radius: 10px;
    font-weight: 600;
}

.status-warning {
    background: #fef3c7;
    color: #92400e;
    padding: 12px 18px;
    border-radius: 10px;
    font-weight: 600;
}

.status-danger {
    background: #fee2e2;
    color: #991b1b;
    padding: 12px 18px;
    border-radius: 10px;
    font-weight: 600;
}

/* Recommendation */

.recommendation-card {
    background: white;
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 12px;
    border-left: 5px solid #2563eb;
    box-shadow: 0 3px 12px rgba(0,0,0,0.06);
}

.recommendation-card h4 {
    margin-top: 0;
    color: #1e3a8a;
}

.recommendation-card p {
    color: #475569;
}

/* Footer */

.footer {
    text-align: center;
    padding: 25px;
    color: #64748b;
    font-size: 13px;
    margin-top: 30px;
}

</style>
""")


# ============================================================
# LOAD DATASET
# ============================================================

if not os.path.exists(DATA_PATH):

    st.error(
        f"Dataset not found.\n\n"
        f"Expected location:\n{DATA_PATH}"
    )

    st.stop()


try:

    df = pd.read_csv(DATA_PATH)

except Exception as e:

    st.error(f"Unable to read dataset: {e}")
    st.stop()


# Clean column names

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# FIND TARGET COLUMN
# ============================================================

target_candidates = [
    "g3",
    "final_grade",
    "final_score",
    "final_marks",
    "target",
    "performance",
    "score",
    "marks",
    "grade",
    "result"
]

target_column = None

for col in target_candidates:

    if col in df.columns:
        target_column = col
        break


# ============================================================
# LOAD MODEL
# ============================================================

model = None
preprocessor = None
model_available = False

if os.path.exists(MODEL_PATH):
    try:
        # The saved model is already a complete Pipeline.
        model = joblib.load(MODEL_PATH)
        model_available = True
    except Exception as e:
        st.warning(f"Performance model could not be loaded: {e}")

# The Pipeline includes its own preprocessing.
preprocessor = None


if PREPROCESSOR_PATH and os.path.exists(PREPROCESSOR_PATH):

    try:

        preprocessor = joblib.load(
            PREPROCESSOR_PATH
        )

    except Exception as e:

        preprocessor = None
        st.warning(
            f"Preprocessor could not be loaded: {e}"
        )


if model is not None:
    model_available = True


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """<div style="text-align:center;">
<div style="font-size:50px;">🎓</div>
<h2>Student Dashboard</h2>
<p>AI Academic Analytics</p>
</div>""",
    unsafe_allow_html=True
)

st.sidebar.markdown("---")


student_name = st.sidebar.text_input(
    "👤 Student Name",
    value="Student"
)

student_id = st.sidebar.text_input(
    "🆔 Student ID",
    value="STU001"
)

department = st.sidebar.text_input(
    "🏫 Department",
    value="Information Technology"
)

semester = st.sidebar.number_input(
    "📚 Semester",
    min_value=1,
    max_value=12,
    value=8
)

st.sidebar.markdown("---")

st.sidebar.info(
    "This dashboard uses Machine Learning "
    "to analyse academic performance and "
    "provide early-warning insights."
)


# ============================================================
# HEADER
# ============================================================

render_html(f"""
<div class="dashboard-header">

    <h1>🎓 Welcome, {student_name}</h1>

    <p>
        Student Academic Performance
        Intelligence Dashboard
    </p>

</div>
""")


# ============================================================
# STUDENT INFORMATION
# ============================================================

render_html("""
<div class="section-title">
    👤 Student Information
</div>
""")

info1, info2, info3, info4 = st.columns(4)

with info1:

    render_html(f"""
    <div class="metric-card">

        <div class="metric-icon">👤</div>

        <div class="metric-title">
            Student
        </div>

        <div class="metric-value">
            {student_name}
        </div>

    </div>
    """)


with info2:

    render_html(f"""
    <div class="metric-card">

        <div class="metric-icon">🆔</div>

        <div class="metric-title">
            Student ID
        </div>

        <div class="metric-value">
            {student_id}
        </div>

    </div>
    """)


with info3:

    render_html(f"""
    <div class="metric-card">

        <div class="metric-icon">🏫</div>

        <div class="metric-title">
            Department
        </div>

        <div class="metric-value">
            {department}
        </div>

    </div>
    """)


with info4:

    render_html(f"""
    <div class="metric-card">

        <div class="metric-icon">📚</div>

        <div class="metric-title">
            Semester
        </div>

        <div class="metric-value">
            {semester}
        </div>

    </div>
    """)


# ============================================================
# DETECT COMMON ACADEMIC FEATURES
# ============================================================

def find_column(possible_names):

    for name in possible_names:

        if name in df.columns:
            return name

    return None


studytime_col = find_column([
    "studytime",
    "study_time",
    "study_hours",
    "studyhours",
    "hours_studied"
])

absences_col = find_column([
    "absences",
    "absence",
    "absent_days",
    "absent",
    "attendance_absences"
])

failures_col = find_column([
    "failures",
    "failure",
    "previous_failures",
    "backlogs"
])

goout_col = find_column([
    "goout",
    "social_activity",
    "social_activities"
])

freetime_col = find_column([
    "freetime",
    "free_time"
])

health_col = find_column([
    "health",
    "health_status"
])

famrel_col = find_column([
    "famrel",
    "family_relationship"
])

traveltime_col = find_column([
    "traveltime",
    "travel_time"
])

attendance_col = find_column([
    "attendance",
    "attendance_percentage",
    "attendance_percent"
])


# ============================================================
# GET SAFE VALUE
# ============================================================

def get_numeric_value(column_name, default=0):

    if column_name is None:
        return default

    try:

        value = pd.to_numeric(
            df[column_name],
            errors="coerce"
        ).median()

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


studytime_value = get_numeric_value(
    studytime_col,
    2
)

absences_value = get_numeric_value(
    absences_col,
    0
)

failures_value = get_numeric_value(
    failures_col,
    0
)

attendance_value = get_numeric_value(
    attendance_col,
    0
)



# PREDICTION
# ============================================================

predicted_score = None
prediction_error = None

if model_available:
    try:
        # Use the exact features expected by the saved model
        if hasattr(model, "feature_names_in_"):
            required_features = list(model.feature_names_in_)
        else:
            required_features = [
                "Department",
                "Semester",
                "Attendance",
                "Internal_Marks",
                "Assignment_Marks",
                "Quiz_Marks",
                "Study_Hours",
                "Previous_CGPA",
                "Backlogs",
                "Assignment_Submission_Rate",
                "Learning_Activity",
            ]

        input_data = {}

        for feature in required_features:
            column_name = feature.lower()

            if column_name in df.columns:
                if pd.api.types.is_numeric_dtype(df[column_name]):
                    value = pd.to_numeric(
                        df[column_name], errors="coerce"
                    ).median()
                    input_data[feature] = (
                        0 if pd.isna(value) else value
                    )
                else:
                    values = df[column_name].mode()
                    input_data[feature] = (
                        values.iloc[0] if not values.empty else ""
                    )
            else:
                input_data[feature] = (
                    department if column_name == "department"
                    else semester if column_name == "semester"
                    else 0
                )

        input_df = pd.DataFrame(
            [input_data], columns=required_features
        )

        predicted_score = float(model.predict(input_df)[0])
        predicted_score = max(0, min(20, predicted_score))

    except Exception as e:
        prediction_error = str(e)
        predicted_score = None





# ============================================================
# FALLBACK PREDICTION
# ============================================================

if predicted_score is None:

    # If the existing model does not match
    # the new dataset, show a safe academic
    # indicator instead of crashing.

    if target_column:

        try:

            target_values = pd.to_numeric(
                df[target_column],
                errors="coerce"
            ).dropna()

            if len(target_values) > 0:

                predicted_score = float(
                    target_values.mean()
                )

            else:

                predicted_score = 10.0

        except Exception:

            predicted_score = 10.0

    else:

        predicted_score = 10.0



# ============================================================
# PERFORMANCE STATUS AND ML RISK PREDICTION
# ============================================================

# Keep the existing performance status based on predicted score
if predicted_score >= 15:
    performance_status = "Excellent Performance"
    performance_class = "status-good"
    score_based_risk = "Low Risk"

elif predicted_score >= 12:
    performance_status = "Good Performance"
    performance_class = "status-good"
    score_based_risk = "Low Risk"

elif predicted_score >= 8:
    performance_status = "Needs Attention"
    performance_class = "status-warning"
    score_based_risk = "Moderate Risk"

else:
    performance_status = "At Risk"
    performance_class = "status-danger"
    score_based_risk = "High Risk"

# Use the trained ML model when available
risk_level = score_based_risk

try:
    if (
        RISK_MODEL is not None
        and RISK_PREPROCESSOR is not None
        and RISK_FEATURES
    ):
        risk_input = {}

        # Use the selected student's dataset row when available
        student_rows = df[
            df["student_id"].astype(str) == str(student_id)
        ] if "student_id" in df.columns else pd.DataFrame()

        for feature in RISK_FEATURES:
            feature_lower = feature.lower()

            if not student_rows.empty and feature_lower in student_rows.columns:
                risk_input[feature] = student_rows.iloc[0][feature_lower]
            elif feature_lower in df.columns:
                if pd.api.types.is_numeric_dtype(df[feature_lower]):
                    risk_input[feature] = df[feature_lower].median()
                else:
                    risk_input[feature] = df[feature_lower].mode().iloc[0]
            else:
                # Safe fallback for a missing feature
                risk_input[feature] = 0

        risk_input_df = pd.DataFrame(
            [risk_input], columns=RISK_FEATURES
        )

        transformed_input = RISK_PREPROCESSOR.transform(risk_input_df)
        predicted_risk = RISK_MODEL.predict(transformed_input)[0]

        risk_map = {
            "Low": "Low Risk",
            "Moderate": "Moderate Risk",
            "High": "High Risk",
        }

        risk_level = risk_map.get(
            str(predicted_risk), score_based_risk
        )

except Exception as e:
    # Keep the dashboard working if risk prediction fails
    risk_level = score_based_risk

# ============================================================
# PERFORMANCE OVERVIEW
# ============================================================

render_html("""
<div class="section-title">
    📊 Academic Performance Overview
</div>
""")

m1, m2, m3, m4 = st.columns(4)


with m1:

    render_html(f"""
    <div class="metric-card">

        <div class="metric-icon">🎯</div>

        <div class="metric-title">
            Predicted Score
        </div>

        <div class="metric-value">
            {predicted_score:.2f}/20
        </div>

        <div class="metric-description">
            AI predicted academic score
        </div>

    </div>
    """)


with m2:

    render_html(f"""
    <div class="metric-card">

        <div class="metric-icon">📈</div>

        <div class="metric-title">
            Performance
        </div>

        <div class="metric-value">
            {performance_status}
        </div>

        <div class="metric-description">
            Based on predicted score
        </div>

    </div>
    """)


with m3:

    render_html(f"""
    <div class="metric-card">

        <div class="metric-icon">⚠️</div>

        <div class="metric-title">
            AI Risk Level
        </div>

        <div class="metric-value">
            {risk_level}
        </div>

        <div class="metric-description">
            Early warning indicator
        </div>

    </div>
    """)


with m4:

    render_html(f"""
    <div class="metric-card">

        <div class="metric-icon">📚</div>

        <div class="metric-title">
            Study Time
        </div>

        <div class="metric-value">
            {studytime_value:.0f}
        </div>

        <div class="metric-description">
            Study-time indicator
        </div>

    </div>
    """)


# ============================================================
# STATUS MESSAGE
# ============================================================

render_html(f"""
<div class="{performance_class}"
     style="margin-top:20px;">

    {("🟢" if risk_level == "Low Risk"
      else "🟡" if risk_level == "Moderate Risk"
      else "🔴")}

    {performance_status}

    — AI Risk Level: {risk_level}

</div>
""")


# ============================================================
# ACADEMIC FACTORS
# ============================================================

render_html("""
<div class="section-title">
    📚 Academic Factors
</div>
""")

f1, f2, f3, f4 = st.columns(4)


with f1:

    render_html(f"""
    <div class="metric-card">

        <div class="metric-icon">📅</div>

        <div class="metric-title">
            Absences
        </div>

        <div class="metric-value">
            {absences_value:.0f}
        </div>

        <div class="metric-description">
            Average dataset value
        </div>

    </div>
    """)


with f2:

    render_html(f"""
    <div class="metric-card">

        <div class="metric-icon">❌</div>

        <div class="metric-title">
            Previous Failures
        </div>

        <div class="metric-value">
            {failures_value:.0f}
        </div>

        <div class="metric-description">
            Academic difficulty indicator
        </div>

    </div>
    """)


with f3:

    render_html(f"""
    <div class="metric-card">

        <div class="metric-icon">📖</div>

        <div class="metric-title">
            Study Time
        </div>

        <div class="metric-value">
            {studytime_value:.0f}
        </div>

        <div class="metric-description">
            Study consistency indicator
        </div>

    </div>
    """)


with f4:

    render_html(f"""
    <div class="metric-card">

        <div class="metric-icon">🎯</div>

        <div class="metric-title">
            Attendance
        </div>

        <div class="metric-value">
            {attendance_value:.0f}%
        </div>

        <div class="metric-description">
            Attendance indicator
        </div>

    </div>
    """)


# ============================================================
# DATASET SUMMARY
# ============================================================

render_html("""
<div class="section-title">
    🗂️ Dataset Summary
</div>
""")

d1, d2, d3, d4 = st.columns(4)


with d1:

    render_html(f"""
    <div class="metric-card">

        <div class="metric-icon">👥</div>

        <div class="metric-title">
            Total Records
        </div>

        <div class="metric-value">
            {len(df)}
        </div>

    </div>
    """)


with d2:

    render_html(f"""
    <div class="metric-card">

        <div class="metric-icon">📊</div>

        <div class="metric-title">
            Total Columns
        </div>

        <div class="metric-value">
            {len(df.columns)}
        </div>

    </div>
    """)


with d3:

    render_html(f"""
    <div class="metric-card">

        <div class="metric-icon">🔢</div>

        <div class="metric-title">
            Numerical Columns
        </div>

        <div class="metric-value">
            {len(df.select_dtypes(include=np.number).columns)}
        </div>

    </div>
    """)


with d4:

    target_display = (
        target_column
        if target_column
        else "Not Found"
    )

    render_html(f"""
    <div class="metric-card">

        <div class="metric-icon">🎯</div>

        <div class="metric-title">
            Target Column
        </div>

        <div class="metric-value"
             style="font-size:18px;">
            {target_display}
        </div>

    </div>
    """)


# ============================================================
# TARGET INFORMATION
# ============================================================

render_html("""
<div class="section-title">
    📈 Target Analysis
</div>
""")

if target_column:

    target_values = pd.to_numeric(
        df[target_column],
        errors="coerce"
    )

    t1, t2, t3 = st.columns(3)

    with t1:

        st.metric(
            "Average Score",
            f"{target_values.mean():.2f}"
        )

    with t2:

        st.metric(
            "Highest Score",
            f"{target_values.max():.2f}"
        )

    with t3:

        st.metric(
            "Lowest Score",
            f"{target_values.min():.2f}"
        )

    chart_data = pd.DataFrame({
        "Score": target_values.dropna()
    })

    st.bar_chart(
        chart_data["Score"].value_counts().sort_index()
    )

else:

    st.info(
        "No standard academic target/score "
        "column was automatically detected in "
        "student_academic_data.csv."
    )


# ============================================================
# FEATURE INFORMATION
# ============================================================

render_html("""
<div class="section-title">
    🔍 Dataset Features
</div>
""")

feature_table = pd.DataFrame({
    "Feature": df.columns,
    "Data Type": [
        str(df[col].dtype)
        for col in df.columns
    ],
    "Missing Values": [
        int(df[col].isna().sum())
        for col in df.columns
    ],
    "Unique Values": [
        int(df[col].nunique())
        for col in df.columns
    ]
})

st.dataframe(
    feature_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# RAW DATA PREVIEW
# ============================================================

render_html("""
<div class="section-title">
    👀 Student Dataset Preview
</div>
""")

st.dataframe(
    df.head(10),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# WEAK AREAS
# ============================================================

render_html("""
<div class="section-title">
    ⚠️ Weak Areas & Early Warning
</div>
""")

weak_areas = []

if predicted_score < 12:

    weak_areas.append(
        "Overall predicted academic performance"
    )

if studytime_value < 3:

    weak_areas.append(
        "Study consistency"
    )

if absences_value > 5:

    weak_areas.append(
        "Attendance / absence management"
    )

if failures_value > 0:

    weak_areas.append(
        "Previous academic failures"
    )

if not weak_areas:

    weak_areas.append(
        "No major academic weakness detected"
    )


for area in weak_areas:

    render_html(f"""
    <div class="recommendation-card">

        <h4>⚠️ {area}</h4>

        <p>
            This area may require additional
            attention to improve academic
            performance.
        </p>

    </div>
    """)


# ============================================================
# RECOMMENDATIONS
# ============================================================

render_html("""
<div class="section-title">
    💡 Personalized Recommendations
</div>
""")

recommendations = []


if studytime_value < 3:

    recommendations.append(
        (
            "📖 Increase Study Time",
            "Try to maintain a consistent daily "
            "study schedule and gradually increase "
            "study time."
        )
    )


if absences_value > 5:

    recommendations.append(
        (
            "🏫 Improve Attendance",
            "Regular class attendance can help "
            "understand concepts and improve "
            "academic performance."
        )
    )


if failures_value > 0:

    recommendations.append(
        (
            "👨‍🏫 Academic Support",
            "Seek guidance from faculty members "
            "for subjects where previous difficulties "
            "were observed."
        )
    )


if predicted_score < 12:

    recommendations.append(
        (
            "🎯 Focus on Weak Subjects",
            "Identify difficult subjects and spend "
            "additional practice time on them."
        )
    )


recommendations.append(
    (
        "📝 Regular Revision",
        "Revise important concepts every week "
        "instead of studying only before examinations."
    )
)


for title, description in recommendations:

    render_html(f"""
    <div class="recommendation-card">

        <h4>{title}</h4>

        <p>{description}</p>

    </div>
    """)


# ============================================================
# AI / MACHINE LEARNING INFORMATION
# ============================================================

render_html("""
<div class="section-title">
    🤖 Artificial Intelligence Analysis
</div>
""")

ai1, ai2, ai3 = st.columns(3)


with ai1:

    render_html("""
    <div class="info-card">

        <h3>🌲 Random Forest</h3>

        <p>
            The project uses a Random Forest
            regression model to predict
            student academic performance.
        </p>

    </div>
    """)


with ai2:

    render_html("""
    <div class="info-card">

        <h3>⚙️ Preprocessing</h3>

        <p>
            Numerical and categorical student
            information is transformed before
            giving it to the Machine Learning model.
        </p>

    </div>
    """)


with ai3:

    render_html("""
    <div class="info-card">

        <h3>🔎 Explainable AI</h3>

        <p>
            Explainable AI techniques such as
            SHAP can help understand which
            factors influence predictions.
        </p>

    </div>
    """)



render_html("""
<div class="section-title">
    ⚙️ Model Status
</div>
""")

if model_available:

    st.success(
        "✅ Machine Learning model and "
        "preprocessor loaded successfully."
    )

else:

    st.warning(
        "⚠️ Saved Machine Learning model "
        "could not be loaded."
    )

if prediction_error:

    st.warning(
        "The saved model could not process "
        "the current dataset schema. "
        "The dashboard is displaying a safe "
        "fallback academic indicator."
    )




render_html("""
<div class="section-title">
    📄 Student Performance Report
</div>
""")

report_text = f"""
STUDENT ACADEMIC PERFORMANCE REPORT
===================================

Student Name     : {student_name}
Student ID       : {student_id}
Department       : {department}
Semester         : {semester}

AI PREDICTION
-------------
Predicted Score  : {predicted_score:.2f}/20
Performance      : {performance_status}
Risk Level       : {risk_level}

ACADEMIC FACTORS
----------------
Study Time       : {studytime_value:.0f}
Absences         : {absences_value:.0f}
Previous Failures: {failures_value:.0f}
Attendance       : {attendance_value:.0f}%

DATASET
-------
Total Records    : {len(df)}
Total Columns    : {len(df.columns)}
Target Column    : {target_column if target_column else "Not Found"}

Generated On     : {datetime.now().strftime("%d-%m-%Y %H:%M")}
"""

st.download_button(
    label="📥 Download Student Report",
    data=report_text,
    file_name="student_performance_report.txt",
    mime="text/plain"
)



render_html("""
<div class="footer">

    🎓 Student Academic Performance
    Intelligence System

    <br><br>

    Powered by Machine Learning •
    Random Forest • Explainable AI

</div>
""")