import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
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

user = require_role(["Faculty", "Admin"])

logout_button()

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Faculty Dashboard",
    page_icon="👨‍🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "student-mat.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tuned_random_forest_regression.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "models",
    "preprocessor.pkl"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .stApp {
        background: linear-gradient(
            135deg,
            #f8fbff 0%,
            #eef5ff 50%,
            #f8fbff 100%
        );
    }

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

    .hero {
        background: linear-gradient(
            135deg,
            #172554,
            #1d4ed8,
            #2563eb
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

    .section-header {
        background: white;
        padding: 20px 24px;
        border-radius: 16px;
        margin-top: 20px;
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

    .alert-card {
        background: #fff7ed;
        border-left: 6px solid #f97316;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
    }

    .alert-card h3 {
        color: #9a3412;
        margin-top: 0;
    }

    .alert-card p {
        color: #7c2d12;
    }

    .success-card {
        background: #f0fdf4;
        border-left: 6px solid #22c55e;
        padding: 20px;
        border-radius: 15px;
    }

    .success-card h3 {
        color: #166534;
        margin-top: 0;
    }

    .success-card p {
        color: #166534;
    }

    .footer {
        text-align: center;
        padding: 30px;
        color: #64748b;
        font-size: 13px;
        margin-top: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_PATH,
        sep=";"
    )

    return df


@st.cache_resource
def load_model():

    return joblib.load(
        MODEL_PATH
    )


@st.cache_resource
def load_preprocessor():

    return joblib.load(
        PREPROCESSOR_PATH
    )


try:

    df = load_data()

except Exception as e:

    st.error("Unable to load student dataset.")
    st.code(str(e))
    st.stop()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>👨‍🏫 Faculty Dashboard</h1>
        <p>Class-level student performance monitoring and early-warning analytics</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 👨‍🏫 Faculty Portal")

    st.markdown("---")

    st.markdown("### 🏫 Class Filters")

    school_filter = st.selectbox(
        "School",
        ["All"] + sorted(df["school"].unique().tolist())
    )

    sex_filter = st.selectbox(
        "Gender",
        ["All"] + sorted(df["sex"].unique().tolist())
    )

    performance_filter = st.selectbox(
        "Performance Group",
        [
            "All",
            "Excellent",
            "Good",
            "Average",
            "Needs Attention"
        ]
    )

    st.markdown("---")

    st.markdown("### 📊 Dashboard")

    st.info(
        "Use the filters to analyse different groups of students."
    )


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()

if school_filter != "All":

    filtered_df = filtered_df[
        filtered_df["school"] == school_filter
    ]

if sex_filter != "All":

    filtered_df = filtered_df[
        filtered_df["sex"] == sex_filter
    ]


# ============================================================
# PERFORMANCE GROUP
# ============================================================

def get_performance_group(score):

    if score >= 16:
        return "Excellent"

    elif score >= 12:
        return "Good"

    elif score >= 10:
        return "Average"

    else:
        return "Needs Attention"


filtered_df = filtered_df.copy()

filtered_df["Performance Group"] = (
    filtered_df["G3"]
    .apply(get_performance_group)
)


if performance_filter != "All":

    filtered_df = filtered_df[
        filtered_df["Performance Group"]
        == performance_filter
    ]


# ============================================================
# KEY METRICS
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>📊 Class Performance Overview</h2>
        <p>Overall academic statistics for the selected student group</p>
    </div>
    """,
    unsafe_allow_html=True
)

total_students = len(filtered_df)

average_score = (
    filtered_df["G3"].mean()
    if total_students > 0
    else 0
)

average_absences = (
    filtered_df["absences"].mean()
    if total_students > 0
    else 0
)

attention_students = (
    len(
        filtered_df[
            filtered_df["Performance Group"]
            == "Needs Attention"
        ]
    )
)

passed_students = (
    len(
        filtered_df[
            filtered_df["G3"] >= 10
        ]
    )
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">👨‍🎓</div>
            <div class="metric-title">Total Students</div>
            <div class="metric-value">{total_students}</div>
            <div class="metric-subtitle">Selected group</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-title">Average Score</div>
            <div class="metric-value">{average_score:.2f}</div>
            <div class="metric-subtitle">Out of 20</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">⚠️</div>
            <div class="metric-title">Need Attention</div>
            <div class="metric-value">{attention_students}</div>
            <div class="metric-subtitle">Students below 10</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">🏫</div>
            <div class="metric-title">Average Absences</div>
            <div class="metric-value">{average_absences:.1f}</div>
            <div class="metric-subtitle">Per student</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CLASS STATUS
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>📌 Class Status</h2>
        <p>Summary of students based on their academic results</p>
    </div>
    """,
    unsafe_allow_html=True
)

if total_students > 0:

    pass_percentage = (
        passed_students /
        total_students
    ) * 100

else:

    pass_percentage = 0


c1, c2 = st.columns(2)

with c1:

    st.metric(
        "Students Meeting Score 10+",
        passed_students
    )

with c2:

    st.metric(
        "Score 10+ Percentage",
        f"{pass_percentage:.1f}%"
    )


# ============================================================
# PERFORMANCE DISTRIBUTION
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>📈 Performance Distribution</h2>
        <p>Number of students in each performance category</p>
    </div>
    """,
    unsafe_allow_html=True
)

performance_counts = (
    filtered_df["Performance Group"]
    .value_counts()
    .reindex(
        [
            "Excellent",
            "Good",
            "Average",
            "Needs Attention"
        ],
        fill_value=0
    )
)

fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.bar(
    performance_counts.index,
    performance_counts.values
)

ax.set_ylabel("Number of Students")
ax.set_xlabel("Performance Category")
ax.set_title("Student Performance Distribution")

plt.xticks(rotation=15)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)


# ============================================================
# SCORE DISTRIBUTION
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>📊 Score Distribution</h2>
        <p>Distribution of final student scores</p>
    </div>
    """,
    unsafe_allow_html=True
)

fig, ax = plt.subplots(
    figsize=(10, 4)
)

ax.hist(
    filtered_df["G3"],
    bins=11
)

ax.set_xlabel("Final Score (G3)")
ax.set_ylabel("Number of Students")
ax.set_title("Distribution of Student Scores")

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)


# ============================================================
# ATTENDANCE ANALYSIS
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>🏫 Attendance Analysis</h2>
        <p>Relationship between absences and academic performance</p>
    </div>
    """,
    unsafe_allow_html=True
)

fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.scatter(
    filtered_df["absences"],
    filtered_df["G3"],
    alpha=0.7
)

ax.set_xlabel("Absences")
ax.set_ylabel("Final Score (G3)")
ax.set_title("Absences vs Student Performance")

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)


# ============================================================
# RISK STUDENTS
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>⚠️ Students Requiring Attention</h2>
        <p>Students with lower scores, high absences, or previous failures</p>
    </div>
    """,
    unsafe_allow_html=True
)

risk_df = filtered_df[
    (
        filtered_df["G3"] < 10
    )
    |
    (
        filtered_df["absences"] >= 20
    )
    |
    (
        filtered_df["failures"] >= 2
    )
].copy()


if len(risk_df) > 0:

    risk_df["Risk Reason"] = ""

    for index in risk_df.index:

        reasons = []

        if risk_df.loc[index, "G3"] < 10:
            reasons.append("Low Score")

        if risk_df.loc[index, "absences"] >= 20:
            reasons.append("High Absence")

        if risk_df.loc[index, "failures"] >= 2:
            reasons.append("Past Failures")

        risk_df.loc[index, "Risk Reason"] = ", ".join(
            reasons
        )

    display_columns = [
        "school",
        "sex",
        "age",
        "studytime",
        "failures",
        "absences",
        "G3",
        "Risk Reason"
    ]

    display_columns = [
        col
        for col in display_columns
        if col in risk_df.columns
    ]

    st.dataframe(
        risk_df[display_columns].sort_values(
            "G3"
        ),
        use_container_width=True,
        hide_index=True
    )

else:

    st.markdown(
        """
        <div class="success-card">
            <h3>✅ No High-Risk Students Found</h3>
            <p>
                No students in the selected group currently
                meet the defined attention criteria.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TOP PERFORMERS
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>🌟 Top Performing Students</h2>
        <p>Students with the highest recorded final scores</p>
    </div>
    """,
    unsafe_allow_html=True
)

top_students = (
    filtered_df
    .sort_values(
        "G3",
        ascending=False
    )
    .head(10)
)

top_columns = [
    "school",
    "sex",
    "age",
    "studytime",
    "absences",
    "G3"
]

top_columns = [
    col
    for col in top_columns
    if col in top_students.columns
]

st.dataframe(
    top_students[top_columns],
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ACADEMIC FACTORS
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>📚 Academic Factors</h2>
        <p>Average values of important student factors</p>
    </div>
    """,
    unsafe_allow_html=True
)

factor_columns = [
    "studytime",
    "failures",
    "absences",
    "goout",
    "freetime",
    "health",
    "famrel",
    "traveltime"
]

available_factors = [
    col
    for col in factor_columns
    if col in filtered_df.columns
]

if len(available_factors) > 0:

    factor_means = (
        filtered_df[available_factors]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.barh(
        factor_means.index,
        factor_means.values
    )

    ax.set_xlabel("Average Value")
    ax.set_title("Average Student Factors")

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# FACULTY INSIGHTS
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>💡 Faculty Insights</h2>
        <p>Automatically generated observations from the selected data</p>
    </div>
    """,
    unsafe_allow_html=True
)

if total_students > 0:

    if average_absences >= 15:

        st.warning(
            "🏫 The selected group has a relatively high average "
            "absence level. Faculty may consider attendance monitoring."
        )

    else:

        st.success(
            "🏫 The selected group's average absence level is "
            "below the high-attendance-risk threshold."
        )

    if attention_students > 0:

        st.warning(
            f"⚠️ {attention_students} student(s) are currently "
            "in the 'Needs Attention' performance category."
        )

    else:

        st.success(
            "✅ No students fall into the 'Needs Attention' "
            "category in the selected group."
        )

    if average_score >= 12:

        st.info(
            "📚 The selected group has an average final score "
            "of 12 or above."
        )

    else:

        st.info(
            "📚 The selected group's average final score is "
            "below 12, so additional academic support may be useful."
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>🤖 Machine Learning Information</h2>
        <p>Technology used in the student performance project</p>
    </div>
    """,
    unsafe_allow_html=True
)

m1, m2, m3 = st.columns(3)

with m1:

    st.markdown(
        """
        <div class="info-card">
            <h3>🌲 Tuned Random Forest</h3>
            <p>
                The project uses a tuned Random Forest regression
                model for student performance prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with m2:

    st.markdown(
        """
        <div class="info-card">
            <h3>🔍 Explainable AI</h3>
            <p>
                SHAP analysis is included in the project to help
                understand important factors affecting predictions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with m3:

    st.markdown(
        """
        <div class="info-card">
            <h3>⚠️ Early Warning</h3>
            <p>
                Score, absence, and previous-failure indicators
                can be used to identify students requiring attention.
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
        👨‍🏫 Faculty Performance Monitoring System
        <br>
        Machine Learning • Data Analytics • Streamlit
        <br><br>
        Trekverse Internship Project
    </div>
    """,
    unsafe_allow_html=True
)