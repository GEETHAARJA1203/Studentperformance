import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Early Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "tuned_random_forest_regression.pkl"
)

PREPROCESSOR_PATH = (
    BASE_DIR
    / "models"
    / "preprocessor.pkl"
)

DATA_PATH = (
    BASE_DIR
    / "data"
    / "student_risk_data.csv"
)


# ============================================================
# LOAD MODEL AND PREPROCESSOR
# ============================================================

@st.cache_resource
def load_models():

    model = joblib.load(MODEL_PATH)

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    return model, preprocessor


try:

    model, preprocessor = load_models()

except Exception as e:

    st.error(
        "Unable to load the trained ML model or preprocessor."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# LOAD STUDENT DATA
# ============================================================

if DATA_PATH.exists():

    try:

        df = pd.read_csv(DATA_PATH)

    except Exception:

        df = pd.DataFrame()

else:

    df = pd.DataFrame()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       MAIN APPLICATION
       ====================================================== */

    .stApp {

        background: #f5f7fb;

    }

    .block-container {

        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;

    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {

        background: #111827;

    }

    section[data-testid="stSidebar"] * {

        color: white !important;

    }


    /* ======================================================
       HERO HEADER
       ====================================================== */

    .hero {

        padding: 32px 38px;

        border-radius: 24px;

        background:
            linear-gradient(
                135deg,
                #111827,
                #1d4ed8
            );

        color: white;

        margin-bottom: 28px;

        box-shadow:
            0 12px 35px
            rgba(29, 78, 216, 0.20);

    }

    .hero h1 {

        font-size: 36px;

        font-weight: 800;

        margin: 0;

    }

    .hero p {

        font-size: 16px;

        margin-top: 10px;

        opacity: 0.85;

    }


    /* ======================================================
       SECTION TITLES
       ====================================================== */

    .section-title {

        font-size: 25px;

        font-weight: 800;

        color: #111827;

        margin-top: 15px;

        margin-bottom: 18px;

    }


    /* ======================================================
       METRIC CARDS
       ====================================================== */

    .metric-card {

        background: white;

        padding: 23px;

        border-radius: 18px;

        border:
            1px solid #e5e7eb;

        box-shadow:
            0 7px 22px
            rgba(15, 23, 42, 0.05);

        min-height: 125px;

    }

    .metric-label {

        color: #64748b;

        font-size: 13px;

        font-weight: 700;

        letter-spacing: 0.5px;

    }

    .metric-number {

        color: #111827;

        font-size: 31px;

        font-weight: 800;

        margin-top: 8px;

    }


    /* ======================================================
       INPUT PANEL
       ====================================================== */

    .input-panel {

        background: white;

        padding: 26px;

        border-radius: 21px;

        border:
            1px solid #e5e7eb;

        box-shadow:
            0 8px 24px
            rgba(15, 23, 42, 0.05);

    }


    /* ======================================================
       PREDICTION CARD
       ====================================================== */

    .prediction-box {

        background: white;

        padding: 32px;

        border-radius: 22px;

        border:
            1px solid #e5e7eb;

        box-shadow:
            0 10px 30px
            rgba(15, 23, 42, 0.07);

        text-align: center;

    }

    .prediction-grade {

        font-size: 58px;

        font-weight: 900;

        color: #2563eb;

        margin-top: 8px;

    }

    .prediction-label {

        color: #64748b;

        font-size: 14px;

        margin-bottom: 20px;

    }


    /* ======================================================
       RISK BADGES
       ====================================================== */

    .high {

        background: #fee2e2;

        color: #b91c1c;

    }

    .medium {

        background: #fef3c7;

        color: #b45309;

    }

    .low {

        background: #dcfce7;

        color: #15803d;

    }

    .risk-badge {

        padding: 13px;

        border-radius: 14px;

        font-weight: 800;

        font-size: 17px;

    }


    /* ======================================================
       INFORMATION BOX
       ====================================================== */

    .info-box {

        background: #eff6ff;

        border-left:
            5px solid #2563eb;

        padding: 17px;

        border-radius: 10px;

        color: #1e3a8a;

        margin-bottom: 15px;

    }


    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button {

        width: 100%;

        height: 48px;

        border-radius: 12px;

        font-weight: 800;

    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {

        text-align: center;

        color: #64748b;

        font-size: 13px;

        padding-top: 30px;

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>
            🎓 Student Early Warning System
        </h1>

        <p>
            AI-powered academic performance monitoring,
            risk detection and early intervention
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🎓 Early Warning"
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊 Overview",
            "🔮 Student Prediction",
            "🔬 What-If Prediction",
            "📋 Risk Analysis"
        ]
    )

    st.markdown("---")

    st.markdown(
        "### 🤖 Machine Learning"
    )

    st.caption(
        "Tuned Random Forest Regression"
    )

    st.caption(
        "Student Performance Prediction"
    )

    st.markdown("---")

    st.caption(
        "Academic Risk Monitoring System"
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_risk_level(grade):

    if grade < 8:

        return "High Risk"

    elif grade <= 10:

        return "Medium Risk"

    else:

        return "Low Risk"


def get_risk_class(risk):

    if risk == "High Risk":

        return "high"

    elif risk == "Medium Risk":

        return "medium"

    return "low"


def get_risk_icon(risk):

    if risk == "High Risk":

        return "🔴"

    elif risk == "Medium Risk":

        return "🟠"

    return "🟢"


def create_student_data(
    studytime,
    failures,
    absences,
    sex="F",
    age=17,
    higher="yes",
    internet="yes"
):

    return {

        "school": "GP",

        "sex": sex,

        "age": age,

        "address": "U",

        "famsize": "GT3",

        "Pstatus": "A",

        "Medu": 2,

        "Fedu": 2,

        "Mjob": "other",

        "Fjob": "other",

        "reason": "course",

        "guardian": "mother",

        "traveltime": 2,

        "studytime": studytime,

        "failures": failures,

        "schoolsup": "no",

        "famsup": "yes",

        "paid": "no",

        "activities": "yes",

        "nursery": "yes",

        "higher": higher,

        "internet": internet,

        "romantic": "no",

        "famrel": 4,

        "freetime": 3,

        "goout": 3,

        "Dalc": 1,

        "Walc": 1,

        "health": 3,

        "absences": absences
    }


def predict_grade(student):

    student_df = pd.DataFrame(
        [student]
    )

    processed = preprocessor.transform(
        student_df
    )

    prediction = model.predict(
        processed
    )[0]

    prediction = max(
        0,
        min(20, prediction)
    )

    return float(prediction)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "📊 Overview":

    st.markdown(
        '<div class="section-title">'
        '📊 Academic Risk Overview'
        '</div>',
        unsafe_allow_html=True
    )

    if df.empty:

        st.warning(
            "No student data found."
        )

    else:

        # ----------------------------------------------------
        # RISK COUNTS
        # ----------------------------------------------------

        total_students = len(df)

        high_risk = len(
            df[df["G3"] < 8]
        )

        medium_risk = len(
            df[
                (df["G3"] >= 8)
                &
                (df["G3"] <= 10)
            ]
        )

        low_risk = len(
            df[df["G3"] > 10]
        )


        # ----------------------------------------------------
        # METRIC CARDS
        # ----------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        TOTAL STUDENTS
                    </div>

                    <div class="metric-number">
                        {total_students}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        🔴 HIGH RISK
                    </div>

                    <div class="metric-number">
                        {high_risk}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        🟠 MEDIUM RISK
                    </div>

                    <div class="metric-number">
                        {medium_risk}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c4:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        🟢 LOW RISK
                    </div>

                    <div class="metric-number">
                        {low_risk}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        st.markdown("")


        # ----------------------------------------------------
        # CHARTS
        # ----------------------------------------------------

        chart1, chart2 = st.columns(2)


        # ----------------------------------------------------
        # RISK DISTRIBUTION
        # ----------------------------------------------------

        with chart1:

            st.markdown(
                "### 📊 Risk Distribution"
            )

            risk_chart = pd.DataFrame(
                {
                    "Risk Level": [
                        "High Risk",
                        "Medium Risk",
                        "Low Risk"
                    ],

                    "Students": [
                        high_risk,
                        medium_risk,
                        low_risk
                    ]
                }
            )

            fig = px.bar(
                risk_chart,
                x="Risk Level",
                y="Students",
                text="Students",
                title=""
            )

            fig.update_traces(
                textposition="outside"
            )

            fig.update_layout(
                height=350,
                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # ----------------------------------------------------
        # GRADE DISTRIBUTION
        # ----------------------------------------------------

        with chart2:

            st.markdown(
                "### 📈 Final Grade Distribution"
            )

            grade_counts = (
                df["G3"]
                .value_counts()
                .sort_index()
            )

            grade_chart = pd.DataFrame(
                {
                    "Grade": grade_counts.index,
                    "Students": grade_counts.values
                }
            )

            fig2 = px.line(
                grade_chart,
                x="Grade",
                y="Students",
                markers=True
            )

            fig2.update_layout(
                height=350,
                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=20
                )
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )


        # ----------------------------------------------------
        # STUDENT DATA
        # ----------------------------------------------------

        st.markdown(
            "### 👥 Student Records"
        )

        display_df = df.copy()

        if "G3" in display_df.columns:

            display_df["Risk Level"] = (
                display_df["G3"]
                .apply(get_risk_level)
            )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PAGE 2 — STUDENT PREDICTION
# ============================================================

elif page == "🔮 Student Prediction":

    st.markdown(
        '<div class="section-title">'
        '🔮 Student Risk Prediction'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="info-box">

            Enter the student's academic information.
            The trained Machine Learning model will estimate
            the student's final grade and academic risk.

        </div>
        """,
        unsafe_allow_html=True
    )


    left, right = st.columns(
        [1.15, 0.85]
    )


    # ========================================================
    # INPUT PANEL
    # ========================================================

    with left:

        st.markdown(
            '<div class="input-panel">',
            unsafe_allow_html=True
        )

        st.markdown(
            "### 👤 Student Profile"
        )

        col1, col2 = st.columns(2)


        with col1:

            student_id = st.number_input(
                "Student ID",
                min_value=1,
                value=101
            )

            age = st.number_input(
                "Age",
                min_value=15,
                max_value=25,
                value=17
            )

            sex = st.selectbox(
                "Gender",
                ["F", "M"]
            )

            studytime = st.selectbox(
                "Study Time",
                [1, 2, 3, 4],
                index=1,
                help=(
                    "1 = Very Low, "
                    "2 = Low, "
                    "3 = High, "
                    "4 = Very High"
                )
            )


        with col2:

            failures = st.selectbox(
                "Previous Failures",
                [0, 1, 2, 3]
            )

            absences = st.number_input(
                "Absences",
                min_value=0,
                max_value=100,
                value=5
            )

            higher = st.selectbox(
                "Plans Higher Education?",
                ["yes", "no"]
            )

            internet = st.selectbox(
                "Internet Access",
                ["yes", "no"]
            )


        st.markdown("")


        predict_button = st.button(
            "🚀 ANALYZE STUDENT RISK"
        )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # ========================================================
    # RESULT PANEL
    # ========================================================

    with right:

        if predict_button:

            student = create_student_data(
                studytime=studytime,
                failures=failures,
                absences=absences,
                sex=sex,
                age=age,
                higher=higher,
                internet=internet
            )

            try:

                prediction = predict_grade(
                    student
                )

            except Exception as e:

                st.error(
                    "Prediction failed."
                )

                st.code(
                    str(e)
                )

                st.stop()


            risk = get_risk_level(
                prediction
            )

            risk_class = get_risk_class(
                risk
            )

            risk_icon = get_risk_icon(
                risk
            )


            st.markdown(
                f"""
                <div class="prediction-box">

                    <h3>
                        🎯 Prediction Result
                    </h3>

                    <div class="prediction-grade">

                        {prediction:.2f}

                    </div>

                    <div class="prediction-label">

                        Predicted Final Grade / 20

                    </div>

                    <div class="risk-badge {risk_class}">

                        {risk_icon} {risk}

                    </div>

                    <br>

                    <div>

                        <b>Student ID:</b>
                        {student_id}

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # PERFORMANCE GAUGE
            # ------------------------------------------------

            st.markdown(
                "### 🎯 Performance Gauge"
            )

            gauge = go.Figure(
                go.Indicator(

                    mode="gauge+number",

                    value=prediction,

                    number={
                        "suffix": " / 20"
                    },

                    gauge={

                        "axis": {
                            "range": [0, 20]
                        },

                        "steps": [

                            {
                                "range": [0, 8],
                                "color": "#fecaca"
                            },

                            {
                                "range": [8, 10],
                                "color": "#fde68a"
                            },

                            {
                                "range": [10, 20],
                                "color": "#bbf7d0"
                            }

                        ],

                        "threshold": {

                            "line": {
                                "color": "#111827",
                                "width": 5
                            },

                            "value": prediction

                        }

                    }

                )
            )

            gauge.update_layout(
                height=280,

                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=20
                )
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )


            # ------------------------------------------------
            # INTERVENTION
            # ------------------------------------------------

            st.markdown(
                "### ⚠️ Early Intervention"
            )


            if risk == "High Risk":

                st.error(
                    "Immediate academic support and "
                    "faculty monitoring are recommended."
                )

            elif risk == "Medium Risk":

                st.warning(
                    "Regular monitoring and additional "
                    "academic support may be helpful."
                )

            else:

                st.success(
                    "The student currently shows a "
                    "lower predicted academic risk."
                )


            # ------------------------------------------------
            # RECOMMENDATIONS
            # ------------------------------------------------

            st.markdown(
                "### 💡 Recommended Actions"
            )

            recommendations = []


            if absences > 10:

                recommendations.append(
                    "Improve attendance and reduce unnecessary absences."
                )


            if studytime <= 1:

                recommendations.append(
                    "Increase weekly study time gradually."
                )


            if failures > 0:

                recommendations.append(
                    "Provide additional support in previously failed subjects."
                )


            if higher == "no":

                recommendations.append(
                    "Discuss higher-education opportunities and career planning."
                )


            if not recommendations:

                recommendations.append(
                    "Continue the current study routine and maintain consistency."
                )


            for recommendation in recommendations:

                st.markdown(
                    f"• {recommendation}"
                )


        else:

            st.markdown(
                """
                <div class="prediction-box">

                    <h3>
                        🎯 Prediction Result
                    </h3>

                    <div class="prediction-grade">
                        --
                    </div>

                    <div class="prediction-label">

                        Predicted Final Grade / 20

                    </div>

                    <div class="risk-badge low">

                        Waiting for student information

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# PAGE 3 — WHAT-IF PREDICTION
# ============================================================

elif page == "🔬 What-If Prediction":

    st.markdown(
        '<div class="section-title">'
        '🔬 What-If Student Analysis'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="info-box">

            Change study time, failures and absences to explore
            how different academic scenarios could affect the
            model's predicted final grade.

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # CURRENT VALUES
    # ========================================================

    st.markdown(
        "### 👤 Current Student Situation"
    )


    current1, current2, current3 = st.columns(3)


    with current1:

        current_studytime = st.slider(
            "Current Study Time",
            min_value=1,
            max_value=4,
            value=2,
            key="current_study"
        )


    with current2:

        current_failures = st.slider(
            "Current Failures",
            min_value=0,
            max_value=3,
            value=0,
            key="current_failure"
        )


    with current3:

        current_absences = st.slider(
            "Current Absences",
            min_value=0,
            max_value=50,
            value=5,
            key="current_absence"
        )


    # ========================================================
    # WHAT-IF VALUES
    # ========================================================

    st.markdown(
        "### 🔄 What-If Scenario"
    )


    what1, what2, what3 = st.columns(3)


    with what1:

        new_studytime = st.slider(
            "What-If Study Time",
            min_value=1,
            max_value=4,
            value=3,
            key="new_study"
        )


    with what2:

        new_failures = st.slider(
            "What-If Failures",
            min_value=0,
            max_value=3,
            value=0,
            key="new_failure"
        )


    with what3:

        new_absences = st.slider(
            "What-If Absences",
            min_value=0,
            max_value=50,
            value=3,
            key="new_absence"
        )


    # ========================================================
    # CALCULATE PREDICTIONS
    # ========================================================

    current_student = create_student_data(
        studytime=current_studytime,
        failures=current_failures,
        absences=current_absences
    )


    new_student = create_student_data(
        studytime=new_studytime,
        failures=new_failures,
        absences=new_absences
    )


    try:

        current_prediction = predict_grade(
            current_student
        )

        what_if_prediction = predict_grade(
            new_student
        )

    except Exception as e:

        st.error(
            "What-if prediction failed."
        )

        st.code(
            str(e)
        )

        st.stop()


    difference = (
        what_if_prediction
        -
        current_prediction
    )


    current_risk = get_risk_level(
        current_prediction
    )

    what_if_risk = get_risk_level(
        what_if_prediction
    )


    # ========================================================
    # RESULT CARDS
    # ========================================================

    st.markdown(
        "### 📊 Scenario Comparison"
    )


    result1, result2, result3 = st.columns(3)


    with result1:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    CURRENT PREDICTION
                </div>

                <div class="metric-number">
                    {current_prediction:.2f} / 20
                </div>

                <p>
                    {get_risk_icon(current_risk)}
                    {current_risk}
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with result2:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    WHAT-IF PREDICTION
                </div>

                <div class="metric-number">
                    {what_if_prediction:.2f} / 20
                </div>

                <p>
                    {get_risk_icon(what_if_risk)}
                    {what_if_risk}
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with result3:

        if difference > 0:

            change = (
                f"⬆️ +{difference:.2f}"
            )

        elif difference < 0:

            change = (
                f"⬇️ {difference:.2f}"
            )

        else:

            change = (
                "➡️ 0.00"
            )


        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    PREDICTED CHANGE
                </div>

                <div class="metric-number">
                    {change}
                </div>

                <p>
                    Difference in predicted grade
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # BEFORE VS AFTER CHART
    # ========================================================

    st.markdown(
        "### 📈 Before vs What-If"
    )


    comparison_df = pd.DataFrame(
        {
            "Scenario": [
                "Current",
                "What-If"
            ],

            "Predicted Grade": [
                current_prediction,
                what_if_prediction
            ]
        }
    )


    comparison_fig = px.bar(
        comparison_df,
        x="Scenario",
        y="Predicted Grade",
        text="Predicted Grade",
        range_y=[0, 20]
    )


    comparison_fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )


    comparison_fig.update_layout(
        height=400,

        yaxis_title="Predicted Grade / 20",

        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        )
    )


    st.plotly_chart(
        comparison_fig,
        use_container_width=True
    )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.markdown(
        "### 💡 What Does This Mean?"
    )


    if difference > 0:

        st.success(
            f"""
            Under the selected what-if scenario, the predicted
            grade increases by **{difference:.2f} points**.

            The predicted risk changes from
            **{current_risk}** to **{what_if_risk}**.

            This is a model-based scenario analysis, not a guarantee
            of the student's actual future result.
            """
        )

    elif difference < 0:

        st.warning(
            f"""
            Under the selected what-if scenario, the predicted
            grade decreases by **{abs(difference):.2f} points**.

            The predicted risk changes from
            **{current_risk}** to **{what_if_risk}**.

            This is a model-based scenario analysis, not a guarantee
            of the student's actual future result.
            """
        )

    else:

        st.info(
            """
            The selected changes do not alter the model's
            predicted grade.
            """
        )


# ============================================================
# PAGE 4 — RISK ANALYSIS
# ============================================================

elif page == "📋 Risk Analysis":

    st.markdown(
        '<div class="section-title">'
        '📋 Student Risk Analysis'
        '</div>',
        unsafe_allow_html=True
    )


    if df.empty:

        st.warning(
            "No student dataset found."
        )

    else:

        analysis_df = df.copy()


        # ----------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------

        analysis_df["Risk Level"] = (
            analysis_df["G3"]
            .apply(get_risk_level)
        )


        # ----------------------------------------------------
        # RISK ICON
        # ----------------------------------------------------

        analysis_df["Status"] = (
            analysis_df["Risk Level"]
            .apply(get_risk_icon)
        )


        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        high_count = len(
            analysis_df[
                analysis_df["Risk Level"]
                == "High Risk"
            ]
        )


        medium_count = len(
            analysis_df[
                analysis_df["Risk Level"]
                == "Medium Risk"
            ]
        )


        low_count = len(
            analysis_df[
                analysis_df["Risk Level"]
                == "Low Risk"
            ]
        )


        r1, r2, r3 = st.columns(3)


        with r1:

            st.metric(
                "🔴 High Risk",
                high_count
            )


        with r2:

            st.metric(
                "🟠 Medium Risk",
                medium_count
            )


        with r3:

            st.metric(
                "🟢 Low Risk",
                low_count
            )


        st.markdown("")


        # ----------------------------------------------------
        # FILTER
        # ----------------------------------------------------

        selected_risk = st.selectbox(
            "Filter by Risk Level",
            [
                "All",
                "High Risk",
                "Medium Risk",
                "Low Risk"
            ]
        )


        if selected_risk != "All":

            filtered_df = analysis_df[
                analysis_df["Risk Level"]
                == selected_risk
            ]

        else:

            filtered_df = analysis_df


        # ----------------------------------------------------
        # TABLE
        # ----------------------------------------------------

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # CSV UPLOAD
        # ----------------------------------------------------

        st.markdown(
            "### 📁 Upload Student Dataset"
        )


        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"]
        )


        if uploaded_file is not None:

            try:

                uploaded_df = pd.read_csv(
                    uploaded_file
                )

                st.success(
                    f"Successfully loaded "
                    f"{len(uploaded_df)} student records."
                )


                st.dataframe(
                    uploaded_df,
                    use_container_width=True,
                    hide_index=True
                )


            except Exception as e:

                st.error(
                    "Unable to read the uploaded CSV."
                )

                st.code(
                    str(e)
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        🎓 Student Early Warning System
        <br>
        Machine Learning • Academic Analytics • Early Intervention

    </div>
    """,
    unsafe_allow_html=True
)