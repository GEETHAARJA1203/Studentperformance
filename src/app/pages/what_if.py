import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="What-If Prediction",
    page_icon="🔮",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f8fafc, #eef2ff, #f0fdf4);
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #172033;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #64748b;
    margin-bottom: 25px;
}

.card {
    background: rgba(255,255,255,0.95);
    padding: 24px;
    border-radius: 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 8px 25px rgba(15,23,42,0.07);
    margin-bottom: 20px;
}

.section-title {
    font-size: 23px;
    font-weight: 750;
    color: #1e293b;
    margin-bottom: 15px;
}

.metric-card {
    background: white;
    padding: 25px 15px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #e2e8f0;
    box-shadow: 0 8px 20px rgba(15,23,42,0.06);
}

.metric-label {
    font-size: 14px;
    font-weight: 700;
    color: #64748b;
    letter-spacing: 0.5px;
}

.metric-value {
    font-size: 32px;
    font-weight: 800;
    color: #1e293b;
    margin-top: 8px;
}

.info-box {
    background: #eff6ff;
    border-left: 5px solid #3b82f6;
    padding: 16px;
    border-radius: 10px;
    margin: 15px 0 25px 0;
    color: #1e3a8a;
}

.warning-box {
    background: #fffbeb;
    border-left: 5px solid #f59e0b;
    padding: 16px;
    border-radius: 10px;
    margin-top: 20px;
    color: #92400e;
}

.success-box {
    background: #f0fdf4;
    border-left: 5px solid #22c55e;
    padding: 16px;
    border-radius: 10px;
    color: #166534;
    margin-top: 15px;
}

.neutral-box {
    background: #f8fafc;
    border-left: 5px solid #64748b;
    padding: 16px;
    border-radius: 10px;
    color: #334155;
    margin-top: 15px;
}

.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 13px;
    font-size: 17px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
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
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)


try:
    model = load_model()
    preprocessor = load_preprocessor()

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔮 What-If Student Performance</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Simulate how changes in student behaviour and academic factors '
    'may affect the predicted performance.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">
💡 <b>What-If Analysis:</b>
Enter the student's current values and create a simulated scenario
by changing selected factors. The machine-learning model will compare
both situations.
</div>
""", unsafe_allow_html=True)


# ============================================================
# FORM
# ============================================================

with st.form("what_if_form"):

    # ========================================================
    # STUDENT INFORMATION
    # ========================================================

    st.markdown(
        '<div class="section-title">👨‍🎓 Student Information</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        school = st.selectbox(
            "School",
            ["GP", "MS"]
        )

    with col2:
        sex = st.selectbox(
            "Gender",
            ["F", "M"]
        )

    with col3:
        age = st.slider(
            "Age",
            15,
            22,
            17
        )

    st.markdown("---")

    # ========================================================
    # CURRENT VALUES
    # ========================================================

    st.markdown(
        '<div class="section-title">📊 Current Student Values</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "These values represent the student's current situation."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        current_studytime = st.slider(
            "📚 Study Time",
            1,
            4,
            2
        )

    with col2:
        current_absences = st.slider(
            "📅 Absences",
            0,
            93,
            5
        )

    with col3:
        current_failures = st.slider(
            "⚠️ Failures",
            0,
            3,
            0
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        current_goout = st.slider(
            "🎉 Going Out",
            1,
            5,
            3
        )

    with col2:
        current_freetime = st.slider(
            "⏰ Free Time",
            1,
            5,
            3
        )

    with col3:
        current_health = st.slider(
            "❤️ Health",
            1,
            5,
            3
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        current_famrel = st.slider(
            "👨‍👩‍👧 Family Relationship",
            1,
            5,
            4
        )

    with col2:
        current_traveltime = st.slider(
            "🚌 Travel Time",
            1,
            4,
            1
        )

    with col3:
        current_dalc = st.slider(
            "🍺 Weekday Alcohol",
            1,
            5,
            1
        )

    current_walc = st.slider(
        "🍺 Weekend Alcohol",
        1,
        5,
        1
    )

    # ========================================================
    # WHAT IF VALUES
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🔄 What-If Scenario</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Change these values to simulate a different student situation."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        whatif_studytime = st.slider(
            "📚 Study Time →",
            1,
            4,
            4
        )

    with col2:
        whatif_absences = st.slider(
            "📅 Absences →",
            0,
            93,
            0
        )

    with col3:
        whatif_failures = st.slider(
            "⚠️ Failures →",
            0,
            3,
            0
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        whatif_goout = st.slider(
            "🎉 Going Out →",
            1,
            5,
            1
        )

    with col2:
        whatif_freetime = st.slider(
            "⏰ Free Time →",
            1,
            5,
            3
        )

    with col3:
        whatif_health = st.slider(
            "❤️ Health →",
            1,
            5,
            2
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        whatif_famrel = st.slider(
            "👨‍👩‍👧 Family Relationship →",
            1,
            5,
            5
        )

    with col2:
        whatif_traveltime = st.slider(
            "🚌 Travel Time →",
            1,
            4,
            1
        )

    with col3:
        whatif_dalc = st.slider(
            "🍺 Weekday Alcohol →",
            1,
            5,
            1
        )

    whatif_walc = st.slider(
        "🍺 Weekend Alcohol →",
        1,
        5,
        1
    )

    # ========================================================
    # BUTTON
    # ========================================================

    st.markdown("---")

    simulate = st.form_submit_button(
        "🚀 RUN WHAT-IF SIMULATION"
    )


# ============================================================
# CREATE INPUT
# ============================================================

def create_input(
    studytime,
    absences,
    failures,
    goout,
    freetime,
    health,
    famrel,
    traveltime,
    dalc,
    walc
):

    data = {
        "school": school,
        "sex": sex,
        "age": age,

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
        "studytime": studytime,
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

        "Dalc": dalc,
        "Walc": walc,

        "health": health,
        "absences": absences
    }

    return pd.DataFrame([data])


# ============================================================
# PREDICT
# ============================================================

def predict_score(data):

    processed = preprocessor.transform(data)

    prediction = model.predict(processed)[0]

    return float(np.clip(prediction, 0, 20))


# ============================================================
# RUN SIMULATION
# ============================================================

if simulate:

    current_data = create_input(
        current_studytime,
        current_absences,
        current_failures,
        current_goout,
        current_freetime,
        current_health,
        current_famrel,
        current_traveltime,
        current_dalc,
        current_walc
    )

    whatif_data = create_input(
        whatif_studytime,
        whatif_absences,
        whatif_failures,
        whatif_goout,
        whatif_freetime,
        whatif_health,
        whatif_famrel,
        whatif_traveltime,
        whatif_dalc,
        whatif_walc
    )

    try:

        # ====================================================
        # PREDICTIONS
        # ====================================================

        current_prediction = predict_score(current_data)

        whatif_prediction = predict_score(whatif_data)

        change = whatif_prediction - current_prediction

        # ====================================================
        # RESULTS
        # ====================================================

        st.markdown("---")

        st.markdown(
            '<div class="section-title">📈 Simulation Results</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">
                    CURRENT PREDICTION
                </div>
                <div class="metric-value">
                    {current_prediction:.2f} / 20
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">
                    WHAT-IF PREDICTION
                </div>
                <div class="metric-value">
                    {whatif_prediction:.2f} / 20
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:

            if change > 0:
                change_display = f"+{change:.2f}"
            else:
                change_display = f"{change:.2f}"

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">
                    CHANGE
                </div>
                <div class="metric-value">
                    {change_display}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ====================================================
        # CHANGED FACTORS
        # ====================================================

        st.markdown("---")

        st.markdown(
            '<div class="section-title">🔍 Changed Factors</div>',
            unsafe_allow_html=True
        )

        factors = [
            ("Study Time", current_studytime, whatif_studytime),
            ("Absences", current_absences, whatif_absences),
            ("Failures", current_failures, whatif_failures),
            ("Going Out", current_goout, whatif_goout),
            ("Free Time", current_freetime, whatif_freetime),
            ("Health", current_health, whatif_health),
            ("Family Relationship", current_famrel, whatif_famrel),
            ("Travel Time", current_traveltime, whatif_traveltime),
            ("Weekday Alcohol", current_dalc, whatif_dalc),
            ("Weekend Alcohol", current_walc, whatif_walc)
        ]

        changed = []

        for name, current, whatif in factors:

            if current != whatif:

                changed.append({
                    "Factor": name,
                    "Current Value": current,
                    "What-If Value": whatif,
                    "Difference": whatif - current
                })

        if len(changed) > 0:

            changed_df = pd.DataFrame(changed)

            st.dataframe(
                changed_df,
                use_container_width=True,
                hide_index=True
            )

            st.success(
                f"✅ {len(changed)} factor(s) changed in this scenario."
            )

        else:

            st.warning(
                "⚠️ No factors were changed. "
                "Please modify at least one What-If value."
            )

        # ====================================================
        # CHART
        # ====================================================

        st.markdown("---")

        st.markdown(
            '<div class="section-title">📊 Current vs What-If</div>',
            unsafe_allow_html=True
        )

        chart = pd.DataFrame({
            "Score": [
                current_prediction,
                whatif_prediction
            ]
        }, index=[
            "Current",
            "What-If"
        ])

        st.bar_chart(
            chart,
            height=350
        )

        # ====================================================
        # INTERPRETATION
        # ====================================================

        st.markdown("---")

        st.markdown(
            '<div class="section-title">🧠 Interpretation</div>',
            unsafe_allow_html=True
        )

        if len(changed) == 0:

            st.warning(
                "No What-If factors were changed. "
                "Change one or more values and run the simulation again."
            )

        elif change > 0.05:

            st.markdown(f"""
            <div class="success-box">
            📈 <b>Simulation Result:</b><br><br>
            The What-If scenario increased the model's predicted score
            from <b>{current_prediction:.2f}</b> to
            <b>{whatif_prediction:.2f}</b>.
            <br><br>
            Predicted change:
            <b>+{change:.2f} points</b>.
            </div>
            """, unsafe_allow_html=True)

        elif change < -0.05:

            st.markdown(f"""
            <div class="warning-box">
            📉 <b>Simulation Result:</b><br><br>
            The What-If scenario decreased the model's predicted score
            from <b>{current_prediction:.2f}</b> to
            <b>{whatif_prediction:.2f}</b>.
            <br><br>
            Predicted change:
            <b>{change:.2f} points</b>.
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="neutral-box">
            ℹ️ <b>Simulation Result:</b><br><br>
            You changed <b>{len(changed)}</b> factor(s), but the trained
            Random Forest model produced the same predicted score:
            <br><br>
            <b>{current_prediction:.2f} → {whatif_prediction:.2f}</b>
            <br><br>
            This can happen because tree-based models can produce the
            same prediction for different input combinations.
            </div>
            """, unsafe_allow_html=True)

        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.markdown("""
        <div class="warning-box">
        ⚠️ <b>Important:</b>
        What-If analysis is a model-based simulation.
        It does not guarantee that a student's actual future score
        will increase or decrease by the predicted amount.
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )