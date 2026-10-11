import streamlit as st
import pandas as pd
import os
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

user = require_role(["Admin", "Faculty"])

logout_button()



# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="New ML Models",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# BEAUTIFUL CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at top left,
            #eef2ff 0%,
            transparent 35%
        ),
        radial-gradient(
            circle at bottom right,
            #ecfdf5 0%,
            transparent 35%
        ),
        #f8fafc;
}

/* Header */

.hero {
    background: linear-gradient(
        135deg,
        #1e293b,
        #334155
    );

    padding: 35px;

    border-radius: 24px;

    margin-bottom: 25px;

    box-shadow:
        0 15px 35px rgba(15,23,42,0.15);
}

.hero-title {
    font-size: 40px;
    font-weight: 800;
    color: white;
}

.hero-text {
    color: #cbd5e1;
    font-size: 17px;
    margin-top: 8px;
}


/* Cards */

.card {
    background: rgba(255,255,255,0.95);

    padding: 24px;

    border-radius: 20px;

    border: 1px solid #e2e8f0;

    box-shadow:
        0 8px 25px rgba(15,23,42,0.07);

    min-height: 180px;
}

.card-icon {
    font-size: 34px;
}

.card-title {
    font-size: 19px;
    font-weight: 750;
    color: #1e293b;
    margin-top: 8px;
}

.card-text {
    color: #64748b;
    font-size: 14px;
    margin-top: 8px;
}


/* Section */

.section-title {
    font-size: 25px;
    font-weight: 800;
    color: #1e293b;

    margin-top: 30px;
    margin-bottom: 15px;
}


/* Metric */

.metric {
    background: white;

    padding: 22px;

    border-radius: 18px;

    text-align: center;

    border: 1px solid #e2e8f0;

    box-shadow:
        0 8px 20px rgba(15,23,42,0.06);
}

.metric-label {
    color: #64748b;
    font-size: 14px;
    font-weight: 700;
}

.metric-value {
    color: #1e293b;
    font-size: 30px;
    font-weight: 800;

    margin-top: 5px;
}


/* Info */

.info {
    background: #eff6ff;

    border-left: 5px solid #3b82f6;

    padding: 17px;

    border-radius: 10px;

    color: #1e3a8a;

    margin: 15px 0;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../.."
    )
)

RESULTS_PATH = os.path.join(
    BASE_DIR,
    "new_model_results.csv"
)


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
🤖 New Machine Learning Models
</div>

<div class="hero-text">
Advanced algorithms for student performance prediction
and model comparison.
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# INTRO
# ============================================================

st.markdown("""
<div class="info">

💡 <b>Purpose:</b>
This module extends the existing student performance prediction
system with additional machine-learning algorithms.

The models are trained using the same student dataset and
preprocessing pipeline used by the original project.

</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD RESULTS
# ============================================================

if not os.path.exists(RESULTS_PATH):

    st.error(
        "New model results were not generated yet."
    )

    st.code(
        "python src\\new_models.py",
        language="powershell"
    )

    st.stop()


results = pd.read_csv(
    RESULTS_PATH
)


# ============================================================
# MODEL CARDS
# ============================================================

st.markdown(
    '<div class="section-title">🧠 Newly Added Models</div>',
    unsafe_allow_html=True
)

model_info = [

    (
        "📐",
        "Ridge Regression",
        "Linear regression with L2 regularization."
    ),

    (
        "🎯",
        "Lasso Regression",
        "Regularized regression using L1 penalty."
    ),

    (
        "🌲",
        "Extra Trees",
        "Ensemble of randomized decision trees."
    ),

    (
        "⚡",
        "Support Vector Regression",
        "Uses support-vector learning for regression."
    )
]


cols = st.columns(4)


for col, info in zip(
    cols,
    model_info
):

    icon, title, description = info

    with col:

        st.markdown(f"""

        <div class="card">

            <div class="card-icon">
                {icon}
            </div>

            <div class="card-title">
                {title}
            </div>

            <div class="card-text">
                {description}
            </div>

        </div>

        """, unsafe_allow_html=True)


# ============================================================
# SUMMARY METRICS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Model Summary</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(f"""
    <div class="metric">

        <div class="metric-label">
            MODELS ADDED
        </div>

        <div class="metric-value">
            {len(results)}
        </div>

    </div>
    """, unsafe_allow_html=True)


with col2:

    avg_rmse = results["RMSE"].mean()

    st.markdown(f"""
    <div class="metric">

        <div class="metric-label">
            AVERAGE RMSE
        </div>

        <div class="metric-value">
            {avg_rmse:.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)


with col3:

    avg_r2 = results["R2"].mean()

    st.markdown(f"""
    <div class="metric">

        <div class="metric-label">
            AVERAGE R²
        </div>

        <div class="metric-value">
            {avg_r2:.2f}
        </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TABLE
# ============================================================

st.markdown(
    '<div class="section-title">📋 Detailed Model Evaluation</div>',
    unsafe_allow_html=True
)

display_df = results.copy()

display_df["MAE"] = display_df["MAE"].map(
    lambda x: f"{x:.4f}"
)

display_df["RMSE"] = display_df["RMSE"].map(
    lambda x: f"{x:.4f}"
)

display_df["R2"] = display_df["R2"].map(
    lambda x: f"{x:.4f}"
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# RMSE CHART
# ============================================================

st.markdown(
    '<div class="section-title">📉 RMSE Comparison</div>',
    unsafe_allow_html=True
)

rmse_data = results.set_index(
    "Model"
)[["RMSE"]]

st.bar_chart(
    rmse_data,
    height=400
)


# ============================================================
# R2 CHART
# ============================================================

st.markdown(
    '<div class="section-title">📈 R² Comparison</div>',
    unsafe_allow_html=True
)

r2_data = results.set_index(
    "Model"
)[["R2"]]

st.bar_chart(
    r2_data,
    height=400
)


# ============================================================
# METRIC EXPLANATION
# ============================================================

st.markdown(
    '<div class="section-title">📚 Understanding the Metrics</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    st.markdown("""
    <div class="card">

    <div class="card-icon">📏</div>

    <div class="card-title">
    MAE
    </div>

    <div class="card-text">

    Mean Absolute Error measures the average
    absolute difference between actual and
    predicted student scores.

    Lower values indicate smaller average errors.

    </div>

    </div>
    """, unsafe_allow_html=True)


with col2:

    st.markdown("""
    <div class="card">

    <div class="card-icon">📐</div>

    <div class="card-title">
    RMSE
    </div>

    <div class="card-text">

    Root Mean Squared Error gives more importance
    to larger prediction errors.

    Lower values indicate smaller errors.

    </div>

    </div>
    """, unsafe_allow_html=True)


with col3:

    st.markdown("""
    <div class="card">

    <div class="card-icon">📈</div>

    <div class="card-title">
    R² Score
    </div>

    <div class="card-text">

    R² measures how much variation in the target
    student score is explained by the model.

    </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Trekverse Student Performance • Machine Learning Module"
)