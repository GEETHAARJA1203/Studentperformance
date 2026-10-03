import streamlit as st
import pandas as pd
import numpy as np
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

user = require_role(["Admin", "Faculty"])

logout_button()

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="College Data Upload",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded"
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
        min-height: 130px;
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

    .upload-card {
        background: white;
        padding: 28px;
        border-radius: 20px;
        border: 2px dashed #93c5fd;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.06);
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

    .warning-card {
        background: #fff7ed;
        border-left: 6px solid #f97316;
        padding: 20px;
        border-radius: 15px;
    }

    .warning-card h3 {
        color: #9a3412;
        margin-top: 0;
    }

    .warning-card p {
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
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>📁 College Data Upload</h1>
        <p>
            Upload student data and generate instant academic analytics
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🏫 College Data")

    st.markdown("---")

    st.markdown("### 📂 Upload Settings")

    show_preview = st.checkbox(
        "Show Data Preview",
        value=True
    )

    show_statistics = st.checkbox(
        "Show Statistics",
        value=True
    )

    st.markdown("---")

    st.info(
        "Upload a CSV file containing student academic information."
    )

# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>📤 Upload Student Dataset</h2>
        <p>
            Select a CSV file containing your college student records
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Upload a CSV file containing student information."
)

# ============================================================
# SAMPLE DATA OPTION
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>🧪 Demo Dataset</h2>
        <p>
            You can also use the existing project dataset for testing
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

use_project_data = st.checkbox(
    "Use existing student-mat.csv",
    value=False
)

# ============================================================
# LOAD DATA
# ============================================================

data = None

if uploaded_file is not None:

    try:

        data = pd.read_csv(
            uploaded_file
        )

        st.success(
            f"✅ File uploaded successfully: {uploaded_file.name}"
        )

    except Exception as e:

        st.error(
            "Unable to read the uploaded CSV file."
        )

        st.code(str(e))

        st.stop()

elif use_project_data:

    base_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            ".."
        )
    )

    project_data_path = os.path.join(
        base_dir,
        "student-mat.csv"
    )

    try:

        data = pd.read_csv(
            project_data_path,
            sep=";"
        )

        st.success(
            "✅ Existing project dataset loaded successfully."
        )

    except Exception as e:

        st.error(
            "Unable to load project dataset."
        )

        st.code(str(e))

# ============================================================
# PROCESS DATA
# ============================================================

if data is not None:

    data = data.copy()

    # --------------------------------------------------------
    # BASIC CLEANING
    # --------------------------------------------------------

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    total_students = len(data)

    total_columns = len(data.columns)

    missing_values = int(
        data.isnull().sum().sum()
    )

    duplicate_rows = int(
        data.duplicated().sum()
    )

    # --------------------------------------------------------
    # DATA OVERVIEW
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-header">
            <h2>📊 Dataset Overview</h2>
            <p>
                Basic information about the uploaded college dataset
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">👨‍🎓</div>
                <div class="metric-title">Students</div>
                <div class="metric-value">{total_students}</div>
                <div class="metric-subtitle">Student records</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">📋</div>
                <div class="metric-title">Features</div>
                <div class="metric-value">{total_columns}</div>
                <div class="metric-subtitle">Dataset columns</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">❓</div>
                <div class="metric-title">Missing Values</div>
                <div class="metric-value">{missing_values}</div>
                <div class="metric-subtitle">Total missing cells</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">🔁</div>
                <div class="metric-title">Duplicates</div>
                <div class="metric-value">{duplicate_rows}</div>
                <div class="metric-subtitle">Duplicate records</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ========================================================
    # DATA QUALITY
    # ========================================================

    st.markdown(
        """
        <div class="section-header">
            <h2>🔎 Data Quality Check</h2>
            <p>
                Automatically check the uploaded dataset before analysis
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if missing_values == 0 and duplicate_rows == 0:

        st.markdown(
            """
            <div class="success-card">
                <h3>✅ Dataset Looks Good</h3>
                <p>
                    No missing values or duplicate records were detected.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        if missing_values > 0:

            st.warning(
                f"⚠️ The dataset contains {missing_values} missing values."
            )

        if duplicate_rows > 0:

            st.warning(
                f"⚠️ The dataset contains {duplicate_rows} duplicate records."
            )

    # ========================================================
    # DATA PREVIEW
    # ========================================================

    if show_preview:

        st.markdown(
            """
            <div class="section-header">
                <h2>👀 Data Preview</h2>
                <p>
                    Preview the uploaded student records
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.dataframe(
            data.head(20),
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # NUMERICAL STATISTICS
    # ========================================================

    numerical_columns = data.select_dtypes(
        include=np.number
    ).columns.tolist()

    if show_statistics and len(numerical_columns) > 0:

        st.markdown(
            """
            <div class="section-header">
                <h2>📈 Numerical Statistics</h2>
                <p>
                    Summary statistics for numerical student attributes
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        statistics = (
            data[numerical_columns]
            .describe()
            .T
        )

        st.dataframe(
            statistics,
            use_container_width=True
        )

    # ========================================================
    # PERFORMANCE ANALYSIS
    # ========================================================

    if "g3" in data.columns:

        st.markdown(
            """
            <div class="section-header">
                <h2>🎯 Academic Performance</h2>
                <p>
                    Analysis based on the final score column
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        average_g3 = data["g3"].mean()

        maximum_g3 = data["g3"].max()

        minimum_g3 = data["g3"].min()

        p1, p2, p3 = st.columns(3)

        with p1:

            st.metric(
                "Average Score",
                f"{average_g3:.2f}"
            )

        with p2:

            st.metric(
                "Highest Score",
                f"{maximum_g3:.0f}"
            )

        with p3:

            st.metric(
                "Lowest Score",
                f"{minimum_g3:.0f}"
            )

        # ----------------------------------------------------
        # PERFORMANCE CATEGORIES
        # ----------------------------------------------------

        def performance_group(score):

            if score >= 16:
                return "Excellent"

            elif score >= 12:
                return "Good"

            elif score >= 10:
                return "Average"

            else:
                return "Needs Attention"

        performance_data = data.copy()

        performance_data[
            "Performance Group"
        ] = performance_data["g3"].apply(
            performance_group
        )

        counts = (
            performance_data[
                "Performance Group"
            ]
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
            counts.index,
            counts.values
        )

        ax.set_xlabel(
            "Performance Category"
        )

        ax.set_ylabel(
            "Number of Students"
        )

        ax.set_title(
            "Student Performance Distribution"
        )

        plt.xticks(
            rotation=15
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    # ========================================================
    # ABSENCE ANALYSIS
    # ========================================================

    if (
        "absences" in data.columns
        and "g3" in data.columns
    ):

        st.markdown(
            """
            <div class="section-header">
                <h2>🏫 Attendance Analysis</h2>
                <p>
                    Relationship between absences and final scores
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.scatter(
            data["absences"],
            data["g3"],
            alpha=0.7
        )

        ax.set_xlabel(
            "Absences"
        )

        ax.set_ylabel(
            "Final Score"
        )

        ax.set_title(
            "Absences vs Final Score"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    # ========================================================
    # AT-RISK STUDENTS
    # ========================================================

    if "g3" in data.columns:

        st.markdown(
            """
            <div class="section-header">
                <h2>⚠️ At-Risk Students</h2>
                <p>
                    Students who may require additional academic attention
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        risk_condition = (
            data["g3"] < 10
        )

        if "absences" in data.columns:

            risk_condition = (
                risk_condition
                |
                (data["absences"] >= 20)
            )

        if "failures" in data.columns:

            risk_condition = (
                risk_condition
                |
                (data["failures"] >= 2)
            )

        risk_students = data[
            risk_condition
        ].copy()

        if len(risk_students) > 0:

            st.warning(
                f"⚠️ {len(risk_students)} student(s) "
                "meet the current attention criteria."
            )

            risk_columns = [
                "school",
                "sex",
                "age",
                "studytime",
                "failures",
                "absences",
                "g3"
            ]

            risk_columns = [
                col
                for col in risk_columns
                if col in risk_students.columns
            ]

            st.dataframe(
                risk_students[
                    risk_columns
                ].sort_values(
                    "g3"
                ),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.markdown(
                """
                <div class="success-card">
                    <h3>✅ No At-Risk Students</h3>
                    <p>
                        No students meet the current attention criteria.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ========================================================
    # DOWNLOAD DATA
    # ========================================================

    st.markdown(
        """
        <div class="section-header">
            <h2>💾 Export Data</h2>
            <p>
                Download the uploaded dataset after processing
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    csv_data = data.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        label="⬇️ Download Processed CSV",
        data=csv_data,
        file_name="processed_student_data.csv",
        mime="text/csv",
        use_container_width=True
    )

else:

    # ========================================================
    # EMPTY STATE
    # ========================================================

    st.markdown(
        """
        <div class="upload-card">
            <h2>📂 No Dataset Uploaded</h2>
            <p>
                Upload a CSV file above to start analysing college
                student performance.
            </p>
            <br>
            <p>
                <strong>Recommended columns:</strong>
                student information, study time, absences,
                failures, and academic scores.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h2>⚙️ How It Works</h2>
        <p>
            College data analysis workflow
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.markdown(
        """
        <div class="info-card">
            <h3>1️⃣ Upload</h3>
            <p>
                Faculty uploads the college student CSV dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:

    st.markdown(
        """
        <div class="info-card">
            <h3>2️⃣ Validate</h3>
            <p>
                The system checks missing values, duplicates,
                and available columns.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:

    st.markdown(
        """
        <div class="info-card">
            <h3>3️⃣ Analyse</h3>
            <p>
                The system generates statistics, charts,
                performance groups, and risk indicators.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:

    st.markdown(
        """
        <div class="info-card">
            <h3>4️⃣ Export</h3>
            <p>
                Faculty can download the processed student data.
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
        📁 College Data Management & Analytics
        <br>
        Machine Learning • Data Analytics • Streamlit
        <br><br>
        Trekverse Internship Project
    </div>
    """,
    unsafe_allow_html=True
)