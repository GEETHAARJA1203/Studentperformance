import streamlit as st
import pandas as pd
import os
import sys
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

user = require_role(["Admin"])

logout_button()


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Student Database",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# IMPORT DATABASE MODULE
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from database import (
    add_student,
    get_students,
    get_student,
    delete_student,
    get_statistics
)


# ============================================================
# CSS
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

        box-shadow:
            0 10px 30px rgba(37, 99, 235, 0.20);
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
    }

    .section-header {
        background: white;
        padding: 20px 24px;
        border-radius: 16px;
        margin-top: 20px;
        margin-bottom: 18px;
        border-left: 6px solid #2563eb;

        box-shadow:
            0 5px 20px rgba(15, 23, 42, 0.07);
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

        box-shadow:
            0 7px 25px rgba(15, 23, 42, 0.08);

        border: 1px solid #e2e8f0;
        min-height: 125px;
    }

    .metric-icon {
        font-size: 27px;
    }

    .metric-title {
        color: #64748b;
        font-size: 14px;
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
    }

    .info-card {
        background: white;
        padding: 24px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;

        box-shadow:
            0 6px 20px rgba(15, 23, 42, 0.06);
    }

    .info-card h3 {
        color: #172554;
    }

    .info-card p {
        color: #64748b;
        line-height: 1.6;
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
        <h1>🗄️ Student Database</h1>
        <p>
            Centralized storage and management of student performance records
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🗄️ Database")

    st.markdown("---")

    st.markdown("### Database Operations")

    menu = st.radio(
        "Select Operation",
        [
            "📊 Database Overview",
            "➕ Add Student",
            "🔍 Search Student",
            "🗑️ Delete Student"
        ]
    )

    st.markdown("---")

    st.info(
        "Student records are stored locally using SQLite."
    )


# ============================================================
# DATABASE OVERVIEW
# ============================================================

if menu == "📊 Database Overview":

    st.markdown(
        """
        <div class="section-header">
            <h2>📊 Database Overview</h2>
            <p>
                Summary of students currently stored in the database
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    stats = get_statistics()

    total_students = stats["total_students"] or 0

    average_score = stats["average_score"] or 0

    average_absences = stats["average_absences"] or 0

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">👨‍🎓</div>
                <div class="metric-title">Total Students</div>
                <div class="metric-value">
                    {total_students}
                </div>
                <div class="metric-subtitle">
                    Database records
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">🎯</div>
                <div class="metric-title">Average Predicted Score</div>
                <div class="metric-value">
                    {average_score:.2f}
                </div>
                <div class="metric-subtitle">
                    Out of 20
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">🏫</div>
                <div class="metric-title">Average Absences</div>
                <div class="metric-value">
                    {average_absences:.1f}
                </div>
                <div class="metric-subtitle">
                    Per student
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # ALL STUDENTS
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-header">
            <h2>👨‍🎓 Student Records</h2>
            <p>
                All students stored in the database
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    rows = get_students()

    if rows:

        students_df = pd.DataFrame(
            [dict(row) for row in rows]
        )

        st.dataframe(
            students_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = students_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Database Records",
            csv_data,
            "student_database.csv",
            "text/csv",
            use_container_width=True
        )

    else:

        st.info(
            "No student records are currently stored."
        )


# ============================================================
# ADD STUDENT
# ============================================================

elif menu == "➕ Add Student":

    st.markdown(
        """
        <div class="section-header">
            <h2>➕ Add Student Record</h2>
            <p>
                Store a student's academic prediction in the database
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("student_form"):

        col1, col2 = st.columns(2)

        with col1:

            student_id = st.text_input(
                "Student ID",
                placeholder="Example: STU001"
            )

            student_name = st.text_input(
                "Student Name",
                placeholder="Enter student name"
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

            study_time = st.slider(
                "Study Time",
                1,
                4,
                2
            )

            absences = st.number_input(
                "Absences",
                min_value=0,
                max_value=100,
                value=5
            )

            failures = st.number_input(
                "Failures",
                min_value=0,
                max_value=4,
                value=0
            )

        with col2:

            goout = st.slider(
                "Going Out",
                1,
                5,
                3
            )

            freetime = st.slider(
                "Free Time",
                1,
                5,
                3
            )

            health = st.slider(
                "Health",
                1,
                5,
                3
            )

            famrel = st.slider(
                "Family Relationship",
                1,
                5,
                4
            )

            traveltime = st.slider(
                "Travel Time",
                1,
                4,
                1
            )

            predicted_score = st.number_input(
                "Predicted Score",
                min_value=0.0,
                max_value=20.0,
                value=10.0,
                step=0.01
            )

        # ----------------------------------------------------
        # PERFORMANCE
        # ----------------------------------------------------

        if predicted_score >= 16:

            performance_level = "Excellent"

        elif predicted_score >= 12:

            performance_level = "Good"

        elif predicted_score >= 10:

            performance_level = "Average"

        else:

            performance_level = "Needs Attention"

        # ----------------------------------------------------
        # WARNING
        # ----------------------------------------------------

        if (
            predicted_score < 10
            or failures >= 2
            or absences >= 20
        ):

            early_warning = "Attention Required"

        else:

            early_warning = "On Track"

        st.markdown("---")

        st.info(
            f"Performance Level: **{performance_level}**"
        )

        st.info(
            f"Early Warning: **{early_warning}**"
        )

        submitted = st.form_submit_button(
            "💾 Save Student",
            use_container_width=True
        )

    if submitted:

        if not student_id.strip():

            st.error(
                "Please enter Student ID."
            )

        elif not student_name.strip():

            st.error(
                "Please enter Student Name."
            )

        else:

            success, message = add_student(
                student_id=student_id.strip(),
                student_name=student_name.strip(),
                department=department,
                study_time=study_time,
                absences=absences,
                failures=failures,
                goout=goout,
                freetime=freetime,
                health=health,
                famrel=famrel,
                traveltime=traveltime,
                predicted_score=predicted_score,
                performance_level=performance_level,
                early_warning=early_warning
            )

            if success:

                st.success(
                    "✅ Student record saved successfully!"
                )

                st.balloons()

            else:

                st.error(
                    f"❌ {message}"
                )


# ============================================================
# SEARCH STUDENT
# ============================================================

elif menu == "🔍 Search Student":

    st.markdown(
        """
        <div class="section-header">
            <h2>🔍 Search Student</h2>
            <p>
                Find a student using their Student ID
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    search_id = st.text_input(
        "Enter Student ID",
        placeholder="Example: STU001"
    )

    if st.button(
        "🔍 Search",
        use_container_width=True
    ):

        if not search_id.strip():

            st.warning(
                "Please enter a Student ID."
            )

        else:

            student = get_student(
                search_id.strip()
            )

            if student:

                st.success(
                    "✅ Student found."
                )

                student_data = dict(student)

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Student Name",
                        student_data["student_name"]
                    )

                    st.metric(
                        "Department",
                        student_data["department"]
                    )

                    st.metric(
                        "Study Time",
                        student_data["study_time"]
                    )

                with col2:

                    st.metric(
                        "Predicted Score",
                        f"{student_data['predicted_score']:.2f}"
                    )

                    st.metric(
                        "Performance",
                        student_data["performance_level"]
                    )

                    st.metric(
                        "Absences",
                        student_data["absences"]
                    )

                with col3:

                    st.metric(
                        "Failures",
                        student_data["failures"]
                    )

                    st.metric(
                        "Early Warning",
                        student_data["early_warning"]
                    )

                    st.metric(
                        "Created",
                        student_data["created_at"]
                    )

            else:

                st.error(
                    "❌ Student ID not found."
                )


# ============================================================
# DELETE STUDENT
# ============================================================

elif menu == "🗑️ Delete Student":

    st.markdown(
        """
        <div class="section-header">
            <h2>🗑️ Delete Student</h2>
            <p>
                Remove a student record from the database
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    delete_id = st.text_input(
        "Student ID",
        placeholder="Example: STU001"
    )

    if st.button(
        "🗑️ Delete Student",
        use_container_width=True
    ):

        if not delete_id.strip():

            st.warning(
                "Please enter a Student ID."
            )

        else:

            deleted = delete_student(
                delete_id.strip()
            )

            if deleted:

                st.success(
                    "✅ Student record deleted."
                )

            else:

                st.error(
                    "❌ Student ID not found."
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🗄️ Student Database Management System
        <br>
        SQLite • Python • Streamlit
        <br><br>
        Trekverse Internship Project
    </div>
    """,
    unsafe_allow_html=True
)