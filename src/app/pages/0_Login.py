import streamlit as st
import os
import sys


# =========================================================
# IMPORT AUTH MODULE
# =========================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from auth import authenticate_user


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Login - Student Performance System",
    page_icon="🎓",
    layout="centered"
)


# =========================================================
# SESSION STATE
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None


# =========================================================
# SIMPLE CSS
# =========================================================

st.markdown(
    """
    <style>
    
    .stApp {
        background-color: #f5f7fb;
    }

    .login-title {
        text-align: center;
        font-size: 34px;
        font-weight: bold;
        margin-top: 40px;
    }

    .login-subtitle {
        text-align: center;
        color: #666666;
        font-size: 16px;
        margin-bottom: 30px;
    }

    .demo-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-top: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# IF ALREADY LOGGED IN
# =========================================================

if st.session_state.logged_in:

    user = st.session_state.user

    st.success(
        f"Welcome {user['username']}! "
        f"You are logged in as {user['role']}."
    )

    st.write("")

    if st.button(
        "🚪 Logout",
        use_container_width=True
    ):

        st.session_state.logged_in = False
        st.session_state.user = None

        st.rerun()

    st.stop()


# =========================================================
# LOGIN HEADER
# =========================================================

st.markdown(
    '<div class="login-title">🎓 Student Performance System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="login-subtitle">'
    'Login to access the Machine Learning Dashboard'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOGIN FORM
# =========================================================

st.subheader("🔐 Login")

username = st.text_input(
    "👤 Username",
    placeholder="Enter username"
)

password = st.text_input(
    "🔒 Password",
    type="password",
    placeholder="Enter password"
)


# =========================================================
# LOGIN BUTTON
# =========================================================

if st.button(
    "🚀 Login",
    use_container_width=True
):

    if username.strip() == "":
        st.warning("Please enter your username.")

    elif password == "":
        st.warning("Please enter your password.")

    else:

        user = authenticate_user(
            username.strip(),
            password
        )

        if user is not None:

            st.session_state.logged_in = True
            st.session_state.user = user

            st.success(
                f"Login successful! Welcome {user['username']}."
            )

            st.rerun()

        else:

            st.error(
                "❌ Invalid username or password."
            )


# =========================================================
# DEMO ACCOUNTS
# =========================================================

st.markdown("---")

st.subheader("🔑 Demo Accounts")

st.info(
    """
    **👑 Admin**

    Username: `admin`

    Password: `admin123`

    **👨‍🏫 Faculty**

    Username: `faculty`

    Password: `faculty123`

    **🎓 Student**

    Username: `student`

    Password: `student123`
    """
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Student Performance Prediction & Analytics System"
)