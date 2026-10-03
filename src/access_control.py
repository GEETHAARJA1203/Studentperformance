import streamlit as st



def require_login():

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if "user" not in st.session_state:
        st.session_state.user = None

    if not st.session_state.logged_in:
        st.warning("🔐 Please login first.")

        st.info(
            "Go to the Login page from the sidebar."
        )

        st.stop()

    return st.session_state.user


def require_role(allowed_roles):

    user = require_login()

    if user["role"] not in allowed_roles:

        st.error(
            "🚫 You do not have permission to access this page."
        )

        st.info(
            f"Your current role is: {user['role']}"
        )

        st.stop()

    return user


def logout_button():

    user = st.session_state.get("user")

    if user:

        st.sidebar.markdown("---")

        st.sidebar.write(
            f"👤 **{user['username']}**"
        )

        st.sidebar.write(
            f"Role: **{user['role']}**"
        )

        if st.sidebar.button(
            "🚪 Logout",
            use_container_width=True
        ):

            st.session_state.logged_in = False
            st.session_state.user = None

            st.rerun()