import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from model import StudentPerformanceModel
from auth import UserManager


# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="📚",
    layout="wide",
)

st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)


# ── Data helpers ─────────────────────────────────────────────────────────────
def load_data(uploaded_file):
    """Load and validate the uploaded CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
        required = [
            "Age", "Gender", "ParentalEducation", "StudyTimeWeekly",
            "Absences", "ParentalSupport", "Extracurricular",
            "Sports", "Music", "Volunteering", "GradeClass", "GPA",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
            return None

        # Make sure GradeClass is integer (CSV may store it as float)
        df["GradeClass"] = df["GradeClass"].astype(int)

        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def train_model(_X, _y):
    """Train the ML model."""
    model = StudentPerformanceModel()
    metrics = model.train(_X, _y)
    return model, metrics


# ── Filters ──────────────────────────────────────────────────────────────────
def apply_filters(df):
    """Add sidebar filters and return the filtered dataframe."""
    st.sidebar.header("📊 Data Filters")

    grade_options = sorted(df["GradeClass"].unique())
    selected_grades = st.sidebar.multiselect(
        "Grade Class", options=grade_options, default=grade_options
    )

    selected_genders = st.sidebar.multiselect(
        "Gender", options=["Male", "Female"], default=["Male", "Female"]
    )

    # Map numeric gender column to labels
    gender_map = {0: "Male", 1: "Female"}
    df["Gender_Label"] = df["Gender"].map(gender_map)

    filtered = df[
        (df["GradeClass"].isin(selected_grades))
        & (df["Gender_Label"].isin(selected_genders))
    ]
    return filtered


# ── Imbalance Info ───────────────────────────────────────────────────────────
def display_imbalance_info(y):
    """Show whether the dataset is imbalanced and the class distribution."""
    info = StudentPerformanceModel.check_imbalance(y)

    if info["is_imbalanced"]:
        st.warning(
            f"⚠️ Dataset is **imbalanced** (max/min class ratio: {info['ratio']}x). "
            f"SMOTE oversampling is applied automatically during training to handle this."
        )
    else:
        st.success(f"Dataset is fairly balanced (ratio: {info['ratio']}x).")

    with st.expander("View class distribution"):
        dist_df = pd.DataFrame([
            {"Grade": grade, "Count": d["count"], "Percent": f"{d['percent']}%"}
            for grade, d in info["distribution"].items()
        ])
        st.dataframe(dist_df, use_container_width=True)


# ── Model Comparison ─────────────────────────────────────────────────────────
def display_model_comparison(model):
    """Show comparison results of all candidate models."""
    if model.comparison_results is None:
        return

    with st.expander("View all model comparison results"):
        rows = []
        for name, scores in model.comparison_results.items():
            rows.append({
                "Model": name,
                "Accuracy": f"{scores['accuracy']:.2%}",
                "Precision": f"{scores['precision']:.2%}",
                "Recall": f"{scores['recall']:.2%}",
                "F1 Score": f"{scores['f1_score']:.2%}",
            })
        comp_df = pd.DataFrame(rows)
        st.dataframe(comp_df, use_container_width=True)

        # Bar chart
        chart_df = pd.DataFrame([
            {"Model": name, "F1 Score": scores["f1_score"]}
            for name, scores in model.comparison_results.items()
        ]).sort_values("F1 Score", ascending=False)

        fig = px.bar(
            chart_df, x="Model", y="F1 Score",
            title="Model Comparison (F1 Score)",
            color="F1 Score", color_continuous_scale="Viridis",
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)


# ── Metrics ──────────────────────────────────────────────────────────────────
def display_metrics(df, metrics, model):
    """Show KPIs and model performance side-by-side."""
    st.subheader("📈 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", len(df))
    col2.metric("Average GPA", f"{df['GPA'].mean():.2f}")
    col3.metric("Avg Study Hours", f"{df['StudyTimeWeekly'].mean():.1f}")
    col4.metric("At-Risk Students", len(df[df["GPA"] < 2.0]))

    st.subheader(f"🤖 Best Model: {metrics.get('best_model', 'N/A')}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    m2.metric("Precision", f"{metrics['precision']:.2%}")
    m3.metric("Recall", f"{metrics['recall']:.2%}")
    m4.metric("F1 Score", f"{metrics['f1_score']:.2%}")

    display_model_comparison(model)


# ── Visualizations ───────────────────────────────────────────────────────────
def display_visualizations(df):
    """Show the essential charts for exploring student data."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Grade Distribution")
        grade_counts = df["GradeClass"].value_counts().reset_index()
        grade_counts.columns = ["Grade", "Count"]
        fig = px.bar(
            grade_counts, x="Grade", y="Count", color="Grade",
            title="Distribution of Grades",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Study Time vs GPA")
        fig = px.scatter(
            df, x="StudyTimeWeekly", y="GPA", color="GradeClass",
            title="Study Time Impact on GPA",
            labels={"StudyTimeWeekly": "Weekly Study Hours", "GPA": "GPA"},
            trendline="ols",
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Absences vs GPA")
        fig = px.scatter(
            df, x="Absences", y="GPA", color="GradeClass",
            size="StudyTimeWeekly", title="Impact of Absences on GPA",
            hover_data=["StudyTimeWeekly"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Feature Importance")
        if (
            "model" in st.session_state
            and st.session_state["model"].feature_importance is not None
        ):
            fi = st.session_state["model"].feature_importance
            fig = px.bar(
                fi, x="importance", y="feature", orientation="h",
                title="What Drives Grade Predictions?",
                labels={"importance": "Importance", "feature": "Feature"},
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model.")


# ── Prediction ───────────────────────────────────────────────────────────────
def prediction_section(model):
    """Sidebar section for predicting a student's grade."""
    with st.sidebar:
        st.markdown("---")
        st.subheader("🤖 Grade Prediction")

        age = st.number_input("Age", 15, 18, 16)
        gender = st.selectbox("Gender ", ["Male", "Female"])
        parental_ed = st.selectbox(
            "Parental Education",
            ["None", "High School", "Some College", "Bachelor's", "Higher"],
        )
        study_time = st.slider("Study Time (hrs/week)", 0, 20, 10)
        absences = st.slider("Absences", 0, 30, 5)
        parental_support = st.selectbox(
            "Parental Support",
            ["None", "Low", "Moderate", "High", "Very High"],
        )
        extracurricular = st.selectbox("Extracurricular", ["No", "Yes"])
        sports = st.selectbox("Sports", ["No", "Yes"])
        music = st.selectbox("Music", ["No", "Yes"])
        volunteering = st.selectbox("Volunteering", ["No", "Yes"])

        if st.button("Predict Grade"):
            input_data = np.array([[
                age,
                1 if gender == "Female" else 0,
                ["None", "High School", "Some College", "Bachelor's", "Higher"].index(parental_ed),
                study_time,
                absences,
                ["None", "Low", "Moderate", "High", "Very High"].index(parental_support),
                1 if extracurricular == "Yes" else 0,
                1 if sports == "Yes" else 0,
                1 if music == "Yes" else 0,
                1 if volunteering == "Yes" else 0,
            ]])

            input_df = pd.DataFrame(input_data, columns=StudentPerformanceModel.FEATURES)
            grade = model.predict_grade_label(input_df)
            st.success(f"Predicted Grade: {grade}")


# ── Auth pages ───────────────────────────────────────────────────────────────
def login_page():
    """Display the login form."""
    st.title("🔐 Login")
    user_manager = UserManager()

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            role = user_manager.verify_user(username, password)
            if role:
                st.session_state["role"] = role
                st.session_state["username"] = username
                st.success(f"Logged in as {role}")
                st.rerun()
            else:
                st.error("Invalid username or password")


def admin_page():
    """Admin dashboard for managing users."""
    st.title("👨‍💼 Admin Dashboard")
    user_manager = UserManager()

    # Add user
    st.subheader("Add New User")
    with st.form("add_user_form"):
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        role = st.selectbox("Role", ["lecturer", "admin"])

        if st.form_submit_button("Add User"):
            if new_username and new_password:
                success, message = user_manager.add_user(new_username, new_password, role)
                if success:
                    st.success(f"Added user: {new_username}")
                else:
                    st.error(message)
            else:
                st.error("Please fill in all fields")

    # Remove user
    st.subheader("Remove User")
    other_users = [u for u in user_manager.users if u != "admin"]

    if other_users:
        user_to_remove = st.selectbox("Select user to remove", other_users)
        if st.button("Remove User"):
            if user_manager.remove_user(user_to_remove):
                st.success(f"Removed user: {user_to_remove}")
                st.rerun()
            else:
                st.error("Could not remove user")
    else:
        st.info("No users to remove")

    # User list
    st.subheader("All Users")
    user_df = pd.DataFrame([
        {"Username": name, "Role": data["role"]}
        for name, data in user_manager.users.items()
    ])
    st.dataframe(user_df)


def lecturer_page():
    """Lecturer dashboard: upload data, view analysis, make predictions."""
    st.title("👨‍🏫 Lecturer Dashboard")

    uploaded_file = st.file_uploader("Upload Student Data (CSV)", type="csv")
    if uploaded_file is None:
        st.info("Please upload a CSV file to get started.")
        return

    df = load_data(uploaded_file)
    if df is None:
        return

    # Train or load model
    X = df[StudentPerformanceModel.FEATURES]
    y = df["GradeClass"]

    # Show imbalance info
    display_imbalance_info(y)

    MODEL_PATH = "trained_model.pkl"
    saved_model = StudentPerformanceModel.load(MODEL_PATH)

    col_load, col_retrain = st.columns(2)

    if saved_model is not None:
        with col_load:
            st.info(f"Saved model found: **{saved_model.best_model_name}**")
        use_saved = not col_retrain.button("Retrain Model")
    else:
        use_saved = False

    if use_saved and saved_model is not None:
        model = saved_model
        metrics = model.comparison_results[model.best_model_name]
        metrics["best_model"] = model.best_model_name
    else:
        with st.spinner("Comparing models and selecting the best one..."):
            model, metrics = train_model(X, y)
            model.save(MODEL_PATH)
            st.success(f"Model trained and saved to disk ({metrics['best_model']})")

    # Apply filters and display dashboard
    filtered_df = apply_filters(df)
    display_metrics(filtered_df, metrics, model)
    st.session_state["model"] = model
    display_visualizations(filtered_df)
    prediction_section(model)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Logout button (only shown when logged in)
    if "role" in st.session_state:
        if st.sidebar.button("🚪 Logout"):
            st.session_state.clear()
            st.rerun()

    # Route based on login state
    if "role" not in st.session_state:
        login_page()
    elif st.session_state["role"] == "admin":
        admin_page()
    else:
        lecturer_page()


if __name__ == "__main__":
    main()