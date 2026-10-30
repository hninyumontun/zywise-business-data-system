from __future__ import annotations

import hashlib
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

# Keep the package importable when Streamlit launches this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from zywise_app.cleaning import clean_dataframe, read_uploaded_file
from zywise_app.analytics import calculate_metrics
from zywise_app.database import authenticate_user, create_user, init_db, refresh_connection_settings
from zywise_app.recommendations import generate_recommendations, recommendation_model_available


st.set_page_config(page_title="ZyWise AI", page_icon="📊", layout="wide")


def apply_design() -> None:
    """Apply the Appendix E visual style to the Streamlit interface."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', Arial, sans-serif;
        }
        .stApp {
            background: #f3f6fa;
            color: #102b46;
        }
        [data-testid="stMainBlockContainer"] {
            max-width: 1440px;
            padding-top: 34px;
            padding-bottom: 34px;
        }
        [data-testid="stSidebar"] {
            background: #102d49;
            border-right: 0;
            min-width: 290px !important;
            max-width: 290px !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding: 30px 22px 24px 22px;
        }
        [data-testid="stSidebar"] * {
            color: #ffffff;
        }
        [data-testid="stSidebar"] .stCaption {
            color: #b9c9d8;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            color: #e8f0f7;
            font-size: 0.98rem !important;
            font-weight: 600;
            min-height: 48px;
            padding: 7px 12px;
            margin: 3px 0;
            border-radius: 10px;
            transition: background 0.15s ease, color 0.15s ease;
            align-items: center;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label p {
            font-size: 0.98rem !important;
            line-height: 1.25 !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            color: #52d39a;
            background: rgba(255, 255, 255, 0.08);
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            color: #ffffff;
            background: #245e91;
            box-shadow: inset 3px 0 0 #42c98b;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] > div {
            gap: 4px;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] > label {
            display: none;
        }
        [data-testid="stSidebar"] button {
            border-radius: 8px;
        }
        .zywise-brand {
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 46px;
        }
        .zywise-brand span {
            color: #42c98b;
            margin-left: 4px;
        }
        .zywise-side-title {
            color: #ffffff;
            font-size: 1.55rem;
            line-height: 1.2;
            font-weight: 700;
            margin-bottom: 12px;
        }
        .zywise-side-copy {
            color: #d3e0eb;
            line-height: 1.45;
            font-size: 0.93rem;
            margin-bottom: 28px;
        }
        .zywise-feature {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #e7f0f7;
            font-size: 0.78rem;
            margin: 15px 0;
        }
        .zywise-feature-icon {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #2ea66d;
            color: #ffffff;
            font-weight: 700;
        }
        .zywise-feature-icon.blue {
            background: #2f70e8;
        }
        .zywise-main-title {
            color: #102d49;
            font-size: 1.95rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            margin: 0 0 4px 0;
        }
        .zywise-main-subtitle {
            color: #718197;
            margin: 0 0 24px 0;
        }
        .zywise-page-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #ffffff;
            margin: 0 -34px 26px -34px;
            padding: 16px 28px;
            min-height: 72px;
            border-bottom: 1px solid #e1e8ef;
        }
        .zywise-avatar {
            width: 38px;
            height: 38px;
            border-radius: 50%;
            background: #2d965f;
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.72rem;
            font-weight: 700;
        }
        .zywise-card {
            background: #ffffff;
            border: 1px solid #d6e0ea;
            border-radius: 12px;
            padding: 18px 20px;
            min-height: 92px;
            box-shadow: 0 3px 12px rgba(16, 45, 73, 0.04);
        }
        .zywise-card-label {
            color: #6d8093;
            font-size: 0.78rem;
            margin-bottom: 9px;
        }
        .zywise-card-value {
            color: #102d49;
            font-size: 1.45rem;
            font-weight: 700;
        }
        .zywise-workspace-label {
            color: #9eb2c5;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin: 28px 0 8px 4px;
        }
        .zywise-status {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            color: #d9e8f5;
            font-size: 0.78rem;
            padding: 7px 10px;
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 999px;
            margin-top: 18px;
        }
        .zywise-status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #42c98b;
            box-shadow: 0 0 0 3px rgba(66, 201, 139, 0.18);
        }
        .zywise-status-dot-error {
            background: #ff7b7b;
            box-shadow: 0 0 0 3px rgba(255, 123, 123, 0.18);
        }
        .zywise-user-card {
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 11px 12px;
            margin-top: 18px;
        }
        .zywise-user-name {
            color: #ffffff;
            font-size: 0.83rem;
            font-weight: 700;
        }
        .zywise-user-caption {
            color: #b9c9d8;
            font-size: 0.7rem;
            margin-top: 3px;
        }
        .auth-heading {
            color: #102d49;
            font-size: 1.45rem;
            margin: 8px 0 2px 0;
        }
        .auth-brand {
            margin-bottom: 10px;
        }
        .auth-left-panel {
            background: #102d49;
            border-radius: 12px;
            width: 100%;
            min-height: 650px;
            padding: 42px 34px;
            box-sizing: border-box;
        }
        .auth-left-panel .zywise-brand {
            margin-bottom: 74px;
            color: #ffffff;
        }
        .auth-left-panel .zywise-side-title {
            font-size: 1.7rem;
        }
        .auth-left-panel .zywise-side-copy {
            max-width: 270px;
        }
        .auth-left-panel .zywise-feature {
            margin-top: 22px;
        }
        [data-testid="stHorizontalBlock"]:has(.auth-left-panel) {
            align-items: stretch;
            gap: 2.25rem;
        }
        [data-testid="stHorizontalBlock"]:has(.auth-left-panel) > div:first-child {
            flex: 0 0 34% !important;
            min-width: 340px !important;
        }
        [data-testid="stHorizontalBlock"]:has(.auth-left-panel) > div:nth-child(2) {
            flex: 1 1 auto !important;
            min-width: 0 !important;
        }
        [data-testid="stTextInput"] input {
            background: #ffffff !important;
            border: 1px solid #cbd7e3 !important;
            border-radius: 7px !important;
            color: #102d49 !important;
            min-height: 42px;
        }
        [data-testid="stTextInput"] input:focus {
            border-color: #2d965f !important;
            box-shadow: 0 0 0 1px #2d965f !important;
        }
        div[data-testid="stFileUploader"] {
            background: #ffffff;
            border: 1px dashed #a9bacb;
            border-radius: 12px;
            padding: 8px;
        }
        .stButton > button, .stDownloadButton > button {
            background: #2d965f;
            color: #ffffff;
            border: 0;
            border-radius: 7px;
            font-weight: 600;
            min-height: 40px;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background: #237a4d;
            color: #ffffff;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #d6e0ea;
            border-radius: 10px;
            overflow: hidden;
        }
        .zywise-upload-card {
            background: #ffffff;
            border: 1px solid #cedae7;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 26px;
        }
        .zywise-section-title {
            color: #102d49;
            font-size: 0.85rem;
            font-weight: 700;
            margin: 16px 0 10px 0;
        }
        .zywise-side-user {
            color: #d6e2ec;
            font-size: 0.55rem;
            line-height: 1.45;
            margin-top: 100px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialise_state() -> None:
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("current_user", None)
    st.session_state.setdefault("db_ready", False)
    st.session_state.setdefault("db_error", None)
    st.session_state.setdefault("raw_df", None)
    st.session_state.setdefault("cleaned_df", None)
    st.session_state.setdefault("quality_summary", None)
    st.session_state.setdefault("metrics", None)
    st.session_state.setdefault("recommendations", None)
    st.session_state.setdefault("source_name", None)
    st.session_state.setdefault("uploaded_file_details", [])
    if not st.session_state.db_ready:
        try:
            refresh_connection_settings()
            init_db()
            st.session_state.db_ready = True
            st.session_state.db_error = None
        except Exception as error:
            st.session_state.db_error = str(error)


def password_hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def clear_current_data() -> None:
    """Remove data from the current browser session."""
    st.session_state.raw_df = None
    st.session_state.cleaned_df = None
    st.session_state.quality_summary = None
    st.session_state.metrics = None
    st.session_state.recommendations = None
    st.session_state.source_name = None
    st.session_state.uploaded_file_details = []


def render_authentication() -> None:
    """Render the Appendix E-style PostgreSQL-backed auth screen."""
    left, right = st.columns([0.82, 1.55], gap="large")
    with left:
        st.markdown(
            '<div class="auth-left-panel">'
            '<div class="zywise-brand">ZyWise <span>AI</span></div>'
            '<div class="zywise-side-title">Prepare better data.</div>'
            '<div class="zywise-side-copy">Clean, analyse and understand business data in one workspace.</div>'
            '<div class="zywise-feature"><span class="zywise-feature-icon">✓</span>Structured data preparation</div>'
            '<div class="zywise-feature"><span class="zywise-feature-icon blue">↗</span>Business-ready insights</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with right:
        with st.container(border=True):
            st.markdown('<div class="zywise-brand auth-brand">ZyWise <span>AI</span></div>', unsafe_allow_html=True)
            register_tab, login_tab = st.tabs(["Create your account", "Log in"])

            with register_tab:
                st.markdown('<h2 class="auth-heading">Create your account</h2>', unsafe_allow_html=True)
                st.caption("Set up your ZyWise workspace to begin.")
                full_name = st.text_input("Full name", key="register_name", placeholder="HNIN YU MON TUN")
                email = st.text_input("Email address", key="register_email", placeholder="student@example.com")
                password = st.text_input("Password", type="password", key="register_password")
                confirmation = st.text_input("Confirm password", type="password", key="register_confirmation")
                if st.button("Register account", type="primary", width="stretch", key="register_button"):
                    if not full_name.strip() or not email.strip() or not password:
                        st.error("Please complete all fields.")
                    elif password != confirmation:
                        st.error("The passwords do not match.")
                    elif not st.session_state.db_ready:
                        st.error("Connect PostgreSQL first.")
                    else:
                        try:
                            user = create_user(full_name.strip(), email.strip().lower(), password_hash(password))
                            if user is None:
                                st.error("An account with this email already exists. Use Log in instead.")
                            else:
                                st.session_state.current_user = user
                                st.session_state.authenticated = True
                                st.success("Account created. Welcome to ZyWise.")
                                st.rerun()
                        except Exception as error:
                            st.error(f"The account could not be created: {error}")

            with login_tab:
                st.markdown('<h2 class="auth-heading">Welcome back</h2>', unsafe_allow_html=True)
                st.caption("Log in to continue to your ZyWise workspace.")
                login_email = st.text_input("Email address", key="login_email", placeholder="student@example.com")
                login_password = st.text_input("Password", type="password", key="login_password")
                if st.button("Log in", type="primary", width="stretch", key="login_button"):
                    if not st.session_state.db_ready:
                        st.error("Connect PostgreSQL first.")
                    else:
                        try:
                            user = authenticate_user(login_email.strip().lower(), password_hash(login_password))
                            if user:
                                st.session_state.current_user = user
                                st.session_state.authenticated = True
                                st.rerun()
                            else:
                                st.error("The email or password is incorrect.")
                        except Exception as error:
                            st.error(f"The login could not be completed: {error}")


def load_sample_data() -> None:
    sample = Path(__file__).resolve().parents[1] / "outputs" / "dashboard_sample_data" / "ZyWise_dashboard_sample_data.csv"
    if not sample.exists():
        st.error("The sample dataset could not be found in the project workspace.")
        return
    st.session_state.raw_df = pd.read_csv(sample)
    st.session_state.source_name = sample.name
    st.session_state.uploaded_file_details = [{
        "File name": sample.name,
        "Type": "CSV",
        "Size": f"{sample.stat().st_size / 1024:.0f} KB",
        "Status": "Ready",
    }]
    process_data()


def load_training_data() -> None:
    """Load the complete synthetic dataset used for model training and app testing."""
    training = Path(__file__).resolve().parents[1] / "data" / "recommendation_training.csv"
    if not training.exists():
        st.error("The complete training dataset could not be found in the project workspace.")
        return
    st.session_state.raw_df = pd.read_csv(training)
    st.session_state.source_name = training.name
    st.session_state.uploaded_file_details = [{
        "File name": training.name,
        "Type": "CSV",
        "Size": f"{training.stat().st_size / 1024:.0f} KB",
        "Status": "Ready",
    }]
    process_data()


def process_data() -> None:
    cleaned, summary = clean_dataframe(st.session_state.raw_df)
    st.session_state.cleaned_df = cleaned
    st.session_state.quality_summary = summary
    st.session_state.metrics = calculate_metrics(cleaned)
    st.session_state.recommendations = generate_recommendations(cleaned)


def excel_bytes(frame: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="Cleaned Data")
    return output.getvalue()


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown('<div class="zywise-brand">ZyWise <span>AI</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="zywise-side-title">Prepare better data.</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="zywise-side-copy">Clean, analyse and understand business data in one workspace.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="zywise-feature"><span class="zywise-feature-icon">✓</span>Structured data preparation</div>'
            '<div class="zywise-feature"><span class="zywise-feature-icon blue">↗</span>Business-ready insights</div>',
            unsafe_allow_html=True,
        )
        status_text = "Database connected" if st.session_state.db_ready else "Database unavailable"
        status_class = "zywise-status-dot" if st.session_state.db_ready else "zywise-status-dot zywise-status-dot-error"
        st.markdown(
            f'<div class="zywise-status"><span class="{status_class}"></span>{status_text}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="zywise-workspace-label">Workspace</div>', unsafe_allow_html=True)
        nav_options = [
            "📊  Dashboard",
            "⬆️  Upload and Clean",
            "🧼  Cleaning Summary",
            "📈  Analytics",
            "💡  Recommendations",
            "🕘  Data History",
            "⬇️  Download",
        ]
        selected = st.radio("Workspace navigation", nav_options, label_visibility="collapsed")
        page = selected.split("  ", 1)[1]
        with st.expander("Workspace tools", expanded=False):
            if st.button("Load dashboard sample data", width="stretch"):
                load_sample_data()
            if st.button("Load complete training dataset", width="stretch"):
                load_training_data()
            if st.button("Clear current data", width="stretch"):
                clear_current_data()
                st.rerun()
        if st.session_state.source_name:
            st.markdown(
                f'<div class="zywise-user-card"><div class="zywise-user-name">Current dataset</div>'
                f'<div class="zywise-user-caption">{st.session_state.source_name}</div></div>',
                unsafe_allow_html=True,
            )
        if st.session_state.current_user:
            st.markdown(
                f'<div class="zywise-user-card"><div class="zywise-user-name">{st.session_state.current_user["name"]}</div>'
                '<div class="zywise-user-caption">Business data workspace</div></div>',
                unsafe_allow_html=True,
            )
        if st.button("Sign out", width="stretch"):
            clear_current_data()
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.rerun()
    return page


def require_data() -> bool:
    if st.session_state.cleaned_df is None:
        st.info("Upload a CSV or Excel file, or load the dashboard sample data from the sidebar.")
        return False
    return True


def render_page_heading(title: str, subtitle: str) -> None:
    user = st.session_state.current_user or {}
    name = str(user.get("name", "ZyWise user"))
    initials = "".join(part[0] for part in name.split()[:2]).upper() or "ZW"
    st.markdown(
        f'<div class="zywise-page-header"><div><div class="zywise-main-title">{title}</div>'
        f'<div class="zywise-main-subtitle">{subtitle}</div></div>'
        f'<div class="zywise-avatar">{initials}</div></div>',
        unsafe_allow_html=True,
    )


def render_dashboard() -> None:
    render_page_heading("Dashboard", "Overview of your cleaned business data and key insights.")
    if not require_data():
        return
    metrics = st.session_state.metrics
    cards = st.columns(4)
    card_values = [
        ("Total revenue", f"{metrics['total_revenue']:,.2f}"),
        ("Total orders", f"{metrics['total_orders']:,}"),
        ("Total quantity", f"{metrics['total_quantity']:,.0f}"),
        ("Average order value", f"{metrics['average_order_value']:,.2f}"),
    ]
    for column, (label, value) in zip(cards, card_values):
        with column:
            st.markdown(
                f'<div class="zywise-card"><div class="zywise-card-label">{label}</div>'
                f'<div class="zywise-card-value">{value}</div></div>',
                unsafe_allow_html=True,
            )
    left, right = st.columns(2)
    with left:
        st.subheader("Monthly sales")
        monthly = metrics["monthly_sales"].set_index("month")
        st.line_chart(monthly["revenue"] if not monthly.empty else monthly)
    with right:
        st.subheader("Sales by category")
        category = metrics["category_sales"].set_index("category")
        st.bar_chart(category["revenue"] if not category.empty else category)
    st.success(f"Top product: {metrics['top_product']}")


def render_upload() -> None:
    render_page_heading("Upload and Clean Data", "Bring your CSV or Excel files into ZyWise and review every cleaning step.")
    uploads = st.file_uploader(
        "Upload one or more CSV or Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )
    if uploads:
        if st.button("Process uploaded file", type="primary"):
            try:
                frames = []
                file_details = []
                for upload in uploads:
                    file_bytes = upload.getvalue()
                    frames.append(read_uploaded_file(upload.name, file_bytes))
                    file_details.append({
                        "File name": upload.name,
                        "Type": upload.name.rsplit(".", 1)[-1].upper(),
                        "Size": f"{len(file_bytes) / 1024:.0f} KB",
                        "Status": "Ready",
                    })

                st.session_state.raw_df = pd.concat(frames, ignore_index=True)
                st.session_state.source_name = ", ".join(upload.name for upload in uploads)
                st.session_state.uploaded_file_details = file_details
                process_data()
                st.success(f"{len(uploads)} file(s) combined and processed successfully.")
            except Exception as error:  # Show a controlled user-facing error.
                st.error(str(error))
    if st.session_state.raw_df is not None:
        st.subheader("Before cleaning")
        st.dataframe(st.session_state.raw_df, width="stretch")
    if st.session_state.quality_summary is not None:
        st.subheader("Cleaning summary")
        st.dataframe(st.session_state.quality_summary, width="stretch", hide_index=True)
        st.subheader("After cleaning")
        st.dataframe(st.session_state.cleaned_df, width="stretch", hide_index=True)


def render_cleaning_summary() -> None:
    render_page_heading("Cleaning Summary", "Review the data-quality checks and transformation results.")
    if not require_data():
        return
    st.markdown('<div class="zywise-section-title">Cleaning rules applied</div>', unsafe_allow_html=True)
    st.dataframe(st.session_state.quality_summary, width="stretch", hide_index=True)
    st.markdown('<div class="zywise-section-title">Cleaned data preview</div>', unsafe_allow_html=True)
    st.dataframe(st.session_state.cleaned_df.head(25), width="stretch", hide_index=True)


def render_history() -> None:
    render_page_heading("Data History", "Review files processed in the current ZyWise workspace.")
    if not st.session_state.uploaded_file_details:
        st.info("No processed files are available yet.")
        return
    st.markdown('<div class="zywise-section-title">Processed files</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(st.session_state.uploaded_file_details), width="stretch", hide_index=True)


def render_analytics() -> None:
    render_page_heading("Business analytics", "Understand revenue, orders, products, categories and monthly performance.")
    if not require_data():
        return
    st.caption(f"Current dataset: {st.session_state.source_name or 'uploaded file'}")
    metrics = st.session_state.metrics
    st.subheader("Product sales")
    if metrics["product_sales"].empty:
        st.warning("No product sales were calculated. Use the complete training dataset or a file containing product and revenue columns.")
    else:
        st.dataframe(metrics["product_sales"], width="stretch", hide_index=True)
    st.subheader("Monthly sales")
    if metrics["monthly_sales"].empty:
        st.warning("No monthly sales were calculated. The dataset must contain a valid order date and revenue column.")
    else:
        st.dataframe(metrics["monthly_sales"], width="stretch", hide_index=True)
    st.subheader("Category sales")
    if metrics["category_sales"].empty:
        st.warning("No category sales were calculated. The dataset must contain category and revenue columns.")
    else:
        st.dataframe(metrics["category_sales"], width="stretch", hide_index=True)


def render_recommendations() -> None:
    render_page_heading("AI recommendations", "Use the cleaned business metrics to support practical decisions.")
    if not require_data():
        return
    recommendations = st.session_state.recommendations
    if recommendation_model_available():
        st.success("A trained decision-tree model is active. Review the confidence and supporting metrics before acting.")
    else:
        st.info("The trained model is not available, so ZyWise is using its transparent rule-based fallback.")
    st.dataframe(recommendations, width="stretch", hide_index=True)


def render_download() -> None:
    render_page_heading("Download results", "Save the cleaned dataset for further use or reporting.")
    if not require_data():
        return
    cleaned = st.session_state.cleaned_df
    st.download_button("Download cleaned CSV", cleaned.to_csv(index=False), "zywise_cleaned_data.csv", "text/csv")
    st.download_button("Download cleaned Excel", excel_bytes(cleaned), "zywise_cleaned_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


apply_design()
initialise_state()
if not st.session_state.authenticated:
    render_authentication()
else:
    page = render_sidebar()
    {
        "Dashboard": render_dashboard,
        "Upload and Clean": render_upload,
        "Cleaning Summary": render_cleaning_summary,
        "Analytics": render_analytics,
        "Recommendations": render_recommendations,
        "Data History": render_history,
        "Download": render_download,
    }[page]()
