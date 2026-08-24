from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
PRIMARY_ARTIFACT = BASE_DIR / "model/lightgbm_churn_production.pkl"
FEATURE_IMPORTANCE_IMAGE = BASE_DIR / "assets/shap_feature_importance.png"
SHAP_BEESWARM_IMAGE = BASE_DIR / "assets/shap_beeswarm_plot.png"


st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400;1,600&display=swap');

    /* Global Font & Background */
    html, body, [class*="css"], [class*="st-"], .stApp {
        font-family: 'IBM Plex Sans', sans-serif !important;
    }

    .stApp {
        background-color: #111827 !important;
        color: #E5E7EB !important;
    }

    header[data-testid="stHeader"] {
        background-color: #111827 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    div[data-testid="stSidebarContent"],
    div[data-testid="stSidebarUserContent"],
    header[data-testid="stSidebarHeader"] {
        background-color: #0F172A !important;
        color: #E5E7EB !important;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid #2A3445 !important;
    }

    /* Sidebar Metrics */
    section[data-testid="stSidebar"] div[data-testid="stMetric"] {
        background-color: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 4px 0 !important;
        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] p,
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] label,
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] span {
        color: #94A3B8 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMetricValue"] div,
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] span,
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #E5E7EB !important;
        font-weight: 600 !important;
    }

    /* Highlight Loaded Model in Sidebar */
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div[data-testid="element-container"]:nth-of-type(2) [data-testid="stMetricValue"] div,
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div:nth-child(2) [data-testid="stMetricValue"] div {
        color: #2DD4BF !important;
    }

    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #94A3B8 !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #E5E7EB !important;
        font-weight: 600 !important;
    }

    /* Sidebar buttons & collapse toggle */
    section[data-testid="stSidebar"] button,
    [data-testid="stSidebarCollapseButton"] button {
        color: #94A3B8 !important;
    }

    section[data-testid="stSidebar"] svg,
    [data-testid="stSidebarCollapseButton"] svg {
        fill: #94A3B8 !important;
    }

    /* Headings */
    h1, [data-testid="stHeadingWithAnchor"] h1 {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 34px !important;
        font-weight: 600 !important;
        color: #E5E7EB !important;
    }

    h2, h3, [data-testid="stHeadingWithAnchor"] h2, [data-testid="stHeadingWithAnchor"] h3 {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 22px !important;
        font-weight: 600 !important;
        color: #E5E7EB !important;
    }

    h4, h5, h6, [data-testid="stHeadingWithAnchor"] h4 {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #E5E7EB !important;
    }

    p, span, label, div {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Body and Secondary Text */
    .stMarkdown p, .stText {
        color: #E5E7EB;
        font-weight: 400;
    }

    [data-testid="stCaptionContainer"], .stCaption {
        color: #94A3B8 !important;
        font-weight: 400 !important;
    }

    /* Form Container / Cards */
    div[data-testid="stForm"] {
        background-color: #151E2E !important;
        border: 1px solid #2A3445 !important;
        border-radius: 8px !important;
        padding: 24px !important;
    }

    /* Widget Labels */
    label[data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"] {
        color: #94A3B8 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }

    /* Selectboxes & Inputs */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    input {
        background-color: #1B2433 !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
        color: #E5E7EB !important;
    }

    div[data-baseweb="select"]:focus-within > div,
    div[data-baseweb="input"]:focus-within > div {
        border-color: #0F766E !important;
        box-shadow: 0 0 0 1px #0F766E !important;
    }

    /* Dropdown Menu */
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[data-testid="stSelectboxVirtualDropdown"] {
        background-color: #1B2433 !important;
        border: 1px solid #2A3445 !important;
    }

    [data-baseweb="menu"] li {
        color: #E5E7EB !important;
        background-color: #1B2433 !important;
    }

    [data-baseweb="menu"] li:hover,
    [data-baseweb="menu"] li[aria-selected="true"] {
        background-color: #151E2E !important;
        color: #2DD4BF !important;
    }

    /* Number Input Controls */
    div[data-testid="stNumberInput"] button {
        background-color: #1B2433 !important;
        border: 1px solid #334155 !important;
        color: #E5E7EB !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background-color: #151E2E !important;
        border-color: #0F766E !important;
        color: #2DD4BF !important;
    }

    /* Buttons */
    button[kind="primary"],
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #0F766E !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }

    button[kind="primary"]:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #0D9488 !important;
        color: #FFFFFF !important;
        border: none !important;
    }

    button[kind="primary"]:active,
    div[data-testid="stFormSubmitButton"] > button:active {
        background-color: #0F766E !important;
    }

    /* Main Area Metric Cards */
    div[data-testid="stMain"] div[data-testid="stMetric"],
    .stMainBlockContainer div[data-testid="stMetric"],
    div[data-testid="stAppViewContainer"] > section:not([data-testid="stSidebar"]) div[data-testid="stMetric"] {
        background-color: #151E2E !important;
        border: 1px solid #2A3445 !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
    }

    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] label {
        color: #94A3B8 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }

    [data-testid="stMetricValue"] div,
    [data-testid="stMetricValue"] {
        color: #E5E7EB !important;
        font-weight: 600 !important;
    }

    /* Progress Bar */
    div[data-testid="stProgress"] > div > div {
        background-color: #1B2433 !important;
    }

    /* Alerts / Recommendations */
    div[data-testid="stAlert"] {
        background-color: #151E2E !important;
        border: 1px solid #2A3445 !important;
        border-left: 3px solid #0F766E !important;
        border-radius: 6px !important;
        color: #E5E7EB !important;
    }

    div[data-testid="stAlert"] p {
        color: #E5E7EB !important;
    }

    div.stAlert:has(div[data-testid="stNotificationErrorIcon"]),
    div.stAlert:has(svg[data-testid="stIconMaterialError"]) {
        background-color: #2A171A !important;
        border: 1px solid #3B1C20 !important;
        border-left: 3px solid #DC2626 !important;
    }

    div.stAlert:has(div[data-testid="stNotificationWarningIcon"]),
    div.stAlert:has(svg[data-testid="stIconMaterialWarning"]) {
        background-color: #261E10 !important;
        border: 1px solid #3D3018 !important;
        border-left: 3px solid #F59E0B !important;
    }

    div.stAlert:has(div[data-testid="stNotificationSuccessIcon"]),
    div.stAlert:has(svg[data-testid="stIconMaterialCheck"]) {
        background-color: #0F241A !important;
        border: 1px solid #1A3D2C !important;
        border-left: 3px solid #15803D !important;
    }

    /* Tabs */
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        color: #94A3B8 !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 500 !important;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        color: #2DD4BF !important;
        font-weight: 600 !important;
    }

    div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {
        background-color: #0F766E !important;
    }

    div[data-testid="stTabs"] div[data-baseweb="tab-border"] {
        background-color: #2A3445 !important;
    }

    /* Horizontal Rule */
    hr {
        border-color: #2A3445 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_production_artifacts():
    """Load the trained model payload produced by model.py."""
    artifact_path = PRIMARY_ARTIFACT
    if not artifact_path.exists():
        st.error(f"Model artifact not found:\n{artifact_path}")
        st.stop()
    payload = joblib.load(artifact_path)

    required_keys = {"model", "threshold", "features"}
    missing_keys = required_keys.difference(payload)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise KeyError(f"Production artifact is missing: {missing}")

    return {
        "path": artifact_path,
        "model": payload["model"],
        "threshold": float(payload["threshold"]),
        "features": list(payload["features"]),
    }


def build_model_input(expected_features, form_values):
    """Mirror the one-hot feature layout generated in model.py."""
    encoded_inputs = {feature: 0 for feature in expected_features}

    numeric_values = {
        "tenure": form_values["tenure"],
        "MonthlyCharges": form_values["monthly_charges"],
        "TotalCharges": form_values["total_charges"],
    }

    for feature, value in numeric_values.items():
        if feature in encoded_inputs:
            encoded_inputs[feature] = value

    categorical_values = {
        "gender": form_values["gender"],
        "Partner": form_values["partner"],
        "Dependents": form_values["dependents"],
        "MultipleLines": form_values["multiple_lines"],
        "InternetService": form_values["internet_service"],
        "OnlineSecurity": form_values["online_security"],
        "OnlineBackup": form_values["online_backup"],
        "DeviceProtection": form_values["device_protection"],
        "TechSupport": form_values["tech_support"],
        "StreamingTV": form_values["streaming_tv"],
        "StreamingMovies": form_values["streaming_movies"],
        "Contract": form_values["contract"],
        "PaperlessBilling": form_values["paperless"],
        "PaymentMethod": form_values["payment_method"],
    }

    for column, value in categorical_values.items():
        feature_name = f"{column}_{value}"
        if feature_name in encoded_inputs:
            encoded_inputs[feature_name] = 1

    return pd.DataFrame([encoded_inputs], columns=expected_features)


def classify_risk(probability, threshold):
    if probability >= threshold:
        return (
            "High Risk",
            "Immediate retention outreach recommended.",
            "#DC2626",
        )
    elif probability >= 0.35:
        return (
            "Moderate Risk",
            "Monitor customer engagement and consider targeted retention strategies.",
            "#F59E0B",
        )
    else:
        return (
            "Low Risk",
            "Customer is likely to stay.",
            "#15803D",
        )


try:
    artifact = load_production_artifacts()
except FileNotFoundError:
    st.error(
        "Production artifact not found. Run model.py first so it creates "
        "`lightgbm_churn_production.pkl`."
    )
    st.stop()
except (KeyError, TypeError, ValueError) as exc:
    st.error(f"Unable to read the production artifact: {exc}")
    st.stop()


model = artifact["model"]
optimal_threshold = artifact["threshold"]
expected_features = artifact["features"]
model_name = type(model).__name__.replace("Classifier", "")


st.sidebar.markdown("## Model Information")
st.sidebar.metric("Loaded Model", model_name)
st.sidebar.metric("Decision Threshold", f"{optimal_threshold * 100:.0f}%")
st.sidebar.metric("Input Features", len(expected_features))
st.sidebar.caption(f"Artifact: {artifact['path'].name}")


st.title("Customer Churn Prediction Dashboard")
st.caption(
    "Predict telecom customer churn from demographics, subscription details, "
    "billing behavior, and service usage with the trained production model."
)
st.markdown("---")


with st.form("churn_prediction_form"):
    st.markdown("### Customer Information")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        partner = st.selectbox("Partner", ["Yes", "No"])
    with col2:
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12,
            step=1,
        )

    st.markdown("### Subscription")
    col1, col2 = st.columns(2)
    with col1:
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    with col2:
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    has_internet = internet_service != "No"

    st.markdown("### Services")
    col1, col2, col3 = st.columns(3)
    with col1:
        online_security = st.selectbox("Online Security", ["Yes", "No"], disabled=not has_internet)
        tech_support = st.selectbox("Tech Support", ["Yes", "No"], disabled=not has_internet)
    with col2:
        online_backup = st.selectbox("Online Backup", ["Yes", "No"], disabled=not has_internet)
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"], disabled=not has_internet)
    with col3:
        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No"],
            disabled=not has_internet,
        )
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"], disabled=not has_internet)

    if not has_internet:
        online_security = "No"
        tech_support = "No"
        online_backup = "No"
        streaming_tv = "No"
        device_protection = "No"
        streaming_movies = "No"

    st.markdown("### Billing")
    col1, col2 = st.columns(2)
    with col1:
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=130.0,
            value=65.0,
            step=0.5,
        )
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=9000.0,
            value=780.0,
            step=1.0,
        )
    with col2:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    submitted = st.form_submit_button("Predict Churn", type="primary")


if submitted:
    form_values = {
        "gender": gender,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "internet_service": internet_service,
        "contract": contract,
        "multiple_lines": multiple_lines,
        "payment_method": payment_method,
        "online_security": online_security,
        "tech_support": tech_support,
        "online_backup": online_backup,
        "streaming_tv": streaming_tv,
        "device_protection": device_protection,
        "streaming_movies": streaming_movies,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "paperless": paperless,
    }

    input_df = build_model_input(expected_features, form_values)

    churn_probability = float(model.predict_proba(input_df)[0, 1])

    risk_label, recommendation, progress_color = classify_risk(
        churn_probability,
        optimal_threshold,
    )

    st.markdown("---")

    st.markdown(
        f"""
        <style>
        div[data-testid="stProgress"] > div > div > div > div {{
            background-color: {progress_color} !important;
        }}
        div[data-testid="column"]:nth-of-type(2) div[data-testid="stMetricValue"] > div {{
            color: {progress_color} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Prediction Result")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric(
            "Churn Probability",
            f"{churn_probability * 100:.1f}%"
        )

    with metric_col2:
        st.metric(
            "Risk Level",
            risk_label
        )

    with metric_col3:
        st.metric(
            "Decision Threshold",
            f"{optimal_threshold * 100:.0f}%"
        )

    st.progress(churn_probability)

    if risk_label == "High Risk":
        st.error(f"**Recommendation:** {recommendation}")
    elif risk_label == "Moderate Risk":
        st.warning(f"**Recommendation:** {recommendation}")
    else:
        st.success(f"**Recommendation:** {recommendation}")
        st.progress(min(max(churn_probability, 0.0), 1.0))

    st.markdown("#### Customer Summary")
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st.write(f"**Tenure:** {tenure} months")
        st.write(f"**Contract:** {contract}")
        st.write(f"**Internet:** {internet_service}")
    with s_col2:
        st.write(f"**Monthly Charge:** ${monthly_charges:.2f}")
        st.write(f"**Total Charge:** ${total_charges:.2f}")
        st.write(f"**Payment Method:** {payment_method}")


st.markdown("---")
st.header("Model Explainability")
st.caption(
    "Understand how the LightGBM model identifies the most influential "
    "features driving customer churn predictions."
)

tab1, tab2 = st.tabs(["Global Feature Importance", "SHAP Value Distribution"])

with tab1:
    if FEATURE_IMPORTANCE_IMAGE.exists():
        c1, c2, c3 = st.columns([1, 4, 1])
        with c2:
            st.info(
                "This chart ranks features by their average contribution to model predictions across the training dataset."
            )
            st.image(str(FEATURE_IMPORTANCE_IMAGE), use_container_width=True)
    else:
        st.warning("`shap_feature_importance.png` was not found in the workspace.")

with tab2:
    if SHAP_BEESWARM_IMAGE.exists():
        c1, c2, c3 = st.columns([1, 4, 1])
        with c2:
            st.info(
                "Each point represents one customer. Features pushing predictions to the right increase churn risk, while those to the left reduce it. Color indicates the feature value."
            )
            st.image(str(SHAP_BEESWARM_IMAGE), use_container_width=True)
    else:
        st.warning("`shap_beeswarm_plot.png` was not found in the workspace.")


st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#94A3B8; font-size:14px; font-family: 'IBM Plex Sans', sans-serif;">
    © 2026 Jayed Ansari<br>
    Built with <b>Python</b>, <b>LightGBM</b>, <b>Optuna</b>,
<b>SHAP</b>, <b>Scikit-learn</b>, and <b>Streamlit</b>.
    </div>
    """,
    unsafe_allow_html=True,
)
