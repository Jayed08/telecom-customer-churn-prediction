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
            "#dc2626",
        )
    elif probability >= 0.35:
        return (
            "Moderate Risk",
            "Monitor customer engagement and consider targeted retention strategies.",
            "#f59e0b",
        )
    else:
        return (
            "Low Risk",
            "Customer is likely to stay.",
            "#16a34a",
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
    <div style="text-align:center; color:gray; font-size:14px;">
    © 2026 Jayed Ansari<br>
    Built with <b>Python</b>, <b>LightGBM</b>, <b>Optuna</b>,
<b>SHAP</b>, <b>Scikit-learn</b>, and <b>Streamlit</b>.
    </div>
    """,
    unsafe_allow_html=True,
)
