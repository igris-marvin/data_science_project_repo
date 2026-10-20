# app.py
# Water Supply Reliability Risk Predictor
 
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from wsa_transformer import WSAFeatureEngineer

# -------------------------------------------------------------------------
# Load the saved best_model pipeline
# -------------------------------------------------------------------------
model_pipeline = joblib.load("models/best_model.pkl")

print("[SUCCESS] best_model.pkl loaded successfully from 'models/' directory!")

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
 
st.set_page_config(
    page_title="Water Supply Reliability Predictor",
    page_icon="💧",
    layout="wide"
)
 
# --------------------------------------------------
# LOAD SAVED MODEL PIPELINE
# --------------------------------------------------
 
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")
 
try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()
 
# --------------------------------------------------
# HEADER
# --------------------------------------------------
 
st.title("💧 Water Supply Reliability Risk Predictor")
 
st.markdown("""
This application predicts the likelihood that a Water Services Authority (WSA)
will experience water service reliability problems.
 
### Risk Categories
- 🔴 High Risk
- 🟡 Medium Risk
- 🟢 Low Risk
 
The prediction is generated using the optimised Deep Tabular Neural Network
developed for the IDA117V project.
""")
 
# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
 
st.sidebar.header("Responsible Use")
 
st.sidebar.info(
    """
    This model is intended to support municipal planning and decision-making.
 
    Predictions should not replace engineering assessments,
    operational reviews, field inspections,
    or professional judgement.
 
    The model is only as reliable as the data provided.
    """
)
 
# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------
 
st.subheader("Enter Water Services Authority Information")
 
col1, col2 = st.columns(2)
 
with col1:
 
    population = st.number_input(
        "Population",
        min_value=0.0,
        value=100000.0,
        help="Total population served by the WSA"
    )
 
    population_access = st.number_input(
        "Population with access to water",
        min_value=0.0,
        value=90000.0,
        help="Population with access to water services"
    )
 
    urban_population = st.number_input(
        "Urban Population",
        min_value=0.0,
        value=70000.0,
        help="Total urban population"
    )
 
with col2:
 
    urban_access = st.number_input(
        "Urban Population with access to water",
        min_value=0.0,
        value=65000.0,
        help="Urban population with water access"
    )
 
    rural_population = st.number_input(
        "Rural Population",
        min_value=0.0,
        value=30000.0,
        help="Total rural population"
    )
 
    rural_access = st.number_input(
        "Rural Population with access to water",
        min_value=0.0,
        value=25000.0,
        help="Rural population with water access"
    )
 
# --------------------------------------------------
# VALIDATION
# --------------------------------------------------
 
def validate_inputs():
 
    errors = []
 
    if population <= 0:
        errors.append("Population must be greater than zero.")
 
    if population_access > population:
        errors.append(
            "Population with access to water cannot exceed Population."
        )
 
    if urban_access > urban_population:
        errors.append(
            "Urban Population with access to water cannot exceed Urban Population."
        )
 
    if rural_access > rural_population:
        errors.append(
            "Rural Population with access to water cannot exceed Rural Population."
        )
 
    if urban_population + rural_population > population:
        errors.append(
            "Urban Population + Rural Population cannot exceed Total Population."
        )
 
    return errors
 
# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------
 
if st.button("Predict Risk Level", type="primary"):
 
    validation_errors = validate_inputs()
 
    if validation_errors:
 
        for err in validation_errors:
            st.error(err)
 
    else:
 
        # EXACT FEATURE NAMES USED BY YOUR MODEL
 
        input_df = pd.DataFrame({
            "Population": [population],
            "Population with access to water": [population_access],
            "Urban Population": [urban_population],
            "Urban Population with access to water": [urban_access],
            "Rural Population": [rural_population],
            "Rural Population with access to water": [rural_access]
        })
 
        try:
 
            prediction = model.predict(input_df)[0]
 
            probabilities = model.predict_proba(input_df)[0]
 
            classes = model.classes_
 
            confidence = np.max(probabilities) * 100
 
            # --------------------------------------
            # SHOW PREDICTION
            # --------------------------------------
 
            st.subheader("Prediction Result")
 
            if prediction == "High Risk":
 
                st.error(
                    f"🔴 HIGH RISK\n\nConfidence: {confidence:.2f}%"
                )
 
            elif prediction == "Medium Risk":
 
                st.warning(
                    f"🟡 MEDIUM RISK\n\nConfidence: {confidence:.2f}%"
                )
 
            else:
 
                st.success(
                    f"🟢 LOW RISK\n\nConfidence: {confidence:.2f}%"
                )
 
            # --------------------------------------
            # PROBABILITY TABLE
            # --------------------------------------
 
            st.subheader("Class Probabilities")
 
            probability_df = pd.DataFrame({
                "Risk Class": classes,
                "Probability (%)": np.round(probabilities * 100, 2)
            })
 
            st.dataframe(
                probability_df,
                use_container_width=True
            )
 
            # --------------------------------------
            # FEATURE CALCULATIONS
            # --------------------------------------
 
            water_access_ratio = (
                population_access / population
            ) * 100
 
            unserved_population = (
                population - population_access
            )
 
            urban_ratio = (
                urban_population / population
            ) * 100
 
            # --------------------------------------
            # EXPLANATION
            # --------------------------------------
 
            st.subheader("Plain Language Explanation")
 
            if prediction == "High Risk":
 
                st.write(
                    f"""
                    The model predicts High Risk because the area has
                    a relatively low water access rate
                    ({water_access_ratio:.1f}%)
                    and approximately
                    {unserved_population:,.0f}
                    people do not have water access.
 
                    These conditions are commonly associated with
                    higher service-delivery vulnerability and reduced
                    water-supply reliability.
                    """
                )
 
            elif prediction == "Medium Risk":
 
                st.write(
                    f"""
                    The model predicts Medium Risk.
 
                    Water access is
                    {water_access_ratio:.1f}%
                    and the municipality may experience reliability
                    challenges that require monitoring.
 
                    Preventative maintenance and infrastructure reviews
                    are recommended.
                    """
                )
 
            else:
 
                st.write(
                    f"""
                    The model predicts Low Risk.
 
                    Water access is relatively high
                    ({water_access_ratio:.1f}%),
                    indicating stronger service coverage and
                    lower likelihood of reliability problems.
 
                    Continued monitoring is recommended.
                    """
                )
 
            # --------------------------------------
            # SUMMARY METRICS
            # --------------------------------------
 
            st.subheader("Derived Indicators")
 
            metric1, metric2, metric3 = st.columns(3)
 
            metric1.metric(
                "Water Access Ratio",
                f"{water_access_ratio:.1f}%"
            )
 
            metric2.metric(
                "Urban Ratio",
                f"{urban_ratio:.1f}%"
            )
 
            metric3.metric(
                "Unserved Population",
                f"{unserved_population:,.0f}"
            )
 
        except Exception as e:

            print(e)
 
            st.error(
                f"""
                Prediction failed.
 
                Possible causes:
                - Incorrect model file
                - Pipeline not saved correctly
                - Feature mismatch
 
                Error: {e}
                """
            )
 
# --------------------------------------------------
# FOOTER
# --------------------------------------------------
 
st.markdown("---")
 
st.caption(
    """
    IDA117V Machine Learning Project
 
    Features used by the model:
    Population,
    Population with access to water,
    Urban Population,
    Urban Population with access to water,
    Rural Population,
    Rural Population with access to water.
 
    No personal information is collected or stored.
    """
)