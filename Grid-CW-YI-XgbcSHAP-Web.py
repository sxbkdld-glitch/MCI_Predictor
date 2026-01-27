import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. Advanced Page Configuration & CSS
# ==========================================
st.set_page_config(
    page_title="MCI Clinical Predictor",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Commercial-Grade UI
st.markdown("""
<style>
    /* Main Background & Font */
    .reportview-container {
        background: #f0f2f6;
    }
    
    /* Header Styling */
    h1 {
        color: #0e1117;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e6e9ef;
    }
    h2, h3 {
        color: #262730;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Card Styling */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e6e9ef;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0068c9;
    }
    .metric-label {
        font-size: 1rem;
        color: #6c757d;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* High Risk Warning Style */
    .risk-high {
        color: #ff4b4b !important;
    }
    
    /* Low Risk Success Style */
    .risk-low {
        color: #09ab3b !important;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        font-weight: 600;
        height: 3em;
        background-color: #0068c9;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0053a0;
        border: none;
        color: white;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e6e9ef;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Header Section
# ==========================================
col_logo, col_title = st.columns([1, 5])

with col_logo:
    # Placeholder for a medical logo (using emoji for now)
    st.markdown("<div style='font-size: 4rem; text-align: center;'>⚕️</div>", unsafe_allow_html=True)

with col_title:
    st.title("MCI Clinical Prediction System")
    st.markdown("""
    **Artificial Intelligence / XGBoost Engine** This application utilizes advanced machine learning algorithms to predict the probability of adverse clinical outcomes based on patient biomarkers and demographics.
    """)

st.markdown("---")

# ==========================================
# 3. Resource Loading (Model & Data)
# ==========================================
@st.cache_resource
def load_model():
    # Load the XGBoost model
    return joblib.load('XGBC.pkl')

@st.cache_data
def load_data():
    # Load test data for feature schema
    return pd.read_csv('X_test.csv')

try:
    model = load_model()
    X_test = load_data()
    feature_names = X_test.columns.tolist()
except Exception as e:
    st.error("System Initialization Failed")
    st.error(f"Error details: {e}")
    st.info("Ensure 'XGBC.pkl' and 'X_test.csv' are in the root directory.")
    st.stop()

# ==========================================
# 4. Sidebar: Patient Configuration
# ==========================================
with st.sidebar:
    st.header("📋 Patient Configuration")
    st.markdown("Configure clinical parameters below:")
    st.markdown("---")

    input_data = {}
    
    # Create a form to group inputs
    with st.form("patient_data_form"):
        # Iterate features and generate appropriate inputs
        for feature in feature_names:
            min_val = float(X_test[feature].min())
            max_val = float(X_test[feature].max())
            default_val = float(X_test[feature].mean())
            
            # Formatting the label for better readability (Replacing underscores)
            label = feature.replace("_", " ").title()

            if X_test[feature].nunique() <= 2:
                # Binary features
                input_data[feature] = st.selectbox(
                    label,
                    options=[0, 1],
                    index=int(default_val),
                    help=f"Select binary value for {label}"
                )
            else:
                # Continuous features
                input_data[feature] = st.number_input(
                    label,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    format="%.2f",
                    help=f"Range: {min_val:.2f} - {max_val:.2f}"
                )
        
        st.markdown("###")
        # The Run button is now inside the sidebar form
        submitted = st.form_submit_button("🚀 Run Analysis")

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# ==========================================
# 5. Main Dashboard Logic
# ==========================================

# Display input summary in an expander
with st.expander("📊 View Current Patient Profile (Input Data)"):
    st.dataframe(input_df, hide_index=True)

if submitted:
    # --- Prediction Logic ---
    try:
        # 1. Get underlying booster to bypass sklearn version mismatch
        booster = model.get_booster()
        
        # 2. Convert to DMatrix (Standard XGBoost format)
        dtest = xgb.DMatrix(input_df)
        
        # 3. Predict
        risk_score = booster.predict(dtest)
        
        # Handle format
        if isinstance(risk_score, np.ndarray):
            risk_score = float(risk_score[0])
        else:
            risk_score = float(risk_score)
            
        prediction_class = 1 if risk_score > 0.5 else 0
        
        # --- UI: Results Dashboard ---
        st.subheader("Diagnostics Result")
        
        # Layout for results
        res_col1, res_col2, res_col3 = st.columns([1, 1, 2])
        
        with res_col1:
            # Custom HTML Card for Class
            class_color = "risk-high" if prediction_class == 1 else "risk-low"
            class_label = "POSITIVE (Risk)" if prediction_class == 1 else "NEGATIVE (Safe)"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Prediction</div>
                <div class="metric-value {class_color}">{class_label}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            # Custom HTML Card for Probability
            prob_color = "risk-high" if risk_score > 0.5 else "risk-low"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Probability</div>
                <div class="metric-value {prob_color}">{risk_score:.2%}</div>
            </div>
            """, unsafe_allow_html=True)

        with res_col3:
            # Progress Bar Visualization
            st.markdown(f"""<div class="metric-card" style="text-align:left; padding-top:25px;">
                            <div class="metric-label" style="margin-bottom:10px;">Risk Assessment Gauge</div>
                        """, unsafe_allow_html=True)
            
            if risk_score > 0.5:
                st.progress(risk_score, text="⚠️ High Risk Detected")
            else:
                st.progress(risk_score, text="✅ Low Risk Profile")
            st.markdown("</div>", unsafe_allow_html=True)

        # --- Explainability Section (SHAP) ---
        st.markdown("###")
        st.subheader("🔍 Model Explainability (SHAP Analysis)")
        st.markdown("The following chart illustrates how each feature contributed to the final risk score.")
        
        with st.container():
            with st.spinner('Calculating feature contributions...'):
                try:
                    explainer = shap.TreeExplainer(booster)
                    shap_values = explainer.shap_values(input_df)
                    
                    # Compatibility handling for SHAP return types
                    if isinstance(shap_values, list):
                        shap_vals_to_plot = shap_values[1]
                    else:
                        shap_vals_to_plot = shap_values
                    
                    if shap_vals_to_plot.ndim > 1:
                        shap_vals_to_plot = shap_vals_to_plot[0]
                    
                    base_val = explainer.expected_value
                    if isinstance(base_val, (list, np.ndarray)) and len(base_val) > 1:
                        pass # Handle list if necessary
                        
                    # Visualization configuration
                    plt.style.use('default') 
                    fig, ax = plt.subplots(figsize=(10, 4))
                    
                    # Generate Waterfall plot
                    shap.plots.waterfall(shap.Explanation(values=shap_vals_to_plot, 
                                                         base_values=base_val, 
                                                         data=input_df.iloc[0], 
                                                         feature_names=feature_names),
                                         show=False)
                    
                    # Customize plot to blend with UI
                    plt.gcf().set_facecolor('none') # Transparent background
                    plt.gca().set_facecolor('none')
                    
                    st.pyplot(plt.gcf(), use_container_width=True)
                    
                except Exception as e_shap:
                    st.warning(f"Could not generate SHAP visualization: {e_shap}")
                    
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.code(f"Debug Info: {type(model)}")

else:
    # Initial State Prompt
    st.info("👈 Please configure patient parameters in the sidebar and click 'Run Analysis'.")

# ==========================================
# 6. Footer / Disclaimer
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.8rem;'>
    <strong>Disclaimer:</strong> This tool is for research and demonstration purposes only. 
    It is not intended to replace professional medical advice, diagnosis, or treatment.
    <br>Developed for MCI Research Project © 2026
</div>
""", unsafe_allow_html=True)
