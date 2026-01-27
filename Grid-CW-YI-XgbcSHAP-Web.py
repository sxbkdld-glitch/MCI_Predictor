import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. 页面基础配置
# ==========================================
st.set_page_config(
    page_title="MCI 6-Year Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. 商业级 UI 设计 (颜色深度修复版)
# ==========================================
st.markdown("""
<style>
    /* 强制重置字体 */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    }

    /* --------------------------------------------------- */
    /* 核心修复区：解决文字看不清的问题 */
    /* --------------------------------------------------- */
    
    /* 1. 强制主区域的所有各级标题为黑色 */
    [data-testid="stAppViewContainer"] h1, 
    [data-testid="stAppViewContainer"] h2, 
    [data-testid="stAppViewContainer"] h3, 
    [data-testid="stAppViewContainer"] h4 {
        color: #000000 !important; /* 纯黑 */
    }

    /* 2. 特别修复：进度条上方的文字 (Elevated risk detected) */
    [data-testid="stAppViewContainer"] .stProgress p {
        color: #000000 !important; /* 纯黑 */
        font-weight: 600 !important; /* 加粗，更清晰 */
        font-size: 1rem !important;
    }

    /* --------------------------------------------------- */
    /* 侧边栏样式 (保持深色高级感) */
    /* --------------------------------------------------- */
    [data-testid="stSidebar"] {
        background-color: #0f172a; /* 深空蓝黑 */
        border-right: 1px solid #1e293b;
    }
    
    /* 侧边栏标题必须保持白色 (否则看不见) */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #f8fafc !important; 
    }
    
    /* 侧边栏普通文字 */
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown {
        color: #cbd5e1 !important; 
    }
    
    /* 侧边栏输入框说明文字 */
    [data-testid="stSidebar"] .stNumberInput label, [data-testid="stSidebar"] .stSelectbox label {
        color: #94a3b8 !important;
    }

    /* --------------------------------------------------- */
    /* 主区域样式 (强制浅色背景) */
    /* --------------------------------------------------- */
    [data-testid="stAppViewContainer"] {
        background-color: #f8fafc; /* 极浅灰白背景 */
    }
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }

    /* 主标题样式 */
    .main-title {
        font-size: 2.5rem;
        color: #0f172a !important; /* 深色文字 */
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #334155 !important; /* 深灰文字 */
        background-color: #ffffff;
        padding: 15px 20px;
        border-radius: 8px;
        border-left: 5px solid #2563eb;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }

    /* 结果卡片样式 */
    .metric-card {
        background-color: #ffffff; /* 强制白底 */
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: all 0.2s ease;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 10px 0;
        letter-spacing: -0.02em;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
    }

    /* 状态颜色 */
    .text-safe { color: #059669 !important; } 
    .text-risk { color: #dc2626 !important; } 
    .bg-safe { background-color: #ecfdf5 !important; border-color: #10b981 !important; }
    .bg-risk { background-color: #fef2f2 !important; border-color: #ef4444 !important; }

    /* 按钮美化 */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1rem;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    }
    .stButton>button:hover {
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. 标题区
# ==========================================
col_header_1, col_header_2 = st.columns([1, 6])
with col_header_1:
    st.markdown("""
        <div style='background: white; border-radius: 100px; width: 80px; height: 80px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: auto;'>
            <span style='font-size: 3rem;'>🧠</span>
        </div>
    """, unsafe_allow_html=True)
with col_header_2:
    st.markdown('<div class="main-title">6-Year MCI Risk Prediction System</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sub-title">
        <b>Target Population:</b> Currently cognitively normal individuals.<br>
        <b>Objective:</b> Predict the probability of developing Mild Cognitive Impairment (MCI) within 6 years.
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 4. 加载资源
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load('XGBC.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('X_test.csv')

try:
    model = load_model()
    X_test = load_data()
    feature_names = X_test.columns.tolist()
except Exception as e:
    st.error("System Initialization Error")
    st.info("Please ensure 'XGBC.pkl' and 'X_test.csv' are uploaded.")
    st.stop()

# ==========================================
# 5. 侧边栏：特征输入 (深色适配版)
# ==========================================
feature_map = {
        "Baseline Cognitive": "Baseline Cognitive Score",
        "IADL": "IADL Impairment Count"
    }


with st.sidebar:
    st.title("📋 Clinical Parameters")
    st.markdown("Please input patient details below:", unsafe_allow_html=True)
    st.markdown("---")

    input_data = {}
    
    with st.form("patient_data_form"):
        for feature in feature_names:
            min_val = float(X_test[feature].min())
            max_val = float(X_test[feature].max())
            default_val = float(X_test[feature].mean())
            label = feature_map.get(feature, feature.replace("_", " ").title())
            
            # 1. 婚姻状况特殊处理
            if 'marital' in feature.lower():
                input_data[feature] = st.selectbox(
                    label,
                    options=[0, 1],
                    index=1 if default_val > 0.5 else 0,
                    format_func=lambda x: "Married & Cohabitating" if x == 1 else "Other",
                    label_visibility="collapsed"
                )
            
            # 2. 整数变量处理
            elif any(x in feature.lower() for x in ['age', 'cognitive', 'score', 'iadl']):
                if 'iadl' in feature.lower():
                     current_min, current_max = 0.0, 8.0
                else:
                     current_min, current_max = min_val, max_val

                input_data[feature] = st.number_input(
                    f"{label}",
                    min_value=int(current_min),
                    max_value=int(current_max),
                    value=int(default_val),
                    step=1,
                    format="%d"
                )

            # 3. 其他变量
            elif X_test[feature].nunique() <= 2:
                input_data[feature] = st.selectbox(label, options=[0, 1], index=int(default_val))
            else:
                input_data[feature] = st.number_input(
                    label,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    format="%.2f"
                )
        
        st.markdown("###")
        submitted = st.form_submit_button("Run Prediction Analysis")

# 转为 DataFrame
input_df = pd.DataFrame([input_data])

# ==========================================
# 6. 预测逻辑与展示
# ==========================================
if submitted:
    try:
        booster = model.get_booster()
        dtest = xgb.DMatrix(input_df)
        risk_score = booster.predict(dtest)
        
        if isinstance(risk_score, np.ndarray):
            risk_score = float(risk_score[0])
        else:
            risk_score = float(risk_score)
            
        prediction_class = 1 if risk_score > 0.5 else 0
        
        # --- 结果展示区 ---
        st.markdown("###")
        st.subheader("Diagnostic Report") # 这里的文字现在会强制变成黑色
        
        res_col1, res_col2, res_col3 = st.columns([1.2, 1.2, 2])
        
        theme_color = "text-risk" if prediction_class == 1 else "text-safe"
        bg_class = "bg-risk" if prediction_class == 1 else "bg-safe"
        status_text = "HIGH RISK" if prediction_class == 1 else "LOW RISK"
        
        with res_col1:
            st.markdown(f"""
            <div class="metric-card {bg_class}">
                <div class="metric-label">Predicted Outcome</div>
                <div class="metric-value {theme_color}">{status_text}</div>
                <div style="font-size:0.8rem; color:#64748b;">@ 6 Years Horizon</div>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Probability</div>
                <div class="metric-value {theme_color}">{risk_score:.2%}</div>
                <div style="font-size:0.8rem; color:#64748b;">Confidence Score</div>
            </div>
            """, unsafe_allow_html=True)

        with res_col3:
            # 进度条
            st.markdown(f"""<div class="metric-card" style="text-align:left; padding: 25px;">
                            <div class="metric-label" style="margin-bottom:12px;">Risk Assessment Gauge</div>
                        """, unsafe_allow_html=True)
            if risk_score > 0.5:
                # 这里的文字现在会强制变成黑色加粗
                st.progress(risk_score, text="⚠️ Elevated risk detected")
            else:
                st.progress(risk_score, text="✅ Patient is likely to remain stable")
            st.markdown("</div>", unsafe_allow_html=True)

        # --- SHAP 可视化 ---
        st.markdown("###")
        st.subheader("Interpretability Analysis (SHAP)") # 这里的文字现在会强制变成黑色
        
        # 容器背景设为白色
        with st.container():
            st.markdown('<div style="background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0;">', unsafe_allow_html=True)
            
            with st.spinner('Calculating feature importance...'):
                explainer = shap.TreeExplainer(booster)
                shap_values = explainer.shap_values(input_df)
                
                if isinstance(shap_values, list): shap_val = shap_values[1]
                else: shap_val = shap_values
                
                if shap_val.ndim > 1: shap_val = shap_val[0]

                base_val = explainer.expected_value
                if isinstance(base_val, (list, np.ndarray)) and len(base_val) > 1: pass 

                # 绘图设置
                plt.style.use('default') 
                fig, ax = plt.subplots(figsize=(10, 5))
                fig.patch.set_facecolor('white') # 强制白底
                ax.set_facecolor('white')
                
                shap.plots.waterfall(shap.Explanation(values=shap_val, 
                                                     base_values=base_val, 
                                                     data=input_df.iloc[0], 
                                                     feature_names=feature_names),
                                     show=False)
                
                # 强制坐标轴和文字为黑色 (解决看不清的问题)
                plt.rcParams['text.color'] = '#000000'
                plt.rcParams['axes.labelcolor'] = '#000000'
                plt.rcParams['xtick.color'] = '#000000'
                plt.rcParams['ytick.color'] = '#000000'
                
                st.pyplot(fig, use_container_width=True)
                st.caption("Chart Guide: Red bars push risk HIGHER, Blue bars push risk LOWER.")
            
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction Error: {e}")

else:
    # 初始状态提示
    st.info("👈 Please enter patient data in the sidebar and click 'Run Prediction Analysis' to start.")

# ==========================================
# 7. 页脚
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #94a3b8; font-size: 0.8rem; padding-bottom: 20px;'>
    <strong>Research Prototype.</strong> Not for clinical diagnosis.<br>
    © 2026 MCI Prediction Research Group
</div>
""", unsafe_allow_html=True)







