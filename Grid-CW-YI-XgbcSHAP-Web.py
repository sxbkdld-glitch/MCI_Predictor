import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. 高级页面配置与 CSS 美化
# ==========================================
st.set_page_config(
    page_title="MCI 6-Year Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入自定义 CSS
st.markdown("""
<style>
    /* 全局字体 */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }

    /* 1. 侧边栏美化：去除突兀的白色，使用柔和的浅灰 */
    [data-testid="stSidebar"] {
        background-color: #f4f5f7; /* 柔和的浅灰 */
        border-right: 1px solid #d1d5db;
    }

    /* 侧边栏标题 */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #1f2937;
    }

    /* 主标题样式 */
    .main-title {
        font-size: 2.2rem;
        color: #111827;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 2rem;
        padding-left: 5px;
        border-left: 4px solid #0068c9;
    }

    /* 结果卡片样式 */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }

    /* 风险颜色定义 */
    .text-safe { color: #059669 !important; } /* 绿色 */
    .text-risk { color: #dc2626 !important; } /* 红色 */
    .bg-safe { background-color: #ecfdf5; border-color: #10b981; }
    .bg-risk { background-color: #fef2f2; border-color: #ef4444; }

    /* 按钮美化 */
    .stButton>button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 标题区
# ==========================================
col_header_1, col_header_2 = st.columns([1, 6])
with col_header_1:
    st.markdown("<div style='font-size: 4.5rem; text-align: center;'>🧠</div>", unsafe_allow_html=True)
with col_header_2:
    st.markdown('<div class="main-title">6-Year MCI Risk Prediction System</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sub-title">
        <b>Target Population:</b> Currently cognitively normal individuals.<br>
        <b>Objective:</b> Predict the probability of developing Mild Cognitive Impairment (MCI) within 6 years.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 3. 加载资源
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
# 4. 侧边栏：特征输入 (逻辑优化版)
# ==========================================
with st.sidebar:
    st.header("📋 Clinical Parameters")
    st.markdown("Please input patient details:")
    st.markdown("---")

    input_data = {}
    
    with st.form("patient_data_form"):
        for feature in feature_names:
            # 数据清洗：获取最小值、最大值、均值
            min_val = float(X_test[feature].min())
            max_val = float(X_test[feature].max())
            default_val = float(X_test[feature].mean())
            
            # 格式化标签显示 (去除下划线，首字母大写)
            label = feature.replace("_", " ").title()
            
            # --- 核心修改 1: 特殊变量处理 (Marital Status) ---
            # 模糊匹配：只要列名里包含 'marital' (忽略大小写)
            if 'marital' in feature.lower():
                st.markdown(f"**{label}**")
                input_data[feature] = st.selectbox(
                    label,
                    options=[0, 1],
                    index=1 if default_val > 0.5 else 0, # 默认值逻辑
                    format_func=lambda x: "Married & Cohabitating" if x == 1 else "Other",
                    label_visibility="collapsed" # 隐藏重复标签
                )
            
            # --- 核心修改 2: 整数变量处理 (Age, Cognitive Score, IADL) ---
            # 只要列名包含这些关键字，强制设为整数输入
            elif any(x in feature.lower() for x in ['age', 'cognitive', 'score', 'iadl']):
                # 针对 IADL 特殊处理范围 (假设 1-5 或 0-8)
                if 'iadl' in feature.lower():
                     # 这里按照您的要求：只能输入整数
                     current_min = 0.0 # 通常 IADL 从 0 开始，您也可以设为 1
                     current_max = 8.0 # 通常最大 8，也可以设为您的最大值
                     current_step = 1.0
                else:
                     current_min = min_val
                     current_max = max_val
                     current_step = 1.0

                input_data[feature] = st.number_input(
                    f"{label} (Integer)",
                    min_value=int(current_min),
                    max_value=int(current_max),
                    value=int(default_val),
                    step=1,          # 强制步长为 1
                    format="%d"      # 强制显示为整数格式 (无小数点)
                )

            # --- 二分类变量 (其他) ---
            elif X_test[feature].nunique() <= 2:
                input_data[feature] = st.selectbox(
                    label,
                    options=[0, 1],
                    index=int(default_val)
                )

            # --- 连续变量 (其他) ---
            else:
                input_data[feature] = st.number_input(
                    label,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    format="%.2f"
                )
        
        st.markdown("###")
        submitted = st.form_submit_button("🚀 Run 6-Year Prediction")

# 转为 DataFrame
input_df = pd.DataFrame([input_data])

# ==========================================
# 5. 预测逻辑与展示
# ==========================================
if submitted:
    # 侧边栏收起提示 (可选)
    # st.toast("Calculating...", icon="⏳")

    try:
        # 1. 获取底层 Booster (避开版本兼容问题)
        booster = model.get_booster()
        
        # 2. 转换为 DMatrix
        dtest = xgb.DMatrix(input_df)
        
        # 3. 预测概率
        risk_score = booster.predict(dtest)
        
        # 格式标准化
        if isinstance(risk_score, np.ndarray):
            risk_score = float(risk_score[0])
        else:
            risk_score = float(risk_score)
            
        prediction_class = 1 if risk_score > 0.5 else 0
        
        # --- 结果展示区 ---
        st.subheader("📊 Prediction Results")
        
        res_col1, res_col2, res_col3 = st.columns([1.2, 1.2, 2])
        
        # 样式逻辑
        theme_color = "text-risk" if prediction_class == 1 else "text-safe"
        bg_class = "bg-risk" if prediction_class == 1 else "bg-safe"
        status_text = "HIGH RISK (MCI)" if prediction_class == 1 else "LOW RISK (Normal)"
        
        with res_col1:
            st.markdown(f"""
            <div class="metric-card {bg_class}">
                <div class="metric-label">Predicted Outcome</div>
                <div class="metric-value {theme_color}">{status_text}</div>
                <div style="font-size:0.8rem; color:#666;">@ 6 Years</div>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Probability</div>
                <div class="metric-value {theme_color}">{risk_score:.2%}</div>
                <div style="font-size:0.8rem; color:#666;">Confidence Score</div>
            </div>
            """, unsafe_allow_html=True)

        with res_col3:
            st.markdown(f"""<div class="metric-card" style="text-align:left; padding: 25px;">
                            <div class="metric-label" style="margin-bottom:8px;">Risk Gauge</div>
                        """, unsafe_allow_html=True)
            if risk_score > 0.5:
                st.progress(risk_score, text="⚠️ Elevated risk of cognitive decline")
            else:
                st.progress(risk_score, text="✅ Likely to remain cognitively normal")
            st.markdown("</div>", unsafe_allow_html=True)

        # --- SHAP 可视化 (修复背景色问题) ---
        st.markdown("###")
        st.subheader("🔍 Personalized Risk Factors (SHAP)")
        st.markdown("This chart explains *why* the model made this prediction based on the patient's data.")
        
        with st.container():
            with st.spinner('Generating interpretability chart...'):
                explainer = shap.TreeExplainer(booster)
                shap_values = explainer.shap_values(input_df)
                
                # 数据兼容性处理
                if isinstance(shap_values, list):
                    shap_val = shap_values[1]
                else:
                    shap_val = shap_values
                
                if shap_val.ndim > 1:
                    shap_val = shap_val[0]

                base_val = explainer.expected_value
                if isinstance(base_val, (list, np.ndarray)) and len(base_val) > 1:
                    pass 

                # --- 核心修改 3: 强制设置 Matplotlib 白底样式 ---
                # 这样可以保证在 Streamlit 的深色模式下，图表依然是白底黑字，清晰可见
                plt.style.use('default') 
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # 设置图表背景为白色，字体为黑色
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')
                
                # 绘制瀑布图
                shap.plots.waterfall(shap.Explanation(values=shap_val, 
                                                     base_values=base_val, 
                                                     data=input_df.iloc[0], 
                                                     feature_names=feature_names),
                                     show=False)
                
                # 再次确保坐标轴颜色正确
                plt.rcParams['text.color'] = 'black'
                plt.rcParams['axes.labelcolor'] = 'black'
                plt.rcParams['xtick.color'] = 'black'
                plt.rcParams['ytick.color'] = 'black'
                
                st.pyplot(fig, use_container_width=True)
                st.caption("Red bars increase MCI risk; Blue bars decrease MCI risk.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

else:
    st.info("👈 Please enter the 6-year baseline data in the sidebar and click 'Run 6-Year Prediction'.")

# ==========================================
# 6. 页脚
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #9ca3af; font-size: 0.8rem;'>
    <strong>Research Use Only.</strong> This model estimates the 6-year risk of Mild Cognitive Impairment (MCI).<br>
    It assumes the subject is currently cognitively normal. Not for clinical diagnosis.
    <br>© 2026 MCI Prediction Research Group
</div>
""", unsafe_allow_html=True)
