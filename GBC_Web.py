import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

import importlib.metadata
import sys

def generate_requirements():
    target_packages = [
        "streamlit",
        "pandas",
        "numpy",
        "joblib",
        "shap",
        "matplotlib",
        "scikit-learn" 
    ]
    
    output_file = "requirements.txt"
    successful_packages = []
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for package in target_packages:
                try:
                    version = importlib.metadata.version(package)
                    line = f"{package}=={version}"
                    f.write(line + "\n")
                    successful_packages.append(line)
                except importlib.metadata.PackageNotFoundError:
                    pass
    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    generate_requirements()

# ==========================================
# 1. 页面配置 (Page Configuration)
# ==========================================
st.set_page_config(
    page_title="6-Year MCI Risk Prediction System",
    page_icon="🧠",
    layout="wide"
)

# ==========================================
# 2. 加载模型与数据 (Load Model & Data)
# ==========================================
@st.cache_resource
def load_model():
    # 将模型文件更换为 GBC.pkl
    model = joblib.load('GBC.pkl')
    return model

@st.cache_data
def load_data():
    data = pd.read_csv('X_test.csv')
    return data

try:
    model = load_model()
    X_test = load_data()
    feature_names = X_test.columns.tolist()
except Exception as e:
    st.error(f"Failed to load model or data. Error: {e}")
    st.stop()

# ==========================================
# 3. 设计用户输入界面 (Sidebar Input)
# ==========================================
st.sidebar.markdown("### 📋 Clinical Parameters")
st.sidebar.markdown("Please input patient details below:")
st.sidebar.markdown("---")

input_data = {}

for feature in feature_names:
    min_val = float(X_test[feature].min())
    max_val = float(X_test[feature].max())
    default_val = float(X_test[feature].mean())
    
    if feature == 'Sex':
        sex_mapping = {"Female": 0, "Male": 1}
        default_int = int(round(default_val))
        default_label = "Male" if default_int == 1 else "Female"
        default_index = list(sex_mapping.keys()).index(default_label)
        selected_label = st.sidebar.selectbox(f"{feature}", options=list(sex_mapping.keys()), index=default_index)
        input_data[feature] = sex_mapping[selected_label]
        
    elif feature == 'Education Level':
        edu_mapping = {"Illiterate": 1, "Primary school": 2, "Middle School": 3, "High School or above": 4}
        default_int = int(round(default_val))
        if default_int not in edu_mapping.values():
            default_int = 1 
        default_label = [k for k, v in edu_mapping.items() if v == default_int][0]
        default_index = list(edu_mapping.keys()).index(default_label)
        selected_label = st.sidebar.selectbox(f"{feature}", options=list(edu_mapping.keys()), index=default_index)
        input_data[feature] = edu_mapping[selected_label]
        
    elif feature in ['Baseline Cognitive', 'Age', 'IADL', 'IADL Impairment Count']:
        input_data[feature] = st.sidebar.number_input(
            f"{feature}",
            min_value=int(min_val),
            max_value=int(max_val),
            value=int(round(default_val)),
            step=1,
            format="%d"
        )
        
    else:
        if X_test[feature].nunique() <= 2:
            input_data[feature] = st.sidebar.selectbox(f"{feature}", options=[0, 1], index=int(round(default_val)))
        else:
            input_data[feature] = st.sidebar.number_input(f"{feature}", min_value=min_val, max_value=max_val, value=default_val, format="%.2f")

input_df = pd.DataFrame([input_data])

# ==========================================
# 4. 主视觉与预测逻辑 (Main UI & Prediction)
# ==========================================
# 顶部标题与描述
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 20px;">
    <span style="font-size: 40px; margin-right: 15px;">🧠</span>
    <h1 style="margin: 0; padding: 0;">6-Year MCI Risk Prediction System</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="border-left: 4px solid #3b82f6; background-color: #ffffff; padding: 15px 20px; margin-bottom: 40px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border-radius: 0 8px 8px 0;">
    <p style="margin: 0 0 5px 0; font-size: 15px; color: #333;"><b>Target Population:</b> Older adults with currently normal cognitive function.</p>
    <p style="margin: 0; font-size: 15px; color: #333;"><b>Objective:</b> Predict the probability of developing Mild Cognitive Impairment (MCI) within 6 years.</p>
</div>
""", unsafe_allow_html=True)

# 侧边栏按钮触发预测
run_prediction = st.sidebar.button("Run Prediction Analysis")

if run_prediction:
    try:
        prediction_proba = model.predict_proba(input_df)[0]
        risk_score = prediction_proba[1] 
        prob_percent = risk_score * 100
        is_high_risk = risk_score > 0.5
        
        # 动态设置卡片样式颜色
        if is_high_risk:
            outcome_text = "HIGH RISK"
            outcome_color = "#dc2626"
            card1_border = "#fca5a5"
            card1_bg = "#fef2f2"
            gauge_icon = "⚠️"
            gauge_text = "Elevated risk detected"
        else:
            outcome_text = "LOW RISK"
            outcome_color = "#16a34a" 
            card1_border = "#86efac"
            card1_bg = "#f0fdf4"
            gauge_icon = "✅"
            gauge_text = "Low risk detected"

        st.markdown("### Diagnostic Report")
        col1, col2, col3 = st.columns(3)
        
        # Card 1: Predicted Outcome
        with col1:
            st.markdown(f"""
            <div style="border: 1px solid {card1_border}; background-color: {card1_bg}; border-radius: 8px; padding: 25px 20px; text-align: center; height: 160px; display: flex; flex-direction: column; justify-content: center; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                <p style="font-size: 11px; color: #64748b; font-weight: 700; margin-bottom: 15px; letter-spacing: 1px; text-transform: uppercase;">Predicted Outcome</p>
                <h2 style="color: {outcome_color}; margin: 0; font-size: 34px; font-weight: 900;">{outcome_text}</h2>
                <p style="font-size: 11px; color: #94a3b8; margin-top: 15px; margin-bottom: 0;">@ 6 Years Horizon</p>
            </div>
            """, unsafe_allow_html=True)

        # Card 2: Probability
        with col2:
            st.markdown(f"""
            <div style="border: 1px solid #e2e8f0; background-color: #ffffff; border-radius: 8px; padding: 25px 20px; text-align: center; height: 160px; display: flex; flex-direction: column; justify-content: center; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                <p style="font-size: 11px; color: #64748b; font-weight: 700; margin-bottom: 15px; letter-spacing: 1px; text-transform: uppercase;">Probability</p>
                <h2 style="color: {outcome_color}; margin: 0; font-size: 34px; font-weight: 900;">{prob_percent:.2f}%</h2>
                <p style="font-size: 11px; color: #94a3b8; margin-top: 15px; margin-bottom: 0;">Confidence Score</p>
            </div>
            """, unsafe_allow_html=True)

        # Card 3: Gauge
        with col3:
            st.markdown(f"""
            <div style="border: 1px solid #e2e8f0; background-color: #ffffff; border-radius: 8px; padding: 25px 20px; height: 160px; display: flex; flex-direction: column; justify-content: center; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                <p style="font-size: 11px; color: #64748b; font-weight: 700; margin-bottom: 30px; letter-spacing: 1px; text-transform: uppercase;">Risk Assessment Gauge</p>
                <p style="font-size: 14px; font-weight: 700; color: #1e293b; margin: 0 0 10px 0;">{gauge_icon} {gauge_text}</p>
                <div style="width: 100%; background-color: #334155; border-radius: 3px; height: 8px; display: flex;">
                    <div style="width: {prob_percent}%; background-color: #3b82f6; height: 100%; border-radius: 3px 0 0 3px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

       # ==========================================
        # 5. 可视化解释 (SHAP Visualization)
        # ==========================================
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### Interpretability Analysis (SHAP)")
        
        with st.spinner('Generating SHAP explanation...'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            
            try:
                fig, ax = plt.subplots(figsize=(10, 3.5))
                
                # 1. 处理 SHAP Values 维度差异
                if isinstance(shap_values, list):
                    shap_val_to_plot = shap_values[1][0]  # 取类别 1 的第一个样本
                    raw_base_val = explainer.expected_value[1]
                else:
                    shap_val_to_plot = shap_values[0]     # GBC 通常走这个分支，取第一个样本
                    raw_base_val = explainer.expected_value
                
                # 2. 强制转换 Base Value 为 Python Scalar (标量)
                if isinstance(raw_base_val, (np.ndarray, list)):
                    base_val = raw_base_val[0]
                else:
                    base_val = raw_base_val
                
                base_val = float(base_val) # 彻底解决 "only 0-dimensional arrays" 报错
                
                # 3. 生成瀑布图
                shap.plots.waterfall(shap.Explanation(
                    values=shap_val_to_plot, 
                    base_values=base_val, 
                    data=input_df.iloc[0], 
                    feature_names=feature_names
                ), show=False)
                
                st.pyplot(plt.gcf())
                
            except Exception as e:
                st.warning(f"Error generating SHAP plot: {e}")
                # 如果依然报错，提供一个降级显示的条形图
                st.bar_chart(pd.Series(shap_val_to_plot, index=feature_names))

    except Exception as e:
        st.error(f"Prediction failed: {e}\nPlease check data formats.")
