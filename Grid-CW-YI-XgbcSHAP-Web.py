import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

import importlib.metadata
import sys

def generate_requirements():
    # 1. 这里是 app.py 中用到的核心库列表
    target_packages = [
        "streamlit",
        "pandas",
        "numpy",
        "joblib",
        "xgboost",
        "shap",
        "matplotlib",
        "scikit-learn" 
    ]
    
    output_file = "requirements.txt"
    successful_packages = []
    
    print(f"正在检测 {len(target_packages)} 个核心包的版本...")
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for package in target_packages:
                try:
                    version = importlib.metadata.version(package)
                    line = f"{package}=={version}"
                    f.write(line + "\n")
                    successful_packages.append(line)
                    print(f"✅ 找到: {line}")
                except importlib.metadata.PackageNotFoundError:
                    print(f"⚠️ 警告: 未找到已安装的包 '{package}'，它不会被写入文件。")
        
        print("-" * 30)
        print(f"🎉 成功生成 '{output_file}'！内容如下：")
        print("\n".join(successful_packages))
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    generate_requirements()

# ==========================================
# 1. 页面配置 (Page Configuration)
# ==========================================
st.set_page_config(
    page_title="临床预测模型 Web 应用",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 机器学习临床预测模型演示")
st.markdown("该应用基于 XGBoost 构建，用于根据患者特征预测风险概率。")

# ==========================================
# 2. 加载模型与数据 (Load Model & Data)
# ==========================================
@st.cache_resource
def load_model():
    model = joblib.load('XGBC.pkl')
    return model

@st.cache_data
def load_data():
    data = pd.read_csv('X_test.csv')
    return data

try:
    model = load_model()
    X_test = load_data()
    feature_names = X_test.columns.tolist()
    st.success("✅ 模型与数据加载成功！")
except Exception as e:
    st.error(f"加载文件失败，请检查目录下是否存在 'XGBC.pkl' 和 'X_test.csv'。错误信息: {e}")
    st.stop()

# ==========================================
# 3. 设计用户输入界面 (User Input Interface)
# ==========================================
st.sidebar.header("📋 请输入特征参数")
st.sidebar.markdown("请在下方调整各个特征的数值：")

# 自动创建输入字典
input_data = {}

# 遍历测试集中的每一列，自动生成对应的输入框
for feature in feature_names:
    min_val = float(X_test[feature].min())
    max_val = float(X_test[feature].max())
    default_val = float(X_test[feature].mean())
    
    # --- 新增：针对特定变量的自定义下拉框处理 ---
    if feature == 'Sex':
        # 定义映射字典
        sex_mapping = {"Female": 0, "Male": 1}
        # 确定默认选项的索引
        default_int = int(round(default_val))
        default_label = "Male" if default_int == 1 else "Female"
        default_index = list(sex_mapping.keys()).index(default_label)
        
        # 显示中文下拉框
        selected_label = st.sidebar.selectbox(
            f"{feature} (Sex)",
            options=list(sex_mapping.keys()),
            index=default_index
        )
        # 存入模型的数据转换为数字
        input_data[feature] = sex_mapping[selected_label]
        
    elif feature == 'Education Level':
        # 定义映射字典
        edu_mapping = {"Illiterate": 1, "Primary school": 2, "Middle School": 3, "High School or above": 4}
        default_int = int(round(default_val))
        # 防止测试集均值四舍五入后不在 1-4 范围内，做个兜底
        if default_int not in edu_mapping.values():
            default_int = 1 
        
        default_label = [k for k, v in edu_mapping.items() if v == default_int][0]
        default_index = list(edu_mapping.keys()).index(default_label)
        
        # 显示中文下拉框
        selected_label = st.sidebar.selectbox(
            f"{feature} (Education Level)",
            options=list(edu_mapping.keys()),
            index=default_index
        )
        # 存入模型的数据转换为数字
        input_data[feature] = edu_mapping[selected_label]
        
    # --- 原有逻辑：处理其他未特别指定的变量 ---
    else:
        if X_test[feature].nunique() <= 2:
            input_data[feature] = st.sidebar.selectbox(
                f"{feature}",
                options=[0, 1],
                index=int(round(default_val)) # 加了 round 防止均值偏离导致越界
            )
        else:
            input_data[feature] = st.sidebar.number_input(
                f"{feature}",
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                format="%.2f"
            )

# 将用户输入转换为 DataFrame
input_df = pd.DataFrame([input_data])

# ==========================================
# 4. 处理输入并调用模型预测 (Prediction)
# ==========================================
with st.expander("查看当前输入数据（传给模型的值）"):
    st.dataframe(input_df)

if st.button("🚀 开始预测 (Run Prediction)", type="primary"):
    
    st.subheader("📊 预测结果")
    
    try:
        prediction_proba = model.predict_proba(input_df)[0]
        prediction_class = model.predict(input_df)[0]
        
        risk_score = prediction_proba[1] 
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="预测类别 (Class)", value=int(prediction_class))
        with col2:
            st.metric(label="风险概率 (Risk Probability)", value=f"{risk_score:.2%}")
            
        if risk_score > 0.5:
            st.error("⚠️ 警告：模型预测为高风险！")
        else:
            st.success("✅ 提示：模型预测为低风险。")

        # ==========================================
        # 5. 展示可视化解释 (Visualization)
        # ==========================================
        st.markdown("---")
        st.subheader("🔍 模型解释 (SHAP Visualization)")
        st.markdown("下图展示了各特征对本次预测结果的具体贡献方向和大小：")
        
        with st.spinner('正在生成 SHAP 解释图...'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            
            try:
                fig, ax = plt.subplots(figsize=(10, 3))
                if isinstance(shap_values, list):
                    shap_val_to_plot = shap_values[1]
                    base_val = explainer.expected_value[1]
                else:
                    shap_val_to_plot = shap_values
                    base_val = explainer.expected_value
                
                shap.plots.waterfall(shap.Explanation(values=shap_val_to_plot[0], 
                                                     base_values=base_val, 
                                                     data=input_df.iloc[0], 
                                                     feature_names=feature_names),
                                     show=False)
                st.pyplot(plt.gcf())
                
            except Exception as e:
                st.warning(f"SHAP 图生成遇到小问题，尝试备用绘图方式... {e}")
                st.bar_chart(pd.Series(shap_val_to_plot[0], index=feature_names))

    except Exception as e:
        st.error(f"预测过程中发生错误: {e}\n可能是输入数据格式与模型不匹配。")

st.markdown("---")
st.caption("Developed for Clinical Prediction Model Research")
