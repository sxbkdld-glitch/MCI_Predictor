import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

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
    # 加载您的 XGBoost 模型
    model = joblib.load('XGBC.pkl')
    return model

@st.cache_data
def load_data():
    # 加载测试集数据，用于获取特征名称和范围
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

input_data = {}

for feature in feature_names:
    min_val = float(X_test[feature].min())
    max_val = float(X_test[feature].max())
    default_val = float(X_test[feature].mean())
    
    if X_test[feature].nunique() <= 2:
        input_data[feature] = st.sidebar.selectbox(
            f"{feature}",
            options=[0, 1],
            index=int(default_val)
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
with st.expander("查看当前输入数据"):
    st.dataframe(input_df)

# 注意：这里的缩进必须是顶格（即没有空格），因为它不是任何函数的一部分
if st.button("🚀 开始预测 (Run Prediction)", type="primary"):
    
    st.subheader("📊 预测结果")
    
    try:
        # 兼容性处理：尝试带列名预测，如果失败则尝试纯数值预测
        try:
            prediction_proba = model.predict_proba(input_df)[0]
            prediction_class = model.predict(input_df)[0]
        except Exception:
            # 如果报错 missing positional argument 'X'，说明版本不兼容，改用 values 传入
            prediction_proba = model.predict_proba(input_df.values)[0]
            prediction_class = model.predict(input_df.values)[0]
        
        # 假设 Class 1 是阳性/患病/高风险
        risk_score = prediction_proba[1] 
        
        # 展示结果卡片
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
        
        try:
            with st.spinner('正在生成 SHAP 解释图...'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)
                
                fig, ax = plt.subplots(figsize=(10, 3))
                
                # SHAP 返回值兼容性处理
                if isinstance(shap_values, list):
                    shap_vals_to_plot = shap_values[1][0]
                    base_val = explainer.expected_value[1]
                else:
                    shap_vals_to_plot = shap_values[0]
                    base_val = explainer.expected_value
                
                # 兼容不同版本的 expected_value 格式
                if isinstance(base_val, (list, np.ndarray)):
                    base_val = base_val[0]

                shap.plots.waterfall(shap.Explanation(values=shap_vals_to_plot, 
                                                     base_values=base_val, 
                                                     data=input_df.iloc[0], 
                                                     feature_names=feature_names),
                                     show=False)
                st.pyplot(plt.gcf())
        except Exception as e_shap:
            st.warning(f"SHAP 图无法生成 (版本兼容性问题)，但不影响预测结果。错误详情: {e_shap}")

    except Exception as e:
        st.error(f"预测过程中发生错误: {e}")
        st.info("排查建议：如果看到 'missing 1 required positional argument'，请检查 requirements.txt 中的 xgboost 版本是否与本地一致。")
