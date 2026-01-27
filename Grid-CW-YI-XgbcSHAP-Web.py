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
# ... (前面的代码保持不变)

if st.button("🚀 开始预测 (Run Prediction)", type="primary"):
    
    st.subheader("📊 预测结果")
    
    try:
        # =======================================================
        # 核心修复：通用预测逻辑 (兼容所有 XGBoost 版本)
        # =======================================================
        # 1. 提取底层 Booster (绕过 sklearn 接口的参数检查 bug)
        booster = model.get_booster()
        
        # 2. 将数据转换为 XGBoost 原生 DMatrix 格式
        # 注意：DMatrix 不会报错 'missing argument X'
        dtest = xgb.DMatrix(input_df)
        
        # 3. 进行预测
        # 原生 booster.predict 直接返回正类概率 (例如 0.85)，而不是 [[0.15, 0.85]]
        risk_score = booster.predict(dtest)
        
        # 处理结果格式（如果输入是多行，risk_score 是数组；如果是单行，可能是浮点数）
        if isinstance(risk_score, np.ndarray):
            risk_score = float(risk_score[0])
        else:
            risk_score = float(risk_score)
            
        # 手动判断类别 (默认阈值 0.5)
        prediction_class = 1 if risk_score > 0.5 else 0
        
        # =======================================================
        
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
                # 对 Booster 进行解释，比对 Model 解释更稳定
                explainer = shap.TreeExplainer(booster)
                shap_values = explainer.shap_values(input_df)
                
                fig, ax = plt.subplots(figsize=(10, 3))
                
                # 针对 Booster 的 shap_values 通常直接就是数值，不需要像 classifier 那样取 [1]
                # 这里做一个兼容性判断
                if isinstance(shap_values, list):
                    shap_vals_to_plot = shap_values[1] # 如果意外返回了 list
                else:
                    shap_vals_to_plot = shap_values # 通常是这个
                
                # 处理单样本维度
                if shap_vals_to_plot.ndim > 1:
                    shap_vals_to_plot = shap_vals_to_plot[0]
                
                # 获取 base_value
                base_val = explainer.expected_value
                if isinstance(base_val, (list, np.ndarray)) and len(base_val) > 1:
                     # 某些版本可能返回 list
                     pass 

                shap.plots.waterfall(shap.Explanation(values=shap_vals_to_plot, 
                                                     base_values=base_val, 
                                                     data=input_df.iloc[0], 
                                                     feature_names=feature_names),
                                     show=False)
                st.pyplot(plt.gcf())
        except Exception as e_shap:
            st.warning(f"SHAP 图生成受阻: {e_shap}")

    except Exception as e:
        st.error(f"预测错误: {e}")
        st.write("调试信息：", type(model))
