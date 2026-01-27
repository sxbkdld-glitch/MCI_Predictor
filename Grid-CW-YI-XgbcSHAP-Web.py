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
# 使用缓存装饰器，避免每次操作都重新加载，提高运行速度
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
    # 获取特征名称列表
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
# 这样无论您的模型有多少个特征，代码都能自动适配
for feature in feature_names:
    # 获取该特征在测试集中的最小值、最大值和均值
    min_val = float(X_test[feature].min())
    max_val = float(X_test[feature].max())
    default_val = float(X_test[feature].mean())
    
    # 针对二分类特征（0/1）使用选择框，连续变量使用滑动条或数字输入
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
# 在主界面展示用户当前的输入数据
with st.expander("查看当前输入数据"):
    st.dataframe(input_df)

if st.button("🚀 开始预测 (Run Prediction)", type="primary"):
    
    st.subheader("📊 预测结果")
    
    # 进行预测
    # predict_proba 返回概率 [类0概率, 类1概率]
    try:
        prediction_proba = model.predict_proba(input_df)[0]
        prediction_class = model.predict(input_df)[0]
        
        # 假设 Class 1 是阳性/患病/高风险
        risk_score = prediction_proba[1] 
        
        # 展示结果卡片
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="预测类别 (Class)", value=int(prediction_class))
        with col2:
            st.metric(label="风险概率 (Risk Probability)", value=f"{risk_score:.2%}")
            
        # 根据概率显示不同提示
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
            # 创建 SHAP 解释器
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            
            # 由于 Streamlit 不直接支持 JS 交互图，我们使用 Matplotlib 静态图
            # Force Plot (展示单个样本的特征贡献)
            try:
                # 注意：SHAP 版本不同，绘图函数调用方式可能略有差异
                fig, ax = plt.subplots(figsize=(10, 3))
                # 如果是二分类，shap_values 可能是 list，取 [1] 对应正类
                if isinstance(shap_values, list):
                    shap_val_to_plot = shap_values[1]
                    base_val = explainer.expected_value[1]
                else:
                    shap_val_to_plot = shap_values
                    base_val = explainer.expected_value
                
                # 绘制瀑布图 (Waterfall plot) 是最清晰的单样本解释
                # 需要 shap >= 0.39.0
                shap.plots.waterfall(shap.Explanation(values=shap_val_to_plot[0], 
                                                     base_values=base_val, 
                                                     data=input_df.iloc[0], 
                                                     feature_names=feature_names),
                                     show=False)
                st.pyplot(plt.gcf())
                
            except Exception as e:
                st.warning(f"SHAP 图生成遇到小问题，尝试备用绘图方式... {e}")
                # 备用方案：简单的条形图
                st.bar_chart(pd.Series(shap_val_to_plot[0], index=feature_names))

    except Exception as e:
        st.error(f"预测过程中发生错误: {e}\n可能是输入数据格式与模型不匹配。")

# 页脚
st.markdown("---")
st.caption("Developed for Clinical Prediction Model Research")