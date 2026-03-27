import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Advanced Data Analytics Platform", page_icon="🚀", layout="wide")

# Custom CSS for premium UI
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px; font-size: 16px; font-weight: 600; }
    [data-testid="stFileUploaderDropzone"] { min-height: 250px; border: 3px dashed #7a5195; border-radius: 12px; background-color: rgba(122, 81, 149, 0.05); }
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; color: #7a5195 !important; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Advanced Data Analytics & AutoML Platform")
st.markdown("Automated EDA, Business Intelligence, and Machine Learning. **Analyze, Predict, and Simulate.**")
st.divider()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2041/2041643.png", width=60)
    st.title("Configuration")
    
    st.header("1. Data Upload")
    uploaded_file_side = st.file_uploader("Upload your data", type=["csv", "xlsx"], help="Limit 200MB per file", key="side")
    
    df_clean = None
    st.header("2. Data Preprocessing")
    drop_na = st.checkbox("Drop rows with missing values", value=True)

# --- MAIN UPLOAD LOGIC ---
uploaded_file_main = None
if uploaded_file_side is None:
    st.info("👋 **Welcome!** Drag & Drop your CSV or Excel (*.xlsx) file directly into the massive area below to initialize the AI Engine.")
    st.markdown("<br>", unsafe_allow_html=True)
    uploaded_file_main = st.file_uploader("⬇️ Drag and Drop your file here to see the Magic! ⬇️", type=["csv", "xlsx"], key="main")

uploaded_file = uploaded_file_side or uploaded_file_main

# --- APP EXECUTION ---
if uploaded_file is not None:
    # Read data safely
    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    except Exception as e:
        st.error(f"Error reading file {uploaded_file.name}: {e}")
        st.stop()
        
    original_len = len(df)
    if drop_na:
        df = df.dropna()
        dropped_count = original_len - len(df)
        if dropped_count > 0:
            st.sidebar.success(f"Cleaned! Removed {dropped_count} rows containing missing values.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("⚠️ The dataset must have at least two numerical columns to perform advanced analytics.")
        st.stop()

    # Create Tabs
    tab1, tab2, tab3 = st.tabs(["🗂️ Smart EDA", "💼 Business Intelligence", "🤖 AutoML & Prediction Engine"])
    
    # --- TAB 1: SMART EDA ---
    with tab1:
        st.subheader("Automated Exploratory Data Analysis (EDA)")
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Rows", f"{len(df):,}")
        kpi2.metric("Total Columns", f"{df.shape[1]:,}")
        kpi3.metric("Numeric Features", len(numeric_cols))
        kpi4.metric("Categorical Features", len(cat_cols))
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_ed1, col_ed2 = st.columns([1, 1], gap="large")
        with col_ed1:
            st.markdown("#### 📄 Dataset Preview")
            st.dataframe(df.head(15), use_container_width=True)
            st.download_button("💾 Download Cleaned Dataset (CSV)", data=df.to_csv(index=False).encode('utf-8'), file_name="cleaned_data.csv", mime="text/csv")
            
        with col_ed2:
            st.markdown("#### 🔥 Feature Correlation Heatmap")
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                fig_corr.update_layout(margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_corr, use_container_width=True)
                
                st.download_button("🖼️ Download Heatmap (PNG)", data=fig_corr.to_image(format="png", engine="kaleido", scale=2), file_name="heatmap.png", mime="image/png", use_container_width=True)
            else:
                st.info("Not enough numeric columns for correlation.")

    # --- TAB 2: BUSINESS INTELLIGENCE ---
    with tab2:
        st.subheader("Business Intelligence & Aggregations")
        
        if len(cat_cols) > 0 and len(numeric_cols) > 0:
            with st.container(border=True):
                biz_col1, biz_col2, biz_col3 = st.columns(3)
                with biz_col1:
                    group_col = st.selectbox("Group By (Category)", cat_cols, help="e.g. Region, Product, Status")
                with biz_col2:
                    metric_col = st.selectbox("Metric to Calculate", numeric_cols, help="e.g. Sales, Profit, Quantity")
                with biz_col3:
                    agg_func = st.selectbox("Aggregation", ["Sum", "Average", "Count", "Max", "Min"])
                    
                agg_dict = {"Sum": "sum", "Average": "mean", "Count": "count", "Max": "max", "Min": "min"}
                
                if group_col and metric_col:
                    grouped_df = df.groupby(group_col)[metric_col].agg(agg_dict[agg_func]).reset_index()
                    grouped_df = grouped_df.sort_values(by=metric_col, ascending=False).head(15)
                    
                    chart_col1, chart_col2 = st.columns([1, 1], gap="large")
                    
                    with chart_col1:
                        fig_bar = px.bar(grouped_df, x=group_col, y=metric_col, text_auto='.2s', color=metric_col, color_continuous_scale="Teal", title=f"📊 Bar Chart: Top 15 {group_col}")
                        fig_bar.update_layout(xaxis_title=group_col, yaxis_title=f"{agg_func} of {metric_col}", showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=0, l=0, r=0))
                        st.plotly_chart(fig_bar, use_container_width=True)
                        st.download_button("🖼️ Download Bar Chart", data=fig_bar.to_image(format="png", engine="kaleido", scale=2), file_name=f"bar_{group_col}.png", mime="image/png", use_container_width=True)
                        
                    with chart_col2:
                        fig_pie = px.pie(grouped_df, names=group_col, values=metric_col, hole=0.35, title=f"🍩 Donut Chart: Share by {group_col}", color_discrete_sequence=px.colors.sequential.Teal)
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        fig_pie.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=0, l=0, r=0))
                        st.plotly_chart(fig_pie, use_container_width=True)
                        st.download_button("🖼️ Download Donut Chart", data=fig_pie.to_image(format="png", engine="kaleido", scale=2), file_name=f"pie_{group_col}.png", mime="image/png", use_container_width=True)
                    
                    st.divider()
                    st.download_button("📄 Download Full Report (CSV)", data=grouped_df.to_csv(index=False).encode('utf-8'), file_name=f"report_{group_col}.csv", mime="text/csv", use_container_width=True)
        else:
            st.info("No categorical columns found for grouping.")

    # --- TAB 3: AUTOML & PREDICTION ENGINE ---
    with tab3:
        st.subheader("🤖 AutoML & Predictive Modeling Engine")
        st.markdown("Train multiple machine learning algorithms simultaneously, evaluate their metrics, and run **Live 'What-If' simulations**.")
        
        with st.container(border=True):
            am_c1, am_c2 = st.columns([1, 2])
            with am_c1:
                target_col = st.selectbox("🎯 Target Variable (Y)", numeric_cols, index=len(numeric_cols)-1)
            with am_c2:
                # Default select first two features to avoid errors
                default_features = [c for c in numeric_cols if c != target_col][:2]
                feature_cols = st.multiselect("🧠 Select Multiple Features (X)", [c for c in numeric_cols if c != target_col], default=default_features)
        
        if len(feature_cols) > 0:
            # Data prep
            ml_df = df[feature_cols + [target_col]].dropna()
            X = ml_df[feature_cols].values
            y = ml_df[target_col].values
            
            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Initialize Models
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42)
            }
            
            # Train and Evaluate
            results = []
            trained_models = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                r2 = r2_score(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                mae = mean_absolute_error(y_test, preds)
                results.append({"Model": name, "R² Score": r2, "RMSE": rmse, "MAE": mae})
                trained_models[name] = model
                
            res_df = pd.DataFrame(results).sort_values(by="R² Score", ascending=False)
            best_model_name = res_df.iloc[0]["Model"]
            best_model = trained_models[best_model_name]
            
            st.markdown("### 🏆 Algorithm Leaderboard")
            st.info(f"The AI evaluated multiple algorithms. **Best Performing Model:** {best_model_name}")
            
            ld1, ld2 = st.columns([3, 2])
            with ld1:
                # Format dataframe nicely
                styled_df = res_df.style.highlight_max(subset=['R² Score'], color='#c3e6cb').highlight_min(subset=['RMSE', 'MAE'], color='#c3e6cb')
                st.dataframe(styled_df, use_container_width=True)
                
            with ld2:
                # Plot Actual vs Predicted for Best Model
                best_preds = best_model.predict(X_test)
                fig_scatter = px.scatter(x=y_test, y=best_preds, labels={'x': 'Actual Values', 'y': 'AI Predictions'}, title=f"Accuracy ({best_model_name})", opacity=0.7)
                fig_scatter.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color="red", dash="dash"))
                fig_scatter.update_layout(margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_scatter, use_container_width=True)
                
            st.divider()
            
            f_col1, f_col2 = st.columns([1, 1], gap="large")
            
            with f_col1:
                # Feature Importance
                st.markdown("### 🔑 Interpretable AI (Feature Importance)")
                st.markdown("Which variables drive the predictions the most?")
                if hasattr(best_model, "feature_importances_"):
                    importances = best_model.feature_importances_
                elif hasattr(best_model, "coef_"):
                    importances = np.abs(best_model.coef_)
                else:
                    importances = np.zeros(len(feature_cols))
                    
                imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values(by="Importance", ascending=True)
                fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation='h', color="Importance", color_continuous_scale="Purp")
                fig_imp.update_layout(margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_imp, use_container_width=True)
                
            with f_col2:
                # What-If Simulator
                st.markdown("### 🎛️ Live 'What-If' Simulation")
                st.markdown(f"Adjust the sliders below. The **{best_model_name}** will predict `{target_col}` in real-time.")
                
                with st.container(border=True):
                    input_data = []
                    for feature in feature_cols:
                        min_val = float(ml_df[feature].min())
                        max_val = float(ml_df[feature].max())
                        mean_val = float(ml_df[feature].mean())
                        step_val = (max_val - min_val) / 100 if max_val > min_val else 1.0
                        val = st.slider(feature, min_value=min_val, max_value=max_val, value=mean_val, step=step_val)
                        input_data.append(val)
                        
                    prediction = best_model.predict([input_data])[0]
                    st.success(f"### 🎯 Predicted {target_col}:\n# **{prediction:,.2f}**")
                
        else:
            st.warning("Please select at least one feature (X) to train the models.")
