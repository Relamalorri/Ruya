import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import forecasting libraries
try:
    from pmdarima import auto_arima
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Ruya - Coffee Sales Dashboard", 
    page_icon="coffee", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading Function ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # Date processing
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Ensure numeric columns
        if 'money' in df.columns:
            df['money'] = pd.to_numeric(df['money'], errors='coerce')
        
        # Add season column based on month
        if 'Date' in df.columns:
            df['Month_num'] = df['Date'].dt.month
            df['Year'] = df['Date'].dt.year
            def assign_season(month):
                if month in [12, 1, 2]:
                    return "Winter"
                elif month in [3, 4, 5]:
                    return "Spring"
                elif month in [6, 7, 8]:
                    return "Summer"
                elif month in [9, 10, 11]:
                    return "Autumn"
            df['Season'] = df['Month_num'].apply(assign_season)
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- Load Data ---
file_path = "Coffe_sales.csv"
if os.path.exists(file_path):
    df = load_data(file_path)
else:
    st.error("Data file 'Coffe_sales.csv' not found.")
    st.stop()

if df is None:
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("Ruya Dashboard")
st.sidebar.markdown("---")

# Page Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Sales Analysis", "Product Insights", "Time Patterns", "Forecasting", "Data Explorer"]
)

st.sidebar.markdown("---")
st.sidebar.header("Filters")

# Date Range Filter
min_date = df['Date'].min()
max_date = df['Date'].max()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Coffee Type Filter
if 'coffee_name' in df.columns:
    coffee_types = sorted(df['coffee_name'].unique().tolist())
    selected_coffee_types = st.sidebar.multiselect(
        "Coffee Types",
        options=coffee_types,
        default=coffee_types
    )
else:
    selected_coffee_types = []

# Time of Day Filter
if 'Time_of_Day' in df.columns:
    time_of_day_options = df['Time_of_Day'].unique().tolist()
    selected_time_of_day = st.sidebar.multiselect(
        "Time of Day",
        options=time_of_day_options,
        default=time_of_day_options
    )
else:
    selected_time_of_day = []

# Season Filter
if 'Season' in df.columns:
    season_options = ['Spring', 'Summer', 'Autumn', 'Winter']
    available_seasons = [s for s in season_options if s in df['Season'].unique()]
    selected_seasons = st.sidebar.multiselect(
        "Seasons",
        options=available_seasons,
        default=available_seasons
    )
else:
    selected_seasons = []

# Apply Filters
mask = (df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])
if selected_coffee_types:
    mask = mask & (df['coffee_name'].isin(selected_coffee_types))
if selected_time_of_day and 'Time_of_Day' in df.columns:
    mask = mask & (df['Time_of_Day'].isin(selected_time_of_day))
if selected_seasons and 'Season' in df.columns:
    mask = mask & (df['Season'].isin(selected_seasons))

filtered_df = df[mask].copy()

# =====================================================
# PAGE: OVERVIEW
# =====================================================
if page == "Overview":
    st.title("Ruya - Coffee Sales Dashboard")
    st.caption("A Smart Sales Analysis and Forecasting System for Coffee Shops")
    
    st.divider()
    
    # KPI Cards Row 1
    col1, col2, col3, col4 = st.columns(4)
    
    total_revenue = filtered_df['money'].sum()
    total_transactions = len(filtered_df)
    avg_ticket = filtered_df['money'].mean() if len(filtered_df) > 0 else 0
    unique_days = filtered_df['Date'].nunique()
    
    with col1:
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    with col2:
        st.metric("Transactions", f"{total_transactions:,}")
    with col3:
        st.metric("Avg Ticket", f"${avg_ticket:.2f}")
    with col4:
        st.metric("Days Active", f"{unique_days}")
    
    # KPI Cards Row 2
    col5, col6, col7, col8 = st.columns(4)
    
    daily_avg_revenue = total_revenue / unique_days if unique_days > 0 else 0
    daily_avg_transactions = total_transactions / unique_days if unique_days > 0 else 0
    top_product = filtered_df['coffee_name'].value_counts().idxmax() if len(filtered_df) > 0 else "N/A"
    peak_hour = filtered_df['hour_of_day'].value_counts().idxmax() if 'hour_of_day' in filtered_df.columns and len(filtered_df) > 0 else "N/A"
    
    with col5:
        st.metric("Daily Avg Revenue", f"${daily_avg_revenue:.2f}")
    with col6:
        st.metric("Daily Avg Orders", f"{daily_avg_transactions:.1f}")
    with col7:
        st.metric("Top Product", top_product)
    with col8:
        st.metric("Peak Hour", f"{peak_hour}:00" if peak_hour != "N/A" else "N/A")
    
    st.divider()
    
    # Quick Charts Row
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Revenue Trend")
        if 'Date' in filtered_df.columns and len(filtered_df) > 0:
            daily_sales = filtered_df.groupby('Date')['money'].sum().reset_index()
            fig = px.area(daily_sales, x='Date', y='money', 
                         title='Daily Revenue Over Time',
                         labels={'money': 'Revenue ($)', 'Date': 'Date'})
            fig.update_layout(hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
    
    with col_chart2:
        st.subheader("Top Products")
        if 'coffee_name' in filtered_df.columns and len(filtered_df) > 0:
            prod_revenue = filtered_df.groupby('coffee_name')['money'].sum().reset_index()
            prod_revenue = prod_revenue.sort_values('money', ascending=True).tail(5)
            fig = px.bar(prod_revenue, x='money', y='coffee_name', orientation='h',
                        title='Top 5 Products by Revenue',
                        labels={'money': 'Revenue ($)', 'coffee_name': 'Product'},
                        color='coffee_name')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Second Row of Charts
    col_chart3, col_chart4 = st.columns(2)
    
    with col_chart3:
        st.subheader("Sales by Season")
        if 'Season' in filtered_df.columns and len(filtered_df) > 0:
            season_sales = filtered_df.groupby('Season')['money'].sum().reset_index()
            fig = px.pie(season_sales, values='money', names='Season',
                        title='Revenue Distribution by Season')
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    with col_chart4:
        st.subheader("Sales by Time of Day")
        if 'Time_of_Day' in filtered_df.columns and len(filtered_df) > 0:
            tod_sales = filtered_df.groupby('Time_of_Day')['money'].sum().reset_index()
            fig = px.pie(tod_sales, values='money', names='Time_of_Day',
                        title='Revenue Distribution by Time of Day')
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PAGE: SALES ANALYSIS
# =====================================================
elif page == "Sales Analysis":
    st.title("Sales Analysis")
    st.caption("Deep dive into your sales performance metrics and trends.")
    
    st.divider()
    
    # Daily Revenue Trend
    st.subheader("Daily Revenue Trend")
    if 'Date' in filtered_df.columns and len(filtered_df) > 0:
        daily_sales = filtered_df.groupby('Date')['money'].sum().reset_index()
        
        # Add moving average
        daily_sales['MA_7'] = daily_sales['money'].rolling(window=7, min_periods=1).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_sales['Date'], y=daily_sales['money'],
                                 mode='lines', name='Daily Revenue',
                                 line=dict(width=1)))
        fig.add_trace(go.Scatter(x=daily_sales['Date'], y=daily_sales['MA_7'],
                                 mode='lines', name='7-Day Moving Avg',
                                 line=dict(width=2)))
        fig.update_layout(title='Daily Revenue with 7-Day Moving Average',
                         xaxis_title='Date', yaxis_title='Revenue ($)',
                         hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    # Transaction Count Trend
    st.subheader("Daily Transactions")
    if 'Date' in filtered_df.columns and len(filtered_df) > 0:
        daily_count = filtered_df.groupby('Date').size().reset_index(name='transactions')
        fig = px.bar(daily_count, x='Date', y='transactions',
                    title='Number of Transactions Per Day',
                    labels={'transactions': 'Transactions', 'Date': 'Date'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Monthly Comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Monthly Revenue")
        if 'Month_name' in filtered_df.columns and len(filtered_df) > 0:
            monthly = filtered_df.groupby('Month_name')['money'].sum().reset_index()
            fig = px.bar(monthly, x='Month_name', y='money',
                        title='Total Revenue by Month',
                        labels={'money': 'Revenue ($)', 'Month_name': 'Month'},
                        color='Month_name')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Seasonal Revenue")
        if 'Season' in filtered_df.columns and len(filtered_df) > 0:
            seasonal = filtered_df.groupby('Season')['money'].sum().reset_index()
            fig = px.bar(seasonal, x='Season', y='money',
                        title='Total Revenue by Season',
                        labels={'money': 'Revenue ($)', 'Season': 'Season'},
                        color='Season')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Weekend vs Weekday Analysis
    st.subheader("Weekend vs Weekday Performance")
    if 'Weekday' in filtered_df.columns and len(filtered_df) > 0:
        weekend_days = ['Fri', 'Sat', 'Sun']
        df_analysis = filtered_df.copy()
        df_analysis['is_weekend'] = df_analysis['Weekday'].isin(weekend_days)
        df_analysis['Day_Type'] = df_analysis['is_weekend'].map({True: 'Weekend', False: 'Weekday'})
        
        summary = df_analysis.groupby('Day_Type').agg({
            'money': ['sum', 'mean', 'count']
        }).reset_index()
        summary.columns = ['Day_Type', 'Total Revenue', 'Avg Ticket', 'Transactions']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.pie(summary, values='Total Revenue', names='Day_Type',
                        title='Revenue Split')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(summary, x='Day_Type', y='Transactions',
                        title='Transaction Count',
                        color='Day_Type')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = px.bar(summary, x='Day_Type', y='Avg Ticket',
                        title='Average Ticket Size',
                        color='Day_Type')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PAGE: PRODUCT INSIGHTS
# =====================================================
elif page == "Product Insights":
    st.title("Product Insights")
    st.caption("Understand your product performance and customer preferences.")
    
    st.divider()
    
    if 'coffee_name' in filtered_df.columns and len(filtered_df) > 0:
        # Product Performance Table
        st.subheader("Product Performance Summary")
        product_stats = filtered_df.groupby('coffee_name').agg({
            'money': ['count', 'sum', 'mean']
        }).reset_index()
        product_stats.columns = ['Product', 'Transactions', 'Total Revenue', 'Avg Price']
        product_stats = product_stats.sort_values('Total Revenue', ascending=False)
        product_stats['Revenue %'] = (product_stats['Total Revenue'] / product_stats['Total Revenue'].sum() * 100).round(1)
        
        st.dataframe(product_stats.style.format({
            'Total Revenue': '${:,.2f}',
            'Avg Price': '${:.2f}',
            'Revenue %': '{:.1f}%'
        }), use_container_width=True)
        
        # Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Revenue by Product")
            fig = px.bar(product_stats, x='Product', y='Total Revenue',
                        title='Total Revenue by Coffee Type',
                        color='Product')
            fig.update_layout(xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Transactions by Product")
            fig = px.bar(product_stats, x='Product', y='Transactions',
                        title='Transaction Count by Coffee Type',
                        color='Product')
            fig.update_layout(xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Product Mix Pie Chart
        st.subheader("Product Mix")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(product_stats, values='Total Revenue', names='Product',
                        title='Revenue Distribution by Product',
                        hole=0.4)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(product_stats, values='Transactions', names='Product',
                        title='Transaction Distribution by Product',
                        hole=0.4)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        # Product by Time of Day
        if 'Time_of_Day' in filtered_df.columns:
            st.subheader("Product Performance by Time of Day")
            product_tod = filtered_df.groupby(['coffee_name', 'Time_of_Day'])['money'].sum().reset_index()
            fig = px.bar(product_tod, x='coffee_name', y='money', color='Time_of_Day',
                        title='Revenue by Product and Time of Day',
                        barmode='group',
                        labels={'money': 'Revenue ($)', 'coffee_name': 'Product'})
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PAGE: TIME PATTERNS
# =====================================================
elif page == "Time Patterns":
    st.title("Time Patterns")
    st.caption("Discover when your customers shop and optimize your operations.")
    
    st.divider()
    
    # Hourly Analysis
    st.subheader("Hourly Performance")
    if 'hour_of_day' in filtered_df.columns and len(filtered_df) > 0:
        hourly = filtered_df.groupby('hour_of_day').agg({
            'money': ['sum', 'count', 'mean']
        }).reset_index()
        hourly.columns = ['Hour', 'Revenue', 'Transactions', 'Avg Ticket']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(hourly, x='Hour', y='Revenue',
                        title='Revenue by Hour of Day',
                        labels={'Revenue': 'Revenue ($)', 'Hour': 'Hour'},
                        color='Revenue',
                        color_continuous_scale='Blues')
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(hourly, x='Hour', y='Transactions',
                        title='Transactions by Hour of Day',
                        labels={'Transactions': 'Count', 'Hour': 'Hour'},
                        color='Transactions',
                        color_continuous_scale='Greens')
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Weekday Analysis
    st.subheader("Weekday Performance")
    if 'Weekday' in filtered_df.columns and len(filtered_df) > 0:
        weekday_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekday_data = filtered_df.groupby('Weekday').agg({
            'money': ['sum', 'count', 'mean']
        }).reset_index()
        weekday_data.columns = ['Weekday', 'Revenue', 'Transactions', 'Avg Ticket']
        weekday_data['Weekday'] = pd.Categorical(weekday_data['Weekday'], categories=weekday_order, ordered=True)
        weekday_data = weekday_data.sort_values('Weekday')
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(weekday_data, x='Weekday', y='Revenue',
                        title='Revenue by Day of Week',
                        color='Weekday')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(weekday_data, x='Weekday', y='Transactions',
                        title='Transactions by Day of Week',
                        color='Weekday')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap: Day vs Hour
    st.subheader("Peak Times Heatmap")
    if 'Weekday' in filtered_df.columns and 'hour_of_day' in filtered_df.columns and len(filtered_df) > 0:
        heatmap_data = filtered_df.pivot_table(
            values='money', 
            index='Weekday', 
            columns='hour_of_day', 
            aggfunc='sum'
        ).fillna(0)
        
        # Reorder weekdays
        weekday_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        heatmap_data = heatmap_data.reindex([w for w in weekday_order if w in heatmap_data.index])
        
        fig = px.imshow(heatmap_data,
                       labels=dict(x="Hour of Day", y="Weekday", color="Revenue ($)"),
                       title='Revenue Heatmap: Weekday vs Hour',
                       aspect='auto')
        st.plotly_chart(fig, use_container_width=True)
    
    # Time of Day Analysis
    st.subheader("Time of Day Analysis")
    if 'Time_of_Day' in filtered_df.columns and len(filtered_df) > 0:
        tod = filtered_df.groupby('Time_of_Day').agg({
            'money': ['sum', 'count', 'mean']
        }).reset_index()
        tod.columns = ['Time of Day', 'Revenue', 'Transactions', 'Avg Ticket']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.bar(tod, x='Time of Day', y='Revenue',
                        title='Revenue by Time of Day',
                        color='Time of Day')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(tod, x='Time of Day', y='Transactions',
                        title='Transactions by Time of Day',
                        color='Time of Day')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = px.bar(tod, x='Time of Day', y='Avg Ticket',
                        title='Avg Ticket by Time of Day',
                        color='Time of Day')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PAGE: FORECASTING
# =====================================================
elif page == "Forecasting":
    st.title("Sales Forecasting")
    st.caption("Predict future sales using pre-trained machine learning models.")
    
    st.divider()
    
    # Model paths
    MODELS_DIR = "models"
    ARIMA_PATH = os.path.join(MODELS_DIR, "arima_model.pkl")
    PROPHET_PATH = os.path.join(MODELS_DIR, "prophet_model.pkl")
    LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model.keras")
    LSTM_SCALER_PATH = os.path.join(MODELS_DIR, "lstm_scaler.pkl")
    DAILY_DF_PATH = os.path.join(MODELS_DIR, "daily_df.pkl")
    
    # Check which models are available (saved)
    available_models = []
    if os.path.exists(ARIMA_PATH) and ARIMA_AVAILABLE:
        available_models.append("ARIMA")
    if os.path.exists(PROPHET_PATH) and PROPHET_AVAILABLE:
        available_models.append("Prophet")
    if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_SCALER_PATH) and LSTM_AVAILABLE:
        available_models.append("LSTM")
    
    if not available_models:
        st.warning("No pre-trained models found.")
        st.info("""
        To use forecasting, please run the notebook first:
        1. Open **Ruya.ipynb**
        2. Run all cells including the final "Train and Save Models" cell
        3. This will create trained models in the `models/` folder
        4. Refresh this page
        """)
        st.stop()
    
    # Load daily data
    if os.path.exists(DAILY_DF_PATH):
        import joblib
        daily_df = joblib.load(DAILY_DF_PATH)
    else:
        daily_df = df.groupby('Date')['money'].sum().reset_index()
        daily_df.columns = ['ds', 'y']
        daily_df = daily_df.sort_values('ds').reset_index(drop=True)
    
    st.subheader("Model Selection")
    
    col_model, col_horizon = st.columns(2)
    
    with col_model:
        selected_model = st.selectbox(
            "Select Forecasting Model",
            options=available_models,
            help="LSTM: Best accuracy (MAE) | ARIMA: Most stable (RMSE) | Prophet: Best % error (MAPE)"
        )
    
    with col_horizon:
        forecast_days = st.slider(
            "Forecast Horizon (days)", 
            7, 30, 14,
            help="Number of days to predict into the future"
        )
    
    st.divider()
    
    # Load pre-trained models and predict
    @st.cache_resource
    def load_arima_model():
        """Load pre-trained ARIMA model"""
        import joblib
        return joblib.load(ARIMA_PATH)
    
    @st.cache_resource
    def load_prophet_model():
        """Load pre-trained Prophet model"""
        import pickle
        with open(PROPHET_PATH, 'rb') as f:
            return pickle.load(f)
    
    @st.cache_resource
    def load_lstm_model():
        """Load pre-trained LSTM model and scaler"""
        import joblib
        from tensorflow.keras.models import load_model
        model = load_model(LSTM_MODEL_PATH)
        scaler = joblib.load(LSTM_SCALER_PATH)
        return model, scaler
    
    def predict_arima(horizon):
        """Generate forecast using pre-trained ARIMA"""
        model = load_arima_model()
        return model.predict(n_periods=horizon)
    
    def predict_prophet(horizon):
        """Generate forecast using pre-trained Prophet"""
        model = load_prophet_model()
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)
        return forecast.tail(horizon)['yhat'].values
    
    def predict_lstm(horizon, window=7):
        """Generate forecast using pre-trained LSTM"""
        model, scaler = load_lstm_model()
        
        # Get last window of data
        ts = daily_df['y'].values.reshape(-1, 1)
        ts_scaled = scaler.transform(ts)
        last_window = ts_scaled[-window:].reshape(1, window, 1)
        
        preds_scaled = []
        for _ in range(horizon):
            next_val = model.predict(last_window, verbose=0)
            preds_scaled.append(next_val[0][0])
            next_val_reshaped = next_val.reshape(1, 1, 1)
            last_window = np.concatenate([last_window[:, 1:, :], next_val_reshaped], axis=1)
        
        preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
        return preds
    
    # Run forecast button
    if st.button("Generate Forecast", type="primary"):
        with st.spinner(f"Loading {selected_model} model and generating forecast..."):
            try:
                last_date = daily_df['ds'].max()
                forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
                
                if selected_model == "ARIMA":
                    forecast_values = predict_arima(forecast_days)
                elif selected_model == "Prophet":
                    forecast_values = predict_prophet(forecast_days)
                elif selected_model == "LSTM":
                    forecast_values = predict_lstm(forecast_days)
                
                # Store in session state
                st.session_state['forecast_values'] = forecast_values
                st.session_state['forecast_dates'] = forecast_dates
                st.session_state['forecast_model'] = selected_model
                st.session_state['daily_df'] = daily_df
                
                st.success(f"{selected_model} forecast generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
        
        # Display forecast if available
        if 'forecast_values' in st.session_state:
            forecast_values = st.session_state['forecast_values']
            forecast_dates = st.session_state['forecast_dates']
            model_used = st.session_state['forecast_model']
            daily_df_cached = st.session_state['daily_df']
            
            st.divider()
            st.subheader(f"{model_used} Forecast Results")
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Forecast': forecast_values
            })
            
            # Visualization
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=daily_df_cached['ds'], y=daily_df_cached['y'],
                mode='lines', name='Historical Revenue',
                line=dict(width=2)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'], y=forecast_df['Forecast'],
                mode='lines+markers', name=f'{model_used} Forecast',
                line=dict(width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f'Revenue Forecast using {model_used} (Next {len(forecast_values)} Days)',
                xaxis_title='Date',
                yaxis_title='Revenue ($)',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast Summary Metrics
            st.subheader("Forecast Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            last_7_avg = daily_df_cached['y'].tail(7).mean()
            forecast_avg = np.mean(forecast_values)
            forecast_total = np.sum(forecast_values)
            change_pct = ((forecast_avg - last_7_avg) / last_7_avg * 100)
            
            with col1:
                st.metric("Forecasted Total", f"${forecast_total:,.2f}")
            with col2:
                st.metric("Daily Average", f"${forecast_avg:,.2f}")
            with col3:
                st.metric("vs Last 7 Days", f"{change_pct:+.1f}%")
            with col4:
                st.metric("Model Used", model_used)
            
            # Forecast Table
            st.subheader("Daily Forecast Details")
            forecast_df['Forecast'] = forecast_df['Forecast'].round(2)
            forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(forecast_df.style.format({'Forecast': '${:,.2f}'}), use_container_width=True)
        
        st.divider()
        
        # Model Performance Summary from notebook
        st.subheader("Model Performance Comparison")
        st.write("Cross-validation results from notebook analysis:")
        
        model_results = pd.DataFrame({
            'Model': ['ARIMA', 'Prophet', 'LSTM'],
            'MAE': [85.42, 112.35, 78.65],
            'RMSE': [102.18, 138.92, 95.43],
            'MAPE (%)': [18.5, 15.2, 19.8],
            'Best For': ['Stability', 'Seasonality', 'Accuracy']
        })
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.bar(model_results, x='Model', y='MAE', 
                        title='MAE (Lower = Better)',
                        color='Model')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(model_results, x='Model', y='RMSE',
                        title='RMSE (Lower = Better)',
                        color='Model')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = px.bar(model_results, x='Model', y='MAPE (%)',
                        title='MAPE % (Lower = Better)',
                        color='Model')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(model_results, use_container_width=True)
        
        st.info("""
        Model Recommendation:
        - LSTM: Best accuracy with lowest MAE (78.65) - recommended for daily predictions
        - ARIMA: Most stable with lowest RMSE (102.18) - good for consistent forecasts  
        - Prophet: Best percentage error with lowest MAPE (15.2%) - good for trend analysis
        """)

# =====================================================
# PAGE: DATA EXPLORER
# =====================================================
elif page == "Data Explorer":
    st.title("Data Explorer")
    st.caption("Explore and download your filtered data.")
    
    st.divider()
    
    # Data Summary
    st.subheader("Data Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{len(filtered_df):,}")
    with col2:
        st.metric("Columns", f"{len(filtered_df.columns)}")
    with col3:
        st.metric("Date Range", f"{(filtered_df['Date'].max() - filtered_df['Date'].min()).days} days")
    with col4:
        st.metric("Unique Products", f"{filtered_df['coffee_name'].nunique()}")
    
    st.divider()
    
    # Column Statistics
    st.subheader("Column Statistics")
    if len(filtered_df) > 0:
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            stats = filtered_df[numeric_cols].describe()
            st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)
    
    st.divider()
    
    # Data Table
    st.subheader("Data Table")
    
    # Column selector
    all_columns = filtered_df.columns.tolist()
    selected_columns = st.multiselect("Select columns to display", all_columns, default=all_columns[:8])
    
    if selected_columns:
        st.dataframe(filtered_df[selected_columns], use_container_width=True)
    
    # Download Button
    st.divider()
    st.subheader("Download Data")
    
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="ruya_filtered_data.csv",
        mime="text/csv"
    )

# =====================================================
# FOOTER
# =====================================================
st.sidebar.markdown("---")
st.sidebar.caption("Ruya v1.0 | Coffee Analytics")

st.divider()
st.caption("Ruya - A Smart Sales Analysis and Forecasting System for Coffee Shops | Powered by Streamlit & Plotly")
