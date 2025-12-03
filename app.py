import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.set_page_config(layout="wide")
st.title("Laptop Price Predictor")

# 1. Brand and Type
st.subheader("Basic Information")
col_brand, col_type = st.columns(2)
with col_brand:
    company = st.selectbox('Brand', df['Company'].unique())
with col_type:
    laptop_type = st.selectbox('Type', df['TypeName'].unique())

# 2. RAM and Memory
st.subheader("Memory Configuration")
col_ram, col_memory = st.columns(2)
with col_ram:
    ram_gb = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
with col_memory:
    memory_options = df['Memory'].unique().tolist()
    memory = st.selectbox('Memory (Storage)', memory_options)

# 3. Screen Details
st.subheader("Screen Specifications")
col_ts_ips, col_res_size = st.columns(2)
with col_ts_ips:
    touchscreen_input = st.selectbox('Touchscreen', ['No', 'Yes'])
    touchscreen_val = 1 if touchscreen_input == 'Yes' else 0
    ips_input = st.selectbox('IPS Display', ['No', 'Yes'])
    ips_val = 1 if ips_input == 'Yes' else 0
with col_res_size:
    screen_inches = st.number_input('Screen Size (in Inches)', value=15.6, min_value=10.0, max_value=20.0, step=0.1)
    resolution_options = ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
                            '2880x1800', '2560x1600', '2560x1440', '2304x1440', '1920x1200',
                            '2400x1600', '2736x1824', '2160x1440', '2256x1504']
    resolution_str = st.selectbox('Screen Resolution', resolution_options)

# 4. CPU Details
st.subheader("Processor Details")
col_cpu_man, col_cpu_model, col_clock_speed = st.columns(3)
with col_cpu_man:
    cpu_manufacturer = st.selectbox('CPU Manufacturer', df['CPU Manufacturer'].unique())
with col_cpu_model:
    cpu_model_options = ['None'] + df['CPU Model'].dropna().unique().tolist()
    cpu_model = st.selectbox('CPU Model', cpu_model_options)
with col_clock_speed:
    clock_speed_ghz = st.number_input('Clock Speed (in GHz)', value=2.5, min_value=0.5, max_value=5.0, step=0.1)

# 5. GPU Details
st.subheader("Graphics Card")
col_gpu_man, col_gpu_model = st.columns(2)
with col_gpu_man:
    gpu_manufacturer = st.selectbox('GPU Manufacturer', df['GPU Manufacturer'].unique())
with col_gpu_model:
    gpu_model_options = ['None'] + df['GPU Model'].dropna().unique().tolist()
    gpu_model = st.selectbox('GPU Model', gpu_model_options)

# 6. Weight and OS
st.subheader("Other Specs & OS")
col_weight, col_os = st.columns(2)
with col_weight:
    weight_kg = st.number_input('Weight (in Kg)', value=1.5, min_value=0.5, max_value=5.0, step=0.1)
with col_os:
    os_val = st.selectbox('Operating System', df['os'].unique())

st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #4CAF50; /* Green */
    color: white;
    padding: 10px 24px;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    font-size: 20px;
    font-weight: bold;
    display: block;
    margin: 0 auto;
}
div.stButton > button:first-child:hover {
    background-color: #45a049;
}
</style>
""", unsafe_allow_html=True)

# Predict Price Button
col_pred_left, col_pred_btn, col_pred_right = st.columns([1,2,1])
with col_pred_btn:
    if st.button('Predict Price', use_container_width=True):
        # 1. PPI calculation
        try:
            X_res = int(resolution_str.split('x')[0])
            Y_res = int(resolution_str.split('x')[1])
            ppi = (((X_res**2) + (Y_res**2))**0.5 / screen_inches)
            ppi = round(ppi, 2)
        except (ValueError, ZeroDivisionError):
            st.error("Invalid screen resolution or screen size. Please provide valid numeric values.")
            st.stop()

        # 2. CPU Prefix extraction
        cpu_prefix_match = re.search(r'(i[3579])', str(cpu_model), re.IGNORECASE)
        cpu_prefix = cpu_prefix_match.group(0) if cpu_prefix_match else 'Other'

        # 3. GPU Order (replicate the mapping used in training)
        sort_order = {"Intel": 0, "AMD": 1, "NVIDIA": 2, "Other": 3}
        gpu_order = sort_order.get(gpu_manufacturer, 3)

        # Create a DataFrame for prediction
        query_data = {
            'Company': [company],
            'TypeName': [laptop_type],
            'Ram': [ram_gb],
            'Memory': [memory],
            'Weight': [weight_kg],
            'Touchscreen': [touchscreen_val],
            'IPS': [ips_val],
            'ppi': [ppi],
            'CPU Manufacturer': [cpu_manufacturer],
            'CPU Model': [cpu_model],
            'CPU Prefix': [cpu_prefix],
            'Clock Speed (GHz)': [clock_speed_ghz],
            'GPU Manufacturer': [gpu_manufacturer],
            'GPU Model': [gpu_model],
            'gpu_order': [gpu_order],
            'os': [os_val]
        }
        query_df = pd.DataFrame(query_data)

        # Make prediction
        predicted_log_price = pipe.predict(query_df)[0]
        predicted_price = np.exp(predicted_log_price)

        st.subheader("The predicted price of this laptop is €" + str(round(predicted_price, 2)))