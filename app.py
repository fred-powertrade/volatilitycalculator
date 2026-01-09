import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime, timedelta
import time  # For Unix timestamps

# ==========================================
# 1. CONFIGURATION & PAGE SETUP
# ==========================================
st.set_page_config(layout="wide", page_title="MM Volatility Dashboard")

st.title("⚡ Dynamic Volatility Dashboard (RMS Model)")
st.markdown("""
**Strategy:** Market Maker (MM) Risk Engine for Volatile Markets (e.g., Memecoins/Altcoins)  
**Metric:** Root Mean Square (RMS) of Custom Historical Volatility Windows  
**Customizable:** By date range, instrument (network/pool), vol windows.
""")

# ==========================================
# 2. DATA ENGINE (Using GeckoTerminal API with params)
# ==========================================
@st.cache_data(ttl=600)  # Longer cache for stability
def get_crypto_data(network, pool_address, timeframe='day', limit=365, before_timestamp=None):
    base_url = f"https://api.geckoterminal.com/api/v2/networks/{network}/pools/{pool_address}/ohlcv/{timeframe}"
    
    params = {
        'aggregate': 1,  # Default aggregation
        'limit': limit,  # Max results
    }
    if before_timestamp:
        params['before_timestamp'] = before_timestamp
    
    try:
        response = requests.get(base_url, params=params)
        
        if response.status_code != 200:
            st.error(f"API Error: {response.status_code}. Check Network/Pool Address.")
            return pd.DataFrame()
        
        data = response.json()
        
        if 'data' not in data or 'attributes' not in data['data'] or 'ohlcv_list' not in data['data']['attributes']:
            st.warning("No OHLCV data available in response.")
            return pd.DataFrame()
        
        ohlcv_list = data['data']['attributes']['ohlcv_list']
        df = pd.DataFrame(ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('timestamp').sort_index()
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# ==========================================
# 3. MATHEMATICAL ENGINE (Customizable windows)
# ==========================================
def calculate_metrics(df, window1=7, window2=14):
    if len(df) < max(window1, window2) + 1:
        return pd.DataFrame()  # Not enough data
    
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    ANNUAL_FACTOR = np.sqrt(365)  # Crypto 365 days/year
    
    df['hv_short'] = df['log_ret'].rolling(window=window1).std() * ANNUAL_FACTOR
    df['hv_long'] = df['log_ret'].rolling(window=window2).std() * ANNUAL_FACTOR
    
    df['rms_vol'] = np.sqrt((df['hv_short']**2 + df['hv_long']**2) / 2)