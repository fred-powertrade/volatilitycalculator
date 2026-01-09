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
    
    return df.dropna()

# ==========================================
# 4. BLACK-SCHOLES PRICER (Unchanged, solid)
# ==========================================
def black_scholes(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        term1 = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        term2 = - r * K * np.exp(-r * T) * norm.cdf(d2)
        theta = (term1 + term2) / 365
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        term1 = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta = (term1 + term2) / 365

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100 
    
    return price, delta, gamma, theta, vega

# ==========================================
# 5. USER INTERFACE LAYOUT
# ==========================================
with st.sidebar:
    st.header("🔧 Settings")
    
    # Instrument selection (support multiple for comparison)
    network = st.text_input("Network (e.g., eth, solana)", value="eth")
    pool_input = st.text_input("Pool Addresses (comma-separated)", value="0xa7bc6c09907fa2ded89f1c8d05374621cb1f88c5")
    pools = [p.strip() for p in pool_input.split(',') if p.strip()]
    
    st.divider()
    
    # Date customization
    today = datetime.today().date()
    end_date = st.date_input("End Date", value=today)
    start_date = st.date_input("Start Date", value=today - timedelta(days=30))
    if start_date > end_date:
        st.error("Start date must be before end date.")
    
    # Calculate params
    days_range = (end_date - start_date).days + 1
    before_unix = int(time.mktime(end_date.timetuple()))
    api_limit = min(days_range, 365)  # Respect max
    
    # Vol windows
    vol_window_short = st.number_input("Short Vol Window (days)", min_value=1, value=7)
    vol_window_long = st.number_input("Long Vol Window (days)", min_value=1, value=14)
    
    st.divider()
    
    st.subheader("Option Pricing Inputs")
    strike_min_pct, strike_max_pct = st.slider("Strike Distance Range (%)", 0.5, 2.0, (0.8, 1.2), 0.01)
    days_expiry = st.number_input("Days to Expiry", min_value=1, value=30)
    risk_free = st.number_input("Risk Free Rate", value=0.05)
    
    # Optional: Upload pricer file for params
    uploaded_file = st.file_uploader("Upload Options Pricer File (CSV/Excel for custom params)", type=['csv', 'xlsx'])
    if uploaded_file:
        # Example integration: read strikes or r from file
        try:
            df_upload = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
            st.info(f"Loaded {len(df_upload)} rows from file. Use columns like 'strike_pct', 'rf_rate' if present.")
            # You can override inputs here, e.g., risk_free = df_upload['rf_rate'].mean()
        except:
            st.warning("Couldn't read file. Ensure it's CSV/Excel.")

# Fetch and process data for each pool
if pools and network:
    all_processed = {}
    for pool in pools:
        raw_df = get_crypto_data(network, pool, limit=api_limit, before_timestamp=before_unix)
        
        if not raw_df.empty:
            processed_df = calculate_metrics(raw_df, vol_window_short, vol_window_long)
            if len(processed_df) > 0:
                all_processed[pool] = processed_df
            else:
                st.warning(f"Not enough data for {pool} (need at least {max(vol_window_short, vol_window_long) + 1} days).")
        else:
            st.warning(f"No data for {pool}. Check inputs.")
    
    if all_processed:
        # Display for first pool (or selected), with option to switch
        selected_pool = st.selectbox("Select Pool to View", list(all_processed.keys()))
        processed_df = all_processed[selected_pool]
        latest = processed_df.iloc[-1]
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Spot Price", f"${latest['close']:.6f}")
        col2.metric(f"RMS Vol ({vol_window_short},{vol_window_long})", f"{latest['rms_vol']*100:.2f}%")
        col3.metric(f"{vol_window_short}d Vol", f"{latest['hv_short']*100:.2f}%")
        col4.metric(f"{vol_window_long}d Vol", f"{latest['hv_long']*100:.2f}%")
        
        # Visualization (with multi-pool overlay if multiple)
        st.subheader("📈 Price vs. Volatility Regime")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=processed_df.index,
                                     open=processed_df['open'], high=processed_df['high'],
                                     low=processed_df['low'], close=processed_df['close'],
                                     name=f'Price ({selected_pool})'))
        
        for pool, df in all_processed.items():
            fig.add_trace(go.Scatter(x=df.index, y=df['rms_vol'],
                                     name=f'RMS Vol ({pool})',
                                     line=dict(width=2), yaxis='y2'))
        
        fig.update_layout(
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Volatility", overlaying='y', side='right', tickformat='.0%'),
            height=500,
            title="Market Structure Analysis"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Option Pricer (with strike range)
        st.subheader("🛠️ Market Maker Inventory Pricer (Shadow Options)")
        current_spot = latest['close']
        time_years = days_expiry / 365.0
        rms_vol = latest['rms_vol']
        
        pricer_data = {"Metric": ["Price", "Delta", "Gamma", "Theta", "Vega"]}
        for strike_pct in np.linspace(strike_min_pct, strike_max_pct, 3):  # Test 3 points in range
            target_strike = current_spot * strike_pct
            c_price, c_delta, c_gamma, c_theta, c_vega = black_scholes(current_spot, target_strike, time_years, risk_free, rms_vol, 'call')
            p_price, p_delta, p_gamma, p_theta, p_vega = black_scholes(current_spot, target_strike, time_years, risk_free, rms_vol, 'put')
            
            col_name = f"Call ({strike_pct:.2f}x Spot)"
            pricer_data[col_name] = [f"{c_price:.4f}", f"{c_delta:.4f}", f"{c_gamma:.4f}", f"{c_theta:.4f}", f"{c_vega:.4f}"]
            
            col_name = f"Put ({strike_pct:.2f}x Spot)"
            pricer_data[col_name] = [f"{p_price:.4f}", f"{p_delta:.4f}", f"{p_gamma:.4f}", f"{p_theta:.4f}", f"{p_vega:.4f}"]
        
        st.table(pd.DataFrame(pricer_data).set_index("Metric"))
    else:
        st.error("No valid data across pools.")
else:
    st.info("Enter network and at least one pool address to start.")
