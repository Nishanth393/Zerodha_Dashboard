import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from kiteconnect import KiteConnect
import datetime

# --- 1. PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Zerodha Command Center", page_icon="🪁")

# --- 2. SIDEBAR: AUTHENTICATION ---
st.sidebar.title("🪁 Kite Connect Setup")

auto_token = None
try:
    with open("access_token.txt", "r") as f:
        auto_token = f.read().strip()
except FileNotFoundError:
    pass

DEFAULT_API_KEY = "" # Optional: Hardcode your API Key here
api_key = st.sidebar.text_input("API Key", value=DEFAULT_API_KEY, type="password")
access_token = st.sidebar.text_input("Access Token", value=auto_token if auto_token else "", type="password")

kite = None
if api_key and access_token:
    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        st.sidebar.success(f"✅ Connected to Zerodha")
    except Exception as e:
        st.sidebar.error(f"Connection Failed: {e}")

st.sidebar.markdown("---")
app_mode = st.sidebar.radio("Select Module", ["Portfolio Manager (New)", "Market Heatmap", "Technical Scanner", "Tax Calculator"])

# --- 3. HELPER FUNCTIONS ---
@st.cache_data(ttl=3600*12) 
def get_instruments():
    if kite: return pd.DataFrame(kite.instruments("NFO"))
    return pd.DataFrame()

def get_underlying_symbol(symbol):
    s = symbol.upper()
    if s == "NIFTY": return "NSE:NIFTY 50"
    if s == "BANKNIFTY": return "NSE:NIFTY BANK"
    return f"NSE:{s}"

def calculate_tax_for_position(row):
    """
    Calculates estimated tax if the position is closed NOW.
    Assumes simple 'Exit' scenario.
    """
    qty = abs(row['quantity'])
    price = row['last_price']
    turnover = qty * price
    
    # Defaults
    brokerage = 20.0
    stt = 0.0
    txn_charge = 0.0
    stamp_duty = 0.0 # Stamp duty is buy side only, assuming we are closing (selling)
    
    # Logic based on Instrument Type
    # Note: This is a robust estimation, not a CA-certified value
    is_fno = row['exchange'] == 'NFO'
    is_option = is_fno and (row['tradingsymbol'].endswith('CE') or row['tradingsymbol'].endswith('PE'))
    is_future = is_fno and not is_option
    is_equity = row['exchange'] == 'NSE'
    
    if is_option:
        # Options Sell: STT 0.1% on Premium, Txn 0.05% (NSE)
        stt = turnover * 0.001
        txn_charge = turnover * 0.0005 
    elif is_future:
        # Futures Sell: STT 0.02%, Txn 0.0019% (NSE)
        stt = turnover * 0.0002
        txn_charge = turnover * 0.000019
    elif is_equity:
        # Equity Intraday Sell: STT 0.025%
        stt = turnover * 0.00025
        txn_charge = turnover * 0.00003
        
    sebi = turnover * 0.000001
    gst = (brokerage + txn_charge + sebi) * 0.18
    
    total_tax = brokerage + stt + txn_charge + sebi + gst
    return total_tax

# --- 4. MODULE: PORTFOLIO MANAGER (Fixed) ---
def render_portfolio():
    st.title("💼 Live Portfolio & Tax Analyzer")
    
    if not kite:
        st.warning("⚠️ Please connect to Zerodha first.")
        return

    if st.button("↻ Refresh Portfolio", type="primary"):
        st.cache_data.clear()

    try:
        # A. Fetch Funds
        margins = kite.margins()
        cash = margins['equity']['available']['live_balance']
        used = margins['equity']['utilised']['debits']
        
        # B. Display Header Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Available Funds", f"₹{cash:,.2f}")
        col2.metric("📉 Used Margin", f"₹{used:,.2f}")
        
        # C. Fetch Positions
        pos = kite.positions()
        net_positions = pos['net']
        
        if not net_positions:
            col3.metric("Total P&L", "₹0.00")
            st.info("You have no open positions.")
            return

        # D. Process Data
        df = pd.DataFrame(net_positions)
        
        # FIX: Ensure we don't drop 'exchange' before calculating tax
        # We perform the calculation on the full dataframe first
        df['Est. Tax'] = df.apply(calculate_tax_for_position, axis=1)
        
        # Calculate Net P&L
        df['Net P&L'] = df['pnl'] - df['Est. Tax']
        
        # Total Metrics
        net_pnl = df['Net P&L'].sum()
        col3.metric("Total Net P&L (Post-Tax)", f"₹{net_pnl:,.2f}", delta=f"{net_pnl:,.2f}")
        
        # E. Styling the Table
        st.markdown("### 📊 Position Details")
        
        # NOW we select only the columns we want to show the user
        columns_to_show = ['tradingsymbol', 'product', 'quantity', 'average_price', 'last_price', 'pnl', 'Est. Tax', 'Net P&L']
        display_df = df[columns_to_show]

        # Apply gradients (Green for profit, Red for loss)
        try:
            st.dataframe(
                display_df.style.format({
                    'average_price': '₹{:.2f}',
                    'last_price': '₹{:.2f}',
                    'pnl': '₹{:.2f}',
                    'Est. Tax': '₹{:.2f}',
                    'Net P&L': '₹{:.2f}'
                }).background_gradient(subset=['Net P&L'], cmap='RdYlGn', vmin=-5000, vmax=5000),
                use_container_width=True,
                height=400
            )
        except Exception:
            # Fallback if matplotlib is missing
            st.dataframe(display_df, use_container_width=True)
        
        st.caption(f"*Est. Tax includes Brokerage (₹20), STT, Exchange Txn, and GST. (Calculated based on exiting NOW)")

    except Exception as e:
        st.error(f"Error fetching portfolio: {e}")

# --- 5. MODULE: HEATMAP (Previous Logic) ---
def render_kite_heatmap():
    st.title("🔥 F&O Heatmap (Pro)")
    if not kite:
        st.warning("⚠️ Please connect to Zerodha first.")
        return
    
    c1, c2 = st.columns([1,1])
    with c1: symbol = st.text_input("Symbol", "NIFTY").upper()
    with c2: 
        if st.button("Generate Heatmap"):
            # ... (Same robust logic as V3.0) ...
            try:
                inst = get_instruments()
                if inst.empty: return
                filtered = inst[(inst['name'] == symbol) & (inst['segment'] == 'NFO-OPT')].copy()
                if filtered.empty: st.error("No options found"); return
                
                filtered['expiry'] = pd.to_datetime(filtered['expiry'])
                nearest = filtered.sort_values('expiry')['expiry'].iloc[0]
                opts = filtered[filtered['expiry'] == nearest].copy()
                
                u_key = get_underlying_symbol(symbol)
                try: ltp = kite.quote([u_key])[u_key]['last_price']
                except: ltp = opts['strike'].median()
                
                opts['diff'] = abs(opts['strike'] - ltp)
                top_opts = opts.sort_values('diff').head(40)
                
                tokens = [int(x) for x in top_opts['instrument_token'].unique()]
                live = kite.quote(tokens)
                
                data = []
                for s in sorted(top_opts['strike'].unique(), reverse=True):
                    row = {'Strike': s}
                    # Helper
                    def get_oi(r):
                        if r.empty: return 0
                        t = r.iloc[0]['instrument_token']
                        return live.get(int(t), {}).get('oi', live.get(str(t), {}).get('oi', 0))
                    
                    row['Call OI'] = get_oi(top_opts[(top_opts['strike']==s) & (top_opts['instrument_type']=='CE')])
                    row['Put OI'] = get_oi(top_opts[(top_opts['strike']==s) & (top_opts['instrument_type']=='PE')])
                    data.append(row)
                
                df = pd.DataFrame(data).set_index('Strike')
                st.success(f"LTP: {ltp}")
                fig = px.imshow(df, aspect="auto", color_continuous_scale="RdBu_r", text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e: st.error(str(e))

# --- 6. MODULE: TECHNICAL SCANNER ---
def render_scanner():
    st.title("📈 Technical Scanner")
    if not kite: st.warning("Connect Zerodha first"); return
    sym = st.text_input("Symbol", "ASHOKLEY").upper()
    if st.button("Chart"):
        try:
            inst = pd.DataFrame(kite.instruments("NSE"))
            tok = inst[inst['tradingsymbol'] == sym].iloc[0]['instrument_token']
            recs = kite.historical_data(tok, datetime.datetime.now()-datetime.timedelta(180), datetime.datetime.now(), "day")
            df = pd.DataFrame(recs)
            df['EMA'] = df['close'].ewm(span=50).mean()
            fig = go.Figure(data=[go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close']), go.Scatter(x=df['date'], y=df['EMA'], line=dict(color='orange'))])
            st.plotly_chart(fig)
        except Exception as e: st.error(str(e))

# --- 7. MAIN EXECUTION ---
if app_mode == "Portfolio Manager (New)": render_portfolio()
elif app_mode == "Market Heatmap": render_kite_heatmap()
elif app_mode == "Technical Scanner": render_scanner()
elif app_mode == "Tax Calculator": st.title("Use Portfolio Manager for Auto-Tax")