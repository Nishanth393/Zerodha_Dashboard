import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from kiteconnect import KiteConnect
import datetime
import re
import time

# --- 1. PAGE CONFIG & STYLING ---
st.set_page_config(layout="wide", page_title="Zerodha Command Center", page_icon="🪁")

st.markdown("""
    <style>
        html, body, [class*="css"] { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
        .block-container { padding-top: 1rem !important; padding-bottom: 5rem !important; }
        h1 { font-size: 1.6rem !important; font-weight: 700 !important; }
        h2 { font-size: 1.4rem !important; font-weight: 600 !important; }
        h3 { font-size: 1.2rem !important; font-weight: 600 !important; }
        .stDataFrame { width: 100% !important; }
        [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
        .stButton button { min-height: 45px; width: 100%; }
        
        /* Pulse Animation for Live Status */
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .live-dot {
            height: 10px; width: 10px;
            background-color: #00ff00;
            border-radius: 50%;
            display: inline-block;
            animation: pulse 2s infinite;
            margin-right: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. AUTHENTICATION ---
st.sidebar.title("🪁 Setup")

def sanitize_key(key):
    if key: return ''.join(e for e in key if e.isalnum())
    return ""

auto_token = None
try:
    with open("access_token.txt", "r") as f:
        auto_token = sanitize_key(f.read().strip())
except FileNotFoundError:
    pass

DEFAULT_API_KEY = "" 
api_key = st.sidebar.text_input("API Key", value=DEFAULT_API_KEY, type="password")
access_token = st.sidebar.text_input("Access Token", value=auto_token if auto_token else "", type="password")

api_key = sanitize_key(api_key)
access_token = sanitize_key(access_token)

kite = None
if api_key and access_token:
    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        st.sidebar.success(f"✅ Connected")
    except Exception as e:
        st.sidebar.error(f"Connection Failed: {e}")

st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Select Module", 
    ["Portfolio Manager", "Tradebook Analyzer", "Technical Scanner", "Market Heatmap"]
)

# --- 3. HELPERS ---
@st.cache_data(ttl=3600*4) 
def get_instruments():
    if kite: return pd.DataFrame(kite.instruments("NFO"))
    return pd.DataFrame()

def get_correct_equity_symbol(sym):
    if sym == "NIFTY": return "NSE:NIFTY 50"
    if sym == "BANKNIFTY": return "NSE:NIFTY BANK"
    return f"NSE:{sym}"

def calculate_tax_for_position(row):
    qty = abs(row['quantity'])
    price = row['last_price']
    turnover = qty * price
    brokerage = 20.0
    is_fno = row['exchange'] == 'NFO'
    is_option = is_fno and (row['tradingsymbol'].endswith('CE') or row['tradingsymbol'].endswith('PE'))
    is_future = is_fno and not is_option
    stt = turnover * 0.000625 if is_option else (turnover * 0.000125 if is_future else turnover * 0.001)
    txn = turnover * 0.00053 if is_option else (turnover * 0.00002 if is_future else turnover * 0.0000345)
    gst = (brokerage + txn) * 0.18
    return brokerage + stt + txn + gst

# --- 4. MODULE: PORTFOLIO MANAGER (AUTO-REFRESH ENABLED) ---
# The @st.fragment decorator makes ONLY this function re-run every 2 seconds
@st.fragment(run_every=2)
def render_portfolio_live():
    # Fetch Data
    try:
        margins = kite.margins()
        cash = margins['equity']['available']['live_balance']
        used = margins['equity']['utilised']['debits']
        
        pos = kite.positions()['net']
        total_pnl = 0
        
        if pos:
            df = pd.DataFrame(pos)
            df['Est. Tax'] = df.apply(calculate_tax_for_position, axis=1)
            df['Net P&L'] = df['pnl'] - df['Est. Tax']
            total_pnl = df['Net P&L'].sum()
        
        # Header Metrics
        st.markdown('<div><span class="live-dot"></span> <b>LIVE UPDATES (2s)</b></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("💰 Available", f"₹{cash:,.0f}")
        c2.metric("📉 Used", f"₹{used:,.0f}")
        c3.metric("Net P&L (Post-Tax)", f"₹{total_pnl:,.0f}", delta=f"{total_pnl:,.0f}")
        
        st.divider()
        
        if pos:
            cols = ['tradingsymbol', 'product', 'quantity', 'average_price', 'last_price', 'pnl', 'Est. Tax', 'Net P&L']
            st.dataframe(
                df[cols].style.format({
                    'average_price': '₹{:.2f}', 'last_price': '₹{:.2f}', 'pnl': '₹{:.0f}', 
                    'Est. Tax': '₹{:.0f}', 'Net P&L': '₹{:.0f}'
                }).background_gradient(subset=['Net P&L'], cmap='RdYlGn'),
                use_container_width=True,
                height=500
            )
        else:
            st.info("No open positions.")
            
    except Exception as e:
        st.error(f"Live Error: {e}")

def render_portfolio():
    st.title("💼 Live Portfolio")
    if not kite: st.warning("⚠️ Connect to Zerodha first."); return
    # Call the auto-refreshing fragment
    render_portfolio_live()

# --- 5. MODULE: TECHNICAL SCANNER ---
def render_scanner():
    st.title("📈 Technical Scanner (Pro)")
    if not kite: st.warning("⚠️ Connect to Zerodha first."); return
    
    c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
    with c1: sym = st.text_input("Symbol", "ASHOKLEY", key="scan_sym").upper()
    with c2: days = st.number_input("Lookback Days", 1, 1000, 5)
    with c3:
        imap = {"Default": None, "1 Min": "minute", "60 Min": "60minute", "Daily": "day"}
        sel_int = st.selectbox("Interval", list(imap.keys()))
    with c4: 
        st.write("") 
        run = st.button("Load Chart", type="primary")
        
    if run:
        try:
            with st.spinner("Fetching..."):
                inst = pd.DataFrame(kite.instruments("NSE"))
                tok_row = inst[inst['tradingsymbol'] == sym]
                if tok_row.empty:
                    inst_nfo = pd.DataFrame(kite.instruments("NFO"))
                    tok_row = inst_nfo[inst_nfo['tradingsymbol'] == sym]
                if tok_row.empty: st.error("Symbol not found"); return
                tok = tok_row.iloc[0]['instrument_token']
                
                manual = imap[sel_int]
                interval = manual if manual else ("minute" if days <= 3 else ("60minute" if days <= 60 else "day"))
                
                buffer = days + 4 + (days // 5) * 2 
                to_date = datetime.datetime.now()
                from_date = to_date - datetime.timedelta(days=buffer)
                from_date = from_date.replace(hour=0, minute=0, second=0) 
                
                recs = kite.historical_data(tok, from_date, to_date, interval)
                df = pd.DataFrame(recs)
                if df.empty: st.warning("No data."); return
                
                if interval == "day": df = df.tail(days)
                
                for s in [10, 20, 50, 100]: df[f'EMA_{s}'] = df['close'].ewm(span=s).mean()
                df['VWAP'] = (df['close']*df['volume']).cumsum() / df['volume'].cumsum()
                
                lc = df.iloc[-2] if len(df)>1 else df.iloc[-1]
                p = (lc['high']+lc['low']+lc['close'])/3
                r = lc['high']-lc['low']
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"))
                
                colors = {10:'cyan', 20:'magenta', 50:'orange', 100:'white'}
                for s, c in colors.items():
                    fig.add_trace(go.Scatter(x=df['date'], y=df[f'EMA_{s}'], line=dict(color=c, width=1), name=f"EMA {s}"))
                fig.add_trace(go.Scatter(x=df['date'], y=df['VWAP'], line=dict(color='purple', width=2, dash='dot'), name="VWAP"))
                
                def add_line(val, name, col):
                    fig.add_shape(type="line", x0=df['date'].iloc[0], x1=df['date'].iloc[-1], y0=val, y1=val, line=dict(color=col, width=1, dash="dash"))
                
                pivots = [
                    (p, "P", "yellow"), (p+0.382*r, "R1", "green"), (p+0.618*r, "R2", "green"), (p+r, "R3", "lightgreen"),
                    (p-0.382*r, "S1", "red"), (p-0.618*r, "S2", "red"), (p-r, "S3", "darkred")
                ]
                for val, name, col in pivots: add_line(val, name, col)

                fig.update_layout(height=600, margin=dict(l=10, r=10, t=30, b=10), template="plotly_dark", legend=dict(orientation="h", y=1.02), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 🔢 Levels")
                cols = st.columns(7)
                cols[0].metric("Pivot", f"{p:.2f}")
                cols[1].metric("R1", f"{p+0.382*r:.2f}"); cols[2].metric("R2", f"{p+0.618*r:.2f}"); cols[3].metric("R3", f"{p+r:.2f}")
                cols[4].metric("S1", f"{p-0.382*r:.2f}"); cols[5].metric("S2", f"{p-0.618*r:.2f}"); cols[6].metric("S3", f"{p-r:.2f}")

        except Exception as e: st.error(str(e))

# --- 6. MODULE: MARKET HEATMAP ---
def render_heatmap():
    st.title("🔥 Market Heatmap")
    if not kite: st.warning("⚠️ Connect to Zerodha first."); return
    
    c1, c2 = st.columns([2, 1])
    with c1: symbol = st.text_input("Symbol", "NIFTY").upper()
    with c2: 
        st.write("")
        if st.button("↻ Clear Cache"): st.cache_data.clear()
    
    if st.button("Generate Heatmap", type="primary"):
        with st.spinner("Scanning..."):
            try:
                inst = get_instruments()
                if inst.empty: return
                filtered = inst[(inst['name'] == symbol) & (inst['segment'] == 'NFO-OPT')].copy()
                if filtered.empty: st.error("No options found."); return
                
                filtered['expiry'] = pd.to_datetime(filtered['expiry'])
                nearest = filtered.sort_values('expiry')['expiry'].iloc[0]
                opts = filtered[filtered['expiry'] == nearest].copy()
                
                u_key = get_correct_equity_symbol(symbol)
                try: ltp = kite.quote([u_key])[u_key]['last_price']
                except: ltp = opts['strike'].median()
                
                opts['diff'] = abs(opts['strike'] - ltp)
                top_opts = opts.sort_values('diff').head(40)
                
                tokens = [int(x) for x in top_opts['instrument_token'].unique()]
                live = kite.quote(tokens)
                
                data = []
                for s in sorted(top_opts['strike'].unique(), reverse=True):
                    row = {'Strike': s}
                    def get_oi(r):
                        if r.empty: return 0
                        t = r.iloc[0]['instrument_token']
                        d = live.get(int(t)) or live.get(str(t))
                        return d.get('oi', 0) if d else 0
                    row['Call OI'] = get_oi(top_opts[(top_opts['strike']==s) & (top_opts['instrument_type']=='CE')])
                    row['Put OI'] = get_oi(top_opts[(top_opts['strike']==s) & (top_opts['instrument_type']=='PE')])
                    data.append(row)
                
                df = pd.DataFrame(data).set_index('Strike')
                st.success(f"LTP: {ltp} | Expiry: {nearest.date()}")
                
                fig = px.imshow(df, aspect="auto", color_continuous_scale="RdBu_r", text_auto=True)
                fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e: st.error(str(e))

# --- 7. MODULE: TRADEBOOK ANALYZER ---
def render_tradebook():
    st.title("📜 F&O Tradebook")
    if not kite: st.warning("⚠️ Connect to Zerodha first."); return

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1: uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    with c2: root_symbol = st.text_input("Root Symbol", "ASHOKLEY").upper()
    with c3: 
        st.write("")
        trigger = st.button("Run Analysis", type="primary")

    if "tb_data" not in st.session_state: st.session_state.tb_data = None
    if "editor_key" not in st.session_state: st.session_state.editor_key = 0

    if trigger:
        # Save Trigger Inputs to Session so they persist during auto-refresh
        st.session_state.tb_inputs = {
            "file": uploaded_file, "symbol": root_symbol
        }
        # Run Calculation Once
        run_tradebook_calc()

    # RENDER THE LIVE FRAGMENT (If data exists)
    if st.session_state.tb_data:
        render_tradebook_live_fragment()

def run_tradebook_calc():
    """Heavy Calculation Logic (FIFO) - Runs only on Button Click"""
    try:
        inputs = st.session_state.tb_inputs
        uploaded_file = inputs["file"]
        root_symbol = inputs["symbol"]
        
        with st.spinner("Processing FIFO..."):
            def clean_sym(s): return re.sub(r'[^A-Z0-9]', '', str(s).upper())
            def clean_time(t):
                t = pd.to_datetime(t)
                if t.tzinfo is not None: t = t.tz_convert('Asia/Kolkata').tz_localize(None)
                return t
            today_date = pd.Timestamp.now().normalize()

            # Metadata (We only need static metadata here)
            inst_df = pd.DataFrame(kite.instruments("NFO"))
            inst_df['clean_sym'] = inst_df['tradingsymbol'].apply(clean_sym)
            stock_inst = inst_df[inst_df['name'] == root_symbol]
            
            # Map Contracts
            contract_map = {}
            for _, r in stock_inst.iterrows():
                contract_map[r['clean_sym']] = {
                    'token': int(r['instrument_token']), 'strike': float(r['strike']), 
                    'type': r['instrument_type'], 'lot_size': int(r['lot_size']),
                    'expiry': r['expiry'] # Keep expiry for futures sort
                }
            
            # Collect Trades
            all_trades = []
            if uploaded_file:
                uploaded_file.seek(0)
                df_h = pd.read_csv(uploaded_file)
                df_h.columns = [c.lower().strip().replace(" ", "_").replace("-", "_") for c in df_h.columns]
                s_col = next((c for c in df_h.columns if 'symbol' in c), None)
                if s_col:
                    df_h['clean_sym'] = df_h[s_col].apply(clean_sym)
                    df_h = df_h[df_h['clean_sym'].str.startswith(clean_sym(root_symbol))]
                    if not df_h.empty:
                        df_h['date'] = pd.to_datetime(df_h['trade_date'], dayfirst=True).apply(clean_time)
                        df_h = df_h[df_h['date'] < today_date]
                        for _, r in df_h.iterrows():
                            all_trades.append({'symbol': r['clean_sym'], 'time': r['date'], 'qty': abs(r['quantity']), 'price': r['price'], 'type': r['trade_type'].upper()})
            
            orders = kite.orders()
            if orders:
                df_l = pd.DataFrame(orders)
                if not df_l.empty:
                    df_l['clean_sym'] = df_l['tradingsymbol'].apply(clean_sym)
                    df_l = df_l[(df_l['clean_sym'].str.startswith(clean_sym(root_symbol))) & (df_l['status'] == 'COMPLETE')]
                    for _, r in df_l.iterrows():
                        all_trades.append({'symbol': r['clean_sym'], 'time': clean_time(r['order_timestamp']), 'qty': abs(r['quantity']), 'price': r['average_price'], 'type': r['transaction_type']})

            if not all_trades: st.warning("No trades."); return

            # FIFO
            df_master = pd.DataFrame(all_trades)
            open_pos = []; ledger = {}

            for cont in df_master['symbol'].unique():
                trades = df_master[df_master['symbol'] == cont]
                buys = trades[trades['type'] == 'BUY'].sort_values('time').to_dict('records')
                sells = trades[trades['type'] == 'SELL'].sort_values('time').to_dict('records')
                cont_ledg = []
                
                while buys and sells:
                    b = buys[0]; s = sells[0]
                    match_q = min(b['qty'], s['qty'])
                    cont_ledg.append(f"MATCH: {match_q}")
                    b['qty'] -= match_q; s['qty'] -= match_q
                    if b['qty'] == 0: buys.pop(0)
                    if s['qty'] == 0: sells.pop(0)
                
                for item in buys:
                    if item['qty'] > 0: item.update({'contract': cont, 'side': 'LONG'}); open_pos.append(item)
                for item in sells:
                    if item['qty'] > 0: item.update({'contract': cont, 'side': 'SHORT'}); open_pos.append(item)
                ledger[cont] = cont_ledg

            # Store Static Data
            st.session_state.tb_data = {
                'df_open_static': pd.DataFrame(open_pos), # The list of open trades
                'ledger': ledger,
                'contract_map': contract_map,
                'root_symbol': root_symbol,
                'agg': pd.DataFrame() # Will build in live view
            }
            
    except Exception as e: st.error(str(e))

@st.fragment(run_every=2)
def render_tradebook_live_fragment():
    # This fragment fetches LIVE PRICES for the OPEN POSITIONS calculated above
    data = st.session_state.tb_data
    if not data: return
    
    root_symbol = data['root_symbol']
    contract_map = data['contract_map']
    df_op = data['df_open_static'].copy() # Work on a copy
    
    try:
        # 1. LIVE PRICES (Spot + Futures + Open Contracts)
        eq_sym = get_correct_equity_symbol(root_symbol)
        
        # Identify Futures
        all_syms = list(contract_map.keys())
        fut_syms = [s for s in all_syms if contract_map[s]['type'] == 'FUT']
        # Sort futures by expiry (we stored expiry in calc step)
        fut_syms.sort(key=lambda s: pd.to_datetime(contract_map[s].get('expiry', '2099-01-01')))
        top_3_futs = fut_syms[:3]
        
        # Identify Open Position Tokens
        op_tokens = []
        if not df_op.empty:
            op_tokens = [contract_map[s]['token'] for s in df_op['contract'].unique() if s in contract_map]
            
        # Get Future Tokens
        fut_tokens = [contract_map[s]['token'] for s in top_3_futs]
        
        # Bulk Fetch
        quote_keys = [eq_sym] + fut_tokens + op_tokens
        quotes = kite.quote(list(set(quote_keys))) # unique
        
        # Helper Price Getter
        def get_price(token_or_sym):
            d = quotes.get(token_or_sym) or quotes.get(str(token_or_sym)) or quotes.get(int(token_or_sym) if str(token_or_sym).isdigit() else 0)
            if d:
                if d.get('last_price', 0) > 0: return d['last_price']
                if d.get('ohlc', {}).get('close', 0) > 0: return d['ohlc']['close']
            return 0

        # Metrics
        eq_ltp = get_price(eq_sym)
        fut_display = [(s, get_price(contract_map[s]['token'])) for s in top_3_futs]
        
        # Update Open Positions with Live Prices
        if not df_op.empty:
            df_op['LTP'] = df_op['contract'].apply(lambda x: get_price(contract_map[x]['token']))
            # If LTP 0, fallback to Spot (Theory)
            df_op['LTP'] = df_op.apply(lambda r: eq_ltp if r['LTP']==0 and contract_map[r['contract']]['type']=='FUT' else r['LTP'], axis=1)
            
            # P&L
            df_op['Unrealized'] = df_op.apply(lambda r: (r['LTP']-r['price'])*r['qty'] if r['side']=='LONG' else (r['price']-r['LTP'])*r['qty'], axis=1)
            
            # Aggregation for Editor
            agg_rows = []
            grp = df_op.groupby(['contract', 'side'])
            for (cont, side), rows in grp:
                agg_rows.append({
                    'Contract': cont, 'Side': side, 'Total Qty': rows['qty'].sum(),
                    'Avg Price': (rows['price']*rows['qty']).sum()/rows['qty'].sum(),
                    'LTP': rows['LTP'].iloc[0],
                    'Lot Size': contract_map[cont]['lot_size']
                })
            st.session_state.tb_data['agg'] = pd.DataFrame(agg_rows)
        
        # --- RENDER HEADER ---
        # Live Dot
        st.markdown('<div><span class="live-dot"></span> <b>LIVE</b></div>', unsafe_allow_html=True)
        
        # Row 1: P&L + Spot
        unrealized = df_op['Unrealized'].sum() if not df_op.empty else 0
        k1, k2, k3 = st.columns(3)
        k1.metric("Realized (Fixed)", f"₹0") # Placeholder as Realized is static in FIFO
        k2.metric("Floating P&L", f"₹{unrealized:,.0f}", delta=f"{unrealized:,.0f}")
        k3.metric("Spot", f"₹{eq_ltp}")
        
        # Row 2: Futures
        if fut_display:
            f_cols = st.columns(len(fut_display))
            for i, (sym, p) in enumerate(fut_display):
                f_cols[i].metric(sym, f"₹{p}")
                
        st.divider()
        
        # --- RENDER LIST ---
        if not df_op.empty:
            st.markdown("### 🔓 Open Positions")
            v = df_op[['contract', 'side', 'time', 'qty', 'price', 'LTP', 'Unrealized']].copy()
            st.dataframe(v.style.format({'price':'{:.2f}', 'LTP':'{:.2f}', 'Unrealized':'{:.0f}'}).background_gradient(subset=['Unrealized'], cmap='RdYlGn'), use_container_width=True)
        else: st.success("No open positions.")
        
    except Exception as e: st.error(f"Live Error: {e}")

    # --- RENDER EDITOR (Outside Exception to persist state) ---
    # Note: Streamlit Editors don't play well inside fragments if they trigger reruns.
    # We render the editor based on the 'agg' data we just calculated live.
    if 'agg' in st.session_state.tb_data and not st.session_state.tb_data['agg'].empty:
        st.divider()
        c_head, c_btn = st.columns([4, 1])
        with c_head: st.markdown("### 🧮 Scenario Planner")
        with c_btn:
            if st.button("Reset Values"):
                st.session_state.editor_key += 1
                st.rerun()

        edit_df = st.session_state.tb_data['agg'].copy()
        if "Add Lots" not in edit_df.columns: edit_df["Add Lots"] = 0
        if "Buy At" not in edit_df.columns: edit_df["Buy At"] = edit_df["LTP"]
        if "Target" not in edit_df.columns: edit_df["Target"] = edit_df["LTP"]

        edited = st.data_editor(
            edit_df,
            column_config={
                "Contract": st.column_config.TextColumn("Contract", disabled=True),
                "Side": st.column_config.TextColumn("Side", disabled=True, width="small"),
                "Total Qty": st.column_config.NumberColumn("Open Qty", disabled=True),
                "Avg Price": st.column_config.NumberColumn("Curr Avg", format="₹%.2f", disabled=True),
                "LTP": st.column_config.NumberColumn("LTP", format="₹%.2f", disabled=True),
                "Add Lots": st.column_config.NumberColumn("➕ Add Lots", min_value=0, step=1, required=True),
                "Buy At": st.column_config.NumberColumn("🛒 Buy At", format="₹%.2f"),
                "Target": st.column_config.NumberColumn("🎯 Target", format="₹%.2f")
            },
            disabled=["Contract", "Side", "Total Qty", "Avg Price", "LTP", "Lot Size"],
            use_container_width=True,
            key=f"scenario_editor_{st.session_state.editor_key}"
        )
        
        if not edited.empty:
            res = []
            for _, r in edited.iterrows():
                if r['Add Lots'] > 0 or r['Target'] != r['LTP']:
                    add_q = r['Add Lots'] * r['Lot Size']
                    new_q = r['Total Qty'] + add_q
                    new_avg = ((r['Total Qty']*r['Avg Price']) + (add_q*r['Buy At'])) / new_q
                    proj_pnl = (r['Target'] - new_avg)*new_q if r['Side']=='LONG' else (new_avg - r['Target'])*new_q
                    res.append({'Contract': r['Contract'], 'New Avg': new_avg, 'Target': r['Target'], 'Proj P&L': proj_pnl})
            
            if res:
                st.dataframe(pd.DataFrame(res).style.format({'New Avg':'{:.2f}', 'Target':'{:.2f}', 'Proj P&L':'{:.0f}'}).background_gradient(subset=['Proj P&L'], cmap='RdYlGn'), use_container_width=True)

# --- 8. MAIN EXECUTION ---
if app_mode == "Portfolio Manager": render_portfolio()
elif app_mode == "Market Heatmap": render_heatmap()
elif app_mode == "Technical Scanner": render_scanner()
elif app_mode == "Tradebook Analyzer": render_tradebook()
