import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from kiteconnect import KiteConnect
import datetime
import pytz # Explicit Timezone handling
import re
import time

# --- 1. PAGE CONFIG ---
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
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
        .live-dot { height: 10px; width: 10px; background-color: #00ff00; border-radius: 50%; display: inline-block; animation: pulse 2s infinite; margin-right: 5px; }
        .atm-row { background-color: #ffffcc !important; color: black !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- 2. SETUP ---
st.sidebar.title("🪁 Setup")
def sanitize_key(key): return ''.join(e for e in key if e.isalnum()) if key else ""

auto_token = None
try:
    with open("access_token.txt", "r") as f: auto_token = sanitize_key(f.read().strip())
except: pass

api_key = sanitize_key(st.sidebar.text_input("API Key", type="password"))
access_token = sanitize_key(st.sidebar.text_input("Access Token", value=auto_token if auto_token else "", type="password"))

kite = None
if api_key and access_token:
    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        st.sidebar.success(f"✅ Connected")
    except Exception as e: st.sidebar.error(f"Connection Failed: {e}")

st.sidebar.markdown("---")
app_mode = st.sidebar.radio("Select Module", ["Portfolio Manager", "Tradebook Analyzer", "Technical Scanner", "Market Heatmap"])

# --- 3. HELPERS ---
@st.cache_data(ttl=3600*4) 
def get_instruments(): return pd.DataFrame(kite.instruments("NFO")) if kite else pd.DataFrame()

def get_correct_equity_symbol(sym):
    if sym == "NIFTY": return "NSE:NIFTY 50"
    if sym == "BANKNIFTY": return "NSE:NIFTY BANK"
    return f"NSE:{sym}"

def calculate_tax_for_position(row):
    qty = abs(row['quantity']); price = row['last_price']; turnover = qty * price
    brokerage = 20.0
    is_opt = (row['exchange'] == 'NFO') and ('CE' in row['tradingsymbol'] or 'PE' in row['tradingsymbol'])
    is_fut = (row['exchange'] == 'NFO') and not is_opt
    stt = turnover * 0.000625 if is_opt else (turnover * 0.000125 if is_fut else turnover * 0.001)
    txn = turnover * 0.00053 if is_opt else (turnover * 0.00002 if is_fut else turnover * 0.0000345)
    gst = (brokerage + txn) * 0.18
    return brokerage + stt + txn + gst

SECTOR_MAP = {
    'ASHOKLEY': ['NSE:NIFTY AUTO', 'NSE:NIFTY 200'],
    'TATAMOTORS': ['NSE:NIFTY AUTO', 'NSE:NIFTY 50'],
    'M&M': ['NSE:NIFTY AUTO', 'NSE:NIFTY 50'],
    'MARUTI': ['NSE:NIFTY AUTO', 'NSE:NIFTY 50'],
    'HDFCBANK': ['NSE:NIFTY BANK', 'NSE:NIFTY 50'],
    'ICICIBANK': ['NSE:NIFTY BANK', 'NSE:NIFTY 50'],
    'SBIN': ['NSE:NIFTY PSU BANK', 'NSE:NIFTY BANK'],
    'INFY': ['NSE:NIFTY IT', 'NSE:NIFTY 50'],
    'TCS': ['NSE:NIFTY IT', 'NSE:NIFTY 50'],
    'RELIANCE': ['NSE:NIFTY ENERGY', 'NSE:NIFTY 50']
}

# --- 4. PORTFOLIO MANAGER ---
@st.fragment(run_every=2)
def render_portfolio_live():
    try:
        m = kite.margins(); c = m['equity']['available']['live_balance']; u = m['equity']['utilised']['debits']
        pos = kite.positions()['net']; total_pnl = 0
        if pos:
            df = pd.DataFrame(pos)
            df['Est. Tax'] = df.apply(calculate_tax_for_position, axis=1)
            df['Net P&L'] = df['pnl'] - df['Est. Tax']
            total_pnl = df['Net P&L'].sum()
        
        st.markdown('<div><span class="live-dot"></span> <b>LIVE</b></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Available", f"₹{c:,.0f}"); c2.metric("Used", f"₹{u:,.0f}"); c3.metric("Net P&L", f"₹{total_pnl:,.0f}", delta=f"{total_pnl:,.0f}")
        st.divider()
        if pos:
            cols = ['tradingsymbol', 'product', 'quantity', 'average_price', 'last_price', 'pnl', 'Est. Tax', 'Net P&L']
            st.dataframe(df[cols].style.format({'average_price':'{:.2f}', 'last_price':'{:.2f}', 'pnl':'{:.0f}', 'Est. Tax':'{:.0f}', 'Net P&L':'{:.0f}'}).background_gradient(subset=['Net P&L'], cmap='RdYlGn'), use_container_width=True)
    except: st.error("Portfolio fetch failed")

def render_portfolio():
    st.title("💼 Live Portfolio"); 
    if not kite: st.warning("⚠️ Connect Zerodha"); return
    render_portfolio_live()

# --- 5. TECHNICAL SCANNER ---
@st.fragment(run_every=2)
def render_sector_monitor(target):
    idx = ["NSE:NIFTY 50", "NSE:NIFTY BANK"]
    if target: idx = ["NSE:NIFTY 50"] + SECTOR_MAP.get(target, [])
    idx = list(set(idx))
    try:
        q = kite.quote(idx); data = []
        for s in idx:
            if s in q:
                d = q[s]; l = d['last_price']; c = d['ohlc']['close']
                data.append({"Index": s.replace("NSE:", ""), "LTP": l, "Prev": c, "Chg": l-c, "%": ((l-c)/c)*100})
        if data:
            st.markdown(f'<div><span class="live-dot"></span> <b>SECTOR ({target})</b></div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(data).style.format({"LTP":"{:.2f}", "Prev":"{:.2f}", "Chg":"{:.2f}", "%":"{:.2f}%"}).background_gradient(subset=['%'], cmap='RdYlGn', vmin=-2, vmax=2), use_container_width=True, hide_index=True)
    except: pass

def render_scanner():
    st.title("📈 Technical Scanner (Pro)")
    if not kite: st.warning("⚠️ Connect Zerodha"); return
    if "scan_sym" not in st.session_state: st.session_state.scan_sym = "ASHOKLEY"
    
    c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
    with c1: st.session_state.scan_sym = st.text_input("Symbol", st.session_state.scan_sym).upper()
    with c2: days = st.number_input("Days", 1, 1000, 5)
    with c3: 
        imap = {"Default": None, "1 Min": "minute", "60 Min": "60minute", "Daily": "day"}
        sel_int = st.selectbox("Interval", list(imap.keys()))
    with c4: st.write(""); run = st.button("Load Charts", type="primary")

    render_sector_monitor(st.session_state.scan_sym)
    st.divider()
    
    if run:
        try:
            with st.spinner("Fetching..."):
                tokens = []
                ie = pd.DataFrame(kite.instruments("NSE"))
                sr = ie[ie['tradingsymbol'] == st.session_state.scan_sym]
                if sr.empty: sr = ie[ie['tradingsymbol'] == get_correct_equity_symbol(st.session_state.scan_sym).replace("NSE:", "")]
                if not sr.empty: tokens.append({"label": f"SPOT: {st.session_state.scan_sym}", "token": sr.iloc[0]['instrument_token']})
                
                nfo = pd.DataFrame(kite.instruments("NFO"))
                f = nfo[(nfo['name'] == st.session_state.scan_sym) & (nfo['instrument_type'] == 'FUT')].copy()
                if not f.empty:
                    f['expiry'] = pd.to_datetime(f['expiry']); f = f.sort_values('expiry')
                    if len(f) >= 1: tokens.append({"label": f"NEAR: {f.iloc[0]['tradingsymbol']}", "token": f.iloc[0]['instrument_token']})
                    if len(f) >= 2: tokens.append({"label": f"NEXT: {f.iloc[1]['tradingsymbol']}", "token": f.iloc[1]['instrument_token']})

                if not tokens: st.error("Symbol not found"); return

                for item in tokens:
                    st.markdown(f"### {item['label']}")
                    manual = imap[sel_int]
                    interval = manual if manual else ("minute" if days <= 3 else ("60minute" if days <= 60 else "day"))
                    from_d = (datetime.datetime.now() - datetime.timedelta(days=days+6)).replace(hour=0, minute=0, second=0)
                    recs = kite.historical_data(item['token'], from_d, datetime.datetime.now(), interval)
                    df = pd.DataFrame(recs)
                    if df.empty: st.warning("No data"); continue
                    if interval == "day": df = df.tail(days)
                    
                    for s in [10, 20, 50, 100]: df[f'EMA_{s}'] = df['close'].ewm(span=s).mean()
                    df['VWAP'] = (df['close']*df['volume']).cumsum() / df['volume'].cumsum()
                    lc = df.iloc[-2] if len(df)>1 else df.iloc[-1]
                    p = (lc['high']+lc['low']+lc['close'])/3; r = lc['high']-lc['low']
                    fmt = '%d-%b %H:%M' if "minute" in interval else '%d-%b'
                    df['date_str'] = df['date'].dt.strftime(fmt)

                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df['date_str'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"))
                    for s, c in {10:'cyan', 20:'magenta', 50:'orange', 100:'white'}.items(): 
                        fig.add_trace(go.Scatter(x=df['date_str'], y=df[f'EMA_{s}'], line=dict(color=c, width=1), name=f"E{s}"))
                    fig.add_trace(go.Scatter(x=df['date_str'], y=df['VWAP'], line=dict(color='purple', width=2, dash='dot'), name="VWAP"))
                    
                    def add_line(val, col): fig.add_shape(type="line", x0=0, x1=1, xref="paper", y0=val, y1=val, line=dict(color=col, width=1, dash="dash"))
                    for v, c in [(p,"yellow"), (p+0.382*r,"green"), (p+0.618*r,"green"), (p+r,"lightgreen"), (p-0.382*r,"red"), (p-0.618*r,"red"), (p-r,"darkred")]: add_line(v, c)
                    
                    fig.update_layout(height=500, margin=dict(l=10,r=10,t=10,b=10), template="plotly_dark", xaxis_rangeslider_visible=False, xaxis_type='category', legend=dict(orientation="h", y=1.02))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    cols = st.columns(7)
                    cols[0].metric("P", f"{p:.2f}")
                    cols[1].metric("R1", f"{p+0.382*r:.2f}"); cols[2].metric("R2", f"{p+0.618*r:.2f}"); cols[3].metric("R3", f"{p+r:.2f}")
                    cols[4].metric("S1", f"{p-0.382*r:.2f}"); cols[5].metric("S2", f"{p-0.618*r:.2f}"); cols[6].metric("S3", f"{p-r:.2f}")
                    st.divider()
        except Exception as e: st.error(str(e))

# --- 6. MODULE: MARKET HEATMAP (V10.0 - IST DATE FIX) ---
def render_heatmap():
    st.title("🔥 Market Heatmap")
    if not kite: st.warning("⚠️ Connect Zerodha"); return
    
    c1, c2 = st.columns([2, 1])
    with c1: symbol = st.text_input("Symbol", "ASHOKLEY").upper()
    with c2: st.write(""); st.button("↻ Clear", on_click=lambda: st.cache_data.clear())
    
    if st.button("Analyze Order Flow", type="primary"):
        with st.spinner("Deep Scan (Slow for Accuracy)..."):
            try:
                # 1. Setup
                inst = get_instruments(); filtered = inst[(inst['name'] == symbol) & (inst['segment'] == 'NFO-OPT')].copy()
                if filtered.empty: st.error("No options"); return
                filtered['expiry'] = pd.to_datetime(filtered['expiry']); nearest = filtered.sort_values('expiry')['expiry'].iloc[0]
                opts = filtered[filtered['expiry'] == nearest].copy()
                
                # 2. Header
                u_key = get_correct_equity_symbol(symbol)
                nfo = pd.DataFrame(kite.instruments("NFO"))
                f = nfo[(nfo['name'] == symbol) & (nfo['instrument_type'] == 'FUT')].copy()
                f['expiry'] = pd.to_datetime(f['expiry']); f = f.sort_values('expiry').head(2)
                f_toks = f['instrument_token'].tolist()
                
                try: qh = kite.quote([u_key] + [int(t) for t in f_toks])
                except: qh = {}
                def gq(t): return qh.get(t) or qh.get(str(t)) or qh.get(int(t) if str(t).isdigit() else 0)
                
                spot_d = gq(u_key); ltp_spot = spot_d.get('last_price', opts['strike'].median()) if spot_d else opts['strike'].median()
                
                st.markdown("### 📊 Market Context")
                h1, h2, h3 = st.columns(3)
                h1.metric("Spot", f"₹{ltp_spot}")
                if len(f_toks)>=1: h2.metric("Near", f"₹{gq(f_toks[0]).get('last_price',0)}")
                if len(f_toks)>=2: h3.metric("Next", f"₹{gq(f_toks[1]).get('last_price',0)}")
                st.divider()

                # 3. Fetch
                opts['diff'] = abs(opts['strike'] - ltp_spot)
                top = opts.sort_values('diff').head(40).sort_values('strike')
                atm_s = top.iloc[0]['strike']; min_d = 9999
                for s in top['strike'].unique():
                    if abs(s-ltp_spot) < min_d: min_d = abs(s-ltp_spot); atm_s = s

                toks = [int(x) for x in top['instrument_token'].unique()]
                live = kite.quote(toks)
                hist_map = {}
                logs = []
                bar = st.progress(0)
                
                # CRITICAL: Define Today in IST (Strict String Comparison)
                ist = pytz.timezone('Asia/Kolkata')
                today_str = datetime.datetime.now(ist).strftime('%Y-%m-%d')
                
                for i, t in enumerate(toks):
                    try:
                        hist = kite.historical_data(t, datetime.datetime.now()-datetime.timedelta(days=10), datetime.datetime.now(), "day", oi=True)
                        found = False
                        
                        # Iterate Backwards
                        for c in reversed(hist):
                            c_date_str = pd.to_datetime(c['date']).tz_convert(ist).strftime('%Y-%m-%d')
                            
                            # SKIP TODAY
                            if c_date_str == today_str:
                                continue
                            
                            # FOUND PREV DAY
                            hist_map[t] = c['oi']
                            logs.append(f"✅ {t}: Used {c_date_str} (OI: {c['oi']})")
                            found = True
                            break
                        
                        if not found: logs.append(f"⚠️ {t}: No valid history found (All dates >= {today_str})")
                            
                    except Exception as e: logs.append(f"❌ {t}: {e}")
                    bar.progress((i+1)/len(toks))
                    time.sleep(0.4) 
                
                bar.empty()

                data = []
                for s in sorted(top['strike'].unique()):
                    r = {'Strike': s, 'ATM': (s==atm_s)}
                    for typ in ['CE', 'PE']:
                        row = top[(top['strike']==s) & (top['instrument_type']==typ)]
                        if not row.empty:
                            tk = row.iloc[0]['instrument_token']
                            d = live.get(int(tk)) or live.get(str(tk))
                            if d:
                                oi = d.get('oi', 0); ltp = d.get('last_price', 0); cl = d.get('ohlc', {}).get('close', 0)
                                poi = hist_map.get(tk, oi) # Fallback to 0 chg
                                
                                oichg = oi - poi
                                pchg = ltp - cl
                                
                                lbl = "Neutral"
                                if pchg>0 and oichg>0: lbl = "Long Buildup"
                                elif pchg<0 and oichg>0: lbl = "Short Buildup"
                                elif pchg<0 and oichg<0: lbl = "Long Unwinding"
                                elif pchg>0 and oichg<0: lbl = "Short Covering"
                                
                                r[f'{typ} Price'] = ltp; r[f'{typ} Label'] = lbl
                                r[f'{typ} OI'] = oi; r[f'{typ} Chg'] = oichg
                            else: r[f'{typ} Price']=0; r[f'{typ} Label']="-"; r[f'{typ} OI']=0; r[f'{typ} Chg']=0
                        else: r[f'{typ} Price']=0; r[f'{typ} Label']="-"; r[f'{typ} OI']=0; r[f'{typ} Chg']=0
                    data.append(r)
                
                rdf = pd.DataFrame(data)
                
                st.markdown("### 🌡️ OI Distribution")
                m = rdf.melt(id_vars=['Strike'], value_vars=['CE OI', 'PE OI'], var_name='Type', value_name='OI')
                if not m.empty:
                    fig = px.bar(m, x='Strike', y='OI', color='Type', barmode='group')
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 📋 Order Flow")
                def col_gen(v):
                    c = 'white'
                    if v=='Long Buildup': c='#00FF00'
                    elif v=='Short Buildup': c='#FF4444'
                    elif v=='Short Covering': c='#90EE90'
                    elif v=='Long Unwinding': c='#FF9999'
                    return f'color: {c}'
                
                def hl_atm(s): return ['background-color: #ffff00; color: black; font-weight: bold' if s.name in rdf[rdf['ATM']].index else '' for _ in s]

                st.dataframe(
                    rdf[['CE Label', 'CE Chg', 'CE Price', 'Strike', 'PE Price', 'PE Chg', 'PE Label']]
                    .style.map(col_gen, subset=['CE Label', 'PE Label']).apply(hl_atm, axis=1, subset=['Strike'])
                    .format({'CE Price':'{:.2f}', 'PE Price':'{:.2f}', 'Strike':'{:.0f}', 'CE Chg':'{:+.0f}', 'PE Chg':'{:+.0f}'}),
                    use_container_width=True, hide_index=True, height=800
                )
                
                with st.expander("🛠️ Debug Logs"): st.write(logs)

            except Exception as e: st.error(str(e))

# --- 7. TRADEBOOK ANALYZER ---
def render_tradebook():
    st.title("📜 F&O Tradebook"); 
    if not kite: st.warning("⚠️ Connect Zerodha"); return
    
    c1, c2, c3 = st.columns([1,1,1])
    with c1: ul = st.file_uploader("Upload CSV", type=['csv'])
    with c2: rs = st.text_input("Root Symbol", "ASHOKLEY").upper()
    with c3: st.write(""); trig = st.button("Run Analysis", type="primary")
    
    if "tb_data" not in st.session_state: st.session_state.tb_data = None
    if "editor_key" not in st.session_state: st.session_state.editor_key = 0
    
    if trig:
        st.session_state.tb_inputs = {"file": ul, "symbol": rs}
        run_tb_calc()
    
    if st.session_state.tb_data: render_tb_live()

def run_tb_calc():
    try:
        inp = st.session_state.tb_inputs; ul = inp["file"]; rs = inp["symbol"]
        with st.spinner("Processing..."):
            def cs(s): return re.sub(r'[^A-Z0-9]', '', str(s).upper())
            def ct(t): return pd.to_datetime(t).tz_convert('Asia/Kolkata').tz_localize(None) if pd.to_datetime(t).tzinfo else pd.to_datetime(t)
            td = pd.Timestamp.now().normalize()
            
            eqs = get_correct_equity_symbol(rs)
            idf = pd.DataFrame(kite.instruments("NFO")); idf['clean'] = idf['tradingsymbol'].apply(cs)
            si = idf[idf['name'] == rs]
            
            cmap = {}
            for _, r in si.iterrows(): cmap[r['clean']] = {'token': int(r['instrument_token']), 'type': r['instrument_type'], 'lot': int(r['lot_size']), 'expiry': r['expiry']}
            
            trades = []
            if ul:
                ul.seek(0); dh = pd.read_csv(ul)
                dh.columns = [c.lower().strip().replace(" ", "_").replace("-", "_") for c in dh.columns]
                sc = next((c for c in dh.columns if 'symbol' in c), None)
                if sc:
                    dh['clean'] = dh[sc].apply(cs); dh = dh[dh['clean'].str.startswith(cs(rs))]
                    if not dh.empty:
                        dh['dt'] = pd.to_datetime(dh['trade_date'], dayfirst=True).apply(ct)
                        dh = dh[dh['dt'] < td]
                        for _, r in dh.iterrows(): trades.append({'s': r['clean'], 't': r['dt'], 'q': abs(r['quantity']), 'p': r['price'], 'ty': r['trade_type'].upper()})
            
            ords = kite.orders()
            if ords:
                dl = pd.DataFrame(ords)
                if not dl.empty:
                    dl['clean'] = dl['tradingsymbol'].apply(cs); dl = dl[(dl['clean'].str.startswith(cs(rs))) & (dl['status'] == 'COMPLETE')]
                    for _, r in dl.iterrows(): trades.append({'s': r['clean'], 't': ct(r['order_timestamp']), 'q': abs(r['quantity']), 'p': r['average_price'], 'ty': r['transaction_type']})
            
            if not trades: st.warning("No trades"); return
            
            dm = pd.DataFrame(trades); op = []; ledg = {}
            for c in dm['s'].unique():
                tr = dm[dm['s'] == c]
                buy = tr[tr['ty']=='BUY'].sort_values('t').to_dict('records')
                sell = tr[tr['ty']=='SELL'].sort_values('t').to_dict('records')
                
                while buy and sell:
                    b = buy[0]; s = sell[0]; m = min(b['q'], s['q'])
                    b['q'] -= m; s['q'] -= m
                    if b['q']==0: buy.pop(0)
                    if s['q']==0: sell.pop(0)
                
                for x in buy: x.update({'contract': c, 'side': 'LONG'}); op.append(x)
                for x in sell: x.update({'contract': c, 'side': 'SHORT'}); op.append(x)
                ledg[c] = [f"{t['ty']} {t['q']}" for _, t in tr.iterrows()]
            
            st.session_state.tb_data = {'op': pd.DataFrame(op), 'ledg': ledg, 'cmap': cmap, 'rs': rs, 'agg': pd.DataFrame()}
    except Exception as e: st.error(str(e))

@st.fragment(run_every=2)
def render_tb_live():
    d = st.session_state.tb_data; rs = d['rs']; cmap = d['cmap']; op = d['op'].copy()
    try:
        eqs = get_correct_equity_symbol(rs)
        syms = list(cmap.keys()); f = [s for s in syms if cmap[s]['type']=='FUT']
        f.sort(key=lambda s: pd.to_datetime(cmap[s].get('expiry', '2099-01-01'))); top3 = f[:3]
        
        otoks = [cmap[s]['token'] for s in op['contract'].unique() if s in cmap] if not op.empty else []
        ftoks = [cmap[s]['token'] for s in top3]
        qk = [eqs] + ftoks + otoks
        q = kite.quote(list(set(qk)))
        
        def gq(t): return q.get(t) or q.get(str(t)) or q.get(int(t) if str(t).isdigit() else 0)
        
        eqp = gq(eqs).get('last_price', 0) if gq(eqs) else 0
        fd = [(s, gq(cmap[s]['token']).get('last_price', 0) if gq(cmap[s]['token']) else 0) for s in top3]
        
        if not op.empty:
            op['LTP'] = op['contract'].apply(lambda x: gq(cmap[x]['token']).get('last_price', 0) if gq(cmap[x]['token']) else 0)
            op['LTP'] = op.apply(lambda r: eqp if r['LTP']==0 and cmap[r['contract']]['type']=='FUT' else r['LTP'], axis=1)
            op['Unrealized'] = op.apply(lambda r: (r['LTP']-r['p'])*r['q'] if r['side']=='LONG' else (r['p']-r['LTP'])*r['q'], axis=1)
            
            ar = []
            for (c, s), r in op.groupby(['contract', 'side']):
                ar.append({'Contract': c, 'Side': s, 'Total Qty': r['q'].sum(), 'Avg Price': (r['p']*r['q']).sum()/r['q'].sum(), 'LTP': r['LTP'].iloc[0], 'Lot Size': cmap[c]['lot']})
            st.session_state.tb_data['agg'] = pd.DataFrame(ar)
        
        st.markdown('<div><span class="live-dot"></span> <b>LIVE</b></div>', unsafe_allow_html=True)
        un = op['Unrealized'].sum() if not op.empty else 0
        k1, k2, k3 = st.columns(3)
        k1.metric("Realized", "₹0"); k2.metric("Floating", f"₹{un:,.0f}", delta=f"{un:,.0f}"); k3.metric("Spot", f"₹{eqp}")
        if fd:
            fc = st.columns(len(fd))
            for i, (n, p) in enumerate(fd): fc[i].metric(n, f"₹{p}")
        st.divider()
        if not op.empty:
            st.markdown("### 🔓 Open Positions")
            v = op[['contract', 'side', 't', 'q', 'p', 'LTP', 'Unrealized']].copy()
            st.dataframe(v.style.format({'p':'{:.2f}', 'LTP':'{:.2f}', 'Unrealized':'{:.0f}'}).background_gradient(subset=['Unrealized'], cmap='RdYlGn'), use_container_width=True)
        else: st.success("No open positions")
    except: st.error("Live update error")

    if 'agg' in st.session_state.tb_data and not st.session_state.tb_data['agg'].empty:
        st.divider()
        c1, c2 = st.columns([4, 1])
        with c1: st.markdown("### 🧮 Scenario Planner")
        with c2: 
            if st.button("Reset Values"): st.session_state.editor_key += 1; st.rerun()
        
        ed = st.session_state.tb_data['agg'].copy()
        if "Add Lots" not in ed.columns: ed["Add Lots"] = 0
        if "Buy At" not in ed.columns: ed["Buy At"] = ed["LTP"]
        if "Target" not in ed.columns: ed["Target"] = ed["LTP"]
        
        res = st.data_editor(ed, key=f"scen_{st.session_state.editor_key}", use_container_width=True, disabled=["Contract", "Side", "Total Qty", "Avg Price", "LTP", "Lot Size"], column_config={"Add Lots": st.column_config.NumberColumn(min_value=0, step=1, required=True)})
        
        if not res.empty:
            out = []
            for _, r in res.iterrows():
                if r['Add Lots'] > 0 or r['Target'] != r['LTP']:
                    nq = r['Total Qty'] + (r['Add Lots']*r['Lot Size'])
                    nav = ((r['Total Qty']*r['Avg Price']) + (r['Add Lots']*r['Lot Size']*r['Buy At'])) / nq
                    pnl = (r['Target'] - nav)*nq if r['Side']=='LONG' else (nav - r['Target'])*nq
                    out.append({'Contract': r['Contract'], 'New Avg': nav, 'Target': r['Target'], 'Proj P&L': pnl})
            if out: st.dataframe(pd.DataFrame(out).style.format({'New Avg':'{:.2f}', 'Proj P&L':'{:.0f}'}).background_gradient(subset=['Proj P&L'], cmap='RdYlGn'), use_container_width=True)
        
        with st.expander("🕵️ Debug"):
            if d['ledg']: s = st.selectbox("Contract", list(d['ledg'].keys())); st.write(d['ledg'][s])

# --- 8. MAIN ---
if app_mode == "Portfolio Manager": render_portfolio()
elif app_mode == "Market Heatmap": render_heatmap()
elif app_mode == "Technical Scanner": render_scanner()
elif app_mode == "Tradebook Analyzer": render_tradebook()
