import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from nifty500 import NIFTY500_STOCKS, get_all_symbols, get_symbols_by_cap, get_symbols_by_sector

# ─────────────────────────────────────────────────────────
# YAHOO FINANCE  — direct HTTP, no yfinance library needed
# ─────────────────────────────────────────────────────────
YF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

TIMEFRAME_CFG = {
    "5m":  {"interval": "5m",   "range": "5d"},
    "15m": {"interval": "15m",  "range": "60d"},
    "30m": {"interval": "30m",  "range": "60d"},
    "1h":  {"interval": "60m",  "range": "730d"},
    "1D":  {"interval": "1d",   "range": "1y"},
    "1W":  {"interval": "1wk",  "range": "2y"},
    "1M":  {"interval": "1mo",  "range": "5y"},
}

_session = None


def get_session():
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(YF_HEADERS)
    return _session


def fetch_stock_data(symbol, timeframe="1W"):
    cfg = TIMEFRAME_CFG.get(timeframe, TIMEFRAME_CFG["1W"])
    ticker = symbol + ".NS"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "interval":          cfg["interval"],
        "range":             cfg["range"],
        "includePrePost":    "false",
        "events":            "div,splits",
    }
    sess = get_session()
    try:
        resp = sess.get(url, params=params, timeout=12)
        if resp.status_code != 200:
            # Try query2 as fallback
            resp = sess.get(url.replace("query1", "query2"),
                            params=params, timeout=12)
        if resp.status_code != 200:
            return None

        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None

        r = result[0]
        timestamps = r.get("timestamp", [])
        quote = r.get("indicators", {}).get("quote", [{}])[0]

        opens = quote.get("open",   [])
        highs = quote.get("high",   [])
        lows = quote.get("low",    [])
        closes = quote.get("close",  [])
        volumes = quote.get("volume", [])

        if not closes or len(closes) < 15:
            return None

        df = pd.DataFrame({
            "Open":   opens,
            "High":   highs,
            "Low":    lows,
            "Close":  closes,
            "Volume": volumes,
        }, index=pd.to_datetime(timestamps, unit="s"))

        # Clean up
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
        df = df[df["Close"] > 0]
        df.sort_index(inplace=True)

        return df if len(df) >= 15 else None

    except Exception as e:
        print(f"  [fetch error] {symbol}: {e}")
        return None


# ─── SuperTrend ────────────────────────────────────────────
def calculate_supertrend(df, period=7, multiplier=3.0):
    df = df.copy()
    hl2 = (df["High"] + df["Low"]) / 2
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"]-df["Low"], (df["High"]-pc).abs(),
                   (df["Low"]-pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    ub_b = (hl2 + multiplier * atr).values.copy()
    lb_b = (hl2 - multiplier * atr).values.copy()
    cl = df["Close"].values
    ub = ub_b.copy()
    lb = lb_b.copy()
    st = [True] * len(df)
    for i in range(1, len(df)):
        ub[i] = ub_b[i] if (ub_b[i] < ub[i-1] or cl[i-1]
                            > ub[i-1]) else ub[i-1]
        lb[i] = lb_b[i] if (lb_b[i] > lb[i-1] or cl[i-1]
                            < lb[i-1]) else lb[i-1]
        st[i] = (cl[i] >= lb[i]) if st[i-1] else (cl[i] > ub[i])
    df["supertrend"] = st
    df["st_upper"] = ub
    df["st_lower"] = lb
    df["supertrend_line"] = [lb[i] if st[i] else ub[i] for i in range(len(df))]
    return df


def detect_supertrend_breakout(df, lookback=3):
    if len(df) < lookback + 2:
        return None
    for i in range(1, lookback + 1):
        p = bool(df["supertrend"].iloc[-(i+1)])
        c = bool(df["supertrend"].iloc[-i])
        if not p and c:
            return "bullish"
        if p and not c:
            return "bearish"
    return None

# ─── RSI ───────────────────────────────────────────────────


def calculate_rsi(close, period=14):
    d = close.diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    ag = g.ewm(alpha=1/period, adjust=False).mean()
    al = l.ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + ag / al.replace(0, np.nan)))

# ─── ADX ───────────────────────────────────────────────────


def calculate_adx(df, period=14):
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    ph = h.shift(1)
    pl = l.shift(1)
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    dmp = pd.Series(
        np.where((h-ph) > (pl-l), np.maximum(h-ph, 0), 0), index=df.index)
    dmm = pd.Series(
        np.where((pl-l) > (h-ph), np.maximum(pl-l, 0), 0), index=df.index)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    dip = 100*dmp.ewm(alpha=1/period, adjust=False).mean()/atr
    dim = 100*dmm.ewm(alpha=1/period, adjust=False).mean()/atr
    dx = 100*(dip-dim).abs()/(dip+dim).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean(), dip, dim


def ema(close, period):
    return close.ewm(span=period, adjust=False).mean()


def check_volume_spike(volume, lookback=20, threshold=1.5):
    if len(volume) < lookback+1:
        return False, 0.0
    avg = volume.iloc[-(lookback+1):-1].mean()
    curr = volume.iloc[-1]
    r = curr/avg if avg > 0 else 0
    return r >= threshold, round(float(r), 2)


# ─── MAIN SCREENER ─────────────────────────────────────────
def run_screener(
    segment="All", timeframe="1W", signal_filter="both", lookback=3,
    st_period=7, st_multiplier=3.0,
    use_rsi=False,    rsi_period=14, rsi_min=45,  rsi_max=70, rsi_bear_min=30, rsi_bear_max=55,
    use_adx=False,    adx_period=14, adx_min=20,
    use_ema=False,    ema_fast=20,   ema_slow=50,
    use_volume=False, vol_lookback=20, vol_threshold=1.5,
    progress_callback=None
):
    caps = ["Large Cap", "Mid Cap", "Small Cap"]
    sectors = sorted(set(d["sector"] for d in NIFTY500_STOCKS.values()))
    if segment == "All":
        symbols = get_all_symbols()
    elif segment in caps:
        symbols = get_symbols_by_cap(segment)
    elif segment in sectors:
        symbols = get_symbols_by_sector(segment)
    else:
        symbols = get_all_symbols()

    results = []
    total = len(symbols)
    ok = skip = no_breakout = 0
    active = [n for n, v in [
        ("RSI", use_rsi), ("ADX", use_adx), ("EMA", use_ema), ("VOL", use_volume)] if v]
    print(
        f"\n[Screener] {total} stocks | TF={timeframe} | ST({st_period},{st_multiplier}) | Filters={active} | lookback={lookback}")

    for idx, symbol in enumerate(symbols):
        if progress_callback:
            progress_callback(idx+1, total, symbol)

        # Throttle: 1 request every ~0.2s to avoid Yahoo rate-limiting
        time.sleep(0.2)

        try:
            df = fetch_stock_data(symbol, timeframe)
            min_rows = max(int(st_period),
                           int(adx_period) if use_adx else 1,
                           int(rsi_period) if use_rsi else 1,
                           int(ema_slow) if use_ema else 1) + 5
            if df is None or len(df) < min_rows:
                skip += 1
                continue
            ok += 1

            df = calculate_supertrend(df, int(st_period), float(st_multiplier))

            # Debug first 3 stocks
            if ok <= 3:
                print(f"  [DEBUG {symbol}] rows={len(df)} close={df['Close'].iloc[-1]:.2f} "
                      f"ST={list(df['supertrend'].tail(6).astype(int).values)}")

            breakout = detect_supertrend_breakout(df, lookback=lookback)
            if breakout is None:
                no_breakout += 1
                continue
            if signal_filter == "bullish" and breakout != "bullish":
                continue
            if signal_filter == "bearish" and breakout != "bearish":
                continue

            lc = float(df["Close"].iloc[-1])
            fr = {}
            passed = True

            if use_rsi:
                rv = round(
                    float(calculate_rsi(df["Close"], int(rsi_period)).iloc[-1]), 1)
                fr["rsi"] = rv
                ok_r = (float(rsi_min) <= rv <= float(rsi_max)) if breakout == "bullish" \
                    else (float(rsi_bear_min) <= rv <= float(rsi_bear_max))
                fr["rsi_pass"] = ok_r
                if not ok_r:
                    passed = False

            if use_adx:
                adx_s, dip, dim = calculate_adx(df, int(adx_period))
                av = round(float(adx_s.iloc[-1]), 1)
                fr["adx"] = av
                fr["di_plus"] = round(float(dip.iloc[-1]), 1)
                fr["di_minus"] = round(float(dim.iloc[-1]), 1)
                ok_a = av >= float(adx_min)
                fr["adx_pass"] = ok_a
                if not ok_a:
                    passed = False

            if use_ema:
                ema_val = round(
                    float(ema(df["Close"], int(ema_fast)).iloc[-1]), 2)
                fr["ema_val"] = ema_val
                ok_e = (lc > ema_val) if breakout == "bullish" else (
                    lc < ema_val)
                fr["ema_pass"] = ok_e
                if not ok_e:
                    passed = False

            if use_volume:
                vs, vr = check_volume_spike(df["Volume"], int(
                    vol_lookback), float(vol_threshold))
                fr["vol_ratio"] = vr
                fr["vol_pass"] = vs
                if not vs:
                    passed = False

            if not passed:
                continue

            info = NIFTY500_STOCKS.get(symbol, {})
            curr_open = float(df["Open"].iloc[-1])
            prev_close = float(df["Close"].iloc[-2])
            # Use previous-close to current-close for % change
            pchg = ((lc - prev_close) / prev_close) * \
                100 if prev_close > 0 else 0.0
            # If current candle is still forming (|pchg| < 0.05), use open→close of current candle
            if abs(pchg) < 0.05 and curr_open > 0:
                pchg = ((lc - curr_open) / curr_open) * 100
            chart_df = df.tail(40).reset_index()
            chart_df["Date"] = chart_df.iloc[:, 0].astype(str)
            candles = chart_df[["Date", "Open", "High", "Low", "Close", "Volume",
                                "supertrend_line", "supertrend"]].to_dict(orient="records")

            print(f"  [✓] {symbol} → {breakout.upper()} | ₹{lc:.2f}")
            results.append({
                "symbol": symbol, "name": info.get("name", symbol),
                "sector": info.get("sector", "Unknown"), "cap": info.get("cap", "Unknown"),
                "signal": breakout, "close": round(lc, 2),
                "high": round(float(df["High"].iloc[-1]), 2), "low": round(float(df["Low"].iloc[-1]), 2),
                "volume": int(df["Volume"].iloc[-1]), "price_change": round(pchg, 2),
                "supertrend_line": round(float(df["supertrend_line"].iloc[-1]), 2),
                "filters": fr, "candles": candles,
            })

        except Exception as e:
            print(f"  [error] {symbol}: {e}")
            continue

    print(
        f"\n[Done] Fetched={ok} | Skipped={skip} | NoBreakout={no_breakout} | Found={len(results)}")
    results.sort(key=lambda x: (
        0 if x["signal"] == "bullish" else 1, -x["price_change"]))
    return results


def get_segments():
    return {
        "caps":    ["All", "Large Cap", "Mid Cap", "Small Cap"],
        "sectors": sorted(set(d["sector"] for d in NIFTY500_STOCKS.values()))
    }
