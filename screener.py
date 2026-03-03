import yfinance as yf
import pandas as pd
import numpy as np
from nifty500 import NIFTY500_STOCKS, get_all_symbols, get_symbols_by_cap, get_symbols_by_sector


def calculate_supertrend(df, period=7, multiplier=3):
    df = df.copy()
    hl2 = (df['High'] + df['Low']) / 2
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    upper = upper_basic.copy()
    lower = lower_basic.copy()
    supertrend = [True] * len(df)

    closes = df['Close'].values
    ub = upper.values.copy()
    lb = lower.values.copy()
    ub_basic = upper_basic.values
    lb_basic = lower_basic.values

    for i in range(1, len(df)):
        # Final upper band
        if ub_basic[i] < ub[i-1] or closes[i-1] > ub[i-1]:
            ub[i] = ub_basic[i]
        else:
            ub[i] = ub[i-1]

        # Final lower band
        if lb_basic[i] > lb[i-1] or closes[i-1] < lb[i-1]:
            lb[i] = lb_basic[i]
        else:
            lb[i] = lb[i-1]

        # Trend
        if supertrend[i-1]:
            supertrend[i] = closes[i] >= lb[i]
        else:
            supertrend[i] = closes[i] > ub[i]

    df['supertrend'] = supertrend
    df['st_upper'] = ub
    df['st_lower'] = lb
    df['supertrend_line'] = [lb[i] if supertrend[i] else ub[i]
                             for i in range(len(df))]
    return df


def detect_breakout(df, lookback=3):
    """Check last N candles for a SuperTrend crossover."""
    if len(df) < lookback + 2:
        return None
    for i in range(1, lookback + 1):
        prev = df.iloc[-(i+1)]
        curr = df.iloc[-i]
        if not prev['supertrend'] and curr['supertrend']:
            return 'bullish'
        elif prev['supertrend'] and not curr['supertrend']:
            return 'bearish'
    return None


TIMEFRAME_MAP = {
    "1D":  {"interval": "1d",  "period": "1y"},
    "1W":  {"interval": "1wk", "period": "2y"},
    "1M":  {"interval": "1mo", "period": "5y"},
}


def fetch_stock_data(symbol, timeframe="1W"):
    tf = TIMEFRAME_MAP.get(timeframe, TIMEFRAME_MAP["1W"])
    try:
        ticker = yf.Ticker(symbol + ".NS")
        df = ticker.history(interval=tf["interval"], period=tf["period"])
        if df.empty:
            return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.dropna(inplace=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        print(f"  [fetch error] {symbol}: {e}")
        return None


def run_screener(segment="All", timeframe="1W", signal_filter="both",
                 st_period=7, st_multiplier=3, lookback=3,
                 progress_callback=None):

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
    ok_count = 0
    no_data_count = 0

    print(
        f"\n[Screener] {total} stocks | TF={timeframe} | ST({st_period},{st_multiplier}) | lookback={lookback} candles")

    for idx, symbol in enumerate(symbols):
        if progress_callback:
            progress_callback(idx + 1, total, symbol)

        try:
            df = fetch_stock_data(symbol, timeframe)
            if df is None or len(df) < int(st_period) + 5:
                no_data_count += 1
                continue

            ok_count += 1
            df = calculate_supertrend(df, period=int(
                st_period), multiplier=float(st_multiplier))
            breakout = detect_breakout(df, lookback=lookback)

            if breakout is None:
                continue
            if signal_filter == "bullish" and breakout != "bullish":
                continue
            if signal_filter == "bearish" and breakout != "bearish":
                continue

            info = NIFTY500_STOCKS.get(symbol, {})
            last = df.iloc[-1]
            prev = df.iloc[-2]
            price_change = (
                (last['Close'] - prev['Close']) / prev['Close']) * 100

            chart_df = df.tail(40).reset_index()
            chart_df['Date'] = chart_df['Date'].astype(str)
            candles = chart_df[['Date', 'Open', 'High', 'Low', 'Close',
                                'Volume', 'supertrend_line', 'supertrend']].to_dict(orient='records')

            print(
                f"  [FOUND] {symbol} → {breakout.upper()} | ₹{last['Close']:.2f}")

            results.append({
                "symbol": symbol,
                "name": info.get("name", symbol),
                "sector": info.get("sector", "Unknown"),
                "cap": info.get("cap", "Unknown"),
                "signal": breakout,
                "close": round(float(last['Close']), 2),
                "high": round(float(last['High']), 2),
                "low": round(float(last['Low']), 2),
                "volume": int(last['Volume']),
                "price_change": round(price_change, 2),
                "supertrend_line": round(float(last['supertrend_line']), 2),
                "candles": candles,
            })

        except Exception as e:
            print(f"  [error] {symbol}: {e}")
            continue

    print(
        f"\n[Done] Scanned={ok_count} | NoData={no_data_count} | Breakouts={len(results)}")
    results.sort(key=lambda x: (
        0 if x['signal'] == 'bullish' else 1, -x['price_change']))
    return results


def get_segments():
    caps = ["All", "Large Cap", "Mid Cap", "Small Cap"]
    sectors = sorted(set(d["sector"] for d in NIFTY500_STOCKS.values()))
    return {"caps": caps, "sectors": sectors}
