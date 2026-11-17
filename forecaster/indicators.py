"""Technical indicators, computed in Python (never by the model).

Used for timing, not direction: the summary gives trend, momentum and volatility
plus an entry zone, stop-loss and target for a paper swing trade.
"""
import pandas as pd


def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n).mean()


def _wilder(x: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing: seed with the simple mean of the first n valid values,
    then avg = (previous * (n - 1) + current) / n. Matches the standard RSI/ATR."""
    vals = x.to_numpy(dtype=float)
    out = [float("nan")] * len(vals)
    valid = [i for i, v in enumerate(vals) if v == v]
    if len(valid) < n:
        return pd.Series(out, index=x.index)
    start = valid[n - 1]
    avg = sum(vals[i] for i in valid[:n]) / n
    out[start] = avg
    for i in range(start + 1, len(vals)):
        avg = (avg * (n - 1) + vals[i]) / n
        out[i] = avg
    return pd.Series(out, index=x.index)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder smoothing."""
    delta = close.diff()
    gain = _wilder(delta.clip(lower=0).where(delta.notna()), n)
    loss = _wilder((-delta.clip(upper=0)).where(delta.notna()), n)
    out = 100 - 100 / (1 + gain / loss)
    return out.where(loss != 0, 100.0).where(gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std()
    return mid - k * sd, mid, mid + k * sd


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Average True Range with Wilder smoothing."""
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    tr = tr.where(prev.notna())
    return _wilder(tr, n)


def technical_summary(df: pd.DataFrame) -> dict:
    """Plain-language summary from daily OHLC data (needs ~200 rows for the
    200-day average). Returns numbers plus short text the model can read."""
    if len(df) < 60:
        return {"error": "not enough price history", "rows": len(df)}
    close, high, low = df["Close"], df["High"], df["Low"]
    last = float(close.iloc[-1])
    s20, s50 = sma(close, 20), sma(close, 50)
    s200 = sma(close, 200)
    r = rsi(close)
    m, sig = macd(close)
    lo, mid, up = bollinger(close)
    a = atr(high, low, close)

    sma20, sma50 = float(s20.iloc[-1]), float(s50.iloc[-1])
    sma200 = float(s200.iloc[-1]) if s200.notna().iloc[-1] else None
    rsi_v = float(r.iloc[-1])
    atr_v = float(a.iloc[-1])
    macd_above = bool(m.iloc[-1] > sig.iloc[-1])
    hist_now = float(m.iloc[-1] - sig.iloc[-1])
    hist_prev = float(m.iloc[-2] - sig.iloc[-2])
    upper, lower = float(up.iloc[-1]), float(lo.iloc[-1])

    above50 = last > sma50
    above200 = (last > sma200) if sma200 is not None else None
    uptrend = above50 and (sma200 is None or sma50 > sma200)

    trend = ("uptrend (price above 50-day average"
             + (", 50-day above 200-day)" if sma200 and sma50 > sma200 else ")")
             if uptrend else
             "weak or downtrend (price below 50-day average)" if not above50
             else "mixed trend")
    momentum = (f"RSI {rsi_v:.0f}"
                + (" (overbought zone)" if rsi_v >= 70 else
                   " (approaching overbought)" if rsi_v >= 65 else
                   " (oversold zone)" if rsi_v <= 30 else "")
                + f"; MACD {'above' if macd_above else 'below'} signal line, "
                  f"{'widening' if abs(hist_now) > abs(hist_prev) else 'narrowing'}")
    band_pos = (last - lower) / (upper - lower) if upper > lower else 0.5
    volatility = (f"price at {band_pos:.0%} of Bollinger range; "
                  f"ATR ${atr_v:.2f} ({atr_v / last:.1%} of price)")

    stretched = rsi_v >= 65 or band_pos >= 0.9
    entry_low = min(sma20, last)
    entry_high = last if not stretched else sma20 + 0.25 * atr_v
    lo_e, hi_e = min(entry_low, entry_high), max(entry_low, entry_high)
    if hi_e - lo_e < 0.25 * atr_v:
        hi_e = lo_e + 0.25 * atr_v
    entry_zone = (round(lo_e, 2), round(hi_e, 2))
    stop = round(entry_zone[0] - 2 * atr_v, 2)
    # Upper Bollinger band, but never beyond 4 x ATR above entry (2:1 reward to the
    # 2 x ATR stop); in a sharp downtrend the band can sit 50% above the price.
    target = round(min(max(upper, entry_zone[1] + 2 * atr_v), entry_zone[1] + 4 * atr_v), 2)
    signal = ("uptrend intact but short-term stretched: wait for a pullback toward "
              "the 20-day average" if uptrend and stretched else
              "uptrend intact and not stretched: entry near current price is reasonable"
              if uptrend else
              "no clear uptrend: timing is unfavorable")

    return {
        "as_of_close": round(last, 2),
        "date": str(df.index[-1].date()) if hasattr(df.index[-1], "date") else None,
        "sma20": round(sma20, 2), "sma50": round(sma50, 2),
        "sma200": round(sma200, 2) if sma200 else None,
        "rsi14": round(rsi_v, 1), "atr14": round(atr_v, 2),
        "macd_above_signal": macd_above,
        "bollinger_upper": round(upper, 2), "bollinger_lower": round(lower, 2),
        "trend": trend, "momentum": momentum, "volatility": volatility,
        "signal": signal,
        "paper_trade": {"entry_zone": entry_zone, "stop_loss": stop, "target": target,
                        "note": "stop = 2 x ATR below entry zone; target capped at 4 x ATR; paper only"},
    }
