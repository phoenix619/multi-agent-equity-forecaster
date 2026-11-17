"""Daily prices via yfinance, cut off at an as-of date to avoid look-ahead."""
from datetime import datetime, timedelta

import pandas as pd

from .. import cache

DAY = 86400


def get_prices(ticker: str, as_of: str, days: int = 400) -> pd.DataFrame:
    """Adjusted daily OHLCV for the `days` calendar days ending on as_of (inclusive)."""
    end = datetime.strptime(as_of, "%Y-%m-%d").date()
    start = end - timedelta(days=days)

    def fetch():
        import yfinance as yf
        df = yf.download(ticker, start=str(start), end=str(end + timedelta(days=1)),
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        return df.to_dict(orient="list")

    data = cache.cached("prices", [ticker, str(start), str(end)], fetch, ttl=30 * DAY)
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"])
    return df[df["Date"] <= pd.Timestamp(end)].set_index("Date")


def excess_return(ticker: str, start: str, end: str, benchmark: str = "SPY") -> dict:
    """Return of ticker minus benchmark between two dates (labels for scoring)."""
    s = get_prices(ticker, end, days=(datetime.strptime(end, "%Y-%m-%d")
                                       - datetime.strptime(start, "%Y-%m-%d")).days + 10)
    b = get_prices(benchmark, end, days=(datetime.strptime(end, "%Y-%m-%d")
                                          - datetime.strptime(start, "%Y-%m-%d")).days + 10)
    s = s[s.index >= pd.Timestamp(start)]["Close"]
    b = b[b.index >= pd.Timestamp(start)]["Close"]
    sr = float(s.iloc[-1] / s.iloc[0] - 1)
    br = float(b.iloc[-1] / b.iloc[0] - 1)
    return {"ticker": ticker, "stock_return": round(sr, 4),
            "benchmark_return": round(br, 4), "excess": round(sr - br, 4),
            "outperformed": sr > br}
