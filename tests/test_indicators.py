import numpy as np
import pandas as pd
import pytest

from forecaster import indicators as ind


def _series(values):
    return pd.Series(values, index=pd.date_range("2025-01-01", periods=len(values), freq="B"))


def test_sma():
    s = ind.sma(_series([1, 2, 3, 4, 5]), 3)
    assert s.iloc[-1] == 4 and np.isnan(s.iloc[0])


def test_rsi_textbook_wilder_example():
    # Classic Wilder/StockCharts worked example: RSI(14) after 15 closes.
    close = _series([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
                     45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28])
    assert ind.rsi(close).iloc[-1] == pytest.approx(70.5, abs=0.6)


def test_rsi_bounds_and_extremes():
    up = ind.rsi(_series(np.arange(1, 60, dtype=float)))
    assert up.dropna().iloc[-1] == 100.0
    down = ind.rsi(_series(np.arange(60, 1, -1, dtype=float)))
    assert down.dropna().iloc[-1] == pytest.approx(0.0, abs=1e-6)
    rng = np.random.default_rng(0)
    r = ind.rsi(_series(100 + rng.normal(0, 1, 200).cumsum())).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_rsi_needs_enough_data():
    assert ind.rsi(_series([1.0, 2.0, 3.0])).isna().all()


def test_atr_constant_range():
    n = 40
    close = _series([100.0] * n)
    high, low = close + 1, close - 1
    assert ind.atr(high, low, close).iloc[-1] == pytest.approx(2.0)


def test_bollinger_ordering():
    rng = np.random.default_rng(1)
    lo, mid, up = ind.bollinger(_series(100 + rng.normal(0, 1, 60).cumsum()))
    assert (lo.dropna() <= mid.dropna()).all() and (mid.dropna() <= up.dropna()).all()


def test_macd_flat_series_is_zero():
    line, sig = ind.macd(_series([50.0] * 80))
    assert line.iloc[-1] == pytest.approx(0) and sig.iloc[-1] == pytest.approx(0)


def _ohlc(close):
    close = _series(close)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close})


def test_technical_summary_uptrend_trade_levels():
    df = _ohlc(np.linspace(100, 160, 260))
    out = ind.technical_summary(df)
    lo, hi = out["paper_trade"]["entry_zone"]
    assert lo < hi
    assert out["paper_trade"]["stop_loss"] < lo          # stop below entry
    assert out["paper_trade"]["target"] > hi             # target above entry
    assert out["trend"].startswith("uptrend")


def test_technical_summary_downtrend_flags_timing():
    out = ind.technical_summary(_ohlc(np.linspace(160, 100, 260)))
    assert "no clear uptrend" in out["signal"]


def test_technical_summary_short_history_errors():
    assert "error" in ind.technical_summary(_ohlc(np.linspace(100, 110, 30)))


def test_target_capped_in_sharp_downtrend():
    close = np.concatenate([np.linspace(200, 210, 200), np.linspace(210, 150, 60)])
    out = ind.technical_summary(_ohlc(close))
    lo, hi = out["paper_trade"]["entry_zone"]
    atr = out["atr14"]
    assert out["paper_trade"]["target"] <= hi + 4 * atr + 0.01
    assert out["paper_trade"]["target"] > hi
