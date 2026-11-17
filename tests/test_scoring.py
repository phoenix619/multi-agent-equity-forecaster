import pandas as pd
import pytest

from forecaster import scoring


def _fake_outcome(label_by_ticker):
    def fn(ticker, as_of):
        y = label_by_ticker.get(ticker)
        return None if y is None else {"outperformed": y, "excess": 0.0}
    return fn


def test_brier_and_baselines():
    rows = [{"ticker": "A", "as_of": "d", "probability": 0.8},
            {"ticker": "B", "as_of": "d", "probability": 0.2},
            {"ticker": "C", "as_of": "d", "probability": 0.6},
            {"ticker": "D", "as_of": "d", "probability": 0.4}]
    res = scoring.score_rows(rows, _fake_outcome({"A": 1, "B": 0, "C": 1, "D": 0}))
    assert res["n"] == 4 and res["pending"] == 0
    assert res["brier"] == pytest.approx((0.04 + 0.04 + 0.16 + 0.16) / 4)
    assert res["brier_always_50"] == 0.25
    assert res["accuracy"] == 1.0
    assert res["base_rate"] == 0.5


def test_pending_and_missing_probability_not_scored():
    rows = [{"ticker": "A", "as_of": "d", "probability": 0.7},
            {"ticker": "B", "as_of": "d", "probability": 0.7},
            {"ticker": "C", "as_of": "d", "probability": None}]
    res = scoring.score_rows(rows, _fake_outcome({"A": 1}))
    assert (res["n"], res["pending"]) == (1, 1)


def test_all_pending():
    res = scoring.score_rows([{"ticker": "A", "as_of": "d", "probability": 0.5}],
                             _fake_outcome({}))
    assert res == {"n": 0, "pending": 1}


def test_calibration_bins():
    rows = [{"ticker": t, "as_of": "d", "probability": p}
            for t, p in [("A", 0.3), ("B", 0.45), ("C", 0.55), ("D", 0.7)]]
    res = scoring.score_rows(rows, _fake_outcome({"A": 0, "B": 0, "C": 1, "D": 1}))
    assert set(res["calibration"]) == {"<40%", "40-50%", "50-60%", ">=60%"}
    assert res["calibration"][">=60%"]["outperformed"] == 1.0


def test_load_rows(tmp_path):
    p = tmp_path / "x.jsonl"
    assert scoring.load_rows(p) == []
    p.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert [r["a"] for r in scoring.load_rows(p)] == [1, 2]


def test_outcome_uses_20_trading_days_and_pending(monkeypatch):
    idx = pd.bdate_range("2026-01-01", periods=40)
    spy = pd.DataFrame({"Close": [100.0 + i for i in range(40)]}, index=idx)      # +20 over horizon
    stk = pd.DataFrame({"Close": [100.0 + 2 * i for i in range(40)]}, index=idx)  # beats SPY
    monkeypatch.setattr(scoring.prices, "get_prices",
                        lambda t, d, days=700: spy if t == "SPY" else stk)
    o = scoring.outcome("XYZ", str(idx[5].date()), horizon=20, today="2026-03-01")
    assert o["outperformed"] == 1
    assert o["stock_return"] > o["spy_return"]
    assert o["end"] == str(idx[25].date())
    # horizon not yet complete -> pending
    assert scoring.outcome("XYZ", str(idx[30].date()), horizon=20, today="2026-03-01") is None
