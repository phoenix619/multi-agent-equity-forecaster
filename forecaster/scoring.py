"""Score forecasts against what happened: excess return vs SPY over N trading days.

Label: 1 if the stock's total return beat SPY's from the as-of close to the close
20 trading days later, else 0. Forecasts without 20 trading days of data yet are
pending. Metrics: Brier score (lower is better; 0.25 = always saying 50%),
accuracy, calibration bins, and comparison with the base-rate predictor.
"""
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

from .config import BENCHMARK, HORIZON_TRADING_DAYS, ROOT
from .data import prices


def outcome(ticker: str, as_of: str, horizon: int = HORIZON_TRADING_DAYS,
            today: str | None = None):
    """Return a dict with the label, or None while the horizon is still open."""
    today = today or date.today().isoformat()
    spy = prices.get_prices(BENCHMARK, today, days=700)["Close"]
    stk = prices.get_prices(ticker, today, days=700)["Close"]
    pos = spy.index.searchsorted(pd.Timestamp(as_of), side="right") - 1
    if pos < 0 or pos + horizon >= len(spy):
        return None
    d0, d1 = spy.index[pos], spy.index[pos + horizon]
    s0, s1 = stk.asof(d0), stk.asof(d1)
    if pd.isna(s0) or pd.isna(s1):
        return None
    sr, br = float(s1 / s0 - 1), float(spy.iloc[pos + horizon] / spy.iloc[pos] - 1)
    return {"start": str(d0.date()), "end": str(d1.date()),
            "stock_return": round(sr, 4), "spy_return": round(br, 4),
            "excess": round(sr - br, 4), "outperformed": int(sr > br)}


def load_rows(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score_rows(rows: list, outcome_fn=outcome) -> dict:
    """Attach outcomes and compute metrics. Pending rows are counted, not scored."""
    scored, pending = [], 0
    for r in rows:
        if r.get("probability") is None:
            continue
        o = outcome_fn(r["ticker"], r["as_of"])
        if o is None:
            pending += 1
            continue
        scored.append(dict(r, **o))
    n = len(scored)
    if n == 0:
        return {"n": 0, "pending": pending}
    y = [s["outperformed"] for s in scored]
    p = [s["probability"] for s in scored]
    base = sum(y) / n
    brier = sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / n
    brier_base = sum((base - yi) ** 2 for yi in y) / n          # always predict the base rate
    brier_half = sum((0.5 - yi) ** 2 for yi in y) / n           # always predict 50%
    acc = sum((pi > 0.5) == bool(yi) for pi, yi in zip(p, y) if pi != 0.5) / max(
        1, sum(pi != 0.5 for pi in p))
    bins = defaultdict(list)
    for pi, yi in zip(p, y):
        b = "<40%" if pi < 0.4 else "40-50%" if pi < 0.5 else "50-60%" if pi < 0.6 else ">=60%"
        bins[b].append((pi, yi))
    calib = {b: {"n": len(v), "mean_forecast": round(sum(a for a, _ in v) / len(v), 3),
                 "outperformed": round(sum(c for _, c in v) / len(v), 3)}
             for b, v in bins.items()}
    return {"n": n, "pending": pending, "base_rate": round(base, 3),
            "brier": round(brier, 4), "brier_always_base_rate": round(brier_base, 4),
            "brier_always_50": round(brier_half, 4), "accuracy": round(acc, 3),
            "calibration": calib, "rows": scored}
