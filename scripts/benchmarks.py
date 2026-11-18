"""Benchmarks for the ablation: non-LLM rules and a permutation null, on the SAME pairs.

Run from the equity-forecaster folder:
    python3 scripts/benchmarks.py

Rules use only prices up to each as-of date (get_prices cuts off at as_of). No parameter is
tuned on outcomes: the three rules and their windows are fixed here before scoring.
  mom_6_1   : 126-day return skipping the latest 21 days, minus SPY's (classic momentum)
  rev_1m    : minus the latest 21-day return, minus SPY's (short-term reversal)
  low_vol   : minus 63-day realised volatility (low-volatility anomaly)
A rule becomes a probability with no fitting: 0.5 +/- 0.05 by above/below the date's median.
Permutation null: shuffle each variant's forecasts across tickers WITHIN each date (keeps the
date-level regime, destroys any stock-level signal) and see where the real score falls.
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecaster.data import prices  # noqa: E402

ROWS = ROOT / "eval" / "backtest" / "scored_backtest_common_rows.json"
OUT = ROOT / "eval" / "backtest" / "benchmarks.json"
STEP = 0.05


_SPY = None


def close(ticker, as_of):
    """Closes up to as_of. SPY comes from ONE long download, sliced by date (no look-ahead)."""
    global _SPY
    if ticker == "SPY":
        if _SPY is None:
            _SPY = prices.get_prices("SPY", "2026-11-18", days=700)["Close"].dropna()
        return _SPY[_SPY.index <= pd.Timestamp(as_of)]
    return prices.get_prices(ticker, as_of, days=400)["Close"].dropna()


def features(ticker, as_of, spy):
    s = close(ticker, as_of)
    if len(s) < 130:
        return None
    r = lambda ser, a, b: float(ser.iloc[-1 - a] / ser.iloc[-1 - b] - 1)   # return from t-b to t-a
    ret = np.log(s).diff().dropna()
    return {"mom_6_1": r(s, 21, 126) - r(spy, 21, 126),
            "rev_1m": -(r(s, 0, 21) - r(spy, 0, 21)),
            "low_vol": -float(ret.iloc[-63:].std())}


def auc(scores, y):
    """Rank-based (Mann-Whitney) AUC; ties count half."""
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(scores).rank().to_numpy()    # average ranks for ties
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(p, y):
    return float(np.mean([(a - b) ** 2 for a, b in zip(p, y)]))


def boot(fn, by, dates, B=3000, seed=0):
    rnd = random.Random(seed)
    out = sorted(fn([k for d in rnd.choices(dates, k=len(dates)) for k in by[d]]) for _ in range(B))
    return [round(out[int(.025 * B)], 3), round(out[int(.975 * B)], 3)]


def main():
    rows = json.loads(ROWS.read_text())
    base = rows["single_agent_react"]
    keys = [(r["ticker"], r["as_of"]) for r in base]
    y = {(r["ticker"], r["as_of"]): r["outperformed"] for r in base}
    dates = sorted({k[1] for k in keys})
    by = defaultdict(list)
    for k in keys:
        by[k[1]].append(k)

    feats = {}
    for d in dates:
        spy = close("SPY", d)
        for t, _ in [k for k in keys if k[1] == d]:
            f = features(t, d, spy)
            if f is None:
                sys.exit(f"not enough cached price history for {t} {d}; run on a machine with network")
            feats[(t, d)] = f

    result = {"n": len(keys), "dates": len(dates), "rules": {}, "permutation": {}}
    print(f"{len(keys)} pairs, {len(dates)} dates\n")
    print("== Non-LLM rules (no tuning)")
    for rule in ("mom_6_1", "rev_1m", "low_vol"):
        prob = {}
        for d in dates:
            med = np.median([feats[k][rule] for k in by[d]])
            for k in by[d]:
                prob[k] = 0.5 + STEP if feats[k][rule] > med else 0.5 - STEP
        a = lambda ks: auc([feats[k][rule] for k in ks], [y[k] for k in ks])
        b = lambda ks: brier([prob[k] for k in ks], [y[k] for k in ks])
        acc = np.mean([(prob[k] > .5) == bool(y[k]) for k in keys])
        res = {"auc": round(a(keys), 3), "auc_ci": boot(a, by, dates),
               "brier": round(b(keys), 4), "brier_ci": boot(b, by, dates), "accuracy": round(float(acc), 3)}
        result["rules"][rule] = res
        print(f"{rule:8s} AUC {res['auc']} {res['auc_ci']}  Brier {res['brier']} {res['brier_ci']}  acc {res['accuracy']:.1%}")
    result["const_50_brier"] = round(brier([.5] * len(keys), [y[k] for k in keys]), 4)
    print(f"const 50% Brier {result['const_50_brier']}")

    print("\n== Permutation null (shuffle forecasts across tickers within each date, 5000 draws)")
    rnd = random.Random(1)
    for v, rs in rows.items():
        p = {(r["ticker"], r["as_of"]): r["probability"] for r in rs}
        ks = list(p)
        obs_a = auc([p[k] for k in ks], [y[k] for k in ks])
        obs_b = brier([p[k] for k in ks], [y[k] for k in ks])
        na, nb = [], []
        for _ in range(5000):
            sh = {}
            for d in dates:
                vals = [p[k] for k in by[d]]
                rnd.shuffle(vals)
                sh.update(zip(by[d], vals))
            na.append(auc([sh[k] for k in ks], [y[k] for k in ks]))
            nb.append(brier([sh[k] for k in ks], [y[k] for k in ks]))
        pa = float(np.mean([x >= obs_a for x in na]))       # P(AUC this good by luck)
        pb = float(np.mean([x <= obs_b for x in nb]))       # P(Brier this good by luck)
        result["permutation"][v] = {"auc": round(obs_a, 3), "p_auc": round(pa, 3),
                                    "null_auc_mean": round(float(np.mean(na)), 3),
                                    "brier": round(obs_b, 4), "p_brier": round(pb, 3),
                                    "null_brier_mean": round(float(np.mean(nb)), 4)}
        print(f"{v:24s} AUC {obs_a:.3f} (null mean {np.mean(na):.3f}, p={pa:.3f})   "
              f"Brier {obs_b:.4f} (null mean {np.mean(nb):.4f}, p={pb:.3f})")
    OUT.write_text(json.dumps(result, indent=1))
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
