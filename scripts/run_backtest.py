"""Historical backtest of the single-agent baseline (resumable).

Runs the baseline for each ticker at past as-of dates (every tool uses only data
known on that date) and appends to eval/backtest/forecasts.jsonl. Already-logged
(ticker, date) pairs are skipped, so you can stop and restart freely.

Run from the equity-forecaster folder:
    python scripts/run_backtest.py                      # default dates, all tickers
    python scripts/run_backtest.py --dates 2026-08-24 2026-09-21
    python scripts/run_backtest.py --tickers AAPL DAL --dates 2026-09-21
    python scripts/run_backtest.py --limit 20           # stop after 20 new forecasts
    python scripts/run_backtest.py --variant multi_critic_verifier --subset
        # ablation subset: 4 dates x all tickers (60 forecasts) with another variant

The last usable as-of date is about 20 trading days before today, so outcomes exist.
Default dates are 4 weeks apart (non-overlapping 20-day windows).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecaster import baseline, variants
from forecaster.config import ROOT, TICKERS

DEFAULT_DATES = ["2026-01-12", "2026-02-09", "2026-03-09", "2026-04-06", "2026-05-04",
                 "2026-06-01", "2026-06-29", "2026-07-27", "2026-08-24", "2026-09-21",
                 "2026-10-12"]
SUBSET_DATES = ["2026-01-12", "2026-04-06", "2026-07-27", "2026-10-12"]  # mixed market regimes
OUT = ROOT / "eval" / "backtest" / "forecasts.jsonl"


def done_pairs() -> set:  # (ticker, as_of, variant) already logged
    if not OUT.exists():
        return set()
    return {(r["ticker"], r["as_of"], r["variant"])
            for r in map(json.loads, OUT.read_text().splitlines()) if r.get("probability") is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="*", default=None)
    ap.add_argument("--variant", default="single_agent_react", choices=variants.VARIANTS)
    ap.add_argument("--subset", action="store_true", help="use the 4 ablation dates")
    ap.add_argument("--tickers", nargs="*", default=TICKERS)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    dates = a.dates or (SUBSET_DATES if a.subset else DEFAULT_DATES)
    done = done_pairs()
    todo = [(t, d) for d in dates for t in a.tickers if (t, d, a.variant) not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(todo)} forecasts to run ({len(done)} already done). Ctrl-C is safe; rerun to resume.")
    ok = 0
    for i, (t, d) in enumerate(todo, 1):
        res = variants.run(a.variant, t, d)
        baseline.save_trace(res)
        f = res["forecast"]
        if f:
            baseline.log_forecast(res, folder="backtest", name="forecasts.jsonl", mode="backtest")
            ok += 1
            print(f"[{i}/{len(todo)}] {t} {d}: {f['probability']:.0%} {f['rating']} "
                  f"{(f.get('verification') or {}).get('status', '')} ({res['seconds']}s)")
        else:
            print(f"[{i}/{len(todo)}] {t} {d}: FAILED {res['error']}")
    print(f"\nDone: {ok} logged, {len(todo) - ok} failed.")


if __name__ == "__main__":
    main()
