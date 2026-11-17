"""Score logged forecasts against realized excess returns vs SPY.

    python scripts/score.py                  # backtest log
    python scripts/score.py live             # live log (most are pending until 20 trading days pass)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecaster import scoring
from forecaster.config import ROOT


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "backtest"
    path = (ROOT / "eval" / "forecasts" / "live.jsonl" if which == "live"
            else ROOT / "eval" / "backtest" / "forecasts.jsonl")
    rows = scoring.load_rows(path)
    res = scoring.score_rows(rows)
    print(f"{which}: {len(rows)} forecasts logged, {res['n']} scored, {res['pending']} pending")
    if not res["n"]:
        return
    print(f"base rate (share that beat SPY): {res['base_rate']:.1%}")
    print(f"Brier score: {res['brier']}   (always 50%: {res['brier_always_50']}, "
          f"always base rate: {res['brier_always_base_rate']})")
    print(f"directional accuracy: {res['accuracy']:.1%}")
    print("calibration (forecast vs what happened):")
    for b in ("<40%", "40-50%", "50-60%", ">=60%"):
        c = res["calibration"].get(b)
        if c:
            print(f"  {b:7s} n={c['n']:3d}  mean forecast {c['mean_forecast']:.0%}  "
                  f"beat SPY {c['outperformed']:.0%}")
    out = path.parent / f"scored_{which}.json"
    out.write_text(json.dumps({k: v for k, v in res.items()}, indent=2, default=str))
    print("saved", out)


if __name__ == "__main__":
    main()
