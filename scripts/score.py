"""Score logged forecasts against realized excess returns vs SPY, per variant.

    python scripts/score.py                  # backtest log, every variant
    python scripts/score.py live             # live log (pending until 20 trading days pass)
    python scripts/score.py --common         # compare variants on the (ticker, date) pairs ALL share
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecaster import scoring
from forecaster.config import ROOT


def show(name, rows):
    res = scoring.score_rows(rows)
    print(f"\n== {name}: {len(rows)} forecasts logged, {res['n']} scored, {res['pending']} pending")
    if not res["n"]:
        return res
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
    return res


def main():
    which = "live" if "live" in sys.argv[1:] else "backtest"
    common = "--common" in sys.argv[1:]
    path = (ROOT / "eval" / "forecasts" / "live.jsonl" if which == "live"
            else ROOT / "eval" / "backtest" / "forecasts.jsonl")
    rows = [r for r in scoring.load_rows(path) if r.get("probability") is not None]
    by = defaultdict(list)
    for r in rows:
        by[r.get("variant", "single_agent_react")].append(r)
    if common and len(by) > 1:
        shared = set.intersection(*[{(r["ticker"], r["as_of"]) for r in v} for v in by.values()])
        by = {k: [r for r in v if (r["ticker"], r["as_of"]) in shared] for k, v in by.items()}
        print(f"comparing on {len(shared)} shared (ticker, date) pairs")
    out, full = {}, {}
    for variant, vrows in sorted(by.items()):
        res = show(variant, vrows)
        out[variant] = {k: v for k, v in res.items() if k != "rows"}
        full[variant] = res.get("rows", [])
    tag = f"{which}{'_common' if common else ''}"
    f = path.parent / f"scored_{tag}.json"          # compact summary: tracked in git
    f.write_text(json.dumps(out, indent=2, default=str))
    rows_f = path.parent / f"scored_{tag}_rows.json"  # per-forecast detail: git-ignored
    rows_f.write_text(json.dumps(full, indent=1, default=str))
    print("\nsaved", f, "and", rows_f.name)


if __name__ == "__main__":
    main()
