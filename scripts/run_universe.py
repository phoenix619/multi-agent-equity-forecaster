"""Run the baseline on every tracked ticker and append timestamped forecasts to
eval/forecasts/live.jsonl (commit that file to git as proof of when each
forecast was made). Skips a ticker already logged for the same as-of date and
variant, so it is safe to re-run.

Run from the equity-forecaster folder:
    python scripts/run_universe.py              # all tickers, as of today
    python scripts/run_universe.py 2026-11-13   # as of a past date
"""
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecaster import baseline
from forecaster.config import ROOT, TICKERS


def already_logged(as_of: str) -> set:
    p = ROOT / "eval" / "forecasts" / "live.jsonl"
    if not p.exists():
        return set()
    done = set()
    for line in p.read_text().splitlines():
        row = json.loads(line)
        if row["as_of"] == as_of and row["variant"] == "single_agent_react":
            done.add(row["ticker"])
    return done


def main():
    as_of = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    done = already_logged(as_of)
    rows = []
    for t in TICKERS:
        if t in done:
            print(f"{t}: already logged for {as_of}, skipping")
            continue
        res = baseline.run(t, as_of)
        baseline.save_trace(res)
        f = res["forecast"]
        if f:
            baseline.log_forecast(res)
            print(f"{t}: {f['probability']:.0%} {f['rating']}  "
                  f"({res['tool_calls']} tools, {res['seconds']}s)")
        else:
            print(f"{t}: FAILED {res['error']}")
        rows.append((t, f["probability"] if f else None))
    ok = [r for r in rows if r[1] is not None]
    print(f"\n{len(ok)} of {len(rows)} new forecasts logged to eval/forecasts/live.jsonl")


if __name__ == "__main__":
    main()
