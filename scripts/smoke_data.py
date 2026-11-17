"""Live check of the data layer. Run from the equity-forecaster folder:
    python scripts/smoke_data.py            # AAPL, today
    python scripts/smoke_data.py XOM 2026-11-13
"""
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecaster.data import edgar, fred, prices


def show(title, obj):
    print(f"\n=== {title} ===")
    print(json.dumps(obj, indent=2, default=str)[:1800])


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    as_of = sys.argv[2] if len(sys.argv) > 2 else date.today().isoformat()

    fin = edgar.get_financials(ticker, as_of)
    show("EDGAR latest quarter ratios", {
        "entity": fin["entity"],
        "revenue": fin["quarter"]["revenue"],
        "ratios": fin["quarter"]["ratios"]})
    show("EDGAR filings", edgar.list_filings(ticker, as_of, limit=3))

    df = prices.get_prices(ticker, as_of)
    print(f"\n=== Prices === {len(df)} rows, last date {df.index[-1].date()}, "
          f"close {df['Close'].iloc[-1]:.2f}")

    show("FRED macro snapshot", fred.macro_snapshot(as_of))


if __name__ == "__main__":
    main()
