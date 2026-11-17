"""FRED macro series, returned as they were known on the as-of date."""
from datetime import datetime, timedelta

import requests

from .. import cache
from ..config import FRED_SERIES, secret

DAY = 86400
RATE_SERIES = {"DGS10", "DFF"}  # interest rates: report changes in percentage points


def get_series(series_id: str, as_of: str, lookback_days: int = 150) -> dict:
    """Observations ending at as_of, using FRED vintage dates so later revisions
    do not leak. Returns the latest value and 30/90-day changes."""
    key = secret("FRED_API_KEY")
    if not key:
        raise RuntimeError("Set FRED_API_KEY in .env")
    end = datetime.strptime(as_of, "%Y-%m-%d").date()
    start = end - timedelta(days=lookback_days)

    def fetch():
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": series_id, "api_key": key, "file_type": "json",
                    "observation_start": str(start), "observation_end": str(end),
                    "realtime_start": str(end), "realtime_end": str(end)},
            timeout=30)
        r.raise_for_status()
        return r.json()["observations"]

    obs = cache.cached("fred", [series_id, str(start), str(end)], fetch, ttl=30 * DAY)
    pts = [(o["date"], float(o["value"])) for o in obs if o["value"] not in (".", "")]
    if not pts:
        return {"series_id": series_id, "as_of": as_of, "latest": None}

    latest_date = datetime.strptime(pts[-1][0], "%Y-%m-%d").date()
    is_rate = series_id in RATE_SERIES

    def change(days):
        # Anchor on the latest observation's date, not on as_of, so monthly
        # series (CPI) compare against their own earlier readings.
        # Rates are quoted as a change in percentage points; others in percent.
        target = (latest_date - timedelta(days=days)).strftime("%Y-%m-%d")
        older = [v for d, v in pts if d <= target]
        if not older:
            return None
        if is_rate:
            return round(pts[-1][1] - older[-1], 2)
        if older[-1] == 0:
            return None
        return round((pts[-1][1] / older[-1] - 1) * 100, 2)

    unit = "pp" if is_rate else "pct"
    return {"series_id": series_id, "as_of": as_of,
            "latest": {"date": pts[-1][0], "value": pts[-1][1]},
            f"change_30d_{unit}": change(30), f"change_90d_{unit}": change(90),
            "change_unit": ("percentage points" if is_rate else "percent"),
            "observations": len(pts)}


def macro_snapshot(as_of: str) -> dict:
    """All configured macro series (oil, rates, inflation) as of a date."""
    return {name: get_series(sid, as_of) for name, sid in FRED_SERIES.items()}
