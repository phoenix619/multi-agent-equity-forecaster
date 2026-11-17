from datetime import date

from forecaster.data import edgar


def f(start, end, val, filed, form, fp=None, fy=2025):
    return {"start": start, "end": end, "val": val, "filed": filed,
            "form": form, "accn": "x", "fp": fp, "fy": fy}


ROWS = [
    f("2025-01-01", "2025-03-31", 100, "2025-05-01", "10-Q", "Q1"),
    f("2025-01-01", "2025-09-30", 330, "2025-11-01", "10-Q", "Q3"),   # 9-month YTD
    f("2025-07-01", "2025-09-30", 120, "2025-11-01", "10-Q", "Q3"),
    f("2025-01-01", "2025-12-31", 460, "2026-02-15", "10-K", "FY"),   # full year
]


def test_q4_derived_from_annual_minus_9m():
    q = edgar._quarter_flow(ROWS, date(2026, 3, 1))
    assert q["val"] == 130 and q["fp"] == "Q4"
    assert "minus 9-month" in q["derived"]


def test_reported_quarter_before_10k_filed():
    q = edgar._quarter_flow(ROWS, date(2025, 12, 1))
    assert q["val"] == 120 and q["derived"] is None


def test_no_lookahead():
    assert edgar._quarter_flow(ROWS, date(2025, 1, 1)) is None
    q = edgar._quarter_flow(ROWS, date(2025, 6, 1))
    assert q["val"] == 100                                           # only Q1 known


def test_cumulative_cash_flow_difference():
    rows = [f("2025-01-01", "2025-03-31", 50, "2025-05-01", "10-Q"),
            f("2025-01-01", "2025-06-30", 130, "2025-08-01", "10-Q")]
    q = edgar._quarter_flow(rows, date(2025, 9, 1))
    assert q["val"] == 80 and "year-to-date" in q["derived"]


def test_pick_annual_requires_10k_and_filed_by_as_of():
    assert edgar._pick(ROWS, date(2026, 1, 1), "annual") is None
    assert edgar._pick(ROWS, date(2026, 3, 1), "annual")["val"] == 460


def test_ratio():
    assert edgar._ratio({"value": 1}, {"value": 4}) == 0.25
    assert edgar._ratio(None, {"value": 4}) is None
    assert edgar._ratio({"value": 1}, {"value": 0}) is None
