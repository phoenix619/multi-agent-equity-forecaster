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


def _facts(concepts):
    return {"facts": {"us-gaap": {c: {"units": {"USD": rows}} for c, rows in concepts.items()}}}


def test_fact_prefers_concept_with_latest_period():
    facts = _facts({
        "Revenues": [f("2013-07-01", "2013-09-30", 1, "2013-11-01", "10-Q", "Q3", 2013)],
        "RevenueFromContractWithCustomerIncludingAssessedTax":
            [f("2026-01-01", "2026-03-31", 5, "2026-04-23", "10-Q", "Q1", 2026)]})
    got = edgar._fact(facts, "revenue", date(2026, 11, 17), "quarter")
    assert got["value"] == 5 and got["period_end"] == "2026-03-31"
    assert got["concept"] == "RevenueFromContractWithCustomerIncludingAssessedTax"


def test_ratio_never_mixes_periods():
    a = {"value": 2, "period_end": "2026-03-31"}
    assert edgar._ratio(a, {"value": 4, "period_end": "2013-09-30"}) is None
    assert edgar._ratio(a, {"value": 4, "period_end": "2026-03-31"}) == 0.5


def _patch_edgar(monkeypatch, filing_period):
    facts = _facts({
        "NetIncomeLoss": [f("2026-01-01", "2026-03-31", 10, "2026-04-23", "10-Q", "Q1", 2026)],
        "Revenues": [f("2026-01-01", "2026-03-31", 40, "2026-04-23", "10-Q", "Q1", 2026)],
        "LongTermDebtNoncurrent": [{"end": "2026-03-31", "val": 90, "filed": "2026-04-23",
                                    "form": "10-Q", "accn": "x", "fp": "Q1", "fy": 2026}],
        "StockholdersEquity": [{"end": "2026-03-31", "val": 30, "filed": "2026-04-23",
                                "form": "10-Q", "accn": "x", "fp": "Q1", "fy": 2026}]})
    monkeypatch.setattr(edgar, "ticker_to_cik", lambda t: "0000000001")
    monkeypatch.setattr(edgar, "_company_facts", lambda cik: dict(facts, entityName="Co"))
    monkeypatch.setattr(edgar, "list_filings", lambda t, a, forms=(), limit=1: [
        {"form": "10-Q", "filed": "2026-10-21", "period": filing_period}])


def test_freshness_flags_lagging_structured_data(monkeypatch):
    _patch_edgar(monkeypatch, "2026-09-30")
    fin = edgar.get_financials("CO", "2026-11-17")
    assert fin["freshness"]["lag"] is True
    assert fin["quarter"]["ratios"]["net_margin"] == 0.25
    assert fin["quarter"]["ratios"]["debt_to_equity"] == 3.0
    assert "debt_to_equity" not in fin["annual"]["ratios"]


def test_freshness_ok_when_data_is_current(monkeypatch):
    _patch_edgar(monkeypatch, "2026-03-31")
    assert edgar.get_financials("CO", "2026-11-17")["freshness"]["lag"] is False


def test_trim_keeps_balance_sheet_only_in_quarter_block(monkeypatch):
    from forecaster import tools
    _patch_edgar(monkeypatch, "2026-09-30")
    trimmed = tools._trim_financials(edgar.get_financials("CO", "2026-11-17"))
    assert "equity" in trimmed["quarter"] and "equity" not in trimmed["annual"]
    assert trimmed["freshness"]["lag"] is True


def test_fact_uses_alternate_concept_when_primary_is_stale():
    facts = _facts({
        "NetIncomeLoss": [f("2011-07-01", "2011-09-30", 9, "2011-11-01", "10-Q", "Q3", 2011)],
        "ProfitLoss": [f("2025-07-01", "2025-09-30", 77, "2025-11-01", "10-Q", "Q3", 2025)]})
    got = edgar._fact(facts, "net_income", date(2026, 1, 12), "quarter")
    assert got["value"] == 77 and got["concept"] == "ProfitLoss"


def test_stale_fact_is_treated_as_not_reported():
    facts = _facts({"NetIncomeLoss": [
        f("2011-07-01", "2011-09-30", 9, "2011-11-01", "10-Q", "Q3", 2011)]})
    assert edgar._fact(facts, "net_income", date(2026, 1, 12), "quarter") is None


def test_predecessor_history_is_merged_for_reorganized_companies(monkeypatch):
    new = _facts({"Revenues": [f("2026-04-01", "2026-06-30", 8, "2026-08-01", "10-Q", "Q2", 2026)]})
    old = _facts({"Revenues": [f("2025-07-01", "2025-09-30", 7, "2025-11-01", "10-Q", "Q3", 2025),
                               f("2026-04-01", "2026-06-30", 8, "2026-08-01", "10-Q", "Q2", 2026)]})
    for r in new["facts"]["us-gaap"]["Revenues"]["units"]["USD"] + \
            old["facts"]["us-gaap"]["Revenues"]["units"]["USD"]:
        r.setdefault("accn", "x")
    monkeypatch.setattr(edgar, "PREDECESSORS", {"ZZZ": ["OLD"]})
    monkeypatch.setattr(edgar, "_company_facts",
                        lambda cik: dict(new if cik == "NEW" else old, entityName="Co"))
    merged = edgar._merged_facts("ZZZ", "NEW")
    rows = merged["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    assert sorted(r["end"] for r in rows) == ["2025-09-30", "2026-06-30"]   # no duplicate
    # an earlier as-of date now has data from the predecessor
    assert edgar._fact(merged, "revenue", date(2025, 12, 1), "quarter")["value"] == 7
    assert edgar._fact(new, "revenue", date(2025, 12, 1), "quarter") is None
