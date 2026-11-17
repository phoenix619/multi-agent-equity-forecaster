from forecaster import verifier as v

TOOLS = {
    "get_financials": {"quarter": {"revenue": {"value": 102466000000, "fiscal_period": "FY2026 Q4"},
                                   "ratios": {"net_margin": 0.2761, "debt_to_equity": 1.87}}},
    "get_technicals": {"rsi14": 75.3, "paper_trade": {"entry_zone": [299.5, 305.1], "stop_loss": 281.2}},
    "get_macro": {"series": {"treasury_10y": {"change_30d_pp": -0.12}}},
}


def draft(rationale="", claims=(), risks=()):
    return {"rationale": rationale,
            "evidence": [{"claim": c, "source": "s"} for c in claims],
            "risks": [{"risk": r, "source": "s"} for r in risks]}


def test_units_and_rounding_match():
    d = draft(claims=["Revenue was $102.5 billion in FY2026 Q4",
                      "Net margin is 27.6%",
                      "Debt-to-equity of 1.87x",
                      "RSI at 75",
                      "Entry zone $299.5 to $305.1"])
    out = v.verify_forecast(d, TOOLS)
    assert out["status"] == "green", out["unverified"]
    assert out["checked"] >= 5


def test_fabricated_number_flagged():
    out = v.verify_forecast(draft(claims=["Revenue grew 18.4% year over year"]), TOOLS)
    assert out["status"] == "orange"
    assert out["unverified"][0]["number"].startswith("18.4")


def test_four_unverified_is_red():
    d = draft(claims=[f"Metric {x}.5% moved" for x in (31, 32, 33, 34)])
    assert v.verify_forecast(d, TOOLS)["status"] == "red"


def test_no_evidence_is_red():
    assert v.verify_forecast(draft(rationale="Looks fine."), TOOLS)["status"] == "red"


def test_ignores_dates_years_windows_and_names():
    text = ("As of 2026-11-16 the 50-day and 200-day averages, FY2026 Q3, a 10-K filing, "
            "the S&P 500 and 20 trading days are context; margin 27.6%")
    assert v.extract_claims(text) == v.extract_claims("margin 27.6%")
    out = v.verify_forecast(draft(claims=[text]), TOOLS)
    assert out["status"] == "green"


def test_negative_and_rate_values_compare_by_magnitude():
    d = draft(claims=["The 10-year yield fell 0.12 percentage points over 30 days"])
    assert v.verify_forecast(d, TOOLS)["status"] == "green"


def test_drop_unverified_removes_only_bad_items():
    d = draft(claims=["Net margin is 27.6%", "Revenue grew 18.4%"], risks=["RSI at 75"])
    res = v.verify_forecast(d, TOOLS)
    cleaned = v.drop_unverified(d, res["unverified"])
    assert [e["claim"] for e in cleaned["evidence"]] == ["Net margin is 27.6%"]
    assert len(cleaned["risks"]) == 1


def test_base_rate_50_percent_is_not_a_data_claim():
    d = draft(rationale="Nudging slightly above the neutral 50% base rate.",
              claims=["Net margin is 27.6%"])
    assert v.verify_forecast(d, TOOLS)["status"] == "green"


def test_sp500_with_odd_spaces_is_not_a_claim():
    for sp in (" ", " ", " "):
        text = f"outperforming the S&P{sp}500 over the next period; margin 27.6%"
        assert v.extract_claims(text) == v.extract_claims("margin 27.6%")


def test_indicator_window_is_not_a_claim_but_real_values_still_checked():
    ok = draft(claims=["RSI 14 is 75.3, SMA 50 is above SMA 200"])
    assert v.verify_forecast(ok, TOOLS)["status"] == "green"
    # a real value is still checked: RSI 20 would match nothing here, 14 is a window
    bad = draft(claims=["RSI at 63"])
    assert v.verify_forecast(bad, TOOLS)["status"] == "orange"
