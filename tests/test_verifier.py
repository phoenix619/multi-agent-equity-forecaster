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


def test_non_breaking_hyphens_do_not_create_false_claims():
    text = ("The 10\u2011year yield, the 50\u2011day and 200\u2011day averages and a 20\u2013day "
            "window are context; margin 27.6%")
    assert v.extract_claims(text) == v.extract_claims("margin 27.6%")
    out = v.verify_forecast(draft(rationale=text, claims=["Net margin is 27.6%"]), TOOLS)
    assert out["status"] == "green", out["unverified"]


def test_percent_sign_with_a_space_is_still_a_percent():
    assert v.extract_claims("above the 50 % base rate") == []          # the 50% benchmark
    claims = v.extract_claims("net margin of 27.6 % this year")
    assert claims and claims[0]["pct"] and claims[0]["value"] == 27.6
    d = draft(rationale="modestly above the 50 % base rate", claims=["Net margin is 27.6 %"])
    assert v.verify_forecast(d, TOOLS)["status"] == "green"


def test_more_non_data_numbers_found_in_real_drafts():
    cases = ["price below the 20\u2011, 50\u2011, and 200\u2011day moving averages",   # compound hyphens
             "filing 0000320193-26-000020 confirms the figures",                          # accession
             "cash of this size as of September 26 2026 is ample",                        # date
             "WTI up 5% 30d and up 81% 90d"]                                              # glued units
    got = [[c["raw"] for c in v.extract_claims(t)] for t in cases]
    assert got == [[], [], [], ["5%", "81%"]]
    # thresholds: unmatched "below 50" is not flagged, but a real RSI value still is checked
    ok = draft(claims=["RSI 14 is 75.3, above 70"])
    assert v.verify_forecast(ok, TOOLS)["status"] == "green"
    assert v.verify_forecast(draft(claims=["RSI is 63"]), TOOLS)["status"] == "orange"
    # a numeric range keeps both ends as claims
    assert [c["raw"] for c in v.extract_claims("a range of $100-$110")] == ["$100", "$110"]


TECH_TEXT = {"get_technicals": {"volatility": "price at 99% of Bollinger range; ATR $11.54 (3.1% of price)",
                                "rsi14": 55.0}}


def test_numbers_inside_tool_text_fields_verify():
    d = draft(claims=["Price is at 99% of Bollinger range with ATR of 11.54"])
    assert v.verify_forecast(d, TECH_TEXT)["status"] == "green"
    assert v.verify_forecast(draft(claims=["Price is at 88% of Bollinger range"]),
                             TECH_TEXT)["status"] == "orange"


def test_models_own_probability_is_not_a_claim():
    d = dict(draft(rationale="keeps the probability near 45%, or 0.45 on a 0-1 scale",
                   claims=["Net margin is 27.6%"]), probability=0.45)
    assert v.verify_forecast(d, TOOLS)["status"] == "green"
    # but a different percentage with no source is still flagged
    d2 = dict(draft(rationale="up 38%", claims=["Net margin is 27.6%"]), probability=0.45)
    assert v.verify_forecast(d2, TOOLS)["status"] == "orange"


def test_treasury_tenor_suffix_is_not_a_claim():
    assert v.extract_claims("Treasury 10Y is 4.18 and the 2Y is 3.9") == [
        c for c in v.extract_claims("Treasury is 4.18 and the is 3.9")]


def test_rounded_long_figures_match_but_misquoted_ones_do_not():
    tools = {"get_financials": {"ocf": {"value": 19556000000}}, "get_technicals": {"rsi14": 66.0}}
    ok = draft(claims=["Operating cash flow of 19,560,000,000"])
    assert v.verify_forecast(ok, tools)["status"] == "green"            # rounded to 4 digits
    off = draft(claims=["Operating cash flow of 19,700,000,000"])
    assert v.verify_forecast(off, tools)["status"] == "orange"          # 0.7% off: not a rounding
    assert v.verify_forecast(draft(claims=["RSI 64 (approaching overbought)"]),
                             tools)["status"] == "orange"                # a real misquote stays flagged


def test_filing_table_numbers_stated_in_millions_verify():
    filing = {"search_filings": {"results": [
        {"text": "Net sales 32,260 and operating income 10,588 (in millions)", "source": "10-K"}]}}
    ok = draft(claims=["Net sales were $32,260 million", "Operating income of $10,588 million"])
    assert v.verify_forecast(ok, filing)["status"] == "green"
    bad = draft(claims=["Net sales were $33,000 million"])
    assert v.verify_forecast(bad, filing)["status"] == "orange"
    # the unit scaling applies only to filing text, never to numeric tool fields
    assert v.verify_forecast(draft(claims=["Revenue of $5.0 million"]),
                             {"get_financials": {"x": 5000}})["status"] == "orange"


def test_drop_unverified_withholds_a_rationale_with_bad_numbers():
    d = draft(rationale="Revenue grew 18.4% year over year.", claims=["Net margin is 27.6%"])
    res = v.verify_forecast(d, TOOLS)
    cleaned = v.drop_unverified(d, res["unverified"])
    assert "18.4" not in cleaned["rationale"] and "withheld" in cleaned["rationale"].lower()
    assert [e["claim"] for e in cleaned["evidence"]] == ["Net margin is 27.6%"]
    ok = v.drop_unverified(draft(rationale="Margins are steady.", claims=["x"]), [])
    assert ok["rationale"] == "Margins are steady."
