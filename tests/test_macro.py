from forecaster import macro


def test_apple_sic_gets_rates_only():
    assert set(macro.exposure_for(3571)) == {"treasury_10y"}   # not caught by machinery range


def test_oil_exposed_industries():
    assert macro.exposure_for(1311)["oil_wti"][0] == "high"    # oil & gas
    assert macro.exposure_for(4512)["oil_wti"][0] == "high"    # airlines


def test_pharma_vs_consumer_products():
    assert "oil_wti" not in macro.exposure_for(2834)           # pharma
    assert "cpi" in macro.exposure_for(2844)                   # consumer products


def test_unknown_or_missing_sic_uses_default():
    assert macro.exposure_for(None) == macro.DEFAULT
    assert macro.exposure_for(9999) == macro.DEFAULT


def test_relevant_macro_excludes_unrelated(monkeypatch):
    monkeypatch.setattr(macro.edgar, "company_profile",
                        lambda t: {"sic": 3571, "sic_description": "Electronic Computers"})
    monkeypatch.setattr(macro.fred, "get_series", lambda sid, as_of: {"id": sid})
    out = macro.relevant_macro("AAPL", "2026-11-16")
    assert list(out["series"]) == ["treasury_10y"]
    assert "oil_wti" in out["excluded_no_identified_exposure"]
    assert out["series"]["treasury_10y"]["why_it_matters"]
