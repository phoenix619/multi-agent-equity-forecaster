"""Which macro series matter for which company, from its SEC industry (SIC) code.

A transparent rule table, not model judgment. Series with no identified link to
the company are excluded so they cannot pad the evidence.
"""
from .config import FRED_SERIES
from .data import edgar, fred

H, M = "high", "medium"

# (SIC range start, end, {series: (relevance, why)}), first match wins.
RULES = [
    (1300, 1399, {"oil_wti": (H, "revenue and margins track crude oil and gas prices"),
                  "treasury_10y": (M, "capital-intensive; financing costs and valuation")}),
    (2900, 2999, {"oil_wti": (H, "refiners' margins depend on crude prices"),
                  "treasury_10y": (M, "financing costs and valuation")}),
    (4500, 4599, {"oil_wti": (H, "jet fuel is a leading airline cost"),
                  "treasury_10y": (M, "leveraged balance sheets, aircraft financing"),
                  "cpi": (M, "ticket pricing power and consumer demand")}),
    (4900, 4999, {"treasury_10y": (H, "regulated, debt-funded utilities; yield competitor for dividends"),
                  "fed_funds": (M, "short-term financing costs")}),
    (2000, 2099, {"cpi": (H, "input costs and pricing power"),
                  "oil_wti": (M, "packaging and transport costs")}),
    (2830, 2839, {"treasury_10y": (M, "valuation of long-duration healthcare earnings")}),
    (2840, 2899, {"cpi": (H, "input costs and pricing power"),
                  "oil_wti": (M, "petrochemical inputs and packaging for consumer products"),
                  "treasury_10y": (M, "valuation of defensive stocks")}),
    (5200, 5299, {"treasury_10y": (H, "housing and renovation demand follow mortgage rates"),
                  "cpi": (M, "consumer prices and demand")}),
    (5300, 5999, {"cpi": (H, "consumer spending and pricing power"),
                  "oil_wti": (M, "fuel prices affect shopper traffic and logistics")}),
    (3570, 3579, {"treasury_10y": (M, "discount rate for long-duration growth stocks")}),
    (3500, 3599, {"oil_wti": (M, "energy and mining customers drive equipment demand"),
                  "treasury_10y": (M, "customers' financing costs and valuation"),
                  "cpi": (M, "input cost inflation")}),
    (3600, 3699, {"treasury_10y": (M, "discount rate for long-duration growth stocks")}),
    (7370, 7379, {"treasury_10y": (M, "discount rate for long-duration growth stocks")}),
    (7300, 7399, {"treasury_10y": (M, "discount rate for long-duration growth stocks")}),
]
DEFAULT = {"treasury_10y": (M, "general discount rate for equity valuations")}


def exposure_for(sic: int | None) -> dict:
    if sic:
        for lo, hi, rule in RULES:
            if lo <= sic <= hi:
                return rule
    return DEFAULT


def relevant_macro(ticker: str, as_of: str) -> dict:
    """Macro series that matter for this company, each with its reason."""
    prof = edgar.company_profile(ticker)
    rule = exposure_for(prof.get("sic"))
    out = {"ticker": ticker.upper(), "industry": prof.get("sic_description"),
           "sic": prof.get("sic"), "series": {}, "excluded_no_identified_exposure": []}
    for name, sid in FRED_SERIES.items():
        if name in rule:
            level, why = rule[name]
            data = fred.get_series(sid, as_of)
            data["relevance"], data["why_it_matters"] = level, why
            out["series"][name] = data
        else:
            out["excluded_no_identified_exposure"].append(name)
    return out
