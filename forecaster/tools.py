"""Tool layer for agents: schemas the model sees + implementations that run in Python.

The as-of date is fixed by the caller, never chosen by the model, so an agent
cannot accidentally (or by prompt injection) read data from the future.
All tools are read-only.
"""
from . import indicators, macro
from .data import edgar, prices

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_financials",
        "description": ("Latest quarterly and annual revenue, income, debt, equity and "
                        "margin ratios for a ticker from SEC filings. Each figure has its "
                        "form, period end and accession number for citation."),
        "parameters": {"type": "object",
                       "properties": {"ticker": {"type": "string"}},
                       "required": ["ticker"]}}},
    {"type": "function", "function": {
        "name": "list_filings",
        "description": "Recent 10-K and 10-Q filings for a ticker (form, date, period, URL).",
        "parameters": {"type": "object",
                       "properties": {"ticker": {"type": "string"}},
                       "required": ["ticker"]}}},
    {"type": "function", "function": {
        "name": "get_technicals",
        "description": ("Trend, momentum and volatility from price history (moving averages, "
                        "RSI, MACD, ATR, Bollinger Bands) plus a paper-trade entry zone, "
                        "stop-loss and target. Used for timing, not direction."),
        "parameters": {"type": "object",
                       "properties": {"ticker": {"type": "string"}},
                       "required": ["ticker"]}}},
    {"type": "function", "function": {
        "name": "get_macro",
        "description": ("Macro series that matter for THIS company, chosen from its industry "
                        "(oil, 10-year yield, fed funds, CPI) with 30 and 90 day changes and the "
                        "reason each matters. Series with no identified exposure are excluded; "
                        "do not cite excluded series."),
        "parameters": {"type": "object",
                       "properties": {"ticker": {"type": "string"}},
                       "required": ["ticker"]}}},
]

BALANCE_SHEET = {"long_term_debt", "equity"}
KEEP = ["revenue", "gross_profit", "operating_income", "net_income",
        "operating_cash_flow", "long_term_debt", "equity"]


def _trim_financials(fin: dict) -> dict:
    out = {"ticker": fin["ticker"], "entity": fin["entity"], "as_of": fin["as_of"]}
    out["freshness"] = fin.get("freshness")
    for kind in ("quarter", "annual"):
        block = {}
        for k in KEEP:
            if kind == "annual" and k in BALANCE_SHEET:
                continue  # point-in-time items appear once, in the quarter block
            f = fin[kind].get(k)
            if f:
                block[k] = {"value": f["value"], "period_end": f["period_end"],
                            "fiscal_period": (f"FY{f['fiscal_year']} {f['fiscal_period']}"
                                              if f.get("fiscal_year") else None),
                            "form": f["form"], "accession": f["accession"],
                            **({"derived": f["derived"]} if f.get("derived") else {})}
        block["ratios"] = fin[kind]["ratios"]
        out[kind] = block
    return out


def make_tools(as_of: str) -> dict:
    """Return {tool_name: callable(**args)} bound to a fixed as-of date."""

    def get_financials(ticker: str):
        return _trim_financials(edgar.get_financials(ticker, as_of))

    def list_filings(ticker: str):
        return edgar.list_filings(ticker, as_of, limit=4)

    def get_technicals(ticker: str):
        df = prices.get_prices(ticker, as_of, days=400)
        return indicators.technical_summary(df)

    def get_macro(ticker: str):
        return macro.relevant_macro(ticker, as_of)

    return {"get_financials": get_financials, "list_filings": list_filings,
            "get_technicals": get_technicals, "get_macro": get_macro}
