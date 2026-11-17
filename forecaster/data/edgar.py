"""SEC EDGAR: financial facts, ratios and filing lists, as of a given date."""
import time
from datetime import date, datetime

import requests

from .. import cache
from ..config import secret

DAY = 86400
_last_call = 0.0


def _get(url: str) -> dict:
    """GET JSON from SEC with the required User-Agent and polite pacing."""
    global _last_call
    ua = secret("EDGAR_USER_AGENT")
    if not ua:
        raise RuntimeError("Set EDGAR_USER_AGENT in .env (name and email).")
    wait = 0.2 - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    r = requests.get(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip"},
                     timeout=60)
    _last_call = time.time()
    r.raise_for_status()
    return r.json()


def ticker_to_cik(ticker: str) -> str:
    data = cache.cached("edgar", ["tickers"],
                        lambda: _get("https://www.sec.gov/files/company_tickers.json"),
                        ttl=30 * DAY)
    for row in data.values():
        if row["ticker"].upper() == ticker.upper():
            return str(row["cik_str"]).zfill(10)
    raise KeyError(f"Ticker not found in SEC list: {ticker}")


def _company_facts(cik: str) -> dict:
    return cache.cached(
        "edgar", ["facts", cik],
        lambda: _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"),
        ttl=3 * DAY)


def _submissions(cik: str) -> dict:
    return cache.cached(
        "edgar", ["subs", cik],
        lambda: _get(f"https://data.sec.gov/submissions/CIK{cik}.json"),
        ttl=1 * DAY)


CONCEPTS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues", "SalesRevenueNet"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "equity": ["StockholdersEquity"],
    "total_assets": ["Assets"],
}
FLOW = {"revenue", "gross_profit", "operating_income", "net_income",
        "operating_cash_flow"}


def _d(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _pick(rows: list, as_of: date, kind: str):
    """Latest fact filed on or before as_of. kind: quarter, annual, instant."""
    best = None
    for f in rows:
        if _d(f["filed"]) > as_of:
            continue
        if kind == "instant":
            if "start" in f:
                continue
        else:
            if "start" not in f:
                continue
            days = (_d(f["end"]) - _d(f["start"])).days
            if kind == "quarter" and not 80 <= days <= 100:
                continue
            if kind == "annual" and not (350 <= days <= 380 and f["form"] == "10-K"):
                continue
        key = (f["end"], f["filed"])
        if best is None or key > (best["end"], best["filed"]):
            best = f
    return best


def _quarter_flow(rows: list, as_of: date):
    """Latest single-quarter value for a flow item (revenue, income, cash flow).

    Candidates, newest period wins (a reported quarter beats a derived one):
    - a reported 3-month figure (10-Q income statement items);
    - Q4 derived as the full-year 10-K value minus the 9-month year-to-date value
      (10-Ks have no standalone Q4);
    - cumulative 10-Q figures (cash flow: 6 or 9 months) minus the prior period.
    """
    facts = [f for f in rows if "start" in f and _d(f["filed"]) <= as_of]

    def days(f):
        return (_d(f["end"]) - _d(f["start"])).days

    cands = []
    q = _pick(rows, as_of, "quarter")
    if q:
        cands.append((q["end"], 2, dict(q, derived=None)))
    ann = _pick(rows, as_of, "annual")
    if ann:
        ytd9 = [f for f in facts if f["start"] == ann["start"]
                and f["form"] == "10-Q" and 250 <= days(f) <= 290]
        if ytd9:
            y = max(ytd9, key=lambda f: (f["end"], f["filed"]))
            cands.append((ann["end"], 1, dict(
                ann, val=ann["val"] - y["val"], fp="Q4",
                derived="full year (10-K) minus 9-month year-to-date")))
    ytd = [f for f in facts if f["form"] == "10-Q" and 150 <= days(f) <= 290]
    if ytd:
        cur = max(ytd, key=lambda f: (f["end"], f["filed"]))
        prev = [f for f in facts if f["start"] == cur["start"] and f["end"] < cur["end"]]
        if prev:
            p = max(prev, key=lambda f: (f["end"], f["filed"]))
            cands.append((cur["end"], 0, dict(
                cur, val=cur["val"] - p["val"],
                derived="cumulative year-to-date minus prior period")))
    if not cands:
        return None
    return max(cands, key=lambda c: (c[0], c[1]))[2]


def _fact(facts: dict, name: str, as_of: date, kind: str):
    for concept in CONCEPTS[name]:
        node = facts.get("facts", {}).get("us-gaap", {}).get(concept)
        if not node:
            continue
        rows = node.get("units", {}).get("USD", [])
        if name in FLOW and kind == "quarter":
            f = _quarter_flow(rows, as_of)
        else:
            f = _pick(rows, as_of, "instant" if name not in FLOW else kind)
        if f:
            return {"value": f["val"], "period_end": f["end"], "filed": f["filed"],
                    "form": f["form"], "accession": f["accn"], "concept": concept,
                    "fiscal_year": f.get("fy"), "fiscal_period": f.get("fp"),
                    "derived": f.get("derived")}
    return None


def _ratio(num, den):
    if num and den and den["value"] != 0:
        return round(num["value"] / den["value"], 4)
    return None


def get_financials(ticker: str, as_of: str) -> dict:
    """Latest quarterly and annual figures and ratios known on as_of (YYYY-MM-DD).

    Ratios are computed in Python from reported values, never by the model.
    Every figure carries its form, period, filing date and accession number.
    """
    cutoff = _d(as_of)
    cik = ticker_to_cik(ticker)
    facts = _company_facts(cik)
    out = {"ticker": ticker.upper(), "cik": cik, "as_of": as_of,
           "entity": facts.get("entityName")}
    for kind in ("quarter", "annual"):
        block = {n: _fact(facts, n, cutoff, kind) for n in CONCEPTS}
        block["ratios"] = {
            "gross_margin": _ratio(block["gross_profit"], block["revenue"]),
            "operating_margin": _ratio(block["operating_income"], block["revenue"]),
            "net_margin": _ratio(block["net_income"], block["revenue"]),
            "debt_to_equity": _ratio(block["long_term_debt"], block["equity"]),
        }
        out[kind] = block
    return out


def company_profile(ticker: str) -> dict:
    """Industry classification (SIC code and description) from SEC submissions."""
    sub = _submissions(ticker_to_cik(ticker))
    sic = sub.get("sic")
    return {"name": sub.get("name"), "sic": int(sic) if sic else None,
            "sic_description": sub.get("sicDescription")}


def list_filings(ticker: str, as_of: str, forms=("10-K", "10-Q"), limit: int = 6) -> list:
    """Recent filings filed on or before as_of, newest first, with URLs."""
    cutoff = _d(as_of)
    cik = ticker_to_cik(ticker)
    recent = _submissions(cik)["filings"]["recent"]
    rows = []
    for i, form in enumerate(recent["form"]):
        filed = recent["filingDate"][i]
        if form in forms and _d(filed) <= cutoff:
            accn = recent["accessionNumber"][i]
            doc = recent["primaryDocument"][i]
            rows.append({
                "form": form, "filed": filed, "period": recent["reportDate"][i],
                "accession": accn,
                "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                       f"{accn.replace('-', '')}/{doc}"})
    rows.sort(key=lambda r: r["filed"], reverse=True)
    return rows[:limit]
