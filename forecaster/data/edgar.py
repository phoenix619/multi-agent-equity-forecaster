"""SEC EDGAR: financial facts, ratios and filing lists, as of a given date."""
import time
from datetime import date, datetime, timedelta

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


# Companies that reorganized into a new registrant: the SEC ticker lookup returns the
# new CIK, whose structured data starts at the reorganization. Merge the predecessor's
# history so earlier as-of dates have fundamentals. (XOM -> ExxonMobil Holdings.)
PREDECESSORS = {"XOM": ["0000034088"]}


def _merged_facts(ticker: str, cik: str) -> dict:
    facts = _company_facts(cik)
    extra = PREDECESSORS.get(ticker.upper(), [])
    if not extra:
        return facts
    gaap = {c: {"units": {u: list(rows) for u, rows in n.get("units", {}).items()}}
            for c, n in facts.get("facts", {}).get("us-gaap", {}).items()}
    for pcik in extra:
        try:
            old = _company_facts(pcik)
        except Exception:  # predecessor unavailable: fall back to the successor alone
            continue
        for concept, node in old.get("facts", {}).get("us-gaap", {}).items():
            units = gaap.setdefault(concept, {"units": {}})["units"]
            for unit, rows in node.get("units", {}).items():
                have = units.setdefault(unit, [])
                seen = {(r["end"], r.get("start"), r["filed"], r["accn"], r["val"])
                        for r in have}
                have.extend(r for r in rows
                            if (r["end"], r.get("start"), r["filed"], r["accn"], r["val"])
                            not in seen)
    return {"entityName": facts.get("entityName"), "facts": {"us-gaap": gaap}}


def _submissions(cik: str) -> dict:
    return cache.cached(
        "edgar", ["subs", cik],
        lambda: _get(f"https://data.sec.gov/submissions/CIK{cik}.json"),
        ttl=1 * DAY)


CONCEPTS = {
    # Companies switch XBRL concepts over time (NEE's "Revenues" stops in 2013), so
    # _fact takes whichever concept has the most recent period, not the first listed.
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "Revenues", "SalesRevenueNet", "RegulatedAndUnregulatedOperatingRevenue"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss",
                   "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt",
                       "LongTermDebtAndCapitalLeaseObligations"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "total_assets": ["Assets"],
}
MAX_AGE_DAYS = 460  # a figure older than ~15 months is "not reported", never "latest"
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
    """Most recent figure across all candidate concepts (latest period, then filing)."""
    best = None
    for concept in CONCEPTS[name]:
        node = facts.get("facts", {}).get("us-gaap", {}).get(concept)
        if not node:
            continue
        rows = node.get("units", {}).get("USD", [])
        if name in FLOW and kind == "quarter":
            f = _quarter_flow(rows, as_of)
        else:
            f = _pick(rows, as_of, "instant" if name not in FLOW else kind)
        if f and (best is None or (f["end"], f["filed"]) > (best[0]["end"], best[0]["filed"])):
            best = (f, concept)
    if not best or (as_of - _d(best[0]["end"])).days > MAX_AGE_DAYS:
        return None
    f, concept = best
    return {"value": f["val"], "period_end": f["end"], "filed": f["filed"],
            "form": f["form"], "accession": f["accn"], "concept": concept,
            "fiscal_year": f.get("fy"), "fiscal_period": f.get("fp"),
            "derived": f.get("derived")}


def _ratio(num, den):
    """num / den, only when both figures cover the same period end (never mix periods)."""
    if num and den and den["value"] != 0 and num.get("period_end") == den.get("period_end"):
        return round(num["value"] / den["value"], 4)
    return None


def get_financials(ticker: str, as_of: str) -> dict:
    """Latest quarterly and annual figures and ratios known on as_of (YYYY-MM-DD).

    Ratios are computed in Python from reported values, never by the model.
    Every figure carries its form, period, filing date and accession number.
    """
    cutoff = _d(as_of)
    cik = ticker_to_cik(ticker)
    facts = _merged_facts(ticker, cik)
    out = {"ticker": ticker.upper(), "cik": cik, "as_of": as_of,
           "entity": facts.get("entityName")}
    for kind in ("quarter", "annual"):
        block = {n: _fact(facts, n, cutoff, kind) for n in CONCEPTS}
        block["ratios"] = {
            "gross_margin": _ratio(block["gross_profit"], block["revenue"]),
            "operating_margin": _ratio(block["operating_income"], block["revenue"]),
            "net_margin": _ratio(block["net_income"], block["revenue"]),
        }
        if kind == "quarter":  # balance-sheet items are point-in-time: one set, latest
            block["ratios"]["debt_to_equity"] = _ratio(block["long_term_debt"], block["equity"])
        out[kind] = block
    out["freshness"] = _freshness(ticker, cutoff, out["quarter"])
    return out


def _freshness(ticker: str, cutoff: date, quarter: dict):
    """Is the newest filing reflected in SEC's structured data? (It sometimes lags.)"""
    latest = list_filings(ticker, cutoff.isoformat(), limit=1)
    if not latest:
        return None
    lf = latest[0]
    periods = [v["period_end"] for v in quarter.values()
               if isinstance(v, dict) and v.get("period_end")]
    have = max(periods) if periods else None
    lag = bool(lf["period"] and have
               and _d(lf["period"]) > _d(have) + timedelta(days=20))
    return {"latest_filing": {"form": lf["form"], "filed": lf["filed"], "period": lf["period"]},
            "latest_period_in_structured_data": have, "lag": lag}


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
