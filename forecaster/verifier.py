"""Deterministic Verifier: checks that numbers in a forecast come from tool results.

No model is involved. It pulls every number out of the draft's rationale, evidence
claims and risks, and looks for a matching figure in the tool results the team
actually retrieved (allowing for display rounding, units like "billion" and "%").
Numbers that match nothing are reported as unverified.

Status (the guardrail light):
  green  - every checked number is backed by a tool result
  orange - 1 to 3 unverified numbers
  red    - 4 or more, or no evidence at all
"""
import re

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_NUM = re.compile(
    r"(?<![\w.])(?P<cur>\$)?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<unit>%|\s?(?:percent|thousand|million|billion|trillion)\b|[bBmMkK]n?\b|x\b)?")
_SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6,
          "billion": 1e9, "b": 1e9, "bn": 1e9, "trillion": 1e12}
# A number followed by these is a window length or a form name, not a data point.
_NOT_DATA = re.compile(r"\s?-?(day|week|month|year|trading|session|K\b|Q\b)", re.I)

# "RSI 14", "SMA 50", "ATR(14)": the number is the indicator's window, not a data point.
_INDICATOR = re.compile(r"(RSI|ATR|SMA|EMA|MA|MACD|ADX|CCI)\s*\(?$", re.I)
_WINDOWS = {5, 9, 10, 12, 14, 20, 21, 26, 50, 100, 200}

GREEN, ORANGE, RED = "green", "orange", "red"


def collect_numbers(obj) -> list:
    """All numeric leaves (absolute values) in nested tool results."""
    out = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(abs(float(obj)))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(collect_numbers(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(collect_numbers(v))
    return out


def extract_claims(text: str) -> list:
    """Numbers in free text that look like data points, with units resolved."""
    text = re.sub(r"[\u00a0\u2009\u202f]", " ", text or "")  # models emit odd spaces
    text = _DATE.sub(" ", text)
    claims = []
    for m in _NUM.finditer(text):
        raw_num, unit = m.group("num"), (m.group("unit") or "").strip()
        after = text[m.end():m.end() + 12]
        if not unit and _NOT_DATA.match(after):
            continue
        if re.search(r"S\s?&\s?P\s*$", text[:m.start()], re.I):
            continue  # "S&P 500"
        decimals = len(raw_num.split(".")[1]) if "." in raw_num else 0
        value = float(raw_num.replace(",", ""))
        u = unit.lower()
        pct = u in ("%", "percent")
        scale = 1.0 if (pct or u in ("", "x")) else _SCALE.get(u, 1.0)
        if pct and value == 50:
            continue  # the 50% base rate / neutral benchmark, not a data point
        if not unit and not m.group("cur") and decimals == 0 and (
                1900 <= value <= 2100 or value < 10):
            continue  # years and small counts
        window = bool(not unit and decimals == 0 and value in _WINDOWS
                      and _INDICATOR.search(text[:m.start()]))
        claims.append({"raw": m.group(0).strip(), "value": value * scale,
                       "tol": 0.5 * 10 ** (-decimals) * scale + 1e-9, "pct": pct,
                       "window": window})
    return claims


def _matches(claim: dict, pool: list) -> bool:
    v, tol = claim["value"], claim["tol"]
    for t in pool:
        if abs(t - v) <= tol:
            return True
        if claim["pct"] and abs(t * 100 - v) <= tol:
            return True
    return False


def forecast_texts(forecast: dict) -> list:
    """(where, text) pairs of the draft fields that carry figures."""
    texts = [("rationale", forecast.get("rationale") or "")]
    for i, e in enumerate(forecast.get("evidence") or []):
        if isinstance(e, dict):
            texts.append((f"evidence[{i}]", e.get("claim") or ""))
    for i, r in enumerate(forecast.get("risks") or []):
        if isinstance(r, dict):
            texts.append((f"risks[{i}]", r.get("risk") or ""))
    return texts


def verify_forecast(forecast: dict, tool_results: dict) -> dict:
    """Check every number in the draft against the tool results."""
    pool = collect_numbers(tool_results)
    checked, unverified = 0, []
    for where, text in forecast_texts(forecast):
        for c in extract_claims(text):
            if _matches(c, pool):
                checked += 1
            elif not c["window"]:  # an unmatched indicator window is not a claim
                checked += 1
                unverified.append({"where": where, "number": c["raw"]})
    n = len(unverified)
    no_evidence = not (forecast.get("evidence") or [])
    status = RED if (n >= 4 or no_evidence) else ORANGE if n else GREEN
    return {"status": status, "checked": checked, "unverified": unverified,
            "no_evidence": no_evidence}


def drop_unverified(forecast: dict, unverified: list) -> dict:
    """Remove evidence and risk items that contain an unverified number."""
    bad = {u["where"] for u in unverified}
    out = dict(forecast)
    out["evidence"] = [e for i, e in enumerate(forecast.get("evidence") or [])
                       if f"evidence[{i}]" not in bad]
    out["risks"] = [r for i, r in enumerate(forecast.get("risks") or [])
                    if f"risks[{i}]" not in bad]
    return out
