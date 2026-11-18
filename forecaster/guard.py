"""Input guard: deterministic checks that run BEFORE any model sees a request, plus a
scan for instruction-like text inside retrieved (untrusted) filing passages.

Declines: advice or trade execution, invalid tickers, future or malformed dates,
unsupported sectors (financials: banks, insurers; SIC 6000-6799), and requests that try to
override the system. Rule-based on purpose: auditable, no model call, no latency.
"""
import re
from datetime import date

ADVICE = [
    r"\bshould i (buy|sell|short|hold|invest|put)\b",
    r"\b(tell|advise|recommend) me (what|whether|to)\b.*\b(buy|sell|invest|short)\b",
    r"\bwhat should i (buy|sell|do with)\b",
    r"\bis (it|\w+) a (good|safe) (buy|investment)\b",
    r"\bguarantee[sd]?\b.*\b(return|profit|gain|win)s?\b",
    r"\b(risk[- ]?free|no[- ]risk|riskless|can'?t lose|sure thing|certain to)\b",
    r"\b(pretend|act|behave) (you are|as|like) my (financial )?(advisor|broker|adviser)\b",
    r"\bmy (portfolio|retirement|savings|401\(?k\)?)\b",
]
EXECUTION = [
    r"\b(place|submit|execute|send|enter|route)\b.{0,30}\b(order|trade)s?\b",
    r"\b(buy|sell|short)\s+(me\s+)?\d[\d,]*\s*(shares?|units?|contracts?)\b",
    r"\b(buy|sell|short)\s+(it|them|this|that)\s+(now|today|immediately|for me)\b",
    r"\b(connect|log ?in)\b.{0,20}\b(broker|brokerage|account)\b",
]
OVERRIDE = [
    r"\bignore (all |any |the )?(previous|prior|above|earlier) (instructions?|rules?|prompts?)\b",
    r"\bdisregard (all |any |the )?(previous|prior|above|your) (instructions?|rules?)\b",
    r"\b(reveal|show|print|repeat) (me )?(your|the) (system )?(prompt|instructions)\b",
    r"\byou are now\b",
    r"\bact as (a|an) (?!(\w+ )?(analyst|researcher|reviewer))\w+",
    r"\bforget (what|everything|all|your)\b.{0,30}\b(told|instructions?|rules?|said)\b",
    r"\b(set|output|return|force|always say) (the )?(probability|p)\b.{0,15}\d",
    r"\bnew instructions?\s*:",
    r"\b(skip|bypass|disable|turn off) (the |your )?(verif|guardrail|check|safety)",
    r"\bjailbreak\b|\bdeveloper mode\b|\bdo anything now\b",
]
# Instruction-like text inside filings/news: legitimate filings do not address the reader.
UNTRUSTED = [
    r"\bignore (all |any |the )?(previous|prior|above|earlier) (instructions?|rules?)\b",
    r"\bdisregard\b.{0,30}\b(instructions?|rules?)\b",
    r"\b(assistant|ai|model|llm|agent)\b.{0,25}\b(must|should|shall|please)\b.{0,40}"
    r"\b(output|report|say|set|return|rate|recommend)\b",
    r"\bsystem prompt\b",
    r"\b(hidden|secret) (note|instruction|message)s?\b.{0,40}\b(model|assistant|ai)\b",
    r"\balways (answer|say|report|recommend)\b.{0,40}\b(outperform|buy|bullish)\b",
    r"\bset the probability\b|\bprobability (to|of) 0?\.\d+ regardless\b",
    r"\bdo not (mention|reveal|tell)\b.{0,30}\b(user|reviewer|critic)\b",
]
FINANCIAL_SIC = range(6000, 6800)
_TICKER = re.compile(r"^[A-Z]{1,5}([.-][A-Z])?$")

_c = lambda pats: [re.compile(p, re.I) for p in pats]
_ADVICE, _EXEC, _OVER, _UNTRUSTED = _c(ADVICE), _c(EXECUTION), _c(OVERRIDE), _c(UNTRUSTED)


def _hit(pats, text):
    return any(p.search(text) for p in pats)


def _no(category: str, reason: str) -> dict:
    return {"allowed": False, "category": category, "reason": reason}


def check_request(ticker, as_of, request: str = "", today=None, sic=None) -> dict:
    """Return {"allowed": bool, "category": str|None, "reason": str}.

    sic: optional SIC code of the company (None = unknown, sector check skipped)."""
    text = request or ""
    if _hit(_OVER, text):
        return _no("injection", "The request tries to override the system's instructions.")
    if _hit(_EXEC, text):
        return _no("execution", "This tool never places trades or connects to brokerages.")
    if _hit(_ADVICE, text):
        return _no("advice", "This is a research tool, not personal investment advice. It can "
                             "estimate P(outperform S&P 500) and show evidence instead.")
    t = (ticker or "").strip().upper()
    if not _TICKER.match(t):
        return _no("invalid_ticker", f"'{ticker}' is not a valid US ticker symbol.")
    try:
        d = date.fromisoformat(str(as_of))
    except ValueError:
        return _no("invalid_date", f"'{as_of}' is not a valid YYYY-MM-DD date.")
    if d > (today or date.today()):
        return _no("future_date", "The as-of date is in the future; no data exists yet.")
    if sic is not None and int(sic) in FINANCIAL_SIC:
        return _no("unsupported_sector", "Banks, insurers and other financials are not "
                   "supported: the fundamentals and ratios here are built for non-financials.")
    return {"allowed": True, "category": None, "reason": ""}


def is_injection(text: str) -> bool:
    return _hit(_UNTRUSTED, text or "")


def drop_injected(results: list) -> tuple:
    """Remove retrieved passages that contain instruction-like text. Returns (kept, n_removed)."""
    kept = [r for r in results if not is_injection(r.get("text", ""))]
    return kept, len(results) - len(kept)


def refusal(res_base: dict, verdict: dict, variant: str) -> dict:
    """A result shaped like a normal run result, with no forecast and a visible refusal."""
    return {**res_base, "variant": variant, "forecast": None, "model": None,
            "model_calls": 0, "tool_calls": 0, "seconds": 0, "trace": [],
            "error": None, "refused": verdict,
            "disclaimer": "Research tool, not financial advice. No trades executed."}
