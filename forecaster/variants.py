"""One entry point for the four architecture variants used in the ablation."""
from . import baseline

VARIANTS = ["single_agent_react", "multi_specialists", "multi_critic_verifier", "multi_rag"]
# A Tree-of-Thought judge, if built, becomes a fifth variant.


from datetime import datetime, timezone


def _sic(ticker: str):
    """SIC code from SEC data; None when unavailable (the sector check is then skipped)."""
    try:
        from .data import edgar
        return edgar.company_profile(ticker).get("sic")
    except Exception:
        return None


def run(variant: str, ticker: str, as_of: str, request: str = "") -> dict:
    from . import guard
    verdict = guard.check_request(ticker, as_of, request)
    if verdict["allowed"] and guard._TICKER.match(ticker.strip().upper()):
        verdict = guard.check_request(ticker, as_of, request, sic=_sic(ticker))
    if not verdict["allowed"]:
        return guard.refusal({"ticker": ticker.upper(), "as_of": as_of,
                              "created_at": datetime.now(timezone.utc).isoformat()},
                             verdict, variant)
    if variant == "single_agent_react":
        return baseline.run(ticker, as_of)
    if variant in VARIANTS:
        from . import multi_agent  # imports langgraph lazily inside build_graph
        return multi_agent.run(ticker, as_of, variant)
    raise ValueError(f"unknown variant {variant}; choose from {VARIANTS}")
