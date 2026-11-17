"""One entry point for the four architecture variants used in the ablation."""
from . import baseline

VARIANTS = ["single_agent_react", "multi_specialists", "multi_critic_verifier"]
# A fourth variant (+ filing RAG and Tree-of-Thought judge) is added after the RAG step.


def run(variant: str, ticker: str, as_of: str) -> dict:
    if variant == "single_agent_react":
        return baseline.run(ticker, as_of)
    if variant in VARIANTS:
        from . import multi_agent  # imports langgraph lazily inside build_graph
        return multi_agent.run(ticker, as_of, variant)
    raise ValueError(f"unknown variant {variant}; choose from {VARIANTS}")
