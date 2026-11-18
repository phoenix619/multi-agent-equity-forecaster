"""Central configuration: paths, secrets from .env, tickers, model roles."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache"


def _load_env() -> dict:
    env = {}
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


_ENV = _load_env()


def secret(name: str, default: str = "") -> str:
    import os
    return os.environ.get(name) or _ENV.get(name, default)


# Local Ollama by default (OpenAI-compatible API). Override in .env to use any
# other OpenAI-style endpoint (LLM_BASE_URL, LLM_API_KEY).
LLM_BASE_URL = secret("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = secret("LLM_API_KEY", "ollama")

# Optional cap on real (uncached) requests per UTC day. 0 = unlimited (local models).
DAILY_CALL_BUDGET = int(secret("DAILY_CALL_BUDGET", "0"))

# Non-financial large caps across sectors (confirm with the user).
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "XOM", "DAL", "MU",
           "JNJ", "WMT", "HD", "CAT", "NEE", "PG"]
BENCHMARK = "SPY"
HORIZON_TRADING_DAYS = 20

# Model per agent role, run locally through Ollama. Both models passed the
# tool-call / follow-up / JSON tests (scripts/test_local.py), have permissive
# open licenses (Apache 2.0), and run on a 24 GB laptop. The Strategist
# (gpt-oss) and the Critic and judge (gemma4) are different model families, so
# the generator never grades itself.
MODELS = {
    "orchestrator": "gpt-oss:20b",
    "strategist": "gpt-oss:20b",
    "specialist": "gemma4:12b",
    "critic": "gemma4:12b",
    "judge": "gemma4:12b",
}

# Embeddings for filing retrieval (RAG), served by Ollama. IBM Granite family;
# the embedding variant is not individually listed in internal model guidance, so
# confirm before any use beyond local testing.
EMBED_MODEL = secret("EMBED_MODEL", "granite-embedding:278m")
RAG_CHUNK_CHARS = 1200      # about 300 tokens per chunk
RAG_OVERLAP_CHARS = 200
RAG_MMR_LAMBDA = 0.6        # 1 = pure relevance, 0 = pure diversity (Module 3 lab 2)
RAG_TOP_K = 4

# How agents reach tools: "direct" Python calls (default) or "mcp" servers over stdio.
# Read at call time: secret("TOOL_BACKEND").

# Macro series from FRED used by the Macro/News agent.
FRED_SERIES = {
    "oil_wti": "DCOILWTICO",
    "treasury_10y": "DGS10",
    "fed_funds": "DFF",
    "cpi": "CPIAUCSL",
}
