# Multi-Agent Equity Forecaster

Capstone project for the CMU Agentic AI program (Forecaster option).

For a given stock, the system estimates **P(stock outperforms the S&P 500 over the next 20 trading days)**, cites the evidence behind the estimate, and proposes a **paper-only** entry zone. It is a research prototype, not investment advice, and it never places trades.

## Current implementation

The repository contains four runnable variants: a single-agent ReAct baseline, parallel specialists, specialists with a Critic and deterministic Verifier, and the Critic/Verifier pipeline with filing RAG. It also includes optional MCP transport, a Streamlit UI, live forecast logs, a scored 165-pair historical comparison, and guardrail evaluation artifacts.

| Component | Current state |
|---|---|
| SEC EDGAR fundamentals and filings; yfinance prices; FRED vintage-aware macro | Implemented |
| Technical indicators and paper-trade timing levels | Implemented in Python |
| Single-agent ReAct and multi-agent LangGraph variants | Implemented |
| Filing RAG over selected 10-K/10-Q sections with local embeddings and MMR | Implemented in `multi_rag` |
| MCP servers for EDGAR, prices, and FRED | Implemented; optional local stdio transport |
| Numeric Verifier, request/injection guard, bounded revision and abstention | Implemented; unsupported claims are removed or withheld, with review flags; Critic failure abstains at 50% |
| Streamlit interface, trace logging, backtests, scoring, and figures | Implemented |
| Analyst ratings and news data | Not connected; the role reports a gap |
| Tree-of-Thought search | Planned in a checkpoint, not implemented in the current runtime |
| Human review queue | Not implemented; the UI can flag a result for review |

## Design

- **Agents:** the orchestrator fans out fundamentals, macro, and technical specialists. Analyst/news is an explicit missing-source stub; the RAG variant adds a filings specialist. A Strategist proposes the probability, a Critic from a different model family challenges it, and a deterministic Verifier checks numeric claims against saved tool results.
- **Look-ahead protection:** the as-of date is fixed by code and never chosen by the model. Every data tool only returns what was known on that date (filing dates, FRED vintages, price cut-off).
- **Deterministic where it matters:** the rating is a fixed function of the probability, coverage gaps and trade levels are computed in code, and probabilities are clamped to [0.05, 0.95].
- **Local models:** runs on Ollama (`gpt-oss:20b`, `gemma4:12b`). The generator and the grader come from different model families.
- **Evaluation:** label = stock return > SPY return over 20 trading days. The checked-in common backtest has 165 ticker/date pairs. `multi_rag` has Brier 0.2578, `multi_critic_verifier` 0.2586, `multi_specialists` 0.2650, and the single-agent baseline 0.2660; the constant 50% baseline scores 0.2500. No variant beats the constant 50% forecast, and none has ranking skill (AUC 0.40 to 0.44). The Critic/Verifier and RAG variants lower Brier versus the single-agent baseline mainly by pulling forecasts toward 50%. With 11 dates the intervals are wide, so small real effects cannot be ruled out.

See [the architecture walkthrough](docs/architecture.md) for a block diagram, request sequence, component contracts, and failure paths.

## Project layout

```
forecaster/        core package
  config.py        tickers, model roles, paths, secrets
  llm.py           OpenAI-compatible chat client (Ollama by default), cached
  baseline.py      single-agent ReAct baseline, forecast logging
  multi_agent.py   LangGraph specialists, Strategist, Critic and routing
  verifier.py      deterministic numeric checks against tool results
  guard.py         request screening and retrieved-text injection filter
  rag.py           filing retrieval, embeddings, and MMR
  tools.py         tool schemas and implementations given to the agent
  indicators.py    technical indicators and paper-trade levels
  macro.py         which macro series matter for which company
  scoring.py       outcomes, Brier score, calibration
  data/            EDGAR, prices, FRED clients
mcp_servers/       read-only FastMCP servers over the same data
scripts/           run_baseline, run_universe, run_backtest, score, smoke tests
ui/                Streamlit forecast and evaluation interface
tests/             offline unit tests
eval/              logged forecasts (JSONL, committed as timestamp proof)
logs/              per-run traces with tool results
```

## Setup

Requires Python 3.11+ and [Ollama](https://ollama.com).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # then fill in FRED_API_KEY and EDGAR_USER_AGENT

ollama pull gpt-oss:20b
ollama pull gemma4:12b
ollama pull granite-embedding:278m   # filing retrieval (multi_rag variant)
```

- `FRED_API_KEY`: free at fred.stlouisfed.org.
- `EDGAR_USER_AGENT`: SEC requires a contact string, e.g. `Your Name you@example.com`.

## Usage

Run from the repo root.

```bash
python scripts/smoke_data.py                      # check the data tools live
python scripts/test_local.py                      # check local models (tool calls, JSON)
python scripts/run_baseline.py AAPL               # one forecast, as of today
python scripts/run_baseline.py AAPL 2026-09-21    # as of a past date
python scripts/run_baseline.py AAPL --variant multi_critic_verifier   # multi-agent
python scripts/run_baseline.py AAPL --variant multi_rag --mcp          # + filing retrieval, tools via MCP
python scripts/run_universe.py                    # all tickers, appended to eval/forecasts/live.jsonl
python scripts/run_backtest.py --dates 2026-09-21 # resumable historical backtest
python scripts/run_baseline.py AAPL --request "Should I buy it?"   # declined by the input guard
python scripts/run_backtest.py --variant multi_rag                  # resumable; one variant at a time
python scripts/score.py [live]                    # score logged forecasts
python scripts/score.py --common                  # compare variants on identical (ticker, date) pairs
python scripts/benchmarks.py                      # non-LLM rules and permutation test (needs network for SPY)
python scripts/eval_guardrails.py                 # input-guard adversarial set (offline)
python scripts/make_figures.py                    # charts into docs/figures
streamlit run ui/app.py                           # demo interface
```

Responses and API data are cached in `.cache/` (git-ignored).

## Evaluation artifacts

`eval/backtest/scored_backtest_common.json` contains the shared 165-pair results; `eval/backtest/benchmarks.json` includes rule baselines and permutation results; `eval/guardrails/results.json` contains development and held-out request/passages checks; and `docs/figures/` contains plots and guardrail-ablation data. The Streamlit Backtest results tab displays a summary.

## Tests

Offline unit tests (no network, no Ollama) cover indicators, scoring, macro rules, cache, EDGAR quarter derivation and look-ahead checks, RAG, input guards, verification, and pipeline behavior.

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Known limitations

- The baseline is noisy run to run (the same ticker can vary by several points) and tends to cluster at round numbers.
- Technical indicators are meant for entry timing, not direction, but the baseline sometimes leans on them for direction.
- Historical performance is below a constant 50% forecast on the checked-in comparison, so predictive value has not been demonstrated.
- A 20-day horizon on 15 large caps gives a small sample, so Brier differences between variants may not be statistically significant.
- The Verifier checks numeric claims against collected tool output. It does not prove every citation supports its qualitative statement.
- Review routing is surfaced as a flag; no person is automatically notified and no approval workflow is connected.
- Forecast logs are scored after outcomes mature, but prior outcomes or lessons are not retrieved by the live forecasting agents.

## Disclaimer

Educational use only. Nothing here is financial advice. Paper trading only.
