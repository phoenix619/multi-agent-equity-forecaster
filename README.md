# Multi-Agent Equity Forecaster

Capstone project for the CMU Agentic AI program (Forecaster option).

For a given stock, the system estimates **P(stock outperforms the S&P 500 over the next 20 trading days)**, cites the evidence behind the estimate, and proposes a **paper-only** entry zone. It is a research prototype, not investment advice, and it never places trades.

## Status

Work in progress (final deliverables due Nov 20, 2026).

| Component | State |
|---|---|
| Data tools: SEC EDGAR (XBRL), prices (yfinance), FRED (vintage-aware) | Done |
| Technical indicators (Wilder RSI/ATR, MACD, Bollinger, SMA) | Done |
| Macro relevance rules by SIC code | Done |
| Single-agent ReAct baseline | Done |
| Forecast log + scorer (Brier, calibration) | Done |
| Unit tests (51, offline) | Done |
| Historical backtest runner | Ready, not yet run |
| MCP servers (EDGAR, prices, FRED) | Written, not yet wired into the agent |
| Deterministic Verifier (numbers vs tool results) | Done |
| Multi-agent LangGraph pipeline: specialists, Strategist, Critic, guardrail routing | Built, first live runs pending |
| Analyst and news data sources | Planned (specialist slot reports a gap) |
| RAG over 10-K/10-Q filings (MMR retrieval) | Planned |
| Tree-of-Thought judge, guardrail routing, ablation | Planned |
| Streamlit chat UI | Planned |

## Design

- **Agents:** an Orchestrator routes work to four specialists (fundamentals, analyst, macro/news, technical). A Strategist proposes the probability, a Critic challenges it, and a Verifier checks every cited number against saved tool results.
- **Look-ahead protection:** the as-of date is fixed by code and never chosen by the model. Every data tool only returns what was known on that date (filing dates, FRED vintages, price cut-off).
- **Deterministic where it matters:** the rating is a fixed function of the probability, coverage gaps and trade levels are computed in code, and probabilities are clamped to [0.05, 0.95].
- **Local models:** runs on Ollama (`gpt-oss:20b`, `gemma4:12b`). The generator and the grader come from different model families.
- **Evaluation:** label = stock return > SPY return over 20 trading days. Scored with Brier score against always-50% and base-rate baselines, plus calibration bins. Historical backtest plus live forecasts.

## Project layout

```
forecaster/        core package
  config.py        tickers, model roles, paths, secrets
  llm.py           OpenAI-compatible chat client (Ollama by default), cached
  baseline.py      single-agent ReAct baseline, forecast logging
  tools.py         tool schemas and implementations given to the agent
  indicators.py    technical indicators and paper-trade levels
  macro.py         which macro series matter for which company
  scoring.py       outcomes, Brier score, calibration
  data/            EDGAR, prices, FRED clients
mcp_servers/       read-only FastMCP servers over the same data
scripts/           run_baseline, run_universe, run_backtest, score, smoke tests
tests/             pytest unit tests
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
python scripts/run_universe.py                    # all tickers, appended to eval/forecasts/live.jsonl
python scripts/run_backtest.py --dates 2026-09-21 # resumable historical backtest
python scripts/score.py [live]                    # score logged forecasts
```

Responses and API data are cached in `.cache/` (git-ignored).

## Tests

Offline unit tests (no network, no Ollama) cover the indicators, scoring and outcome logic, macro rules, cache, EDGAR quarter derivation (including look-ahead checks), and the baseline's rating and coverage-gap logic.

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Known limitations

- The baseline is noisy run to run (the same ticker can vary by several points) and tends to cluster at round numbers.
- Technical indicators are meant for entry timing, not direction, but the baseline sometimes leans on them for direction.
- Generic risks remain until filing risk-factor text is retrievable (RAG step).
- A 20-day horizon on 15 large caps gives a small sample, so Brier differences between variants may not be statistically significant.

## Disclaimer

Educational use only. Nothing here is financial advice. Paper trading only.
