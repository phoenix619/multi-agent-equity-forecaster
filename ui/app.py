"""Streamlit demo for the Multi-Agent Equity Forecaster.
Run from the equity-forecaster folder:
    streamlit run ui/app.py
Research prototype: paper-only, not investment advice.
"""
import json
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecaster import baseline, variants          # noqa: E402
from forecaster.config import TICKERS              # noqa: E402

VARIANT_LABELS = {
    "single_agent_react": "1. Single agent (baseline)",
    "multi_specialists": "2. + Parallel specialists",
    "multi_critic_verifier": "3. + Critic and Verifier",
    "multi_rag": "4. + Filing RAG (final)",
}
LIGHT = {"green": "🟢 green", "orange": "🟠 orange", "red": "🔴 red"}

st.set_page_config(page_title="Multi-Agent Equity Forecaster", layout="wide")
st.title("Multi-Agent Equity Forecaster")
st.caption("P(stock beats the S&P 500 over the next 20 trading days). Paper-only research "
           "prototype, not investment advice.")

with st.sidebar:
    st.header("Forecast")
    ticker = st.selectbox("Ticker", TICKERS)
    variant = st.selectbox("Architecture", list(VARIANT_LABELS), index=3,
                           format_func=VARIANT_LABELS.get)
    as_of = st.date_input("As of", value=date.today(), max_value=date.today())
    request = st.text_input("Optional question (screened by the input guard)", "")
    use_mcp = st.checkbox("Call tools through MCP servers", value=False)
    run = st.button("Run forecast", type="primary")
    st.markdown("---")
    traces = sorted((ROOT / "logs").glob("*.json"), key=lambda p: p.stat().st_mtime,
                    reverse=True) if (ROOT / "logs").exists() else []
    saved = st.selectbox("Or load a saved run", [""] + [p.name for p in traces[:50]])


def show(res: dict):
    f = res.get("forecast")
    st.subheader(f"{res['ticker']} as of {res['as_of']}  ·  {VARIANT_LABELS.get(res['variant'], res['variant'])}")
    if res.get("refused"):
        st.warning(f"Request declined by the input guard ({res['refused']['category']}): "
                   f"{res['refused']['reason']}")
        return
    if res.get("error"):
        st.error(f"Run failed: {res['error']}")
    if not f:
        return
    ver = f.get("verification") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P(outperform S&P 500, 20d)", f"{f['probability']:.0%}")
    c2.metric("Rating", f.get("rating", "-"))
    c3.metric("Guardrail", LIGHT.get(ver.get("status"), "n/a (baseline)"))
    c4.metric("Time", f"{res.get('seconds', 0):.0f}s", f"{res.get('model_calls')} model / "
              f"{res.get('tool_calls')} tool calls", delta_color="off")
    if ver.get("abstained"):
        st.warning("The guardrail abstained (red): too many unverified numbers, so the "
                   "forecast was reset to the 50% base rate.")
    st.markdown("**Rationale**")
    st.write(f.get("rationale", ""))

    left, right = st.columns(2)
    with left:
        st.markdown("**Evidence** (every number is checked against tool results)")
        ev = pd.DataFrame(f.get("evidence") or [])
        st.dataframe(ev, use_container_width=True, hide_index=True) if len(ev) else st.write("None")
    with right:
        st.markdown("**Risks** (each backed by a tool result)")
        rk = pd.DataFrame(f.get("risks") or [])
        st.dataframe(rk, use_container_width=True, hide_index=True) if len(rk) else st.write("None")

    pt = f.get("paper_trade")
    if pt:
        st.markdown("**Paper trade** (computed from technicals, not by the model)")
        a, b, c = st.columns(3)
        a.metric("Entry zone", f"{pt['entry_zone'][0]} to {pt['entry_zone'][1]}")
        b.metric("Stop", pt["stop_loss"])
        c.metric("Target", pt["target"])
        st.caption(pt.get("signal", ""))

    st.markdown("**Gaps** (computed from what was actually available)")
    for g in f.get("gaps", []):
        st.write(f"- {g}")

    if ver:
        with st.expander("Verification and Critic detail"):
            st.write(f"Verifier: {ver['status']}, {ver['checked']} numbers checked, "
                     f"{len(ver['unverified'])} unverified, {f.get('revisions', 0)} revision(s).")
            for u in ver["unverified"]:
                st.write(f"- unverified `{u['number']}` in {u['where']}")
            crit = f.get("critic") or {}
            if crit:
                st.write(f"Critic verdict: **{crit.get('verdict')}**"
                         f"{' (no valid answer, defaulted)' if crit.get('failed') else ''}")
                if crit.get("weakest_claim"):
                    st.write(f"Weakest claim: {crit['weakest_claim']}")
                for i in crit.get("issues", []):
                    st.write(f"- {i.get('type')}: {i.get('detail')}")
    with st.expander("Agent timing and tool calls"):
        rows = [{"agent": t["agent"], "seconds": t["seconds"]}
                for t in res.get("trace", []) if t.get("type") == "timing"]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        for t in res.get("trace", []):
            if t.get("type") == "tool":
                st.write(f"{t.get('agent', 'agent')}: `{t['tool']}({t['args']})` -> "
                         f"{'ok' if t['ok'] else 'ERROR'}")
    st.caption(res.get("disclaimer", ""))


tab1, tab2, tab3 = st.tabs(["Forecast", "Backtest results", "Live forecasts"])

with tab1:
    if run:
        if use_mcp:
            os.environ["TOOL_BACKEND"] = "mcp"
        else:
            os.environ.pop("TOOL_BACKEND", None)
        with st.spinner("Specialists, strategist, critic and verifier are working "
                        "(1 to 3 minutes on local models)..."):
            res = variants.run(variant, ticker, as_of.isoformat(), request)
            if not res.get("refused"):
                baseline.save_trace(res)
        st.session_state["res"] = res
    elif saved:
        st.session_state["res"] = json.loads((ROOT / "logs" / saved).read_text())
    if "res" in st.session_state:
        show(st.session_state["res"])
    else:
        st.info("Pick a ticker and architecture in the sidebar, then press Run forecast, "
                "or load a saved run.")

with tab2:
    p = ROOT / "eval" / "backtest" / "scored_backtest_common.json"
    if not p.exists():
        st.info("No scored backtest found.")
    else:
        d = json.loads(p.read_text())
        st.markdown("11 dates x 15 tickers = 165 shared (ticker, date) pairs. Outcome: stock "
                    "return beat SPY over the next 20 trading days. Lower Brier is better; the "
                    "constant 50% forecast scores 0.2500.")
        tbl = pd.DataFrame([{"Variant": VARIANT_LABELS.get(k, k), "n": v["n"],
                             "Brier": v["brier"], "Accuracy": v["accuracy"]}
                            for k, v in d.items()])
        st.dataframe(tbl, hide_index=True, use_container_width=True)
        st.bar_chart(tbl.set_index("Variant")["Brier"])
        st.markdown("**Calibration**: what actually happened vs the forecast")
        order = ["<40%", "40-50%", "50-60%", ">=60%"]
        v = st.selectbox("Variant", list(d), format_func=lambda k: VARIANT_LABELS.get(k, k),
                         index=list(d).index("multi_rag") if "multi_rag" in d else 0)
        cal = pd.DataFrame([{"bin": b, "mean forecast": d[v]["calibration"][b]["mean_forecast"],
                             "share that beat SPY": d[v]["calibration"][b]["outperformed"],
                             "n": d[v]["calibration"][b]["n"]}
                            for b in order if b in d[v]["calibration"]])
        st.dataframe(cal, hide_index=True)
        st.line_chart(cal.set_index("bin")[["mean forecast", "share that beat SPY"]])
        st.info("Honest result: no variant beats the constant 50% forecast. The Critic and "
                "RAG variants lower Brier mainly by shrinking forecasts toward 50%; ranking "
                "ability (AUC) stays at or below 0.5.")

with tab3:
    p = ROOT / "eval" / "forecasts" / "live.jsonl"
    if not p.exists():
        st.info("No live forecasts logged yet.")
    else:
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        df = pd.DataFrame(rows)
        cols = [c for c in ("ticker", "as_of", "variant", "probability", "rating",
                            "verification", "created_at") if c in df]
        st.caption("Timestamped before outcomes exist; scored after the 20-day window closes.")
        st.dataframe(df[cols].sort_values(["as_of", "ticker"], ascending=[False, True]),
                     hide_index=True, use_container_width=True)
