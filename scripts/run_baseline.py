"""Run one forecast for one ticker with any architecture variant.
Run from the equity-forecaster folder:
    python scripts/run_baseline.py AAPL                  # single-agent baseline, as of today
    python scripts/run_baseline.py DAL 2026-11-13        # as of a past date
    python scripts/run_baseline.py AAPL --log            # also append to the live forecast log
    python scripts/run_baseline.py AAPL --variant multi_critic_verifier
    python scripts/run_baseline.py AAPL --variant multi_rag --mcp   # filing RAG, tools via MCP
Variants: single_agent_react (default), multi_specialists, multi_critic_verifier, multi_rag
"""
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecaster import baseline, variants


def main():
    argv = sys.argv[1:]
    variant = "single_agent_react"
    if "--mcp" in argv:                       # call tools through MCP servers instead
        import os
        os.environ["TOOL_BACKEND"] = "mcp"
        argv.remove("--mcp")
    if "--variant" in argv:
        i = argv.index("--variant")
        variant = argv[i + 1]
        del argv[i:i + 2]
    args = [a for a in argv if not a.startswith("--")]
    ticker = args[0] if args else "AAPL"
    as_of = args[1] if len(args) > 1 else date.today().isoformat()

    res = variants.run(variant, ticker, as_of)
    trace_path = baseline.save_trace(res)

    print(f"\n{res['ticker']} as of {as_of}  variant={res['variant']}  model={res['model']}  "
          f"{res['model_calls']} model calls, {res['tool_calls']} tool calls, {res['seconds']}s")
    for t in res["trace"]:
        if t["type"] == "tool":
            print(f"  {t.get('agent', 'agent')} step {t['step']}: {t['tool']}({t['args']}) -> "
                  f"{'ok' if t['ok'] else 'ERROR'} ({t['result_chars']} chars)")
        elif t.get("type") == "timing":
            print(f"  {t['agent']}: {t['seconds']}s")
        elif t.get("agent") == "critic" and t.get("type") == "review":
            print(f"  critic: {t['verdict']}{'  (NO VALID ANSWER - defaulted)' if t.get('failed') else ''}")
            if t.get("weakest_claim"):
                print(f"    weakest claim: {t['weakest_claim']}")
            for i in t.get("issues", []):
                print(f"    {i.get('type')}: {i.get('detail')}")
        elif t.get("agent") == "verifier":
            print(f"  verifier: {t.get('status')}")
    if res["error"]:
        print("\nERROR:", res["error"])
    f = res["forecast"]
    if f:
        print(f"\nP(outperform S&P 500, 20 trading days): {f['probability']:.0%}   "
              f"rating: {f.get('rating')}")
        print("Rationale:", f.get("rationale"))
        print("Evidence:")
        for e in f.get("evidence", []):
            print(f"  - {e.get('claim')}  [{e.get('source')}]")
        print("Risks (each backed by a tool result):")
        for r in f.get("risks", []):
            print(f"  - {r['risk']}  [{r['source']}]")
        if f.get("risks_dropped_unsourced"):
            print(f"  ({f['risks_dropped_unsourced']} unsourced risk(s) dropped)")
        print("Gaps (computed from what was actually available):")
        for g in f.get("gaps", []):
            print(f"  - {g}")
        pt = f.get("paper_trade")
        if pt:
            print(f"Paper trade (from technicals): entry {pt['entry_zone'][0]} to "
                  f"{pt['entry_zone'][1]}, stop {pt['stop_loss']}, target {pt['target']}")
            print(f"  signal: {pt.get('signal')}")
        v = f.get("verification")
        if v:
            print(f"Verifier: {v['status']} ({v['checked']} numbers checked, "
                  f"{len(v['unverified'])} unverified"
                  f"{', ABSTAINED' if v.get('abstained') else ''}); revisions: {f.get('revisions')}")
            for u in v["unverified"]:
                print(f"  unverified: {u['number']} in {u['where']}")
        if "--log" in sys.argv:
            print("Logged to", baseline.log_forecast(res))
    print("\nTrace saved:", trace_path)
    print(res["disclaimer"])


if __name__ == "__main__":
    main()
