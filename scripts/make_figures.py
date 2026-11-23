"""Figures for the report and slides, from the saved backtest results.
    python3 scripts/make_figures.py      (needs matplotlib; writes docs/figures/*.png)
Inputs: eval/backtest/scored_backtest_common_rows.json (local), benchmarks.json, forecasts.jsonl.
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
rows = json.loads((ROOT / "eval/backtest/scored_backtest_common_rows.json").read_text())
bench = json.loads((ROOT / "eval/backtest/benchmarks.json").read_text())

NAMES = {"single_agent_react": "1 Single agent", "multi_specialists": "2 + Specialists",
         "multi_critic_verifier": "3 + Critic & Verifier", "multi_rag": "4 + Filing RAG"}
ORDER = list(NAMES)
COL = {"single_agent_react": "#8a8a8a", "multi_specialists": "#6a9fd1",
       "multi_critic_verifier": "#2f6fb0", "multi_rag": "#1b4a82"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})

P = {v: {(r["ticker"], r["as_of"]): (r["probability"], r["outperformed"]) for r in rows[v]}
     for v in ORDER}
keys = sorted(P[ORDER[0]])
dates = sorted({k[1] for k in keys})
by = defaultdict(list)
for k in keys:
    by[k[1]].append(k)


def brier(v, ks):
    return float(np.mean([(P[v][k][0] - P[v][k][1]) ** 2 for k in ks]))


def auc(v, ks):
    s = pd.Series([P[v][k][0] for k in ks]).rank().to_numpy()
    y = np.array([P[v][k][1] for k in ks], dtype=bool)
    n1, n0 = y.sum(), (~y).sum()
    return float((s[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot(fn, B=3000, seed=0):
    rnd = random.Random(seed)
    xs = sorted(fn([k for d in rnd.choices(dates, k=len(dates)) for k in by[d]]) for _ in range(B))
    return xs[int(.025 * B)], xs[int(.975 * B)]


# 1. Brier with date-clustered CI vs constant 50%
fig, ax = plt.subplots(figsize=(7, 4))
for i, v in enumerate(ORDER):
    b = brier(v, keys); lo, hi = boot(lambda ks: brier(v, ks))
    ax.errorbar(i, b, yerr=[[b - lo], [hi - b]], fmt="o", color=COL[v], capsize=5, ms=8)
ax.axhline(0.25, color="crimson", ls="--", lw=1.3, label="Always 50% (0.2500)")
ax.set_xticks(range(4)); ax.set_xticklabels([NAMES[v] for v in ORDER], rotation=12)
ax.set_ylabel("Brier score (lower is better)")
ax.set_title("Brier score, 165 pairs (95% CI, resampling dates)")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT / "brier_by_variant.png"); plt.close(fig)

# 2. AUC with CI, agents and non-LLM rules
labels, vals, los, his, cols = [], [], [], [], []
for v in ORDER:
    a = auc(v, keys); lo, hi = boot(lambda ks: auc(v, ks))
    labels.append(NAMES[v]); vals.append(a); los.append(a - lo); his.append(hi - a); cols.append(COL[v])
RN = {"mom_6_1": "Momentum (rule)", "rev_1m": "1-month reversal (rule)", "low_vol": "Low volatility (rule)"}
for r, name in RN.items():
    a = bench["rules"][r]["auc"]; lo, hi = bench["rules"][r]["auc_ci"]
    labels.append(name); vals.append(a); los.append(a - lo); his.append(hi - a); cols.append("#d9a441")
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.errorbar(range(len(vals)), vals, yerr=[los, his], fmt="none", ecolor="#555", capsize=5)
ax.scatter(range(len(vals)), vals, c=cols, s=70, zorder=3)
ax.axhline(0.5, color="crimson", ls="--", lw=1.3, label="No skill (AUC 0.5)")
ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, rotation=25, ha="right")
ax.set_ylabel("AUC (ranking ability)"); ax.set_ylim(0.25, 0.75)
ax.set_title("Ranking ability: agents vs simple rules (95% CI)")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT / "auc_agents_vs_rules.png"); plt.close(fig)

# 3. Calibration (reliability) plot, pooled variants shown separately
edges = [0, .40, .50, .60, 1.01]
fig, ax = plt.subplots(figsize=(5.8, 5.2))
ax.plot([0.3, 0.7], [0.3, 0.7], color="crimson", ls="--", lw=1.2, label="Perfect calibration")
for v in ORDER:
    xs, ys, ns = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        ks = [k for k in keys if a <= P[v][k][0] < b]
        if ks:
            xs.append(np.mean([P[v][k][0] for k in ks])); ys.append(np.mean([P[v][k][1] for k in ks])); ns.append(len(ks))
    ax.plot(xs, ys, "-", color=COL[v], alpha=.5)
    ax.scatter(xs, ys, s=[12 + 5 * n for n in ns], color=COL[v], label=NAMES[v], alpha=.9, zorder=3)
ax.axhline(0.509, color="#999", lw=1, ls=":", label="Base rate 50.9%")
ax.set_xlabel("Mean forecast probability"); ax.set_ylabel("Share that beat the S&P 500")
ax.set_xlim(0.3, 0.7); ax.set_ylim(0.1, 1.0); ax.set_title("Calibration (marker size = number of forecasts)")
ax.legend(frameon=False, fontsize=9, loc="upper left"); fig.tight_layout()
fig.savefig(OUT / "calibration.png"); plt.close(fig)

# 4. Benchmark ladder: Brier of every reference and variant
items = [("Always 50%", 0.25, "#555"), ("1-month reversal", bench["rules"]["rev_1m"]["brier"], "#d9a441"),
         ("Low volatility", bench["rules"]["low_vol"]["brier"], "#d9a441"),
         ("Momentum", bench["rules"]["mom_6_1"]["brier"], "#d9a441")]
items += [(NAMES[v], brier(v, keys), COL[v]) for v in ORDER[::-1][::-1]]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.barh(range(len(items))[::-1], [i[1] for i in items], color=[i[2] for i in items])
ax.set_yticks(range(len(items))[::-1]); ax.set_yticklabels([i[0] for i in items])
ax.axvline(0.25, color="crimson", ls="--", lw=1.2)
ax.set_xlim(0.24, 0.27); ax.set_xlabel("Brier score (lower is better)")
for y, it in zip(range(len(items))[::-1], items):
    ax.text(it[1] + 0.0003, y, f"{it[1]:.4f}", va="center", fontsize=9)
ax.set_title("Benchmark ladder"); fig.tight_layout()
fig.savefig(OUT / "benchmark_ladder.png"); plt.close(fig)

# 5. Forecast spread by variant (shrinkage story)
fig, ax = plt.subplots(figsize=(7, 4))
ax.boxplot([[P[v][k][0] for k in keys] for v in ORDER], tick_labels=[NAMES[v] for v in ORDER], showfliers=True)
ax.axhline(0.5, color="crimson", ls="--", lw=1)
ax.set_ylabel("Forecast P(outperform S&P 500)"); ax.set_title("Forecasts narrow toward 50% as guardrails are added")
plt.setp(ax.get_xticklabels(), rotation=12); fig.tight_layout()
fig.savefig(OUT / "forecast_spread.png"); plt.close(fig)

# 6. Share of stocks beating SPY per date (why 165 pairs is closer to 11 samples)
share = [np.mean([P[ORDER[0]][k][1] for k in by[d]]) for d in dates]
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.bar(range(len(dates)), share, color="#6a9fd1"); ax.axhline(0.509, color="crimson", ls="--", lw=1)
ax.set_xticks(range(len(dates))); ax.set_xticklabels([d[5:] for d in dates], rotation=45)
ax.set_ylabel("Share of 15 stocks beating SPY"); ax.set_ylim(0, 1)
ax.set_title("Outcomes cluster by date (as-of dates, 2026)"); fig.tight_layout()
fig.savefig(OUT / "outcomes_by_date.png"); plt.close(fig)

# 7. Guardrail ablation: unverified numbers with and without routing
fc = [json.loads(l) for l in (ROOT / "eval/backtest/forecasts.jsonl").read_text().splitlines() if l.strip()]
abl = {}
for v in ("multi_specialists", "multi_critic_verifier", "multi_rag"):
    rs = [r for r in fc if r["variant"] == v and r.get("probability") is not None]
    abl[v] = {"n": len(rs), "with_unverified": sum(r.get("unverified", 0) > 0 for r in rs),
              "revised": sum(r.get("revisions", 0) > 0 for r in rs),
              "status": dict(Counter(r.get("verification") for r in rs))}
(OUT / "guardrail_ablation.json").write_text(json.dumps(abl, indent=1))
print("figures written:", sorted(p.name for p in OUT.iterdir()))


# 8. Architecture diagram (block flow of the routed multi-agent variants)
from matplotlib.patches import FancyBboxPatch  # noqa: E402


def box(ax, x, y, w, h, text, fc="#eaf1fb", ec="#2f6fb0", fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, style="-|>", ls="-", color="#444"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=1.3, ls=ls))


fig, ax = plt.subplots(figsize=(10.5, 7.2))
ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis("off")
box(ax, 0.3, 8.6, 2.6, 1.0, "User / CLI /\nStreamlit UI", "#f4f4f4", "#777")
box(ax, 3.7, 8.6, 3.0, 1.0, "Input guard (rules)\nadvice, execution, injection,\nticker, date, sector", "#fdecea", "#c0392b", 8.5)
box(ax, 7.5, 8.6, 3.0, 1.0, "Orchestrator (code)\nfixed plan, as-of date\nset by code", "#eaf1fb", "#2f6fb0", 8.5)
arrow(ax, 2.9, 9.1, 3.7, 9.1); arrow(ax, 6.7, 9.1, 7.5, 9.1)
ax.text(5.2, 8.25, "declined: reason, no forecast", fontsize=8, color="#c0392b", ha="center")
# specialists
sp = [("Fundamentals\n(EDGAR XBRL)", 0.5), ("Macro\n(FRED vintage)", 3.85), ("Technical\n(prices, indicators)", 7.2),
      ("Filings RAG\n(multi_rag only)", 10.55)]
SPW = 2.7
for t, x in sp:
    box(ax, x, 6.3, SPW, 1.2, t, "#e8f6ec", "#2e8b57", 9)
    arrow(ax, 9.0, 8.6, x + SPW / 2, 7.5)
# strategist -> critic -> verifier
box(ax, 5.2, 4.3, 3.6, 1.1, "Strategist (gpt-oss:20b)\nprobability, evidence, risks", "#eaf1fb", "#2f6fb0", 9.5, True)
for t, x in sp:
    arrow(ax, x + SPW / 2, 6.3, 7.0, 5.4, color="#999")
box(ax, 9.7, 4.3, 3.6, 1.1, "Critic (gemma4:12b)\ndifferent model family", "#f3eafb", "#7d3c98", 9.5, True)
arrow(ax, 8.8, 4.85, 9.7, 4.85)
box(ax, 9.7, 2.5, 3.6, 1.1, "Verifier (deterministic code)\nevery number vs tool results", "#e8f6ec", "#2e8b57", 9.5, True)
arrow(ax, 11.5, 4.3, 11.5, 3.6)
arrow(ax, 9.7, 3.05, 7.0, 4.3, ls="--", color="#c0392b")
ax.text(8.0, 3.35, "one bounded\nrevision", fontsize=8, color="#c0392b", ha="center")
# routing
box(ax, 4.3, 0.4, 4.7, 1.5, "Guardrail routing\ngreen: deliver as is\norange: drop unsupported text, flag review\nred or Critic failure: abstain at 50%", "#fff6e5", "#d9a441", 8.5)
arrow(ax, 9.7, 2.8, 9.0, 1.9)
box(ax, 9.7, 0.4, 3.6, 1.5, "Finalize (code)\nrating, gaps, paper-trade\nlevels, log and trace", "#f4f4f4", "#777", 8.5)
arrow(ax, 9.0, 1.15, 9.7, 1.15)
# tools
box(ax, 0.3, 2.4, 4.2, 2.2, "Read-only tools (as-of fixed by code)\nSEC EDGAR / filing text\nprices + indicators\nFRED macro (vintage dates)\nlocal embeddings (Granite)\ndirect Python or MCP servers", "#f4f4f4", "#777", 8.5)
arrow(ax, 2.4, 4.6, 2.4, 6.3, ls="--", color="#777")
ax.text(0.3, 1.5, "Variants compared:\n1 single agent\n2 specialists + Strategist\n    (Verifier measures only)\n3 + Critic and routing\n4 + filing RAG", fontsize=8.5, color="#333", va="center")
fig.tight_layout(); fig.savefig(OUT / "architecture.png", dpi=170); plt.close(fig)
print("architecture diagram written")
