"""Live check of the filing RAG on one ticker (needs Ollama with the embedding model and
network access to SEC).

    ollama pull granite-embedding:278m
    python scripts/test_rag.py AAPL 2026-11-17 "supply chain concentration risk"

Prints what was extracted from each filing, then the retrieved passages.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecaster import rag
from forecaster.data import edgar

ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
as_of = sys.argv[2] if len(sys.argv) > 2 else "2026-11-17"
query = sys.argv[3] if len(sys.argv) > 3 else "principal risks to the business"

for f in rag.latest_filings(ticker, as_of):
    text = rag.html_to_text(edgar.fetch_filing_html(f["url"]))
    print(f"\n{f['form']} {f['accession']} (period {f['period']}, filed {f['filed']}): "
          f"{len(text):,} characters of text")
    for name, start, end in rag.SECTIONS[f["form"]]:
        sec = rag.extract_section(text, start, end)
        n = len(rag.chunk_text(sec)) if sec else 0
        print(f"   {name:13s} {len(sec or ''):>8,} chars -> {n} chunks"
              + ("" if sec else "   <-- NOT FOUND"))
        if sec:
            print("      starts:", sec[:110].replace("\n", " "))

print(f"\nQuery: {query!r}")
out = rag.search_filings(ticker, as_of, query)
print("Filings searched:", out.get("filings_searched"))
for i, r in enumerate(out["results"], 1):
    print(f"\n[{i}] similarity {r['similarity']}  {r['source']}\n    {r['text']}")
if not out["results"]:
    print("NO RESULTS:", out.get("error"))
