import zlib

import numpy as np

from forecaster import cache, llm, rag
from forecaster.data import edgar

PARA = "{} " * 14  # filler to make each paragraph longer than the 80 character minimum


def _html(sections):
    body = ["Table of Contents", "Item 1A. Risk Factors", "Item 1B. Unresolved Staff Comments",
            "Item 7. Management's Discussion and Analysis", "Item 7A. Quantitative Disclosures"]
    body += ["Item 1A. Risk Factors"] + sections["risk"] + ["Item 1B. Unresolved Staff Comments",
                                                           "none"]
    body += ["Item 7. Management's Discussion and Analysis"] + sections["mdna"] + [
        "Item 7A. Quantitative and Qualitative Disclosures"]
    return "<html><head><title>x</title></head><body>" + "".join(f"<p>{b}</p>" for b in body) + \
           "</body></html>"


RISK = ["Our supply chain depends on a small number of suppliers in Asia and a disruption "
        "there would reduce product availability and hurt revenue and margins.",
        "Changes in foreign exchange rates could harm our reported results because a large "
        "part of our sales are made in currencies other than the dollar."]
MDNA = ["Liquidity and capital resources: we had cash and marketable securities sufficient to "
        "fund operations and repurchase shares over the next twelve months and beyond."]


def test_html_to_text_strips_scripts_and_page_numbers():
    html = "<html><head><style>x{}</style></head><body><p>Hello&nbsp;world</p><p>12</p>" \
           "<script>bad()</script><p>Second line</p></body></html>"
    assert rag.html_to_text(html) == "Hello world\nSecond line"


def test_extract_section_skips_table_of_contents():
    text = rag.html_to_text(_html({"risk": RISK, "mdna": MDNA}))
    start, end = rag.SECTIONS["10-K"][0][1:]
    sec = rag.extract_section(text, start, end)
    assert "supply chain" in sec and "foreign exchange" in sec
    assert "Item 1B" not in sec and "Liquidity" not in sec
    start, end = rag.SECTIONS["10-K"][1][1:]
    assert "Liquidity and capital" in rag.extract_section(text, start, end)
    assert rag.extract_section("nothing relevant here", start, end) is None


def test_chunk_text_covers_everything_with_bounded_size():
    paras = [f"Sentence number {i} says something fairly long about risk factors. " * 3
             for i in range(40)]
    text = "\n".join(paras)
    chunks = rag.chunk_text(text, size=500, overlap=80)
    assert all(len(c) <= 500 + 80 for c in chunks)
    joined = " ".join(chunks)
    assert all(f"Sentence number {i} " in joined for i in range(40))
    assert len(chunks) > 5


def test_chunk_text_splits_one_huge_paragraph():
    big = "This is a sentence about the business and its risks. " * 80
    chunks = rag.chunk_text(big, size=500, overlap=50)
    assert len(chunks) >= 6 and all(len(c) <= 560 for c in chunks)


def test_mmr_prefers_diverse_passages():
    q = np.array([1.0, 0.0, 0.0])
    docs = np.array([[0.99, 0.10, 0.0], [0.98, 0.12, 0.0], [0.97, 0.14, 0.0],   # near duplicates
                     [0.70, 0.0, 0.71]])                                         # different topic
    docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)
    assert rag.mmr(q, docs, 2, lam=1.0) == [0, 1]          # pure relevance: duplicates
    assert 3 in rag.mmr(q, docs, 2, lam=0.4)                # MMR brings in the other topic


def _fake_embed(texts, model=None, batch=16):
    out = []
    for t in texts:
        v = np.zeros(64)
        for w in t.lower().replace(".", " ").replace(",", " ").split():
            v[zlib.crc32(w.encode()) % 64] += 1
        out.append(v.tolist())
    return out


def _setup(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm, "embed", _fake_embed)
    filings = [
        {"form": "10-K", "filed": "2026-02-13", "period": "2025-12-31", "accession": "0001-26-000015",
         "url": "u10k"},
        {"form": "10-Q", "filed": "2026-10-21", "period": "2026-09-30", "accession": "0001-26-000099",
         "url": "u10q"},
        {"form": "10-Q", "filed": "2026-04-23", "period": "2026-03-31", "accession": "0001-26-000031",
         "url": "u10q1"}]
    monkeypatch.setattr(edgar, "list_filings", lambda t, a, forms=(), limit=6: sorted(
        [f for f in filings if f["filed"] <= a], key=lambda f: f["filed"], reverse=True)[:limit])
    html = _html({"risk": RISK, "mdna": MDNA})
    monkeypatch.setattr(edgar, "fetch_filing_html", lambda url: html)


def test_search_returns_relevant_cited_passages(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    out = rag.search_filings("AAA", "2026-11-17", "supplier disruption supply chain", k=2)
    assert out["results"] and "supply chain" in out["results"][0]["text"]
    assert out["results"][0]["source"].startswith(("10-K", "10-Q"))
    assert "Risk Factors" in out["results"][0]["source"] or "MD&A" in out["results"][0]["source"]


def test_search_never_uses_filings_after_as_of(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    early = rag.search_filings("AAA", "2026-05-01", "supply chain")
    assert all("0001-26-000099" not in s for s in early["filings_searched"])
    assert all("0001-26-000099" not in r["source"] for r in early["results"])
    assert any("0001-26-000031" in s for s in early["filings_searched"])
    none = rag.search_filings("AAA", "2025-01-01", "supply chain")
    assert none["results"] == [] and "error" in none


def test_index_is_cached_per_filing(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    calls = []
    real = rag.edgar.fetch_filing_html
    monkeypatch.setattr(rag.edgar, "fetch_filing_html", lambda u: calls.append(u) or real(u))
    rag.search_filings("AAA", "2026-11-17", "risk")
    n = len(calls)
    rag.search_filings("AAA", "2026-11-17", "liquidity")
    assert n >= 1 and len(calls) == n        # second search re-uses the saved index


def test_chunks_start_and_end_on_word_boundaries():
    words = [f"word{i:03d}" for i in range(900)]
    big = " ".join(words)                                    # one huge paragraph
    mixed = "\n".join(" ".join(words[i:i + 40]) for i in range(0, 400, 40))
    for text in (big, mixed):
        vocab = set(words)
        for c in rag.chunk_text(text, size=500, overlap=90):
            toks = c.split()
            assert toks[0] in vocab and toks[-1] in vocab, (toks[0], toks[-1])
    # overlap exists between consecutive chunks of the huge paragraph
    cs = rag.chunk_text(big, size=500, overlap=90)
    assert set(cs[0].split()) & set(cs[1].split())


def test_relevance_floor_drops_far_off_topic_passages(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    out = rag.search_filings("AAA", "2026-11-17", "supplier disruption supply chain", k=4)
    sims = [r["similarity"] for r in out["results"]]
    assert sims and max(sims) - min(sims) <= rag.MAX_SIM_DROP + 1e-6
