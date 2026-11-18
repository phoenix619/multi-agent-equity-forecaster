"""Retrieval over SEC filing text (10-K and 10-Q): risk factors and MD&A.

Pipeline: fetch the filing HTML -> plain text -> pull out Item 1A (risk factors) and
the MD&A section -> chunk with overlap -> embed (Ollama, Granite) -> cache the index per
filing -> at query time rank by cosine similarity and re-rank with Maximal Marginal
Relevance (Module 3 lab 2) so the returned passages are relevant AND different.

As-of safety: only filings filed on or before the as-of date are indexed or searched.
"""
import json
import re

import numpy as np

from . import llm
from .config import (CACHE_DIR, RAG_CHUNK_CHARS, RAG_MMR_LAMBDA, RAG_OVERLAP_CHARS,
                     RAG_TOP_K)
from .data import edgar

MAX_SECTION_CHARS = 120_000
CANDIDATES = 20          # passages ranked by similarity before MMR picks the final k
MAX_SIM_DROP = 0.08      # drop passages this far below the best match: MMR must not
                         # promote an irrelevant passage just because it is different
SNIPPET_CHARS = 1400      # above the chunk size (about 1200 + overlap) so passages are never cut

_DASH = r"[\s.:\-–—]*"
# (section name, start heading, end heading) per form; matched at line starts.
SECTIONS = {
    "10-K": [
        ("Risk Factors", rf"item\s*1a{_DASH}risk\s+factors", r"item\s*1b|item\s*1c|item\s*2\b"),
        ("MD&A", rf"item\s*7{_DASH}management", r"item\s*7a|item\s*8\b"),
    ],
    "10-Q": [
        ("MD&A", rf"item\s*2{_DASH}management", r"item\s*3\b|item\s*4\b"),
        ("Risk Factors", rf"item\s*1a{_DASH}risk\s+factors", r"item\s*2\b|item\s*5\b|item\s*6\b"),
    ],
}


# ---------------------------------------------------------------- text
def html_to_text(html: str) -> str:
    import warnings

    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
    # Inline-XBRL filings are XHTML; the HTML parser handles them fine.
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = text.replace(" ", " ").replace("’", "'").replace("‘", "'")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not re.fullmatch(r"\d{1,3}|page \d+( of \d+)?", ln, re.I)]
    return "\n".join(lines)


def extract_section(text: str, start_pat: str, end_pat: str, cap: int = MAX_SECTION_CHARS):
    """Text between a start heading and the next end heading.

    Headings also appear in the table of contents, so every start match is tried and
    the longest span wins (the TOC entry is a few lines long, the real section is not).
    """
    start_re = re.compile(rf"^\s*{start_pat}", re.I | re.M)
    end_re = re.compile(rf"^\s*(?:{end_pat})", re.I | re.M)
    best = ""
    for m in start_re.finditer(text):
        e = end_re.search(text, m.end())
        span = text[m.start():(e.start() if e else min(len(text), m.start() + cap))]
        if len(span) > len(best):
            best = span
    return best[:cap].strip() or None


def _split_long(p: str, size: int, overlap: int) -> list:
    """Split one very long paragraph at sentence (or word) boundaries, with overlap."""
    pieces = []
    while len(p) > size:
        cut = p.rfind(". ", 0, size)
        if cut > size // 2:
            cut += 1
        else:
            sp = p.rfind(" ", 0, size)
            cut = sp if sp > 0 else size
        pieces.append(p[:cut].strip())
        nxt = p.find(" ", max(cut - overlap, 1))          # restart on a word boundary
        p = p[(nxt + 1 if 0 <= nxt < cut else cut):].strip()
    if p:
        pieces.append(p)
    return pieces


def chunk_text(text: str, size: int = RAG_CHUNK_CHARS, overlap: int = RAG_OVERLAP_CHARS) -> list:
    """Paragraph-aware chunks of about `size` characters with `overlap` between them.
    Chunks always start and end on word boundaries."""
    chunks, cur = [], ""
    for p in (x.strip() for x in text.split("\n")):
        if not p:
            continue
        if len(p) > size:
            if cur:
                chunks.append(cur)
                cur = ""
            pieces = _split_long(p, size, overlap)
            chunks.extend(pieces[:-1])
            cur = pieces[-1]
            continue
        if cur and len(cur) + len(p) + 1 > size:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cur = tail.split(" ", 1)[1] if " " in tail and len(cur) > overlap else ""
        cur = f"{cur} {p}".strip() if cur else p
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(c) > 80]


# ---------------------------------------------------------------- MMR
def mmr(query: np.ndarray, docs: np.ndarray, k: int, lam: float = RAG_MMR_LAMBDA) -> list:
    """Indices chosen by Maximal Marginal Relevance (rows must be unit length)."""
    rel = docs @ query
    chosen, remaining = [], list(range(len(docs)))
    while remaining and len(chosen) < k:
        if not chosen:
            best = max(remaining, key=lambda i: rel[i])
        else:
            sel = docs[chosen]
            best = max(remaining, key=lambda i: lam * rel[i] - (1 - lam) * float((sel @ docs[i]).max()))
        chosen.append(best)
        remaining.remove(best)
    return chosen


# ---------------------------------------------------------------- index
def _rag_dir():
    d = CACHE_DIR / "rag"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(n == 0, 1, n)


def build_index(filing: dict) -> dict:
    """Chunks and embeddings for one filing (cached by accession number)."""
    key = filing["accession"].replace("-", "")
    meta_p, vec_p = _rag_dir() / f"{key}.json", _rag_dir() / f"{key}.npy"
    if meta_p.exists() and vec_p.exists():
        return {"chunks": json.loads(meta_p.read_text()), "vectors": np.load(vec_p)}
    text = html_to_text(edgar.fetch_filing_html(filing["url"]))
    chunks = []
    for name, start, end in SECTIONS.get(filing["form"], []):
        sec = extract_section(text, start, end)
        for j, c in enumerate(chunk_text(sec or "")):
            chunks.append({"text": c, "section": name, "form": filing["form"],
                           "accession": filing["accession"], "filed": filing["filed"],
                           "period": filing["period"], "chunk": j})
    if not chunks:
        return {"chunks": [], "vectors": np.zeros((0, 1), dtype="float32")}
    vecs = _normalize(np.array(llm.embed([c["text"] for c in chunks]), dtype="float32"))
    meta_p.write_text(json.dumps(chunks))
    np.save(vec_p, vecs)
    return {"chunks": chunks, "vectors": vecs}


def latest_filings(ticker: str, as_of: str) -> list:
    """The newest 10-K and the newest 10-Q filed on or before as_of."""
    rows = edgar.list_filings(ticker, as_of, limit=12)
    out = []
    for form in ("10-K", "10-Q"):
        pick = next((r for r in rows if r["form"] == form), None)
        if pick:
            out.append(pick)
    return out


def search_filings(ticker: str, as_of: str, query: str, k: int = RAG_TOP_K) -> dict:
    """Most relevant, non-redundant passages from the latest 10-K and 10-Q."""
    filings = latest_filings(ticker, as_of)
    chunks, mats = [], []
    for f in filings:
        idx = build_index(f)
        if len(idx["chunks"]):
            chunks.extend(idx["chunks"])
            mats.append(idx["vectors"])
    if not chunks:
        return {"ticker": ticker.upper(), "as_of": as_of, "query": query, "results": [],
                "error": "no filing text could be indexed for this ticker and date"}
    vectors = np.vstack(mats)
    q = _normalize(np.array(llm.embed([query]), dtype="float32"))[0]
    top = np.argsort(-(vectors @ q))[:CANDIDATES]
    best = float(vectors[top[0]] @ q)
    picked = [int(top[i]) for i in mmr(q, vectors[top], k)
              if float(vectors[top[i]] @ q) >= best - MAX_SIM_DROP]
    results = [{"text": chunks[i]["text"][:SNIPPET_CHARS],
                "source": f"{chunks[i]['form']} {chunks[i]['accession']} "
                          f"{chunks[i]['section']} (period {chunks[i]['period']}, "
                          f"filed {chunks[i]['filed']})",
                "similarity": round(float(vectors[i] @ q), 3)} for i in picked]
    from . import guard
    results, removed = guard.drop_injected(results)   # filings are untrusted input
    out = {"ticker": ticker.upper(), "as_of": as_of, "query": query,
           "filings_searched": [f"{f['form']} {f['accession']} filed {f['filed']}" for f in filings],
           "results": results}
    if removed:
        out["passages_removed_as_instruction_like"] = removed
    return out
