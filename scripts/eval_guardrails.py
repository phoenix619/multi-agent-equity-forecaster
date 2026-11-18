"""Score the input guard on the adversarial set (offline, no models).
    python3 scripts/eval_guardrails.py
Reports catch rate (should-block cases blocked), false-block rate (should-allow cases blocked),
category accuracy, and for retrieved passages: injected ones dropped / benign ones kept.
"""
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forecaster import guard  # noqa: E402

SRC = ROOT / "eval" / "guardrails" / "adversarial_cases.json"
OUT = ROOT / "eval" / "guardrails" / "results.json"


def main():
    full = json.loads(SRC.read_text())
    out = {}
    for name, rk, uk in (("development", "request_cases", "untrusted_cases"),
                         ("held_out", "heldout_request_cases", "heldout_untrusted_cases")):
        print(f"\n######## {name} set")
        out[name] = score(full, rk, uk)
    OUT.write_text(json.dumps(out, indent=1))


def score(d, rk, uk):
    today = date.fromisoformat(d["today"])
    blk = [c for c in d[rk] if c["expect"] != "allow"]
    alw = [c for c in d[rk] if c["expect"] == "allow"]
    miss, wrong_cat, fb = [], [], []
    for c in d[rk]:
        v = guard.check_request(c["ticker"], c["as_of"], c["request"], today=today, sic=c["sic"])
        c["got"] = "allow" if v["allowed"] else "block:" + v["category"]
        if c["expect"] == "allow" and not v["allowed"]:
            fb.append(c)
        elif c["expect"] != "allow":
            if v["allowed"]:
                miss.append(c)
            elif c["got"] != c["expect"]:
                wrong_cat.append(c)
    caught = len(blk) - len(miss)
    print(f"Should block: {len(blk)}  caught {caught} ({caught / len(blk):.0%});  "
          f"right category {caught - len(wrong_cat)} of {caught}")
    print(f"Should allow: {len(alw)}  falsely blocked {len(fb)} ({len(fb) / len(alw):.0%})")
    for c in miss:
        print(f"  MISSED   {c['id']}: {c['request'] or c['ticker']!r} expected {c['expect']}")
    for c in wrong_cat:
        print(f"  WRONG CATEGORY {c['id']}: expected {c['expect']} got {c['got']}")
    for c in fb:
        print(f"  FALSE BLOCK {c['id']}: {c['request'] or c['ticker']!r} -> {c['got']}")

    u = d[uk]
    inj = [c for c in u if c["expect"] == "drop"]
    ben = [c for c in u if c["expect"] == "keep"]
    missed = [c for c in inj if not guard.is_injection(c["text"])]
    wrongly = [c for c in ben if guard.is_injection(c["text"])]
    print(f"\nRetrieved passages: injected {len(inj)}, dropped {len(inj) - len(missed)};  "
          f"benign {len(ben)}, wrongly dropped {len(wrongly)}")
    for c in missed:
        print(f"  MISSED INJECTION {c['id']}: {c['text'][:70]!r}")
    for c in wrongly:
        print(f"  WRONGLY DROPPED {c['id']}: {c['text'][:70]!r}")
    return ({
        "request": {"should_block": len(blk), "caught": caught,
                    "catch_rate": round(caught / len(blk), 3),
                    "category_correct": caught - len(wrong_cat),
                    "should_allow": len(alw), "false_blocks": len(fb),
                    "false_block_rate": round(len(fb) / len(alw), 3)},
        "retrieved_passages": {"injected": len(inj), "dropped": len(inj) - len(missed),
                               "benign": len(ben), "wrongly_dropped": len(wrongly)},
        "missed": [c["id"] for c in miss], "false_block_ids": [c["id"] for c in fb],
        "missed_injection_ids": [c["id"] for c in missed],
        "wrongly_dropped_ids": [c["id"] for c in wrongly]})


if __name__ == "__main__":
    main()
