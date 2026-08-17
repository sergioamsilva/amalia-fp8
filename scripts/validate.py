"""Check that every number claimed in the write-up exists in the raw result files.

Run this before publishing anything. If a number in README.md does not match the JSON
files, it fails here instead of in public.

Usage:
    python3 scripts/validate.py
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
failures = []
passed = []


def load(p):
    with open(os.path.join(ROOT, p)) as f:
        return json.load(f)


def check(desc, condition, actual=""):
    if condition:
        passed.append(desc)
    else:
        failures.append(f"{desc}  (actual: {actual})")


bf = load("results/bf16.json")
fp = load("results/fp8.json")
w8 = load("results/w8a16.json")
rep = load("results/fp8_repeat.json")
pbf = load("results/ppl_bf16.json")
pfp = load("results/ppl_fp8.json")
pw8 = load("results/ppl_w8a16.json")
pfrag = load("results/ppl_fp8_resharded.json")

doc = open(os.path.join(ROOT, "README.md")).read()

print("=" * 74)
print("1. PERPLEXITY: numbers in the write-up against the files")
print("=" * 74)

g_bf = pbf["_global"]["ppl"]
g_fp = pfp["_global"]["ppl"]
g_w8 = pw8["_global"]["ppl"]
d_fp = 100 * (g_fp - g_bf) / g_bf
d_w8 = 100 * (g_w8 - g_bf) / g_bf

print(f"  BF16 global:  {g_bf}")
print(f"  W8A8 global:  {g_fp}  ({d_fp:+.2f}%)")
print(f"  W8A16 global: {g_w8}  ({d_w8:+.2f}%)")

check("write-up cites 11.0987 for BF16", "11.0987" in doc, g_bf)
check("write-up cites 11.2033 for W8A8", "11.2033" in doc, g_fp)
check("write-up cites 11.3553 for W8A16", "11.3553" in doc, g_w8)
check("W8A8 delta rounds to +0.94%", abs(d_fp - 0.94) < 0.005, f"{d_fp:.4f}%")
check("W8A16 delta rounds to +2.31%", abs(d_w8 - 2.31) < 0.005, f"{d_w8:.4f}%")
check("write-up states +0.94%", "+0.94%" in doc)
check("write-up states +2.31%", "+2.31%" in doc)
check("W8A8 really is better than W8A16", d_fp < d_w8, f"{d_fp:.2f} vs {d_w8:.2f}")
check("W8A8 degradation really is under 1%", d_fp < 1.0, f"{d_fp:.4f}%")
check("401 perplexity tokens", pbf["_global"]["tokens"] == 401, pbf["_global"]["tokens"])
check("all three measured over the same tokens",
      pbf["_global"]["tokens"] == pfp["_global"]["tokens"] == pw8["_global"]["tokens"])

for reg, expected in (("literary", 2.30), ("administrative", 0.53),
                      ("colloquial", 2.52), ("technical", -1.82)):
    d = 100 * (pfp[reg]["ppl"] - pbf[reg]["ppl"]) / pbf[reg]["ppl"]
    check(f"delta for the {reg} register = {expected:+.2f}%", abs(d - expected) < 0.006, f"{d:+.4f}%")
check("the technical register really is negative for W8A8",
      pfp["technical"]["ppl"] < pbf["technical"]["ppl"])

print()
print("=" * 74)
print("2. RESHARDING DID NOT CHANGE THE MODEL")
print("=" * 74)
same = all(abs(pfp[k]["ppl"] - pfrag[k]["ppl"]) < 1e-9
           for k in ("literary", "administrative", "colloquial", "technical", "_global"))
print(f"  perplexity before resharding: {pfp['_global']['ppl']}")
print(f"  perplexity after resharding:  {pfrag['_global']['ppl']}")
check("resharding left perplexity identical", same)

print()
print("=" * 74)
print("3. THROUGHPUT AND DIALECT MARKERS")
print("=" * 74)
for label, d in (("BF16", bf), ("W8A8", fp), ("W8A16", w8)):
    s = d["summary"]
    print(f"  {label:6s} {s['tokens_per_s']:>6} tok/s | PT={s['markers_pt']} BR={s['markers_br']} "
          f"| ger_pt={s['gerund_pt']} ger_br={s['gerund_br']}")

check("write-up cites 12.2 tok/s for BF16",
      abs(bf["summary"]["tokens_per_s"] - 12.2) < 0.05 and "12.2 tok/s" in doc)
check("write-up cites 22.3 tok/s for W8A8",
      abs(fp["summary"]["tokens_per_s"] - 22.3) < 0.05 and "22.3 tok/s" in doc)
check("write-up cites 23.2 tok/s for W8A16",
      abs(w8["summary"]["tokens_per_s"] - 23.2) < 0.05 and "23.2 tok/s" in doc)
check("W8A16 really is faster than W8A8",
      w8["summary"]["tokens_per_s"] > fp["summary"]["tokens_per_s"])
check("zero Brazilian markers in BF16", bf["summary"]["markers_br"] == 0)
check("zero Brazilian markers in W8A8", fp["summary"]["markers_br"] == 0)
check("7 European markers in both, as claimed",
      bf["summary"]["markers_pt"] == 7 and fp["summary"]["markers_pt"] == 7)

print()
print("=" * 74)
print("4. DETERMINISM: the null hypothesis test")
print("=" * 74)
identical = sum(1 for k in fp["answers"]
                if fp["answers"][k].get("answer") == rep["answers"].get(k, {}).get("answer"))
total = len(fp["answers"])
print(f"  FP8 against itself:  {identical}/{total} identical answers")
check("vLLM really is deterministic (12/12)", identical == total == 12, f"{identical}/{total}")

differ = sum(1 for k in bf["answers"]
             if bf["answers"][k].get("answer") != fp["answers"].get(k, {}).get("answer"))
print(f"  BF16 against W8A8:   {total-differ}/{total} identical")
check("0 of 12 identical between BF16 and FP8, as claimed", differ == 12, f"{total-differ} identical")

print()
print("=" * 74)
print("5. THE MADEIRA QUOTE")
print("=" * 74)
geo_bf = bf["answers"]["geography"]["answer"]
geo_fp = fp["answers"]["geography"]["answer"]
check("BF16 really says 'a oeste de Marrocos'", "Marrocos" in geo_bf, geo_bf[:80])
check("FP8 really says 'Estreito de Gibraltar'", "Gibraltar" in geo_fp, geo_fp[:80])
check("the quote in the write-up is accurate",
      "Strait of Gibraltar" in doc and "west of Morocco" in doc)
print(f"  BF16: ...{geo_bf[geo_bf.find('Madeira'):][:70]}")
print(f"  FP8 : ...{geo_fp[geo_fp.find('Madeira'):][:70]}")

print()
print("=" * 74)
print("VALIDATION RESULT")
print("=" * 74)
print(f"  checks passed: {len(passed)}")
if failures:
    print(f"  FAILURES: {len(failures)}")
    for f in failures:
        print(f"    x {f}")
    sys.exit(1)
print("  no failures. Every number in the write-up exists in the raw data.")
