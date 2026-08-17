"""Compare AMALIA variants against the BF16 reference.

Usage:
    python3 scripts/compare.py results/bf16.json results/fp8.json [...]

The first file is always the reference. For each variant it reports:
  - exact agreement with the reference
  - common prefix length and token-set overlap (Jaccard)
  - European vs Brazilian Portuguese markers
  - throughput

Caveat, and it matters: exact agreement is NOT a measure of quantization quality. Greedy
decoding cascades, so one flipped token early makes the whole rest of the answer differ
while remaining equally correct. Use perplexity.py for quality. This script is useful for
detecting non-determinism: run the same model twice and it should report 100%.
"""
import json, sys, unicodedata


def load(path):
    with open(path) as f:
        return json.load(f)


def normalise(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c)).strip()


def tokens(s):
    return [t for t in normalise(s).replace("\n", " ").split(" ") if t]


def common_prefix(a, b):
    ta, tb = tokens(a), tokens(b)
    n = 0
    for x, y in zip(ta, tb):
        if x != y:
            break
        n += 1
    return n, max(len(ta), len(tb))


def jaccard(a, b):
    sa, sb = set(tokens(a)), set(tokens(b))
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

ref_path, *others = sys.argv[1:]
ref = load(ref_path)
names = list(ref["answers"].keys())

print(f"reference: {ref_path}")
print("=" * 92)
print(f"  {'variant':<26} {'exact':>8} {'prefix':>9} {'jaccard':>9} {'PT':>4} {'BR':>4} {'tok/s':>7}")
print("-" * 92)
r = ref["summary"]
print(f"  {'BF16 (reference)':<26} {'--':>8} {'--':>9} {'--':>9} "
      f"{r['markers_pt']:>4} {r['markers_br']:>4} {r['tokens_per_s']:>7}")

details = []
for path in others:
    v = load(path)
    exact = 0
    prefix_sum = prefix_max = 0
    jac = []
    detail = []
    for name in names:
        a = (ref["answers"].get(name) or {}).get("answer", "")
        b = (v["answers"].get(name) or {}).get("answer", "")
        if not a or not b:
            continue
        same = normalise(a) == normalise(b)
        exact += same
        n, m = common_prefix(a, b)
        prefix_sum += n; prefix_max += m
        j = jaccard(a, b)
        jac.append(j)
        detail.append((name, same, n, m, round(j, 3)))
    vs = v["summary"]
    label = path.split("/")[-1].replace(".json", "")
    print(f"  {label:<26} {exact:>3}/{len(jac):<4} "
          f"{100*prefix_sum/prefix_max if prefix_max else 0:>8.1f}% "
          f"{100*sum(jac)/len(jac) if jac else 0:>8.1f}% "
          f"{vs['markers_pt']:>4} {vs['markers_br']:>4} {vs['tokens_per_s']:>7}")
    details.append((label, detail))

print()
for label, detail in details:
    print(f"detail for {label}:")
    for name, same, n, m, j in detail:
        mark = "identical" if same else f"diverges at token {n}"
        print(f"    {name:<18} {mark:<24} jaccard={j}")
    print()
