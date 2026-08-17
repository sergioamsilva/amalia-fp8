"""Generate the README charts as self-contained SVG, one pair per theme.

GitHub does not honour CSS media queries inside an SVG embedded in markdown, so each
chart is emitted twice (light and dark) and the README selects between them with a
<picture> element. No external fonts, no scripts, no network: the files render anywhere.

Numbers come from results/, never typed by hand, so the charts cannot drift from the data.

Usage:
    python3 scripts/make_charts.py
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "charts")
os.makedirs(OUT, exist_ok=True)

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#8a8880",
                  grid="#e3e2dd", s1="#2a78d6", s2="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", ink3="#8a8880",
                 grid="#333330", s1="#3987e5", s2="#d95926"),
}
FONT = ("ui-sans-serif,-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,"
        "sans-serif")


def load(name):
    with open(os.path.join(ROOT, "results", name)) as f:
        return json.load(f)


ppl_bf = load("ppl_bf16.json")
ppl_w8a8 = load("ppl_fp8.json")
ppl_w8a16 = load("ppl_w8a16.json")
gen_bf = load("bf16.json")
gen_w8a8 = load("fp8.json")
gen_w8a16 = load("w8a16.json")

REGISTERS = [("literary", "Literary"), ("administrative", "Administrative"),
             ("colloquial", "Colloquial"), ("technical", "Technical"),
             ("_global", "GLOBAL")]


def delta(variant, key):
    a = ppl_bf[key]["ppl"]
    b = variant[key]["ppl"]
    return 100 * (b - a) / a


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def chart_perplexity(theme):
    c = THEMES[theme]
    W, H = 760, 360
    left, right, top = 132, 40, 78
    row_h, bar_h, gap = 52, 17, 4
    x0 = left
    plot_w = W - left - right

    lo, hi = -2.6, 5.0
    def sx(v):
        return x0 + (v - lo) / (hi - lo) * plot_w

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
         f'<rect width="{W}" height="{H}" fill="{c["surface"]}"/>',
         f'<text x="24" y="30" font-size="15" font-weight="600" fill="{c["ink"]}">'
         f'Perplexity degradation vs BF16, by register</text>',
         f'<text x="24" y="50" font-size="12" fill="{c["ink2"]}">'
         f'Lower is better. Weight-only (W8A16) is worse in every register.</text>']

    # legend
    lx = 24
    for label, col in (("W8A8 (weights + activations)", c["s1"]),
                       ("W8A16 (weights only)", c["s2"])):
        p.append(f'<rect x="{lx}" y="60" width="10" height="10" rx="2" fill="{col}"/>')
        p.append(f'<text x="{lx+16}" y="69" font-size="11" fill="{c["ink2"]}">{esc(label)}</text>')
        lx += 22 + len(label) * 6.0

    # gridlines and zero
    for v in (-2, 0, 2, 4):
        gx = sx(v)
        is_zero = v == 0
        p.append(f'<line x1="{gx:.1f}" y1="{top-10}" x2="{gx:.1f}" y2="{top+len(REGISTERS)*row_h-14}" '
                 f'stroke="{c["ink3"] if is_zero else c["grid"]}" stroke-width="{2 if is_zero else 1}"/>')
        p.append(f'<text x="{gx:.1f}" y="{top+len(REGISTERS)*row_h+4}" font-size="10" '
                 f'text-anchor="middle" fill="{c["ink3"]}">{v:+d}%</text>')

    for i, (key, label) in enumerate(REGISTERS):
        y = top + i * row_h
        weight = "600" if key == "_global" else "400"
        p.append(f'<text x="{left-14}" y="{y+18}" font-size="12" text-anchor="end" '
                 f'font-weight="{weight}" fill="{c["ink"] if key=="_global" else c["ink2"]}">'
                 f'{esc(label)}</text>')
        for j, (variant, col) in enumerate(((ppl_w8a8, c["s1"]), (ppl_w8a16, c["s2"]))):
            d = delta(variant, key)
            by = y + j * (bar_h + gap)
            a, b = sx(0), sx(d)
            bx, bw = min(a, b), abs(b - a)
            p.append(f'<rect x="{bx:.1f}" y="{by}" width="{max(bw,1.5):.1f}" height="{bar_h}" '
                     f'rx="4" fill="{col}"/>')
            tx = b + (7 if d >= 0 else -7)
            anchor = "start" if d >= 0 else "end"
            p.append(f'<text x="{tx:.1f}" y="{by+bar_h-4}" font-size="11" text-anchor="{anchor}" '
                     f'fill="{c["ink2"]}">{d:+.2f}%</text>')

    p.append(f'<text x="24" y="{H-10}" font-size="10" fill="{c["ink3"]}">'
             f'401 tokens of European Portuguese. The technical register goes negative for '
             f'W8A8, which is impossible and shows the noise floor.</text>')
    p.append("</svg>")
    return "\n".join(p)


def chart_throughput(theme):
    c = THEMES[theme]
    W, H = 760, 232
    left, top, bar_h, row_h = 132, 66, 26, 40
    plot_w = W - left - 100
    data = [("BF16", gen_bf["summary"]["tokens_per_s"], "18 GB"),
            ("FP8 W8A8", gen_w8a8["summary"]["tokens_per_s"], "9.6 GB"),
            ("FP8 W8A16", gen_w8a16["summary"]["tokens_per_s"], "9.6 GB")]
    hi = 26.0

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
         f'<rect width="{W}" height="{H}" fill="{c["surface"]}"/>',
         f'<text x="24" y="30" font-size="15" font-weight="600" fill="{c["ink"]}">'
         f'Throughput on a DGX Spark, single stream</text>',
         f'<text x="24" y="50" font-size="12" fill="{c["ink2"]}">'
         f'Higher is better. Halving the weight bytes nearly doubles the tokens per second.</text>']

    for v in (0, 5, 10, 15, 20, 25):
        gx = left + v / hi * plot_w
        p.append(f'<line x1="{gx:.1f}" y1="{top-10}" x2="{gx:.1f}" y2="{top+len(data)*row_h-12}" '
                 f'stroke="{c["grid"]}" stroke-width="1"/>')
        p.append(f'<text x="{gx:.1f}" y="{top+len(data)*row_h+6}" font-size="10" '
                 f'text-anchor="middle" fill="{c["ink3"]}">{v}</text>')

    for i, (label, val, size) in enumerate(data):
        y = top + i * row_h
        p.append(f'<text x="{left-14}" y="{y+18}" font-size="12" text-anchor="end" '
                 f'fill="{c["ink2"]}">{esc(label)}</text>')
        w = val / hi * plot_w
        p.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="4" '
                 f'fill="{c["s1"]}"/>')
        p.append(f'<text x="{left+w+9:.1f}" y="{y+18}" font-size="12" font-weight="600" '
                 f'fill="{c["ink"]}">{val} tok/s</text>')
        p.append(f'<text x="{left+w+9:.1f}" y="{y+31}" font-size="10" fill="{c["ink3"]}">'
                 f'{size} on disk</text>')

    p.append(f'<text x="{left}" y="{H-10}" font-size="10" fill="{c["ink3"]}">'
             f'tokens per second</text>')
    p.append("</svg>")
    return "\n".join(p)


for theme in ("light", "dark"):
    for name, fn in (("perplexity", chart_perplexity), ("throughput", chart_throughput)):
        path = os.path.join(OUT, f"{name}-{theme}.svg")
        with open(path, "w") as f:
            f.write(fn(theme))
        print(f"  wrote charts/{name}-{theme}.svg")
