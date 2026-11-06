"""Build the at-a-glance results page (KNN_summary.html) from results/*.csv.

All numbers in the tables and charts are read from the CSV files; none are typed in.
Only the interpretation text (FINDINGS) was written by hand from the results.

Run: python make_summary.py [--artifact-out PATH]
"""
import argparse
import csv
import html
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"
OUT = BASE_DIR / "KNN_summary.html"

METRICS = ["l1", "cosine", "l2", "linf"]
LABEL = {"l1": "L1", "l2": "L2", "cosine": "Cosine", "linf": "L∞"}
COLOR = {"l1": "var(--s1)", "cosine": "var(--s3)", "l2": "var(--s2)", "linf": "var(--s4)"}
CLASS_LABEL = {"airplane": "Airplane", "automobile": "Automobile", "bird": "Bird", "cat": "Cat", "deer": "Deer",
               "dog": "Dog", "frog": "Frog", "horse": "Horse", "ship": "Ship", "truck": "Truck"}

FINDINGS = [
    "<b>Distance metric:</b> Test accuracy ranks L1 39.2% &gt; cosine 38.1% &gt; L2 35.6% &gt; L∞ 18.3%. "
    "L∞ looks only at the single largest pixel difference, discarding the information in the other 3,071 values. "
    "L2 squares differences, so a few large pixel differences dominate it, while L1 adds all differences with equal weight. "
    "Cosine normalizes vector length, which reduces differences in overall image brightness scale.",
    "<b>Size of the differences:</b> L1 beats L2 by 3.0-4.3 percentage points in all five folds, a clear difference. "
    "L1 and cosine differ by only 0.88 points in cross-validation, about the size of the standard deviation across folds, so neither can be called better.",
    "<b>Effect of k:</b> F1 for L1, cosine and L2 peaks at k=5-7 and falls as k grows to 101. "
    "k=1 depends on a single nearest image, while a very large k lets unrelated images into the vote.",
    "<b>Precision and F1:</b> As k grows, F1 falls while precision stays high. "
    "Predictions concentrate on deer, bird and ship, so the other classes are predicted rarely, which raises their precision and lowers their recall. "
    "This is why k was chosen by F1.",
    "<b>Differences between classes:</b> Ship (F1 52.6%) and airplane (50.0%) score highest and cat (26.9%) lowest. "
    "Trucks, automobiles and airplanes are often mistaken for ships (244, 219 and 219 images), and frogs and horses for deer (293 and 261 images). "
    "This suggests that pixel comparison is drawn to backgrounds and color layout more than to object shape.",
    "<b>Overall level:</b> 39.2% is about four times random guessing, but still low. "
    "Raw pixel distances struggle to handle changes in object position, color and background.",
]
LIMITS = [
    "<b>Data:</b> 50,000 training images and 10,000 test images, as 3,072-dimensional raw pixels with no normalization. "
    "To limit CPU time, cross-validation used 20,000 images, 2,000 per class (full run: <code>python knn_cifar10.py --n-cv 50000</code>).",
    "<b>Choosing k:</b> k was chosen on 20,000 images, so it may differ from the best value for 50,000. "
    "For L1, test F1 at k=3 (39.1%) was slightly higher than at the chosen k=7 (38.7%), but test results were not used to change k.",
    "<b>L∞:</b> Its distances are integers with many ties, so results can depend on how tied neighbors are picked.",
    "<b>Vote ties:</b> When votes tie, the class of the nearest neighbor is chosen.",
    "<b>Implementation check:</b> Distances, evaluation metrics and the confusion matrix matched scikit-learn on a sample of real images.",
]


def read_csv(name):
    path = RESULT_DIR / name
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pct(x, digits=1):
    return f"{float(x) * 100:.{digits}f}"


def line_chart(summary, field, title, selected, mark_selected=False):
    """One line per distance metric. The x axis is k (categorical); the y axis starts at 0."""
    ks = sorted({int(r["k"]) for r in summary})
    w, h, left, right, y0, bottom = 400, 215, 34, 54, 12, 38
    top_value = max(float(r[f"{field}_mean"]) + float(r[f"{field}_std"]) for r in summary)
    ymax = (int(top_value * 20) + 1) / 20
    x = lambda k: left + ks.index(k) * (w - left - right) / (len(ks) - 1)
    y = lambda v: y0 + (1 - v / ymax) * (h - y0 - bottom)
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}">']

    tick = 0.0
    while tick <= ymax + 1e-9:
        parts.append(f'<line x1="{left}" x2="{w - right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 6}" y="{y(tick) + 4:.1f}" text-anchor="end" class="tick">{tick * 100:.0f}</text>')
        tick += 0.1
    parts.append(f'<line x1="{left}" x2="{w - right}" y1="{y(0.1):.1f}" y2="{y(0.1):.1f}" class="ref"/>')
    parts.append(f'<text x="{w - right}" y="{y(0.1) + 13:.1f}" text-anchor="end" class="tick">Random 10%</text>')
    for k in ks:
        parts.append(f'<text x="{x(k):.1f}" y="{h - bottom + 16}" text-anchor="middle" class="tick">{k}</text>')
    parts.append(f'<text x="{(left + w - right) / 2}" y="{h - 6}" text-anchor="middle" class="tick">k</text>')

    ends = []
    for metric in METRICS:
        rows = sorted((r for r in summary if r["metric"] == metric), key=lambda r: int(r["k"]))
        if not rows:
            continue
        points = [(x(int(r["k"])), float(r[f"{field}_mean"]), float(r[f"{field}_std"]), r) for r in rows]
        band = " ".join(f"{px:.1f},{y(v + s):.1f}" for px, v, s, _ in points) + " " + \
               " ".join(f"{px:.1f},{y(v - s):.1f}" for px, v, s, _ in reversed(points))
        line = " ".join(f"{px:.1f},{y(v):.1f}" for px, v, _, _ in points)
        parts.append(f'<g><polygon points="{band}" fill="{COLOR[metric]}" fill-opacity=".14"/>')
        parts.append(f'<polyline points="{line}" fill="none" stroke="{COLOR[metric]}" stroke-width="2"/>')
        for px, v, s, r in points:
            is_sel = mark_selected and int(r["k"]) == selected[metric]
            ring = f'<circle cx="{px:.1f}" cy="{y(v):.1f}" r="7" fill="none" stroke="{COLOR[metric]}" stroke-width="1.5"/>' if is_sel else ""
            parts.append(f'{ring}<circle cx="{px:.1f}" cy="{y(v):.1f}" r="3.5" fill="{COLOR[metric]}" class="dot">'
                         f'<title>{LABEL[metric]}, k={r["k"]}: {pct(v)}% ± {pct(s)}</title></circle>')
        parts.append("</g>")
        ends.append([y(points[-1][1]) + 4, metric])

    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 13)
    for ly, metric in ends:
        parts.append(f'<text x="{w - right + 8}" y="{ly:.1f}" class="endlabel">{LABEL[metric]}</text>')
    parts.append("</svg>")
    return "".join(parts)


def panel(summary, field, title, selected, mark_selected=False):
    best = max(summary, key=lambda r: float(r[f"{field}_mean"]))
    note = f'Best: {LABEL[best["metric"]]}, k={best["k"]}, {pct(best[f"{field}_mean"])}%'
    return (f'<figure class="panel"><figcaption><strong>{title}</strong><span>{note}</span></figcaption>'
            f'{line_chart(summary, field, title, selected, mark_selected)}</figure>')


def best_table(summary, test_rows, selected, best_metric):
    body = []
    for metric in METRICS:
        cv = next(r for r in summary if r["metric"] == metric and int(r["k"]) == selected[metric])
        test = next((r for r in test_rows if r["metric"] == metric and int(r["k"]) == selected[metric]), None)
        cells = "".join(f'<td>{pct(cv[f"{m}_mean"])}<small> ±{pct(cv[f"{m}_std"])}</small></td>'
                        for m in ["accuracy", "precision", "recall", "f1"])
        if test:
            tests = "".join(f'<td class="t">{pct(test[m])}</td>' for m in ["accuracy", "precision", "recall", "f1"])
        else:
            tests = '<td class="t na" colspan="4">Not evaluated (lowest in cross-validation)</td>'
        cls = ' class="win"' if metric == best_metric else ""
        body.append(f'<tr{cls}><th scope="row">{LABEL[metric]}</th><td>{selected[metric]}</td>{cells}{tests}</tr>')
    return ('<div class="scroll"><table class="data"><thead>'
            '<tr><th rowspan="2" scope="col">Distance</th><th rowspan="2" scope="col">Chosen k</th>'
            '<th colspan="4" scope="colgroup">5-fold cross-validation (20,000 images)</th>'
            '<th colspan="4" scope="colgroup" class="t">Test (50,000 train / 10,000 test)</th></tr>'
            '<tr><th>Acc</th><th>Prec</th><th>Rec</th><th>F1</th>'
            '<th class="t">Acc</th><th class="t">Prec</th><th class="t">Rec</th><th class="t">F1</th></tr>'
            '</thead><tbody>' + "".join(body) + "</tbody></table></div>")


def full_table(summary, selected):
    ks = sorted({int(r["k"]) for r in summary})
    head1 = "".join(f'<th colspan="4" scope="colgroup">{LABEL[m]}</th>' for m in METRICS)
    head2 = "<th>Acc</th><th>Prec</th><th>Rec</th><th>F1</th>" * len(METRICS)
    rows = []
    for k in ks:
        cells = ""
        for m in METRICS:
            r = next(x for x in summary if x["metric"] == m and int(x["k"]) == k)
            cls = ' class="sel"' if selected[m] == k else ""
            cells += (f'<td{cls}>{pct(r["accuracy_mean"])}</td><td{cls}>{pct(r["precision_mean"])}</td>'
                      f'<td{cls}>{pct(r["recall_mean"])}</td><td{cls}>{pct(r["f1_mean"])}</td>')
        rows.append(f'<tr><th scope="row">{k}</th>{cells}</tr>')
    return ('<div class="scroll"><table class="data dense"><thead>'
            f'<tr><th rowspan="2" scope="col">k</th>{head1}</tr><tr>{head2}</tr></thead>'
            '<tbody>' + "".join(rows) + "</tbody></table></div>")


def per_class_table(rows):
    rows = sorted(rows, key=lambda r: -float(r["f1"]))
    body = "".join(f'<tr><th scope="row">{CLASS_LABEL[r["class"]]}</th><td>{pct(r["precision"])}</td>'
                   f'<td>{pct(r["recall"])}</td><td>{pct(r["f1"])}</td></tr>' for r in rows)
    return ('<table class="data"><thead><tr><th scope="col">Class</th><th>Prec</th><th>Rec</th><th>F1</th></tr>'
            f'</thead><tbody>{body}</tbody></table>')


def confusion_table(path, class_names):
    with open(path, encoding="utf-8") as f:
        matrix = [[int(v) for v in line] for line in csv.reader(f)]
    peak = max(max(row) for row in matrix)
    header = "".join(f'<th scope="col">{CLASS_LABEL[c]}</th>' for c in class_names)
    rows = []
    for name, row in zip(class_names, matrix):
        cells = ""
        for value in row:
            share = round(value / peak * 85)
            cls = ' class="on"' if share > 50 else ""
            cells += f'<td{cls} style="background:color-mix(in srgb,var(--s1) {share}%,transparent)">{value}</td>'
        rows.append(f'<tr><th scope="row">{CLASS_LABEL[name]}</th>{cells}</tr>')
    return ('<div class="scroll"><table class="cm"><thead><tr><th scope="col">True \\ Predicted</th>'
            f'{header}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


STYLE = """
/* Two-page result sheet read on a laptop in the classroom: conclusion on top, evidence below */
:root{
  --bg:#f6f7f8;--paper:#ffffff;--ink:#14181c;--ink2:#4d555d;--muted:#868c92;--line:#dfe2e5;--axis:#bfc4c9;
  --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#9a62d8;--tint:#e8f1fb;
  --sans:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#101214;--paper:#1a1c1f;--ink:#f0f1f2;--ink2:#c0c5ca;--muted:#8d9399;--line:#2d3135;--axis:#3e4348;
  --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#a06fdc;--tint:#1c2c40;color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#101214;--paper:#1a1c1f;--ink:#f0f1f2;--ink2:#c0c5ca;--muted:#8d9399;--line:#2d3135;--axis:#3e4348;
  --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#a06fdc;--tint:#1c2c40;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 var(--sans);word-break:keep-all;
  padding-block:20px 32px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
main{max-width:1160px;margin:0 auto;display:grid;gap:14px}
section{background:var(--paper);border:1px solid var(--line);border-radius:6px;padding:16px 18px;display:grid;gap:10px;min-width:0}
h1{font-size:22px;margin:0;font-weight:700;text-wrap:balance}
h2{font-size:15px;margin:0;font-weight:700}
p{margin:0}
.top{display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px 20px;align-items:baseline}
.answer{font-size:17px;font-weight:600;text-wrap:balance}
.answer b{color:var(--s1)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px 20px}
.stat span{display:block;color:var(--ink2);font-size:12.5px}
.stat strong{font-size:27px;font-weight:700;line-height:1.2}
.stat strong small{font-size:14px;font-weight:500}
.checks{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:4px 18px;margin:0;padding:0;list-style:none;font-size:13px}
.checks li{display:grid;grid-template-columns:18px 1fr;gap:2px}
.checks li::before{content:"✓";color:var(--s1);font-weight:700}
.checks b{font-weight:600}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
.panel{margin:0;min-width:0}
.panel figcaption{display:flex;flex-wrap:wrap;justify-content:space-between;gap:0 8px;font-size:13px}
.panel figcaption span{color:var(--ink2);font-size:12px}
svg{display:block;width:100%;height:auto}
svg text{font-family:var(--sans)}
.grid{stroke:var(--line);stroke-width:1}
.ref{stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3}
.tick{fill:var(--ink2);font-size:11px;font-variant-numeric:tabular-nums}
.endlabel{fill:var(--ink);font-size:12px;font-weight:500}
.dot{stroke:var(--paper);stroke-width:1.5}
.key{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:12.5px;color:var(--ink2)}
.key i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:baseline}
.scroll{overflow-x:auto;min-width:0}
table{border-collapse:collapse;font-variant-numeric:tabular-nums}
.data{width:100%;font-size:13px}
.data th,.data td{padding:4px 7px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
.data thead th{font-weight:600;color:var(--ink2);text-align:center;border-bottom:1px solid var(--axis)}
.data tbody th{text-align:left;font-weight:600}
.data small{color:var(--ink2);font-size:11px}
.data .t{background:color-mix(in srgb,var(--line) 35%,transparent)}
.data .na{color:var(--ink2);text-align:center;font-size:12px}
.data tr.win th,.data tr.win td{font-weight:700;background:var(--tint)}
.data tr.win small{font-weight:400}
.dense{font-size:12.5px}
.dense th,.dense td{padding:3px 6px}
.dense td.sel{background:var(--tint);font-weight:700}
.dense thead th[colspan],.dense tbody td:nth-child(4n+2){border-left:1px solid var(--axis)}
.two{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,8fr);gap:18px}
.two>div{display:grid;gap:6px;align-content:start;min-width:0}
h3{font-size:13px;margin:0;font-weight:600}
.cm{font-size:12px;width:100%}
.cm th{font-weight:500;color:var(--ink2);padding:3px 4px;white-space:nowrap;font-size:11.5px}
.cm tbody th{text-align:right}
.cm td{padding:5px 2px;text-align:center;border:1px solid var(--paper);min-width:34px}
.cm td.on{color:#fff;font-weight:600}
.note{color:var(--ink2);font-size:12.5px}
ul.find{margin:0;padding-left:18px;display:grid;gap:5px}
code{font-size:12.5px;background:color-mix(in srgb,var(--line) 45%,transparent);padding:1px 4px;border-radius:3px;overflow-wrap:anywhere}
.sheet2{break-before:page}
@media (max-width:760px){.two{grid-template-columns:minmax(0,1fr)}}
@page{size:A4;margin:11mm}
@media print{
  body{background:#fff;padding:0;font-size:10.5px;line-height:1.4}
  main{max-width:none;gap:6px}
  section{border:0;padding:3px 0;gap:5px;break-inside:avoid}
  h1{font-size:17px}
  h2{font-size:12px}
  h3,.data .na{font-size:10px}
  .answer{font-size:13px}
  .stat strong{font-size:18px}
  .checks{font-size:10px}
  .panels{grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
  .panel figcaption,.panel figcaption span,.key,.note{font-size:9.5px}
  .data,.dense,.cm{font-size:9.5px}
  .data th,.data td,.dense th,.dense td{padding:2px 4px}
  .cm td{padding:2px 1px;min-width:0}
  .cm th{font-size:9px}
  .two{grid-template-columns:minmax(0,3fr) minmax(0,8fr)}
  code{font-size:9.5px}
  *{-webkit-print-color-adjust:exact;print-color-adjust:exact}
}
"""


def build(summary, test_rows, per_class, cm_path, class_names):
    ks = sorted({int(r["k"]) for r in summary})
    selected = {}
    for metric in METRICS:
        rows = [r for r in summary if r["metric"] == metric]
        selected[metric] = int(max(rows, key=lambda r: (round(float(r["f1_mean"]), 10), -int(r["k"])))["k"])

    # With equal class sizes, macro recall must equal accuracy.
    assert all(abs(float(r["accuracy_mean"]) - float(r["recall_mean"])) < 1e-9 for r in summary)

    chosen = [r for r in test_rows if r["selected_by_cv"] == "True"]
    for r in chosen:
        assert int(r["k"]) == selected[r["metric"]]
    # The best setting is chosen by cross-validation F1, not test results (same rule as run_test).
    best_metric = max(chosen, key=lambda r: next(
        float(s["f1_mean"]) for s in summary if s["metric"] == r["metric"] and int(s["k"]) == int(r["k"])))["metric"]
    best = next(r for r in chosen if r["metric"] == best_metric)
    assert per_class[0]["metric"] == best_metric and int(per_class[0]["k"]) == int(best["k"])
    ratio = float(best["accuracy"]) / 0.10
    answer = (f'The best setting chosen by cross-validation is <b>{LABEL[best_metric]} distance with k={best["k"]}</b>. '
              f'Its test accuracy is <b>{pct(best["accuracy"])}%</b>, {ratio:.1f} times random guessing (10%).')
    stats = "".join(
        f'<div class="stat"><span>{name}</span><strong>{pct(best[key])}<small>%</small></strong></div>'
        for name, key in [("Accuracy", "accuracy"), ("Precision (macro)", "precision"),
                          ("Recall (macro)", "recall"), ("F1-score (macro)", "f1")])

    checks = [
        ("KNN implementation", "written in NumPy (knn_cifar10.py)"),
        ("Values of k", f"{len(ks)} values: {', '.join(map(str, ks))}"),
        ("Distance metrics", "L1, L2, L∞, cosine"),
        ("5-fold cross-validation", f"{len(summary)} combinations, stratified folds"),
        ("Evaluation metrics", "Accuracy, Precision, Recall, F1"),
        ("Comparison and explanation", "see Interpretation below"),
    ]
    check_html = "".join(f"<li><span><b>{a}</b> · {b}</span></li>" for a, b in checks)
    key_html = "".join(f'<span><i style="background:{COLOR[m]}"></i>{LABEL[m]}</span>' for m in METRICS)

    findings = "".join(f"<li>{x}</li>" for x in FINDINGS)
    limits = "".join(f"<li>{x}</li>" for x in LIMITS)
    detail = (f'<section><h2>Per-class test results for the best setting ({LABEL[best_metric]}, k={best["k"]})</h2>'
              f'<div class="two"><div><h3>Per-class scores (%), by F1</h3>{per_class_table(per_class)}</div>'
              f'<div><h3>Confusion matrix (1,000 images per class; darker means more)</h3>{confusion_table(cm_path, class_names)}</div></div></section>')

    return f"""<title>CIFAR-10 KNN Results</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>{STYLE}</style>
<main>
<section>
  <div class="top"><h1>CIFAR-10 K-Nearest Neighbors Results</h1>
  <span>Yu Ho-jun, 2023112519</span></div>
  <p class="answer">{answer}</p>
  <div class="stats">{stats}</div>
</section>
<section>
  <h2>Assignment requirements</h2>
  <ul class="checks">{check_html}</ul>
</section>
<section>
  <h2>5-fold cross-validation mean by k and distance metric (%)</h2>
  <div class="key">{key_html}<span>Shading: ±1 standard deviation across folds</span><span>Circle on F1: chosen k</span></div>
  <div class="panels">
    {panel(summary, "accuracy", "Accuracy = Recall", selected)}
    {panel(summary, "precision", "Precision", selected)}
    {panel(summary, "f1", "F1-score", selected, True)}
  </div>
</section>
<section>
  <h2>Best k and performance by distance metric (%)</h2>
  {best_table(summary, test_rows, selected, best_metric)}
  <p class="note">k was chosen only by mean cross-validation F1; test results were not used. Because every class has the same number of images, macro recall equals accuracy.</p>
</section>
<section class="sheet2">
  <h2>Interpretation</h2>
  <ul class="find">{findings}</ul>
</section>
{detail}
<section>
  <h2>Full cross-validation results (5-fold mean, %)</h2>
  {full_table(summary, selected)}
  <p class="note">Blue cells mark the chosen k for each distance metric. Recall equals accuracy because every class has the same number of images. Per-fold values and standard deviations are in results/cv_per_fold.csv and results/cv_summary.csv.</p>
</section>
<section>
  <h2>Conditions and limitations</h2>
  <ul class="find">{limits}</ul>
</section>
</main>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-out", help="path for saving the page body without the document skeleton")
    args = parser.parse_args()

    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    page = build(read_csv("cv_summary.csv"), read_csv("test_results.csv"), read_csv("test_per_class.csv"),
                 RESULT_DIR / "test_confusion_matrix.csv", classes)
    OUT.write_text('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
                   '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
                   f"{page}\n</html>\n", encoding="utf-8")
    if args.artifact_out:
        Path(args.artifact_out).write_text(page, encoding="utf-8")
    print("wrote", OUT)
