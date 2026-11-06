"""results/*.csv를 읽어 한눈에 보는 결과 페이지(KNN_결과정리.html)를 만든다.

표와 그래프의 모든 숫자는 CSV에서 읽으므로 직접 입력한 값이 없다.
해석 문장(FINDINGS)만 결과를 보고 작성했다.

실행: python make_summary.py [--artifact-out 경로]
"""
import argparse
import csv
import html
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"
OUT = BASE_DIR / "KNN_결과정리.html"

METRICS = ["l1", "cosine", "l2", "linf"]
LABEL = {"l1": "L1", "l2": "L2", "cosine": "코사인", "linf": "L∞"}
COLOR = {"l1": "var(--s1)", "cosine": "var(--s3)", "l2": "var(--s2)", "linf": "var(--muted)"}
KR_CLASS = {"airplane": "비행기", "automobile": "자동차", "bird": "새", "cat": "고양이", "deer": "사슴",
            "dog": "개", "frog": "개구리", "horse": "말", "ship": "배", "truck": "트럭"}

FINDINGS = [
    "<b>거리 척도:</b> 테스트 정확도는 L1 39.2% &gt; 코사인 38.1% &gt; L2 35.6% &gt; L∞ 18.3% 순입니다. "
    "L∞는 가장 큰 픽셀 차이 하나만 보므로 나머지 3,071개 값의 정보를 버립니다. "
    "L2는 차이를 제곱해 일부 픽셀의 큰 차이에 끌리지만, L1은 모든 차이를 같은 비중으로 더합니다. "
    "코사인은 벡터 크기를 정규화해 사진 전체의 밝기 배율 차이를 줄입니다.",
    "<b>차이의 크기:</b> L1은 L2보다 5개 fold 모두에서 3.0~4.3%p 높아 차이가 분명합니다. "
    "반면 L1과 코사인의 교차검증 차이는 0.88%p로 fold 간 표준편차와 비슷해, 우열을 단정하기 어렵습니다.",
    "<b>k의 영향:</b> L1, 코사인, L2의 F1은 k=5~7에서 가장 높았고, k=101까지 가면서 떨어졌습니다. "
    "k=1은 가장 가까운 사진 한 장에 좌우되고, k가 너무 크면 관련 없는 사진까지 투표에 들어옵니다.",
    "<b>Precision과 F1:</b> k가 커지면 F1은 떨어지는데 Precision은 높게 유지됩니다. "
    "예측이 사슴·새·배로 쏠리면서 나머지 클래스는 드물게만 예측되어 정밀도가 오르고 재현율이 떨어지기 때문입니다. "
    "그래서 k는 F1로 골랐습니다.",
    "<b>클래스별 차이:</b> 배(F1 52.6%)와 비행기(50.0%)가 가장 높고 고양이(26.9%)가 가장 낮습니다. "
    "트럭·자동차·비행기는 배로(각 244, 219, 219장), 개구리·말은 사슴으로(293, 261장) 많이 틀렸습니다. "
    "픽셀 비교가 사물의 형태보다 배경과 색 배치에 끌린 결과로 추정됩니다.",
    "<b>전체 수준:</b> 39.2%는 무작위 추측의 약 4배지만 낮은 편입니다. "
    "원시 픽셀 거리로는 물체의 위치·색·배경 변화를 구분하기 어렵습니다.",
]
LIMITS = [
    "<b>데이터:</b> 학습 50,000장, 테스트 10,000장, 정규화 없는 원시 픽셀 3,072차원. "
    "교차검증은 CPU 실행 시간 때문에 클래스별 2,000장씩 뽑은 20,000장으로 했습니다(전체 실행: <code>python knn_cifar10.py --n-cv 50000</code>).",
    "<b>k 선택:</b> k를 20,000장 기준으로 골라 50,000장의 최적값과 다를 수 있습니다. "
    "L1의 테스트 F1은 k=3(39.1%)이 선택한 k=7(38.7%)보다 조금 높았지만, 테스트 결과로 k를 바꾸지 않았습니다.",
    "<b>L∞:</b> 거리 값이 정수라 동률이 많아, 동률 이웃을 고르는 방식에 따라 결과가 달라질 수 있습니다.",
    "<b>동점 처리:</b> 득표가 같으면 가장 가까운 이웃의 클래스를 고릅니다.",
    "<b>구현 검증:</b> 거리 값, 평가 지표, 혼동행렬을 실제 이미지 일부에서 scikit-learn과 비교해 일치함을 확인했습니다.",
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
    """거리 척도별 꺾은선. x축은 k(범주), y축은 0부터 시작한다."""
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
    parts.append(f'<text x="{w - right}" y="{y(0.1) + 13:.1f}" text-anchor="end" class="tick">무작위 10%</text>')
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
    note = f'최고 {LABEL[best["metric"]]}, k={best["k"]}: {pct(best[f"{field}_mean"])}%'
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
            tests = '<td class="t na" colspan="4">교차검증 최하위로 미평가</td>'
        cls = ' class="win"' if metric == best_metric else ""
        body.append(f'<tr{cls}><th scope="row">{LABEL[metric]}</th><td>{selected[metric]}</td>{cells}{tests}</tr>')
    return ('<div class="scroll"><table class="data"><thead>'
            '<tr><th rowspan="2" scope="col">거리 척도</th><th rowspan="2" scope="col">선택 k</th>'
            '<th colspan="4" scope="colgroup">5-fold 교차검증 (20,000장)</th>'
            '<th colspan="4" scope="colgroup" class="t">테스트 (학습 50,000 / 테스트 10,000)</th></tr>'
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
    body = "".join(f'<tr><th scope="row">{KR_CLASS[r["class"]]}</th><td>{pct(r["precision"])}</td>'
                   f'<td>{pct(r["recall"])}</td><td>{pct(r["f1"])}</td></tr>' for r in rows)
    return ('<table class="data"><thead><tr><th scope="col">클래스</th><th>Prec</th><th>Rec</th><th>F1</th></tr>'
            f'</thead><tbody>{body}</tbody></table>')


def confusion_table(path, class_names):
    with open(path, encoding="utf-8") as f:
        matrix = [[int(v) for v in line] for line in csv.reader(f)]
    peak = max(max(row) for row in matrix)
    header = "".join(f'<th scope="col">{KR_CLASS[c]}</th>' for c in class_names)
    rows = []
    for name, row in zip(class_names, matrix):
        cells = ""
        for value in row:
            share = round(value / peak * 85)
            cls = ' class="on"' if share > 50 else ""
            cells += f'<td{cls} style="background:color-mix(in srgb,var(--s1) {share}%,transparent)">{value}</td>'
        rows.append(f'<tr><th scope="row">{KR_CLASS[name]}</th>{cells}</tr>')
    return ('<div class="scroll"><table class="cm"><thead><tr><th scope="col">실제＼예측</th>'
            f'{header}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


STYLE = """
/* 강의실에서 노트북 화면으로 확인하는 2쪽 분량의 결과지: 위에서 결론, 아래로 근거 */
:root{
  --bg:#f6f7f8;--paper:#ffffff;--ink:#14181c;--ink2:#4d555d;--muted:#868c92;--line:#dfe2e5;--axis:#bfc4c9;
  --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--tint:#e8f1fb;
  --sans:"IBM Plex Sans KR","Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#101214;--paper:#1a1c1f;--ink:#f0f1f2;--ink2:#c0c5ca;--muted:#8d9399;--line:#2d3135;--axis:#3e4348;
  --s1:#3987e5;--s2:#d95926;--s3:#199e70;--tint:#1c2c40;color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#101214;--paper:#1a1c1f;--ink:#f0f1f2;--ink2:#c0c5ca;--muted:#8d9399;--line:#2d3135;--axis:#3e4348;
  --s1:#3987e5;--s2:#d95926;--s3:#199e70;--tint:#1c2c40;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 var(--sans);word-break:keep-all;
  padding-block:20px 32px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
main{max-width:1160px;margin:0 auto;display:grid;gap:14px}
section{background:var(--paper);border:1px solid var(--line);border-radius:6px;padding:16px 18px;display:grid;gap:10px;min-width:0}
h1{font-size:22px;margin:0;font-weight:700;text-wrap:balance}
h2{font-size:15px;margin:0;font-weight:700}
p{margin:0}
.top{display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px 20px;align-items:baseline}
.top input{font:inherit;color:var(--ink);background:transparent;border:0;border-bottom:1px solid var(--axis);padding:2px 4px;width:190px;max-width:100%}
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
input:focus-visible{outline:2px solid var(--s1);outline-offset:2px}
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

    # 클래스 수가 같으면 macro Recall과 Accuracy는 같아야 한다.
    assert all(abs(float(r["accuracy_mean"]) - float(r["recall_mean"])) < 1e-9 for r in summary)

    chosen = [r for r in test_rows if r["selected_by_cv"] == "True"]
    for r in chosen:
        assert int(r["k"]) == selected[r["metric"]]
    # 최고 설정은 테스트가 아니라 교차검증 F1로 정한다 (run_test와 같은 기준).
    best_metric = max(chosen, key=lambda r: next(
        float(s["f1_mean"]) for s in summary if s["metric"] == r["metric"] and int(s["k"]) == int(r["k"])))["metric"]
    best = next(r for r in chosen if r["metric"] == best_metric)
    assert per_class[0]["metric"] == best_metric and int(per_class[0]["k"]) == int(best["k"])
    ratio = float(best["accuracy"]) / 0.10
    answer = (f'교차검증으로 고른 최고 설정은 <b>{LABEL[best_metric]} 거리, k={best["k"]}</b>이고, '
              f'테스트 정확도는 <b>{pct(best["accuracy"])}%</b>로 무작위 추측(10%)의 {ratio:.1f}배입니다.')
    stats = "".join(
        f'<div class="stat"><span>{name}</span><strong>{pct(best[key])}<small>%</small></strong></div>'
        for name, key in [("Accuracy", "accuracy"), ("Precision (macro)", "precision"),
                          ("Recall (macro)", "recall"), ("F1-score (macro)", "f1")])

    checks = [
        ("KNN 구현", "NumPy로 직접 작성 (knn_cifar10.py)"),
        ("k 값 실험", f"k {len(ks)}개: {', '.join(map(str, ks))}"),
        ("거리 척도 실험", "L1, L2, L∞, 코사인"),
        ("5-fold 교차검증", f"{len(summary)}개 조합, 층화 분할"),
        ("평가 지표", "Accuracy, Precision, Recall, F1"),
        ("비교 및 설명", "아래 결과 해석"),
    ]
    check_html = "".join(f"<li><span><b>{a}</b> · {b}</span></li>" for a, b in checks)
    key_html = "".join(f'<span><i style="background:{COLOR[m]}"></i>{LABEL[m]}</span>' for m in METRICS)

    findings = "".join(f"<li>{x}</li>" for x in FINDINGS)
    limits = "".join(f"<li>{x}</li>" for x in LIMITS)
    detail = (f'<section><h2>최고 설정의 클래스별 테스트 결과 ({LABEL[best_metric]}, k={best["k"]})</h2>'
              f'<div class="two"><div><h3>클래스별 지표 (%), F1 높은 순</h3>{per_class_table(per_class)}</div>'
              f'<div><h3>혼동행렬 (클래스당 1,000장, 진할수록 많음)</h3>{confusion_table(cm_path, class_names)}</div></div></section>')

    return f"""<title>CIFAR-10 KNN 결과 정리</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap">
<style>{STYLE}</style>
<main>
<section>
  <div class="top"><h1>CIFAR-10 K-Nearest Neighbors 결과 정리</h1>
  <label>이름·학번 <input id="student" autocomplete="off"></label></div>
  <p class="answer">{answer}</p>
  <div class="stats">{stats}</div>
</section>
<section>
  <h2>과제 요구사항</h2>
  <ul class="checks">{check_html}</ul>
</section>
<section>
  <h2>k와 거리 척도별 5-fold 교차검증 평균 (%)</h2>
  <div class="key">{key_html}<span>음영: fold 간 ±1 표준편차</span><span>F1의 동그라미: 선택된 k</span></div>
  <div class="panels">
    {panel(summary, "accuracy", "Accuracy = Recall", selected)}
    {panel(summary, "precision", "Precision", selected)}
    {panel(summary, "f1", "F1-score", selected, True)}
  </div>
</section>
<section>
  <h2>거리 척도별 최적 k와 성능 (%)</h2>
  {best_table(summary, test_rows, selected, best_metric)}
  <p class="note">k는 교차검증 평균 F1로만 골랐고, 테스트 결과는 선택에 사용하지 않았습니다. 클래스별 장수가 같아 macro Recall은 Accuracy와 같습니다.</p>
</section>
<section class="sheet2">
  <h2>결과 해석</h2>
  <ul class="find">{findings}</ul>
</section>
{detail}
<section>
  <h2>전체 교차검증 결과 (5-fold 평균, %)</h2>
  {full_table(summary, selected)}
  <p class="note">파란 칸은 거리 척도별 선택 k입니다. 클래스별 장수가 같아 Recall과 Accuracy는 같은 값입니다. fold별 값과 표준편차는 results/cv_per_fold.csv, results/cv_summary.csv에 있습니다.</p>
</section>
<section>
  <h2>실험 조건과 한계</h2>
  <ul class="find">{limits}</ul>
</section>
</main>
<script>
(function(){{var el=document.getElementById("student");
try{{el.value=localStorage.getItem("student")||"";}}catch(e){{}}
el.addEventListener("input",function(){{try{{localStorage.setItem("student",el.value);}}catch(e){{}}}});}})();
</script>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-out", help="문서 골격 없이 본문만 저장할 경로")
    args = parser.parse_args()

    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    page = build(read_csv("cv_summary.csv"), read_csv("test_results.csv"), read_csv("test_per_class.csv"),
                 RESULT_DIR / "test_confusion_matrix.csv", classes)
    OUT.write_text('<!doctype html>\n<html lang="ko"><head><meta charset="utf-8">'
                   '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
                   f"{page}\n</html>\n", encoding="utf-8")
    if args.artifact_out:
        Path(args.artifact_out).write_text(page, encoding="utf-8")
    print("wrote", OUT)
