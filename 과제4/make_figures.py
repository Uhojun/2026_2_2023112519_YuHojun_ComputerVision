"""검사용 이미지 2장을 만든다. 모든 숫자는 results/*.csv에서 읽는다.

  그림1_k_거리척도별_교차검증.png : k와 거리 척도별 5-fold 교차검증 평균
  그림2_거리척도별_최적k_성능.png : 거리 척도별 최적 k와 교차검증·테스트 성능

실행: python make_figures.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"
METRICS = ["l1", "cosine", "l2", "linf"]
LABEL = {"l1": "L1", "cosine": "코사인", "l2": "L2", "linf": "L∞"}
COLOR = {"l1": "#2a78d6", "cosine": "#1baf7a", "l2": "#eb6834", "linf": "#868c92"}
SCORES = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 12})


def read_csv(name):
    with open(RESULT_DIR / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_k(summary):
    """거리 척도별로 평균 F1이 가장 높은 k. knn_cifar10.select_k와 같은 기준이다."""
    return {m: int(max((r for r in summary if r["metric"] == m),
                       key=lambda r: (round(float(r["f1_mean"]), 10), -int(r["k"])))["k"]) for m in METRICS}


def figure_cv(summary, selected):
    ks = sorted({int(r["k"]) for r in summary})
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.4), sharex=True, sharey=True)
    fig.suptitle("k와 거리 척도별 5-fold 교차검증 평균 (20,000장, 음영: ±1 표준편차)", fontsize=17, x=0.015, ha="left")
    ymax = 50

    for ax, (key, name) in zip(axes.flat, SCORES):
        ends = []
        for metric in METRICS:
            rows = sorted((r for r in summary if r["metric"] == metric), key=lambda r: int(r["k"]))
            xs = [ks.index(int(r["k"])) for r in rows]
            mean = [float(r[f"{key}_mean"]) * 100 for r in rows]
            std = [float(r[f"{key}_std"]) * 100 for r in rows]
            ax.fill_between(xs, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)],
                            color=COLOR[metric], alpha=0.15, linewidth=0)
            ax.plot(xs, mean, color=COLOR[metric], linewidth=2, marker="o", markersize=5, label=LABEL[metric])
            if key == "f1":
                i = ks.index(selected[metric])
                ax.plot(i, mean[i], marker="o", markersize=13, markerfacecolor="none",
                        markeredgecolor=COLOR[metric], markeredgewidth=1.8, zorder=5)
            ends.append([mean[-1], metric])

        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 2.6)
        for value, metric in ends:
            ax.text(len(ks) - 1 + 0.2, value, LABEL[metric], va="center", fontsize=11.5)

        best = max(summary, key=lambda r: float(r[f"{key}_mean"]))
        ax.set_title(f'{name}   (최고: {LABEL[best["metric"]]}, k={best["k"]}, {float(best[f"{key}_mean"]) * 100:.1f}%)',
                     fontsize=14)
        ax.plot([0, len(ks) - 1], [10, 10], color="#868c92", linestyle="--", linewidth=1)
        ax.text(0, 8.2, "무작위 추측 10%", fontsize=10.5, color="#4d555d", va="top")
        ax.set_xticks(range(len(ks)), ks)
        ax.set_xlim(-0.3, len(ks) - 1 + 1.0)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", color="#dfe2e5", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[1]:
        ax.set_xlabel("k")
    for ax in axes[:, 0]:
        ax.set_ylabel("점수 (%)")
    axes[1, 1].annotate("동그라미: 선택된 k", xy=(0.99, 0.97), xycoords="axes fraction",
                        ha="right", va="top", fontsize=11, color="#4d555d")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=4, frameon=False, bbox_to_anchor=(0.985, 0.975))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(BASE_DIR / "그림1_k_거리척도별_교차검증.png", dpi=130)
    plt.close(fig)


def figure_table(summary, test_rows, selected):
    keys = [k for k, _ in SCORES]
    fig, ax = plt.subplots(figsize=(14, 4.1))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.05)
    ax.axis("off")
    fig.suptitle("거리 척도별 최적 k와 성능 (%)", fontsize=17, x=0.015, ha="left")

    col_x = [0.7, 1.9] + [3.45 + i * 1.5 for i in range(4)] + [9.55 + i * 1.08 for i in range(4)]
    y_group, y_head, y_rows = 5.6, 4.85, [3.95, 3.05, 2.15, 1.25]

    chosen = [r for r in test_rows if r["selected_by_cv"] == "True"]
    best_metric = max(chosen, key=lambda r: next(float(s["f1_mean"]) for s in summary
                                                 if s["metric"] == r["metric"] and int(s["k"]) == int(r["k"])))["metric"]

    ax.fill_between([8.95, 13.75], 0.8, 6.05, color="#f1f2f4", linewidth=0, zorder=0)
    ax.text(5.7, y_group, "5-fold 교차검증 (20,000장)", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(11.17, y_group, "테스트 (학습 50,000 / 테스트 10,000)", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(col_x[0], y_head, "거리 척도", ha="center", va="center", fontsize=12.5, fontweight="bold")
    ax.text(col_x[1], y_head, "선택 k", ha="center", va="center", fontsize=12.5, fontweight="bold")
    for x, (_, name) in zip(col_x[2:6], SCORES):
        ax.text(x, y_head, name, ha="center", va="center", fontsize=12.5)
    for x, (_, name) in zip(col_x[6:], SCORES):
        ax.text(x, y_head, name.replace("-score", "").replace("Accuracy", "Acc").replace("Precision", "Prec").replace("Recall", "Rec"),
                ha="center", va="center", fontsize=12.5)
    ax.plot([0.15, 13.75], [4.45, 4.45], color="#4d555d", linewidth=1.2)

    for y, metric in zip(y_rows, METRICS):
        cv = next(r for r in summary if r["metric"] == metric and int(r["k"]) == selected[metric])
        test = next((r for r in test_rows if r["metric"] == metric and int(r["k"]) == selected[metric]), None)
        is_best = metric == best_metric
        weight = "bold" if is_best else "normal"
        if is_best:
            ax.fill_between([0.15, 13.75], y - 0.42, y + 0.42, color="#e8f1fb", linewidth=0, zorder=0.5)
        ax.text(col_x[0], y, LABEL[metric], ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(col_x[1], y, str(selected[metric]), ha="center", va="center", fontsize=13, fontweight=weight)
        for x, key in zip(col_x[2:6], keys):
            ax.text(x, y, f'{float(cv[f"{key}_mean"]) * 100:.1f} ± {float(cv[f"{key}_std"]) * 100:.1f}',
                    ha="center", va="center", fontsize=13, fontweight=weight)
        if test:
            for x, key in zip(col_x[6:], keys):
                ax.text(x, y, f"{float(test[key]) * 100:.1f}", ha="center", va="center", fontsize=13, fontweight=weight)
        else:
            ax.text(11.17, y, "평가 진행 중", ha="center", va="center", fontsize=12, color="#4d555d")
        ax.plot([0.15, 13.75], [y - 0.45, y - 0.45], color="#dfe2e5", linewidth=0.8)

    ax.text(0.15, 0.3, "k는 교차검증 평균 F1로만 골랐고, 테스트 결과는 선택에 사용하지 않았습니다. "
                       "클래스별 장수가 같아 macro Recall은 Accuracy와 같습니다.",
            ha="left", va="center", fontsize=11.5, color="#4d555d")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.91, bottom=0.01)
    fig.savefig(BASE_DIR / "그림2_거리척도별_최적k_성능.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    cv_summary = read_csv("cv_summary.csv")
    tests = read_csv("test_results.csv")
    best_k = select_k(cv_summary)
    figure_cv(cv_summary, best_k)
    figure_table(cv_summary, tests, best_k)
    print("selected k:", best_k)
