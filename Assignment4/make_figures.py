"""Create the figures for the assignment. All numbers are read from results/*.csv.

  figure1_cv_by_k_and_metric.png  : 5-fold cross-validation mean by k and distance metric
  figure2_best_k_performance.png  : best k per distance metric, cross-validation and test scores
  figure3_input_representation.png: extra experiment, scores per input representation
  figure4_input_examples.png      : extra experiment, what each input looks like (needs the dataset)

Run: python make_figures.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"
REPR_DIR = RESULT_DIR / "input_representation"
METRICS = ["l1", "cosine", "l2", "linf"]
LABEL = {"l1": "L1", "cosine": "Cosine", "l2": "L2", "linf": "L∞"}
COLOR = {"l1": "#2a78d6", "cosine": "#1baf7a", "l2": "#eb6834", "linf": "#9a62d8"}
SCORES = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]
REPRESENTATIONS = ["raw", "gray", "he", "hog"]
REPR_LABEL = {"raw": "Raw color", "gray": "Grayscale", "he": "HE", "hog": "HOG"}
REPR_DIMS = {"raw": "3,072", "gray": "1,024", "he": "1,024", "hog": "1,764"}

plt.rcParams.update({"font.size": 12, "axes.unicode_minus": False})


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_k(summary):
    """The k with the highest mean F1 per distance metric, the same rule as knn_cifar10.select_k."""
    return {m: int(max((r for r in summary if r["metric"] == m),
                       key=lambda r: (round(float(r["f1_mean"]), 10), -int(r["k"])))["k"]) for m in METRICS}


def figure_cv(summary, selected):
    ks = sorted({int(r["k"]) for r in summary})
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.4), sharex=True, sharey=True)
    fig.suptitle("5-fold cross-validation mean by k and distance metric (20,000 images, shading: ±1 SD)",
                 fontsize=16, x=0.015, ha="left")
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
        ax.set_title(f'{name}   (best: {LABEL[best["metric"]]}, k={best["k"]}, {float(best[f"{key}_mean"]) * 100:.1f}%)',
                     fontsize=14)
        ax.plot([0, len(ks) - 1], [10, 10], color="#868c92", linestyle="--", linewidth=1)
        ax.text(0, 8.2, "Random guessing 10%", fontsize=10.5, color="#4d555d", va="top")
        ax.set_xticks(range(len(ks)), ks)
        ax.set_xlim(-0.3, len(ks) - 1 + 1.0)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", color="#dfe2e5", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[1]:
        ax.set_xlabel("k")
    for ax in axes[:, 0]:
        ax.set_ylabel("Score (%)")
    axes[1, 1].annotate("Circle: chosen k", xy=(0.99, 0.97), xycoords="axes fraction",
                        ha="right", va="top", fontsize=11, color="#4d555d")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=4, frameon=False, bbox_to_anchor=(0.985, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(BASE_DIR / "figure1_cv_by_k_and_metric.png", dpi=130)
    plt.close(fig)


def score_table(path, title, row_names, row_labels, selected, get_cv, get_test, best_name, first_column, note, extra=None):
    """Table figure with cross-validation and test scores, one row per setting."""
    keys = [k for k, _ in SCORES]
    n = len(row_names)
    fig, ax = plt.subplots(figsize=(14, 4.1))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.05)
    ax.axis("off")
    fig.suptitle(title, fontsize=17, x=0.015, ha="left")

    col_x = [0.7, 1.9] + [3.45 + i * 1.5 for i in range(4)] + [9.55 + i * 1.08 for i in range(4)]
    y_group, y_head, y_rows = 5.6, 4.85, [3.95 - i * 0.9 for i in range(n)]

    ax.fill_between([8.95, 13.75], 0.8, 6.05, color="#f1f2f4", linewidth=0, zorder=0)
    ax.text(5.7, y_group, "5-fold cross-validation (20,000 images)", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(11.17, y_group, "Test (50,000 train / 10,000 test)", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(col_x[0], y_head, first_column, ha="center", va="center", fontsize=12.5, fontweight="bold")
    ax.text(col_x[1], y_head, "Chosen k", ha="center", va="center", fontsize=12.5, fontweight="bold")
    for x, (_, name) in zip(col_x[2:6], SCORES):
        ax.text(x, y_head, name, ha="center", va="center", fontsize=12.5)
    for x, short in zip(col_x[6:], ["Acc", "Prec", "Rec", "F1"]):
        ax.text(x, y_head, short, ha="center", va="center", fontsize=12.5)
    ax.plot([0.15, 13.75], [4.45, 4.45], color="#4d555d", linewidth=1.2)

    for y, name in zip(y_rows, row_names):
        cv, test = get_cv(name), get_test(name)
        weight = "bold" if name == best_name else "normal"
        if name == best_name:
            ax.fill_between([0.15, 13.75], y - 0.42, y + 0.42, color="#e8f1fb", linewidth=0, zorder=0.5)
        ax.text(col_x[0], y, row_labels[name], ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(col_x[1], y, str(selected[name]), ha="center", va="center", fontsize=13, fontweight=weight)
        for x, key in zip(col_x[2:6], keys):
            ax.text(x, y, f'{float(cv[f"{key}_mean"]) * 100:.1f} ± {float(cv[f"{key}_std"]) * 100:.1f}',
                    ha="center", va="center", fontsize=13, fontweight=weight)
        for x, key in zip(col_x[6:], keys):
            ax.text(x, y, f"{float(test[key]) * 100:.1f}", ha="center", va="center", fontsize=13, fontweight=weight)
        ax.plot([0.15, 13.75], [y - 0.45, y - 0.45], color="#dfe2e5", linewidth=0.8)

    ax.text(0.15, 0.3, note, ha="left", va="center", fontsize=11.5, color="#4d555d")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.91, bottom=0.01)
    fig.savefig(BASE_DIR / path, dpi=130)
    plt.close(fig)


def figure_table(summary, test_rows, selected):
    chosen = [r for r in test_rows if r["selected_by_cv"] == "True"]
    best_metric = max(chosen, key=lambda r: next(float(s["f1_mean"]) for s in summary
                                                 if s["metric"] == r["metric"] and int(s["k"]) == int(r["k"])))["metric"]
    score_table(
        "figure2_best_k_performance.png", "Best k and performance by distance metric (%)",
        METRICS, LABEL, selected,
        lambda m: next(r for r in summary if r["metric"] == m and int(r["k"]) == selected[m]),
        lambda m: next(r for r in test_rows if r["metric"] == m and int(r["k"]) == selected[m]),
        best_metric, "Distance",
        "k was chosen only by mean cross-validation F1; test results were not used. "
        "Every class has the same number of images, so macro recall equals accuracy.")


def figure_input_representation():
    summary = read_csv(REPR_DIR / "cv_summary.csv")
    test_rows = read_csv(REPR_DIR / "test_results.csv")
    selected = {r["representation"]: int(r["k"]) for r in test_rows if r["selected_by_cv"] == "True"}
    best = max(REPRESENTATIONS, key=lambda n: next(float(r["f1_mean"]) for r in summary
                                                   if r["representation"] == n and int(r["k"]) == selected[n]))
    labels = {n: REPR_LABEL[n] for n in REPRESENTATIONS}
    score_table(
        "figure3_input_representation.png", "Extra experiment: KNN (L1) performance by input representation (%)",
        REPRESENTATIONS, labels, selected,
        lambda n: next(r for r in summary if r["representation"] == n and int(r["k"]) == selected[n]),
        lambda n: next(r for r in test_rows if r["representation"] == n and int(r["k"]) == selected[n]),
        best, "Input",
        "Same images, folds and L1 distance as the main experiment; only the input changes. "
        "k was chosen for each input by mean cross-validation F1.")


def figure_input_examples():
    """Show the four inputs for three test images. Skipped if the dataset is not in this folder."""
    import knn_cifar10 as K
    import input_representation as R
    from skimage.feature import hog
    from skimage.exposure import rescale_intensity

    if not K.ARCHIVE.exists():
        print("dataset not found: skipped figure4_input_examples.png")
        return
    _, _, X_test, y_test = K.load_cifar10()
    picks = [np.flatnonzero(y_test == K.CLASS_NAMES.index(name))[0] for name in ["automobile", "horse", "ship"]]
    fig, axes = plt.subplots(3, 4, figsize=(9, 7.2))
    fig.suptitle("What KNN compares for each input representation", fontsize=15, x=0.02, ha="left")
    for row, index in zip(axes, picks):
        x = X_test[index:index + 1]
        gray = R.to_gray(x)
        he = R.to_he(gray)
        _, view = hog(gray[0].reshape(32, 32) / 255.0, visualize=True, **{**R.HOG_PARAMS, "feature_vector": True})
        view = rescale_intensity(view, out_range=(0, 1))
        images = [x[0].reshape(3, 32, 32).transpose(1, 2, 0), gray[0].reshape(32, 32), he[0].reshape(32, 32), view]
        for ax, image in zip(row, images):
            ax.imshow(image, cmap="gray" if image.ndim == 2 else None, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(K.CLASS_NAMES[y_test[index]], fontsize=12)
    for ax, name in zip(axes[0], REPRESENTATIONS):
        ax.set_title(f"{REPR_LABEL[name]}\n({REPR_DIMS[name]} dimensions)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(BASE_DIR / "figure4_input_examples.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    cv_summary = read_csv(RESULT_DIR / "cv_summary.csv")
    tests = read_csv(RESULT_DIR / "test_results.csv")
    best_k = select_k(cv_summary)
    figure_cv(cv_summary, best_k)
    figure_table(cv_summary, tests, best_k)
    figure_input_representation()
    figure_input_examples()
    print("selected k:", best_k)
