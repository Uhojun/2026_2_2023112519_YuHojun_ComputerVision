"""Create the two figures for the input representation experiment.

Scores are read from results/input_representation/*.csv.

  figure3_input_examples.png   : the same images as the four inputs
  figure4_input_performance.png: cross-validation F1 by k and final scores for each input

Run: python make_repr_figures.py
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog

import input_representation as R
import knn_cifar10 as K

OUT_DIR = R.OUT_DIR
ORDER = ["hog", "raw", "he", "gray"]  # highest cross-validation F1 first
LABEL = {"raw": "Raw color", "gray": "Grayscale", "he": "HE", "hog": "HOG"}
DIMS = {"raw": "3,072", "gray": "1,024", "he": "1,024", "hog": "1,764"}
COLOR = {"raw": "#2a78d6", "hog": "#eb6834", "he": "#1baf7a", "gray": "#eda100"}
SCORES = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]
EXAMPLE_CLASSES = ["automobile", "ship", "frog"]

plt.rcParams.update({"axes.unicode_minus": False, "font.size": 12})


def read_csv(name):
    with open(OUT_DIR / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def figure_examples():
    """Transform the first test image of each class."""
    _, _, X_test, y_test = K.load_cifar10()
    fig, axes = plt.subplots(len(EXAMPLE_CLASSES), 4, figsize=(9.5, 7.6))
    fig.suptitle("The same images as four inputs", fontsize=17, x=0.02, ha="left")
    for row, name in zip(axes, EXAMPLE_CLASSES):
        index = np.flatnonzero(y_test == K.CLASS_NAMES.index(name))[0]
        x = X_test[index:index + 1]
        gray = R.to_gray(x)
        panels = [x[0].reshape(3, 32, 32).transpose(1, 2, 0), gray[0].reshape(32, 32),
                  R.to_he(gray)[0].reshape(32, 32)]
        _, picture = hog(gray[0].reshape(32, 32) / 255.0, visualize=True, **R.HOG_PARAMS)
        panels.append(picture)
        for ax, image in zip(row, panels):
            ax.imshow(image, cmap="gray", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
        row[0].set_ylabel(name, fontsize=13)
    titles = ["Raw color (3,072 dim.)", "Grayscale (1,024 dim.)", "HE (1,024 dim.)", "HOG (1,764 dim.)"]
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(K.BASE_DIR / "figure3_input_examples.png", dpi=130)
    plt.close(fig)


def draw_curves(ax, summary, tests):
    ks = K.K_VALUES
    selected = {r["representation"]: int(r["k"]) for r in tests if r["selected_by_cv"] == "True"}
    ends = []
    for name in ORDER:
        rows = sorted((r for r in summary if r["representation"] == name), key=lambda r: int(r["k"]))
        mean = np.array([float(r["f1_mean"]) * 100 for r in rows])
        std = np.array([float(r["f1_std"]) * 100 for r in rows])
        xs = np.arange(len(ks))
        ax.fill_between(xs, mean - std, mean + std, color=COLOR[name], alpha=0.15, linewidth=0)
        ax.plot(xs, mean, color=COLOR[name], linewidth=2, marker="o", markersize=5, label=LABEL[name])
        i = ks.index(selected[name])
        ax.plot(i, mean[i], marker="o", markersize=13, markerfacecolor="none",
                markeredgecolor=COLOR[name], markeredgewidth=1.8, zorder=5)
        ends.append([mean[-1], name])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 3)
    for value, name in ends:
        ax.text(len(ks) - 1 + 0.2, value, LABEL[name], va="center", fontsize=12)
    ax.set_title("5-fold cross-validation mean F1 by k (L1 distance, 20,000 images, shading: ±1 SD, circle: chosen k)",
                 fontsize=13, loc="left")
    ax.set_xticks(range(len(ks)), ks)
    ax.set_xlim(-0.3, len(ks) - 1 + 1.0)
    ax.set_ylim(0, 60)
    ax.set_xlabel("k")
    ax.set_ylabel("F1 (%)")
    ax.grid(axis="y", color="#dfe2e5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", ncol=4, frameon=False)


def draw_table(ax, summary, tests):
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.05)
    ax.axis("off")
    col_x = [0.75, 1.95, 2.95] + [4.2 + i * 1.28 for i in range(4)] + [9.7 + i * 1.05 for i in range(4)]
    y_group, y_head, y_rows = 5.6, 4.85, [3.95, 3.05, 2.15, 1.25]
    ax.fill_between([9.1, 13.75], 0.8, 6.05, color="#f1f2f4", linewidth=0, zorder=0)
    ax.text(6.1, y_group, "5-fold cross-validation (20,000 images)", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(11.27, y_group, "Test (50,000 train / 10,000 test)", ha="center", va="center", fontsize=13, fontweight="bold")
    for x, text in zip(col_x[:3], ["Input", "Dim.", "Chosen k"]):
        ax.text(x, y_head, text, ha="center", va="center", fontsize=12.5, fontweight="bold")
    for x, (_, name) in zip(col_x[3:7], SCORES):
        ax.text(x, y_head, name, ha="center", va="center", fontsize=12.5)
    for x, name in zip(col_x[7:], ["Acc", "Prec", "Rec", "F1"]):
        ax.text(x, y_head, name, ha="center", va="center", fontsize=12.5)
    ax.plot([0.15, 13.75], [4.45, 4.45], color="#4d555d", linewidth=1.2)

    keys = [k for k, _ in SCORES]
    for y, name in zip(y_rows, ORDER):
        test = next(r for r in tests if r["representation"] == name and r["selected_by_cv"] == "True")
        cv = next(r for r in summary if r["representation"] == name and r["k"] == test["k"])
        best = name == ORDER[0]
        weight = "bold" if best else "normal"
        if best:
            ax.fill_between([0.15, 13.75], y - 0.42, y + 0.42, color="#fdeee7", linewidth=0, zorder=0.5)
        ax.text(col_x[0], y, LABEL[name], ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(col_x[1], y, DIMS[name], ha="center", va="center", fontsize=13, fontweight=weight)
        ax.text(col_x[2], y, test["k"], ha="center", va="center", fontsize=13, fontweight=weight)
        for x, key in zip(col_x[3:7], keys):
            ax.text(x, y, f'{float(cv[f"{key}_mean"]) * 100:.1f} ± {float(cv[f"{key}_std"]) * 100:.1f}',
                    ha="center", va="center", fontsize=12.5, fontweight=weight)
        for x, key in zip(col_x[7:], keys):
            ax.text(x, y, f"{float(test[key]) * 100:.1f}", ha="center", va="center", fontsize=13, fontweight=weight)
        ax.plot([0.15, 13.75], [y - 0.45, y - 0.45], color="#dfe2e5", linewidth=0.8)
    ax.text(0.15, 0.3, "The distance is fixed to L1. k was chosen for each input only by mean cross-validation F1; test results were not used.",
            ha="left", va="center", fontsize=11.5, color="#4d555d")


def figure_performance():
    summary, tests = read_csv("cv_summary.csv"), read_csv("test_results.csv")
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(14, 9.4), height_ratios=[1.25, 1])
    fig.suptitle("KNN performance by input representation (%)", fontsize=17, x=0.015, ha="left")
    draw_curves(top, summary, tests)
    draw_table(bottom, summary, tests)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(K.BASE_DIR / "figure4_input_performance.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    figure_examples()
    figure_performance()
