"""
CIFAR-10 Linear Classification with randomly initialized weights (no training).

f(x) = Wx + b
  - x : flattened image, shape (3072,)  (32*32*3)
  - W : weight matrix, shape (10, 3072), small random values
  - b : bias vector,  shape (10,), zeros

Covers: score computation, prediction, multiclass SVM hinge loss,
Softmax probabilities, correct / misclassified examples,
and visualization of the class weight vectors as 32x32x3 images.
"""
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "cifar-10-batches-py")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
SEED = 42
NUM_EXAMPLES = 10          # number of example images to analyse
WEIGHT_SCALE = 0.0001      # "small random values"


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def load_cifar10_test():
    batch = unpickle(os.path.join(DATA_DIR, "test_batch"))
    meta = unpickle(os.path.join(DATA_DIR, "batches.meta"))
    X = batch[b"data"].astype(np.float64)          # (10000, 3072) in [0, 255]
    y = np.array(batch[b"labels"])                 # (10000,)
    class_names = [c.decode() for c in meta[b"label_names"]]
    return X, y, class_names


def to_image(row):
    """3072-vector (R plane, G plane, B plane) -> 32x32x3 uint8 image."""
    return row.reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)


# ----------------------------------------------------------------------------
# Model pieces
# ----------------------------------------------------------------------------
def init_params(num_classes=10, dim=3072, scale=WEIGHT_SCALE, seed=SEED):
    rng = np.random.default_rng(seed)
    W = scale * rng.standard_normal((num_classes, dim))
    b = np.zeros(num_classes)
    return W, b


def linear_scores(X, W, b):
    """f(x) = Wx + b for a batch. X: (N, D) -> scores (N, C)."""
    return X @ W.T + b


def svm_hinge_loss(scores, y, delta=1.0):
    """
    Multiclass SVM loss:  L_i = sum_{j != y_i} max(0, s_j - s_{y_i} + delta)
    Returns per-example losses (N,) and the average loss.
    """
    N = scores.shape[0]
    correct = scores[np.arange(N), y][:, None]
    margins = np.maximum(0, scores - correct + delta)
    margins[np.arange(N), y] = 0                  # skip j == y_i
    losses = margins.sum(axis=1)
    return losses, losses.mean()


def softmax(scores):
    """Numerically stable softmax over classes."""
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.set_printoptions(precision=4, suppress=True, linewidth=160)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    X_raw, y_all, class_names = load_cifar10_test()

    # Preprocessing: scale pixels to [0, 1] (no training, so keep it simple)
    X_all = X_raw / 255.0

    # 1) Initialise W (small random) and b (zeros) with a fixed seed
    W, b = init_params()
    log(f"Random seed: {SEED}")
    log(f"W shape: {W.shape}, mean={W.mean():.2e}, std={W.std():.2e}")
    log(f"b shape: {b.shape}, values={b}")
    log()

    # Choose example images reproducibly: the first image of each class
    idx = np.array([np.where(y_all == c)[0][0] for c in range(10)])[:NUM_EXAMPLES]
    X, y = X_all[idx], y_all[idx]

    # 2) Scores and predictions
    scores = linear_scores(X, W, b)
    preds = scores.argmax(axis=1)

    header = "idx    true        pred        | " + " ".join(f"{c[:6]:>8}" for c in class_names)
    log("=== Class scores f(x) = Wx + b ===")
    log(header)
    for i in range(len(idx)):
        row = " ".join(f"{s:8.4f}" for s in scores[i])
        log(f"{idx[i]:<6} {class_names[y[i]]:<11} {class_names[preds[i]]:<11} | {row}")
    log()

    # 3) SVM hinge loss
    losses, avg_loss = svm_hinge_loss(scores, y)
    log("=== Multiclass SVM hinge loss (delta = 1) ===")
    for i in range(len(idx)):
        log(f"image {idx[i]:<6} ({class_names[y[i]]:<10}) loss = {losses[i]:.4f}")
    log(f"Average SVM loss over {len(idx)} examples = {avg_loss:.4f}")
    log(f"(Expected approx. C-1 = 9 when all scores are ~0)")
    log()

    # 4) Softmax
    probs = softmax(scores)
    log("=== Softmax: scores vs probabilities ===")
    for i in range(len(idx)):
        log(f"image {idx[i]} ({class_names[y[i]]})")
        log(f"  scores: {scores[i]}")
        log(f"  probs : {probs[i]}")
        log(f"  sum(probs) = {probs[i].sum():.10f}  -> equals 1: {np.isclose(probs[i].sum(), 1.0)}")
    ce = -np.log(probs[np.arange(len(idx)), y])
    log(f"All rows sum to 1: {np.allclose(probs.sum(axis=1), 1.0)}")
    log(f"(Reference) average softmax cross-entropy loss = {ce.mean():.4f}, ln(10) = {np.log(10):.4f}")
    log()

    # 5) Correct vs misclassified
    correct_mask = preds == y
    log("=== Correct / misclassified among the selected examples ===")
    log(f"Correct     : {[int(i) for i in idx[correct_mask]]}")
    log(f"Misclassified: {[int(i) for i in idx[~correct_mask]]}")
    log()

    # Score bar chart + image per example
    fig, axes = plt.subplots(len(idx), 2, figsize=(11, 2.0 * len(idx)),
                             gridspec_kw={"width_ratios": [1, 4]})
    for i in range(len(idx)):
        axes[i, 0].imshow(to_image(X_raw[idx[i]]))
        axes[i, 0].axis("off")
        ok = "O" if correct_mask[i] else "X"
        axes[i, 0].set_title(f"#{idx[i]} [{ok}]\ntrue: {class_names[y[i]]}\npred: {class_names[preds[i]]}",
                             fontsize=8)
        colors = ["tab:gray"] * 10
        colors[y[i]] = "tab:green"
        if not correct_mask[i]:
            colors[preds[i]] = "tab:red"
        axes[i, 1].bar(class_names, scores[i], color=colors)
        axes[i, 1].axhline(0, color="black", lw=0.5)
        axes[i, 1].tick_params(axis="x", labelsize=7)
        axes[i, 1].set_ylabel("score", fontsize=7)
    fig.suptitle("Class scores (green = true class, red = wrong prediction)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "scores_per_example.png"), dpi=120)
    plt.close(fig)

    # Scores vs Softmax probabilities for the first 3 examples
    # All probabilities are ~0.1, so the probability axis is zoomed in around
    # 1/10 and every bar is labelled; otherwise the bars look identical.
    lo = probs[:3].min() - 0.0003
    hi = probs[:3].max() + 0.0004
    fig, axes = plt.subplots(3, 2, figsize=(14, 9))
    for k in range(3):
        colors = ["#9db4d0"] * 10
        colors[y[k]] = "tab:green"
        if preds[k] != y[k]:
            colors[preds[k]] = "tab:red"
        else:
            colors[preds[k]] = "tab:green"
        ax_s, ax_p = axes[k]
        ax_s.bar(class_names, scores[k], color=colors)
        ax_s.axhline(0, color="black", lw=0.6)
        for c in range(10):
            ax_s.text(c, scores[k, c], f"{scores[k, c]:.4f}", ha="center",
                      va="bottom" if scores[k, c] >= 0 else "top", fontsize=7)
        ax_s.set_ylim(scores[:3].min() * 1.35, scores[:3].max() * 1.25)
        ax_s.set_title(f"#{idx[k]} (true: {class_names[y[k]]}) raw scores", fontsize=10)

        ax_p.bar(class_names, probs[k], color=colors, bottom=0)
        ax_p.axhline(0.1, color="black", ls="--", lw=0.8)
        for c in range(10):
            ax_p.text(c, probs[k, c] + 0.00003, f"{probs[k, c]:.4f}", ha="center", va="bottom", fontsize=7)
        ax_p.set_ylim(lo, hi)
        ax_p.set_title(f"softmax probabilities (sum = {probs[k].sum():.4f})  "
                       f"[y-axis zoomed: {lo:.4f}–{hi:.4f}, dashed = 1/10]", fontsize=10)
        for a in (ax_s, ax_p):
            a.tick_params(axis="x", labelsize=8)
    fig.suptitle("Raw scores (left) vs Softmax probabilities (right)  "
                 "green = true class, red = wrong prediction", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "scores_vs_softmax.png"), dpi=120)
    plt.close(fig)

    # Correct / misclassified gallery (from the selected examples; if none is
    # correct, look further into the test set so both cases can be shown)
    corr_ids = list(idx[correct_mask])
    wrong_ids = list(idx[~correct_mask])
    if len(corr_ids) == 0:
        all_pred = linear_scores(X_all[:200], W, b).argmax(axis=1)
        corr_ids = list(np.where(all_pred == y_all[:200])[0][:3])
        log(f"No correct example among selected; extra correct ones from first 200 test images: {corr_ids}")
    show_c, show_w = corr_ids[:4], wrong_ids[:4]
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.5))
    for row, (ids, title) in enumerate([(show_c, "Correct"), (show_w, "Misclassified")]):
        for j in range(4):
            ax = axes[row, j]
            ax.axis("off")
            if j < len(ids):
                n = ids[j]
                p = linear_scores(X_all[n:n + 1], W, b).argmax()
                pr = softmax(linear_scores(X_all[n:n + 1], W, b))[0]
                ax.imshow(to_image(X_raw[n]))
                ax.set_title(f"{title} #{n}\ntrue: {class_names[y_all[n]]}\npred: {class_names[p]} ({pr[p]:.3f})",
                             fontsize=8, color="green" if row == 0 else "red")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "correct_vs_misclassified.png"), dpi=120)
    plt.close(fig)

    # 6) Weight visualisation as 32x32x3 images
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.5))
    for c in range(10):
        w = W[c].reshape(3, 32, 32).transpose(1, 2, 0)
        w_img = (w - w.min()) / (w.max() - w.min())   # min-max rescale to [0,1]
        ax = axes[c // 5, c % 5]
        ax.imshow(w_img)
        ax.set_title(class_names[c])
        ax.axis("off")
    fig.suptitle("Randomly initialised class weight vectors W[c] reshaped to 32x32x3")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "weights_visualization.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(OUT_DIR, "results.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved figures and results.txt to {OUT_DIR}")


if __name__ == "__main__":
    main()
