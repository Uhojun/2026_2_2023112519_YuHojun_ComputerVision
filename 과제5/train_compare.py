"""
Extra (not required by the assignment): train the same linear classifier with
the multiclass SVM loss for a short time, then compare its weights with the
random ones.

Raw pixels minus the mean image, W ~ 0.0001*N(0,1) (seed 42), b = 0,
minibatch SGD: batch 200, lr 1e-7, L2 reg 2.5e4, 1500 iterations.
Output: results/random_vs_trained_weights.png
"""
import os, pickle, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "cifar-10-batches-py")
OUT = os.path.join(HERE, "results", "random_vs_trained_weights.png")
def up(f):
    with open(os.path.join(D, f), "rb") as h: return pickle.load(h, encoding="bytes")
Xtr = np.concatenate([up(f"data_batch_{i}")[b"data"] for i in range(1, 6)]).astype(np.float32)
ytr = np.concatenate([up(f"data_batch_{i}")[b"labels"] for i in range(1, 6)])
te = up("test_batch"); Xte = te[b"data"].astype(np.float32); yte = np.array(te[b"labels"])
names = [c.decode() for c in up("batches.meta")[b"label_names"]]
mean = Xtr.mean(0, dtype=np.float64).astype(np.float32); Xtr -= mean; Xte -= mean
rng = np.random.default_rng(42)
W = 0.0001 * rng.standard_normal((3072, 10)); b = np.zeros(10); W0 = W.copy()
lr, reg, B = 1e-7, 2.5e4, 200
def svm(W, b, X, y):
    N = len(X); s = X @ W + b; c = s[np.arange(N), y][:, None]
    m = np.maximum(0, s - c + 1); m[np.arange(N), y] = 0
    loss = m.sum() / N + reg * (W * W).sum()
    g = (m > 0).astype(float); g[np.arange(N), y] = -g.sum(1)
    return loss, X.T @ g / N + 2 * reg * W, g.sum(0) / N
for it in range(1500):
    i = rng.integers(0, len(Xtr), B)
    loss, dW, db = svm(W, b, Xtr[i].astype(np.float64), ytr[i]); W -= lr * dW; b -= lr * db
    if it % 300 == 0 or it == 1499: print(f"iter {it:4d} loss {loss:.3f}")
acc0 = ((Xte @ W0).argmax(1) == yte).mean(); acc = ((Xte @ W + b).argmax(1) == yte).mean()
print(f"test acc random W: {acc0:.3f}  trained W: {acc:.3f}")
fig, ax = plt.subplots(2, 10, figsize=(16, 4.2))
for row, (M, t) in enumerate([(W0, "random"), (W, "trained")]):
    for c in range(10):
        w = M[:, c].reshape(3, 32, 32).transpose(1, 2, 0)
        a = ax[row, c]; a.imshow((w - w.min()) / (w.max() - w.min())); a.axis("off")
        a.set_title(f"{names[c]}" if row == 0 else "", fontsize=10)
    ax[row, 0].text(-6, 16, t, rotation=90, va="center", ha="right", fontsize=12, weight="bold")
fig.suptitle(f"Linear SVM weights: random (top, test acc {acc0:.1%}) vs trained 1500 SGD steps (bottom, test acc {acc:.1%})")
fig.tight_layout(); fig.savefig(OUT, dpi=120); print("saved", OUT)
