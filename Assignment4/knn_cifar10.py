"""CIFAR-10 K-Nearest Neighbors: implementation and evaluation.

KNN, the evaluation metrics and the 5-fold split are implemented in NumPy (no scikit-learn).

Experiment steps
  1. Run 5-fold cross-validation on a class-balanced training sample (20,000 images
     by default) to evaluate every combination of distance metric and k.
  2. For each distance metric, pick the k with the highest cross-validation macro F1.
  3. Evaluate the 10,000 test images using all 50,000 training images.
     Test results are not used to choose k.

Examples
  python knn_cifar10.py --threads 16   # default settings (about 1 hour)
  python knn_cifar10.py --n-cv 50000   # cross-validate on all training data (several hours)

After an interruption, a rerun reuses the neighbor results saved in results/cache.
"""
import argparse
import csv
import hashlib
import pickle
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
ARCHIVE = BASE_DIR / "cifar-10-python.tar.gz"
ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"  # value from the official CIFAR-10 page
RESULT_DIR = BASE_DIR / "results"
CACHE_DIR = RESULT_DIR / "cache"

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
N_CLASSES = len(CLASS_NAMES)
METRICS = ["l2", "cosine", "l1", "linf"]
TEST_METRICS = METRICS
K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51, 101]
N_FOLDS = 5
SEED = 42

BATCH_SIZE = 256   # query images processed at once
TRAIN_BLOCK = 512  # training-data block size for L1/L∞ (saves memory and cache)
DOT_BLOCK = 4096   # training-data block size for L2/cosine


# ------------------------------------------------------------------ Data ----
def load_cifar10(path=ARCHIVE):
    """Read directly from the archive and return (X_train, y_train, X_test, y_test).

    X is an (N, 3072) uint8 array. Scaling pixels to [0, 1] does not change the neighbor
    ranking for L1, L2, L∞ or cosine, so the 0-255 integers are kept to save memory.
    The checksum confirms this is the official file before the pickle is opened.
    """
    digest = hashlib.md5(Path(path).read_bytes()).hexdigest()
    if digest != ARCHIVE_MD5:
        raise ValueError(f"Checksum mismatch: {digest}")

    parts = {}
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if name.startswith("data_batch") or name == "test_batch":
                batch = pickle.load(tar.extractfile(member), encoding="bytes")
                parts[name] = (batch[b"data"], np.array(batch[b"labels"], dtype=np.uint8))

    train_names = [f"data_batch_{i}" for i in range(1, 6)]
    X_train = np.concatenate([parts[n][0] for n in train_names])
    y_train = np.concatenate([parts[n][1] for n in train_names])
    X_test, y_test = parts["test_batch"]
    return X_train, y_train, X_test, y_test


def stratified_subset(y, n, rng):
    """Return indices with the same number of images from each class."""
    per_class = n // N_CLASSES
    picked = [rng.permutation(np.flatnonzero(y == c))[:per_class] for c in range(N_CLASSES)]
    return rng.permutation(np.concatenate(picked))


def stratified_kfold(y, n_folds, rng):
    """Return n_folds validation index arrays with equal class proportions."""
    folds = [[] for _ in range(n_folds)]
    for c in range(N_CLASSES):
        indices = rng.permutation(np.flatnonzero(y == c))
        for fold, chunk in zip(folds, np.array_split(indices, n_folds)):
            fold.append(chunk)
    return [np.sort(np.concatenate(fold)) for fold in folds]


# ------------------------------------------------------------ Distances ----
def _absolute_differences(queries, X_train, reduce, dtype, n_threads):
    """Compute |x - q| per query and aggregate it with reduce (shared by L1 and L∞)."""
    def one_query(q):
        q = q.astype(np.int16)
        out = np.empty(len(X_train), dtype=dtype)
        for s in range(0, len(X_train), TRAIN_BLOCK):
            block = X_train[s:s + TRAIN_BLOCK].astype(np.int16)
            out[s:s + TRAIN_BLOCK] = reduce(np.abs(block - q))
        return out

    with ThreadPoolExecutor(n_threads) as pool:
        return np.stack(list(pool.map(one_query, queries)))


def l1_distance(queries, X_train, n_threads):
    """Manhattan distance: sum_i |q_i - x_i|"""
    return _absolute_differences(
        queries, X_train, lambda d: d.sum(axis=1, dtype=np.int32), np.int32, n_threads)


def linf_distance(queries, X_train, n_threads):
    """Chebyshev distance: max_i |q_i - x_i|"""
    return _absolute_differences(
        queries, X_train, lambda d: d.max(axis=1), np.int16, n_threads)


def l2_distance(queries, X_train, n_threads=None):
    """Squared Euclidean distance: |q|^2 + |x|^2 - 2 q·x

    The square root does not change the neighbor ranking, so it is omitted. Integer pixels
    are computed in float64, so there is no rounding error.
    """
    q = queries.astype(np.float64)
    q_sq = (q * q).sum(axis=1, keepdims=True)
    out = np.empty((len(q), len(X_train)), dtype=np.float64)
    for s in range(0, len(X_train), DOT_BLOCK):
        x = X_train[s:s + DOT_BLOCK].astype(np.float64)
        out[:, s:s + DOT_BLOCK] = q_sq + (x * x).sum(axis=1) - 2.0 * (q @ x.T)
    return out


def cosine_distance(queries, X_train, n_threads=None):
    """Cosine distance: 1 - q·x / (|q||x|)"""
    q = queries.astype(np.float64)
    q_norm = np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    out = np.empty((len(q), len(X_train)), dtype=np.float64)
    for s in range(0, len(X_train), DOT_BLOCK):
        x = X_train[s:s + DOT_BLOCK].astype(np.float64)
        x_norm = np.maximum(np.linalg.norm(x, axis=1), 1e-12)
        out[:, s:s + DOT_BLOCK] = 1.0 - (q @ x.T) / (q_norm * x_norm)
    return out


DISTANCES = {"l1": l1_distance, "l2": l2_distance,
             "linf": linf_distance, "cosine": cosine_distance}


# ------------------------------------------------------------------- KNN ----
def majority_vote(neighbor_labels, k):
    """Take a majority vote over the first k of the neighbor labels, sorted nearest first.

    On a tie, pick the class of the nearest neighbor among the tied classes.
    """
    top_k = neighbor_labels[:, :k]
    n = len(top_k)
    rows = np.arange(n)
    counts = np.zeros((n, N_CLASSES), dtype=np.int32)
    first_rank = np.full((n, N_CLASSES), k, dtype=np.int32)
    for rank in range(k - 1, -1, -1):  # overwriting in reverse leaves each class's lowest rank
        labels = top_k[:, rank]
        counts[rows, labels] += 1
        first_rank[rows, labels] = rank
    return np.argmax(counts * (k + 1) - first_rank, axis=1)


class KNNClassifier:
    """KNN that stores the training data and computes all distances at prediction time."""

    def __init__(self, metric="l2", n_threads=12):
        self.distance = DISTANCES[metric]
        self.n_threads = n_threads

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def neighbor_labels(self, X, max_k):
        """Return the labels of the max_k nearest neighbors of each query, nearest first."""
        result = np.empty((len(X), max_k), dtype=np.uint8)
        for s in range(0, len(X), BATCH_SIZE):
            d = self.distance(X[s:s + BATCH_SIZE], self.X_train, self.n_threads)
            nearest = np.argpartition(d, max_k - 1, axis=1)[:, :max_k]
            order = np.argsort(np.take_along_axis(d, nearest, axis=1), axis=1, kind="stable")
            result[s:s + BATCH_SIZE] = self.y_train[np.take_along_axis(nearest, order, axis=1)]
        return result

    def predict(self, X, k):
        return majority_vote(self.neighbor_labels(X, k), k)


# ------------------------------------------------------ Evaluation metrics ----
def confusion_matrix(y_true, y_pred):
    """Rows are true classes; columns are predicted classes."""
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    np.add.at(cm, (y_true.astype(np.int64), y_pred.astype(np.int64)), 1)
    return cm


def per_class_scores(cm):
    """Per-class (precision, recall, f1). A value is 0 when its denominator is 0."""
    tp = np.diag(cm).astype(np.float64)
    predicted = cm.sum(axis=0)
    actual = cm.sum(axis=1)
    precision = np.divide(tp, predicted, out=np.zeros(N_CLASSES), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros(N_CLASSES), where=actual > 0)
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros(N_CLASSES), where=total > 0)
    return precision, recall, f1


def evaluate(y_true, y_pred):
    """Return accuracy and macro-averaged precision, recall and F1."""
    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1 = per_class_scores(cm)
    return {"accuracy": np.trace(cm) / cm.sum(), "precision": precision.mean(),
            "recall": recall.mean(), "f1": f1.mean()}


# ------------------------------------------------------------- Experiment ----
def cached_neighbors(name, X_train, y_train, X_query, metric, max_k, n_threads):
    """Compute and save the neighbor labels, or load them if they already exist.

    Finding the neighbors up to the largest k once lets every k be evaluated without more distance calculations.
    """
    path = CACHE_DIR / f"{name}.npz"
    if path.exists():
        saved = np.load(path)
        return saved["labels"], float(saved["seconds"])
    start = time.perf_counter()
    model = KNNClassifier(metric, n_threads).fit(X_train, y_train)
    labels = model.neighbor_labels(X_query, max_k)
    seconds = time.perf_counter() - start
    np.savez_compressed(path, labels=labels, seconds=seconds)
    return labels, seconds


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_cross_validation(X, y, n_threads):
    """Evaluate every combination of distance metric and k with 5-fold cross-validation."""
    rng = np.random.default_rng(SEED)
    subset = stratified_subset(y, args.n_cv, rng)
    X, y = X[subset], y[subset]
    folds = stratified_kfold(y, N_FOLDS, rng)

    fold_rows = []
    for metric in METRICS:
        for fold, val_idx in enumerate(folds):
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[val_idx] = False
            labels, seconds = cached_neighbors(
                f"cv{args.n_cv}_{metric}_fold{fold}", X[train_mask], y[train_mask],
                X[val_idx], metric, max(K_VALUES), n_threads)
            for k in K_VALUES:
                scores = evaluate(y[val_idx], majority_vote(labels, k))
                fold_rows.append({"metric": metric, "k": k, "fold": fold,
                                  "seconds": seconds, **scores})
            best = max(r["accuracy"] for r in fold_rows[-len(K_VALUES):])
            print(f"[CV] {metric} fold {fold}: {seconds:.0f}s, best acc {best:.4f}", flush=True)

    summary = []
    for metric in METRICS:
        for k in K_VALUES:
            rows = [r for r in fold_rows if r["metric"] == metric and r["k"] == k]
            row = {"metric": metric, "k": k}
            for name in ["accuracy", "precision", "recall", "f1"]:
                values = [r[name] for r in rows]
                row[f"{name}_mean"] = np.mean(values)
                row[f"{name}_std"] = np.std(values, ddof=1)
            summary.append(row)

    write_csv(RESULT_DIR / "cv_per_fold.csv", fold_rows)
    write_csv(RESULT_DIR / "cv_summary.csv", summary)
    return summary


def select_k(summary):
    """For each distance metric, pick the k with the highest mean macro F1. Ties go to the smaller k."""
    selected = {}
    for metric in METRICS:
        rows = [r for r in summary if r["metric"] == metric]
        selected[metric] = max(rows, key=lambda r: (round(r["f1_mean"], 10), -r["k"]))
    return selected


def run_test(X_train, y_train, X_test, y_test, selected, n_threads):
    """Evaluate the test set using all of the training data."""
    best_metric = max(TEST_METRICS, key=lambda m: selected[m]["f1_mean"])
    rows = []
    for metric in TEST_METRICS:
        labels, seconds = cached_neighbors(
            f"test_{metric}", X_train, y_train, X_test, metric, max(K_VALUES), n_threads)
        chosen_k = selected[metric]["k"]
        for k in K_VALUES:
            pred = majority_vote(labels, k)
            rows.append({"metric": metric, "k": k, "selected_by_cv": k == chosen_k,
                         "seconds": seconds, **evaluate(y_test, pred)})
            if metric == best_metric and k == chosen_k:
                cm = confusion_matrix(y_test, pred)
                precision, recall, f1 = per_class_scores(cm)
                np.savetxt(RESULT_DIR / "test_confusion_matrix.csv", cm, fmt="%d", delimiter=",")
                write_csv(RESULT_DIR / "test_per_class.csv", [
                    {"class": name, "metric": metric, "k": k, "precision": p, "recall": r, "f1": f}
                    for name, p, r, f in zip(CLASS_NAMES, precision, recall, f1)])
        chosen = next(r for r in rows if r["metric"] == metric and r["k"] == chosen_k)
        print(f"[TEST] {metric} k={chosen_k}: {seconds:.0f}s, acc {chosen['accuracy']:.4f}, "
              f"f1 {chosen['f1']:.4f}", flush=True)
    write_csv(RESULT_DIR / "test_results.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-cv", type=int, default=20000,
                        help="training images used for cross-validation (at most 50000)")
    parser.add_argument("--threads", type=int, default=12, help="threads for L1/L∞ computation")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train, X_test, y_test = load_cifar10()
    print(f"train {X_train.shape}, test {X_test.shape}", flush=True)

    cv_summary = run_cross_validation(X_train, y_train, args.threads)
    best_k = select_k(cv_summary)
    run_test(X_train, y_train, X_test, y_test, best_k, args.threads)
    print("done", flush=True)
