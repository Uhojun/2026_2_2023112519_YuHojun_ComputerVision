"""Extra experiment: how does KNN performance change with the input representation?

The KNN implementation, evaluation metrics, 20,000-image sample and 5-fold split are the
same as in knn_cifar10.py; only the input changes. The distance is fixed to L1, the best
metric in the main experiment.

  raw  : original color pixels (3,072 dimensions). Reuses the main experiment's cache.
  gray : grayscale pixels (1,024 dimensions)
  he   : per-image histogram equalization of the grayscale image (1,024 dimensions)
  hog  : HOG features of the grayscale image (1,764 dimensions), from scikit-image.

k is chosen per input by mean cross-validation macro F1; results at a fixed k=7 are also
recorded. Test results are not used to choose k.

Run: python input_representation.py --threads 14
Results: results/input_representation/
"""
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from skimage.feature import hog

import knn_cifar10 as K

OUT_DIR = K.RESULT_DIR / "input_representation"
REPRESENTATIONS = ["raw", "gray", "he", "hog"]
SCORES = ["accuracy", "precision", "recall", "f1"]
FIXED_K = 7
HOG_PARAMS = {"orientations": 9, "pixels_per_cell": (4, 4), "cells_per_block": (2, 2),
              "block_norm": "L2-Hys", "feature_vector": True}


# ------------------------------------------------------- Input transforms ----
def to_gray(X):
    """Convert (N, 3072) color to (N, 1024) grayscale. Y = 0.299R + 0.587G + 0.114B"""
    r, g, b = X[:, :1024], X[:, 1024:2048], X[:, 2048:]
    return np.rint(0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)


def equalize_one(image):
    """Histogram equalization of one 8-bit image.

    lut(v) = round((cdf(v) - cdf_min) * 255 / (N - cdf_min))
    If all pixels are equal the denominator is 0, so the image is left unchanged.
    """
    cdf = np.bincount(image, minlength=256).cumsum()
    cdf_min = cdf[cdf > 0][0]
    if cdf_min == image.size:
        return image.copy()
    lut = np.rint((cdf - cdf_min) * 255.0 / (image.size - cdf_min)).clip(0, 255).astype(np.uint8)
    return lut[image]


def to_he(gray):
    return np.stack([equalize_one(image) for image in gray])


def to_hog(gray, n_threads):
    """Extract HOG features from 32×32 grayscale images into an (N, 1764) float32 array."""
    def one(image):
        return hog(image.reshape(32, 32) / 255.0, **HOG_PARAMS).astype(np.float32)

    with ThreadPoolExecutor(n_threads) as pool:
        return np.stack(list(pool.map(one, gray)))


# ------------------------------------------------------------ Distances ----
def l1_distance_float(queries, X_train, n_threads):
    """Manhattan distance for real-valued features (HOG). Same formula as l1_distance for integer pixels."""
    def one_query(q):
        out = np.empty(len(X_train), dtype=np.float32)
        for s in range(0, len(X_train), K.TRAIN_BLOCK):
            out[s:s + K.TRAIN_BLOCK] = np.abs(X_train[s:s + K.TRAIN_BLOCK] - q).sum(axis=1)
        return out

    with ThreadPoolExecutor(n_threads) as pool:
        return np.stack(list(pool.map(one_query, queries)))


K.DISTANCES["l1_float"] = l1_distance_float


def distance_name(representation):
    return "l1_float" if representation == "hog" else "l1"


def cache_name(representation, part):
    """raw reuses the neighbors computed in the main experiment."""
    prefix = "" if representation == "raw" else f"repr_{representation}_"
    return f"{prefix}{part}"


# ------------------------------------------------------------- Experiment ----
def run_cross_validation(features, y, n_threads):
    rng = np.random.default_rng(K.SEED)
    subset = K.stratified_subset(y, 20000, rng)  # uses random numbers in the same order as knn_cifar10.py
    y_cv = y[subset]
    folds = K.stratified_kfold(y_cv, K.N_FOLDS, rng)

    fold_rows = []
    for name in REPRESENTATIONS:
        X_cv = features[name][subset]
        for fold, val_idx in enumerate(folds):
            train_mask = np.ones(len(y_cv), dtype=bool)
            train_mask[val_idx] = False
            labels, seconds = K.cached_neighbors(
                cache_name(name, f"cv20000_l1_fold{fold}"), X_cv[train_mask], y_cv[train_mask],
                X_cv[val_idx], distance_name(name), max(K.K_VALUES), n_threads)
            for k in K.K_VALUES:
                scores = K.evaluate(y_cv[val_idx], K.majority_vote(labels, k))
                fold_rows.append({"representation": name, "k": k, "fold": fold,
                                  "seconds": seconds, **scores})
            best = max(r["accuracy"] for r in fold_rows[-len(K.K_VALUES):])
            print(f"[CV] {name} fold {fold}: {seconds:.0f}s, best acc {best:.4f}", flush=True)

    summary = []
    for name in REPRESENTATIONS:
        for k in K.K_VALUES:
            rows = [r for r in fold_rows if r["representation"] == name and r["k"] == k]
            row = {"representation": name, "k": k}
            for score in SCORES:
                values = [r[score] for r in rows]
                row[f"{score}_mean"] = np.mean(values)
                row[f"{score}_std"] = np.std(values, ddof=1)
            summary.append(row)

    K.write_csv(OUT_DIR / "cv_per_fold.csv", fold_rows)
    K.write_csv(OUT_DIR / "cv_summary.csv", summary)
    return summary


def select_k(summary):
    """For each input, pick the k with the highest mean macro F1. Ties go to the smaller k."""
    return {name: max((r for r in summary if r["representation"] == name),
                      key=lambda r: (round(r["f1_mean"], 10), -r["k"]))["k"]
            for name in REPRESENTATIONS}


def run_test(train_features, y_train, test_features, y_test, selected, n_threads):
    rows, class_rows = [], []
    for name in REPRESENTATIONS:
        labels, seconds = K.cached_neighbors(
            cache_name(name, "test_l1"), train_features[name], y_train, test_features[name],
            distance_name(name), max(K.K_VALUES), n_threads)
        for k in K.K_VALUES:
            pred = K.majority_vote(labels, k)
            rows.append({"representation": name, "k": k, "selected_by_cv": k == selected[name],
                         "seconds": seconds, **K.evaluate(y_test, pred)})
            if k == selected[name]:
                precision, recall, f1 = K.per_class_scores(K.confusion_matrix(y_test, pred))
                class_rows += [{"representation": name, "k": k, "class": c,
                                "precision": p, "recall": r, "f1": f}
                               for c, p, r, f in zip(K.CLASS_NAMES, precision, recall, f1)]
        chosen = next(r for r in rows if r["representation"] == name and r["k"] == selected[name])
        print(f"[TEST] {name} k={selected[name]}: {seconds:.0f}s, acc {chosen['accuracy']:.4f}, "
              f"f1 {chosen['f1']:.4f}", flush=True)
    K.write_csv(OUT_DIR / "test_results.csv", rows)
    K.write_csv(OUT_DIR / "test_per_class.csv", class_rows)


def build_features(X, n_threads):
    gray = to_gray(X)
    return {"raw": X, "gray": gray, "he": to_he(gray), "hog": to_hog(gray, n_threads)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, default=14, help="threads for L1 computation")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    K.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train, X_test, y_test = K.load_cifar10()
    train_features = build_features(X_train, args.threads)
    test_features = build_features(X_test, args.threads)
    print({name: value.shape for name, value in train_features.items()}, flush=True)

    cv_summary = run_cross_validation(train_features, y_train, args.threads)
    best_k = select_k(cv_summary)
    print("selected k:", best_k, flush=True)
    run_test(train_features, y_train, test_features, y_test, best_k, args.threads)
    print("done", flush=True)
