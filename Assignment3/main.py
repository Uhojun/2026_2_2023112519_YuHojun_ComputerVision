from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# 1. Settings
# ==================================================
# Put the two images in the same folder as main.py.
BASE_DIR = Path(__file__).resolve().parent
IMAGE_FILES = ["image.jpg", "image1.jpg"]
OUTPUT_DIR = BASE_DIR / "results"

# As in Assignment 2, each basic operation is applied twice.
# Change this to 1 if the effect is too strong.
ITERATIONS = 2
SHOW_WINDOWS = True  # False: save the results without opening windows


# ==================================================
# 2. Structuring elements (SE): same as Assignment 2
# ==================================================
KERNELS = {
    "Cross SE (5x5)": np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8),

    "Square SE (7x7)": np.ones((7, 7), dtype=np.uint8),
    "Vertical SE (7x1)": np.ones((7, 1), dtype=np.uint8),
    "Horizontal SE (1x7)": np.ones((1, 7), dtype=np.uint8),

    "Diamond SE (7x7)": np.array([
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8),
}


# ==================================================
# 3. Read a grayscale image
# ==================================================
def read_grayscale(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    # Decode from bytes so that paths with non-ASCII characters also work.
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"Image file is empty: {path}")

    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


# ==================================================
# 4. Apply erosion / opening / closing
# ==================================================
def apply_morphology(image: np.ndarray) -> dict:
    results = {"Erosion": [], "Opening": [], "Closing": []}

    for kernel in KERNELS.values():
        # Erosion: original -> erosion
        erosion = cv2.erode(image, kernel, iterations=ITERATIONS)

        # Opening: original -> erosion -> dilation
        # The erosion result above is the input to the dilation.
        opening = cv2.dilate(erosion, kernel, iterations=ITERATIONS)

        # Closing: original -> dilation -> erosion
        # Starts again from the original, not from the opening result.
        dilation = cv2.dilate(image, kernel, iterations=ITERATIONS)
        closing = cv2.erode(dilation, kernel, iterations=ITERATIONS)

        results["Erosion"].append(erosion)
        results["Opening"].append(opening)
        results["Closing"].append(closing)

    return results


# ==================================================
# 5. Show and save the original, the result per SE, and each SE array
# ==================================================
def plot_comparison(original: np.ndarray, processed: list,
                    operation: str, image_name: str) -> None:
    titles = ["Original"] + list(KERNELS.keys())
    images = [original] + processed
    kernels = [None] + list(KERNELS.values())

    # Top: images / bottom: the 0/1 array of each SE
    fig = plt.figure(figsize=(24, 9))
    grid = fig.add_gridspec(
        2, 6, height_ratios=[3, 1.5], hspace=0.08, wspace=0.25
    )
    fig.suptitle(
        f"{image_name} - {operation} (iterations={ITERATIONS})",
        fontsize=20
    )

    for i, (title, image, kernel) in enumerate(zip(titles, images, kernels)):
        ax = fig.add_subplot(grid[0, i])
        # All results are compared on the same intensity range (0-255).
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=14)
        ax.axis("off")

        se_ax = fig.add_subplot(grid[1, i])
        se_ax.axis("off")

        if kernel is None:
            se_text = "No Structuring Element"
        else:
            rows = [" ".join(map(str, row)) for row in kernel]
            se_text = "SE\n\n" + "\n".join(rows)

        se_ax.text(
            0.5, 0.95, se_text,
            transform=se_ax.transAxes,
            ha="center", va="top",
            fontsize=12, family="monospace",
            bbox=dict(
                facecolor="white", edgecolor="black",
                boxstyle="round,pad=0.4"
            )
        )

    fig.subplots_adjust(top=0.88, bottom=0.04, left=0.03, right=0.99)
    output_path = OUTPUT_DIR / f"{Path(image_name).stem}_{operation.lower()}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")

    if not SHOW_WINDOWS:
        plt.close(fig)


# ==================================================
# 6. Run
# ==================================================
def main() -> None:
    if not isinstance(ITERATIONS, int) or ITERATIONS < 1:
        raise ValueError("ITERATIONS must be an integer of at least 1.")

    # Check and read all the required images first.
    inputs = [
        (name, read_grayscale(BASE_DIR / name)) for name in IMAGE_FILES
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for image_name, image in inputs:
        results = apply_morphology(image)
        for operation, processed in results.items():
            plot_comparison(image, processed, operation, image_name)

    if SHOW_WINDOWS:
        plt.show()


if __name__ == "__main__":
    main()