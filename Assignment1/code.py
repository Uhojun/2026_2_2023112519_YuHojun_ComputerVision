

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import rank
from skimage.morphology import disk


# =========================================================
# 1. Load the image
# =========================================================
image_path = "input images.jpg"

img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(
        f"Image not found: {image_path}"
    )


# =========================================================
# 2. Convert to grayscale
# =========================================================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# =========================================================
# 3. HE (Histogram Equalization)
# =========================================================
he_img = cv2.equalizeHist(gray)


# =========================================================
# 4. AHE (Adaptive Histogram Equalization)
#    Histogram equalization over each local neighborhood
# =========================================================
ahe_img = rank.equalize(
    gray,
    footprint=disk(15)
)


# =========================================================
# 5. MSE function
# =========================================================
def calculate_mse(original, processed):
    original = original.astype(np.float64)
    processed = processed.astype(np.float64)

    mse = np.mean((original - processed) ** 2)

    return mse


# =========================================================
# 6. PSNR function
# =========================================================
def calculate_psnr(original, processed):
    mse = calculate_mse(original, processed)

    if mse == 0:
        return float("inf")

    psnr = 10 * np.log10((255.0 ** 2) / mse)

    return psnr


# =========================================================
# 7. Compute MSE / PSNR
# =========================================================

# Original vs Original
original_mse = calculate_mse(gray, gray)
original_psnr = calculate_psnr(gray, gray)

# Original vs HE
he_mse = calculate_mse(gray, he_img)
he_psnr = calculate_psnr(gray, he_img)

# Original vs AHE
ahe_mse = calculate_mse(gray, ahe_img)
ahe_psnr = calculate_psnr(gray, ahe_img)


# =========================================================
# 8. Compute histograms
# =========================================================
hist_original = cv2.calcHist(
    [gray],
    [0],
    None,
    [256],
    [0, 256]
)

hist_he = cv2.calcHist(
    [he_img],
    [0],
    None,
    [256],
    [0, 256]
)

hist_ahe = cv2.calcHist(
    [ahe_img],
    [0],
    None,
    [256],
    [0, 256]
)


# =========================================================
# 9. Plot the results in a 3 x 2 grid
# =========================================================
fig, axes = plt.subplots(
    3,
    2,
    figsize=(12, 17)
)


# =========================================================
# Row 1 - Original
# =========================================================

# Original Image
axes[0, 0].imshow(
    gray,
    cmap="gray",
    vmin=0,
    vmax=255
)

axes[0, 0].set_title(
    f"Original\nPSNR: {original_psnr} dB",
    fontsize=15
)

axes[0, 0].axis("off")


# Original Histogram
axes[0, 1].bar(
    np.arange(256),
    hist_original.ravel(),
    width=1
)

axes[0, 1].set_title(
    "Original Histogram",
    fontsize=15
)

axes[0, 1].set_xlim([0, 255])
axes[0, 1].set_xlabel("Pixel Intensity")
axes[0, 1].set_ylabel("Frequency")


# =========================================================
# Row 2 - HE
# =========================================================

# HE Image
axes[1, 0].imshow(
    he_img,
    cmap="gray",
    vmin=0,
    vmax=255
)

axes[1, 0].set_title(
    f"HE\n"
    f"MSE: {he_mse:.2f} | "
    f"PSNR: {he_psnr:.2f} dB",
    fontsize=15
)

axes[1, 0].axis("off")


# HE Histogram
axes[1, 1].bar(
    np.arange(256),
    hist_he.ravel(),
    width=1
)

axes[1, 1].set_title(
    "HE Histogram",
    fontsize=15
)

axes[1, 1].set_xlim([0, 255])
axes[1, 1].set_xlabel("Pixel Intensity")
axes[1, 1].set_ylabel("Frequency")


# =========================================================
# Row 3 - AHE
# =========================================================

# AHE Image
axes[2, 0].imshow(
    ahe_img,
    cmap="gray",
    vmin=0,
    vmax=255
)

axes[2, 0].set_title(
    f"AHE\n"
    f"MSE: {ahe_mse:.2f} | "
    f"PSNR: {ahe_psnr:.2f} dB",
    fontsize=15
)

axes[2, 0].axis("off")


# AHE Histogram
axes[2, 1].bar(
    np.arange(256),
    hist_ahe.ravel(),
    width=1
)

axes[2, 1].set_title(
    "AHE Histogram",
    fontsize=15
)

axes[2, 1].set_xlim([0, 255])
axes[2, 1].set_xlabel("Pixel Intensity")
axes[2, 1].set_ylabel("Frequency")


# =========================================================
# 10. Tidy the layout
# =========================================================
plt.tight_layout()


# =========================================================
# 11. Save the figure
# =========================================================
plt.savefig(
    "he_ahe_result.png",
    dpi=300,
    bbox_inches="tight"
)


# =========================================================
# 12. Show the figure
# =========================================================
plt.show()


# =========================================================
# 13. Print the results to the terminal
# =========================================================
print("====================================")
print("          IMAGE QUALITY")
print("====================================")

print("\n[Original]")
print(f"MSE  : {original_mse:.2f}")
print(f"PSNR : {original_psnr} dB")

print("\n[HE]")
print(f"MSE  : {he_mse:.2f}")
print(f"PSNR : {he_psnr:.2f} dB")

print("\n[AHE]")
print(f"MSE  : {ahe_mse:.2f}")
print(f"PSNR : {ahe_psnr:.2f} dB")