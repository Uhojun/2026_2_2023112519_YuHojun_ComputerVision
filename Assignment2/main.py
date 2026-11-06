import cv2
import numpy as np
import matplotlib.pyplot as plt



# 1. Read Grayscale Image

img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise Exception("Image not found.")


# 2. Define Structuring Elements (SE)


# (1) Cross SE (5x5)

cross = np.array([
    [0,0,1,0,0],
    [0,0,1,0,0],
    [1,1,1,1,1],
    [0,0,1,0,0],
    [0,0,1,0,0]
], dtype=np.uint8)




# (2) Square SE (7x7)


square = np.ones((7,7), dtype=np.uint8)


# (3) Vertical Line SE (7x1)


vertical = np.ones((7,1), dtype=np.uint8)



# (4) Horizontal Line SE (1x7)


horizontal = np.ones((1,7), dtype=np.uint8)

# (5) Diamond SE (7x7)

diamond = np.array([
    [0,0,0,1,0,0,0],
    [0,0,1,1,1,0,0],
    [0,1,1,1,1,1,0],
    [1,1,1,1,1,1,1],
    [0,1,1,1,1,1,0],
    [0,0,1,1,1,0,0],
    [0,0,0,1,0,0,0]
], dtype=np.uint8)




# 3. Apply Dilation


iterations = 2     # Two iterations make the differences easier to see

d_cross = cv2.dilate(img, cross, iterations=iterations)
d_square = cv2.dilate(img, square, iterations=iterations)
d_vertical = cv2.dilate(img, vertical, iterations=iterations)
d_horizontal = cv2.dilate(img, horizontal, iterations=iterations)
d_diamond = cv2.dilate(img, diamond, iterations=iterations)




# 4. Display Results

titles = [
    "Original",
    "Cross SE (5x5)",
    "Square SE (7x7)",
    "Vertical SE (7x1)",
    "Horizontal SE (1x7)",
    "Diamond SE (7x7)"
]

images = [
    img,
    d_cross,
    d_square,
    d_vertical,
    d_horizontal,
    d_diamond
]

kernels = [
    None,
    cross,
    square,
    vertical,
    horizontal,
    diamond
]

plt.figure(figsize=(24,9))

for i in range(len(images)):

    ax = plt.subplot(1,6,i+1)

    ax.imshow(images[i], cmap='gray')
    ax.set_title(titles[i], fontsize=14)
    ax.axis("off")

    if kernels[i] is None:
        se_text = "No Structuring Element"

    else:

        rows = []

        for row in kernels[i]:
            rows.append(" ".join(map(str,row)))

        se_text = "SE\n\n" + "\n".join(rows)

    ax.text(
        0.5,
        -0.20,
        se_text,
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=11,
        family='monospace',
        bbox=dict(
            facecolor='white',
            edgecolor='black',
            boxstyle='round,pad=0.4'
        )
    )

plt.subplots_adjust(bottom=0.35, wspace=0.30)
plt.show()