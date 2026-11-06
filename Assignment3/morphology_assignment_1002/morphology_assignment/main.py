"""Morphology assignment: Dilation / Erosion / Opening / Closing implemented from scratch.

Run: python main.py
Compare all five structuring elements: python main.py --all-se
Use another image: python main.py --input "my_image.png"

Required packages: numpy, Pillow
The morphology functions of OpenCV and SciPy are not used.
Binary and grayscale images are both handled, with no thresholding or resizing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------- 1. Morphology operations implemented from scratch ----------
def _check(image: np.ndarray, se: np.ndarray) -> np.ndarray:
    """The input is uint8 grayscale; the SE is an odd-sized 0/1 array whose center is 1."""
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        raise ValueError("image must be a nonempty 2-D NumPy array.")
    if image.dtype != np.uint8:
        raise TypeError("image must have dtype=np.uint8 (0..255).")
    se = np.asarray(se)
    if se.ndim != 2 or se.size == 0 or any(n % 2 == 0 for n in se.shape):
        raise ValueError("SE must be a nonempty 2-D array with odd dimensions.")
    if not np.all((se == 0) | (se == 1)):
        raise ValueError("SE must contain only 0 and 1.")
    if se[se.shape[0] // 2, se.shape[1] // 2] != 1:
        raise ValueError("The center of the SE must be 1.")
    return se.astype(bool)


def _extreme_filter(image: np.ndarray, se: np.ndarray, *, dilate: bool,
                    iterations: int = 1) -> np.ndarray:
    """Find the max/min by directly comparing the values at the positions where the SE is 1.

    The whole image is compared at once for each SE offset, which is faster than
    a double loop over pixels. np.maximum/minimum compare two arrays and are not
    morphology functions. The input array is not overwritten.
    """
    mask = _check(image, se)
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)):
        raise TypeError("iterations must be an integer.")
    if iterations < 1:
        raise ValueError("iterations must be at least 1.")

    # Mathematical dilation reflects the SE by 180 degrees.
    # All five SEs here are centrally symmetric, so reflecting keeps their shape.
    if dilate:
        mask = mask[::-1, ::-1]
    ry, rx = mask.shape[0] // 2, mask.shape[1] // 2
    height, width = image.shape
    # Values outside the border are set so that they do not affect the result.
    # For uint8, the neutral value is 0 for max and 255 for min.
    border = 0 if dilate else 255
    combine = np.maximum if dilate else np.minimum
    current = image.copy()
    for _ in range(iterations):
        padded = np.pad(current, ((ry, ry), (rx, rx)),
                        mode="constant", constant_values=border)
        result = np.full_like(current, border)
        for y, x in np.argwhere(mask):
            neighbor = padded[y:y + height, x:x + width]
            combine(result, neighbor, out=result)
        current = result
    return current


def dilation(image: np.ndarray, se: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Dilation: local maximum. Bright objects expand."""
    return _extreme_filter(image, se, dilate=True, iterations=iterations)


def erosion(image: np.ndarray, se: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Erosion: local minimum. Bright objects shrink."""
    return _extreme_filter(image, se, dilate=False, iterations=iterations)


def opening(image: np.ndarray, se: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Opening: erosion -> dilation with the same SE. Removes small bright noise and thin bridges."""
    return dilation(erosion(image, se, iterations), se, iterations)


def closing(image: np.ndarray, se: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Closing: dilation -> erosion with the same SE. Fills small dark holes and narrow gaps."""
    return erosion(dilation(image, se, iterations), se, iterations)


# ---------- 2. Original image designed for this experiment ----------
def create_test_image() -> np.ndarray:
    """A 384x288 test image designed to show the four effects, not a random image.

    Background=0, white objects=255, gray objects=170. No randomness, thresholding or interpolation.
    Top left: a large rectangle and six isolated bright specks of size 3x3 and 5x5.
    Top right: a large rectangle with three small 5x5 holes and one large 15x15 hole.
    Bottom left: a 5-pixel-wide vertical gap between two rectangles of intensity 170.
    Bottom right: a 3-pixel-thick horizontal bridge between two white rectangles.
    """
    image = np.zeros((288, 384), dtype=np.uint8)
    image[34:118, 38:114] = 255
    for y, x, size in [(18, 15, 3), (31, 138, 3), (67, 146, 5),
                       (105, 133, 3), (136, 21, 5), (125, 171, 3)]:
        image[y:y + size, x:x + size] = 255

    image[26:133, 224:356] = 255
    for y, x in [(44, 242), (75, 280), (107, 328)]:
        image[y:y + 5, x:x + 5] = 0
    image[88:103, 260:275] = 0

    image[180:254, 30:97] = 170
    image[180:254, 102:169] = 170

    image[182:254, 228:266] = 255
    image[182:254, 310:348] = 255
    image[216:219, 266:310] = 255
    return image


def structuring_elements() -> dict[str, np.ndarray]:
    """Only positions with 1 take part. A 0 is an unused position, not a black pixel.

    Square: the default SE, for comparing expansion/shrinking in all directions.
    Cross: uses neighbors along the cross.
    Diamond: uses horizontal, vertical and nearby diagonal neighbors.
    Vertical / Horizontal: compare the vertical and horizontal effects.
    Their sizes also differ, so the extra experiment is not a fair comparison of shape alone.
    """
    cross = np.zeros((5, 5), dtype=np.uint8)
    cross[2, :] = 1
    cross[:, 2] = 1
    y, x = np.ogrid[-3:4, -3:4]
    return {
        "Square_7x7": np.ones((7, 7), dtype=np.uint8),
        "Cross_5x5": cross,
        "Diamond_7x7": ((np.abs(y) + np.abs(x)) <= 3).astype(np.uint8),
        "Vertical_7x1": np.ones((7, 1), dtype=np.uint8),
        "Horizontal_1x7": np.ones((1, 7), dtype=np.uint8),
    }


OPERATIONS = {"Dilation": dilation, "Erosion": erosion,
              "Opening": opening, "Closing": closing}


# ---------- 3. Saving results and the comparison image ----------
def _font(size: int, mono: bool = False):
    candidates = ["DejaVuSansMono.ttf", "consola.ttf"] if mono else [
        "DejaVuSans.ttf", "arial.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def save_comparison(original: np.ndarray, results: dict[str, np.ndarray],
                    se: np.ndarray, name: str, iterations: int, path: Path) -> None:
    """Each image is shown at the same scale and intensity range (0-255).

    Only the comparison figure is resized, using nearest-neighbor interpolation.
    The individual PNGs keep the original resolution.
    """
    gap, margin, card_w, card_h = 28, 36, 800, 738
    sheet = Image.new("RGB", (2 * card_w + gap + 2 * margin,
                              3 * card_h + 2 * gap + 164), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 24), "MORPHOLOGICAL IMAGE PROCESSING", font=_font(34), fill="black")
    draw.text((margin, 76), f"{name} | iterations per stage: {iterations} | bright objects on dark background",
              font=_font(21), fill="black")
    notes = {
        "Original": "Designed features: dots, holes, gap and thin bridge.",
        "Dilation": "Local maximum: bright regions expand.",
        "Erosion": "Local minimum: bright regions shrink.",
        "Opening": "Erosion then dilation: small bright details may vanish.",
        "Closing": "Dilation then erosion: small dark details may fill.",
    }
    panels = [("Original", original)] + list(results.items())
    for index, (title, array) in enumerate(panels):
        row, col = divmod(index, 2)
        left, top = margin + col * (card_w + gap), 132 + row * (card_h + gap)
        draw.rectangle((left, top, left + card_w, top + card_h), outline="black", width=1)
        draw.text((left + 18, top + 16), title, font=_font(29), fill="black")
        draw.text((left + 18, top + 60), notes[title], font=_font(19), fill="black")
        frame = Image.fromarray(array)
        scale = min((card_w - 32) / frame.width, 576 / frame.height)
        resized = frame.resize((max(1, round(frame.width * scale)),
                                max(1, round(frame.height * scale))), Image.Resampling.NEAREST)
        sheet.paste(resized, (left + (card_w - resized.width) // 2,
                             top + 104 + (576 - resized.height) // 2))
        draw.text((left + 18, top + 698), f"Nonzero pixels: {np.count_nonzero(array):,}",
                  font=_font(20), fill="black")

    left, top = margin + card_w + gap, 132 + 2 * (card_h + gap)
    draw.rectangle((left, top, left + card_w, top + card_h), outline="black", width=1)
    draw.text((left + 22, top + 18), "Structuring element (SE)", font=_font(29), fill="black")
    draw.text((left + 22, top + 62), name, font=_font(23), fill="black")
    for row, values in enumerate(se):
        draw.text((left + 38, top + 112 + row * 37), " ".join(str(int(v)) for v in values),
                  font=_font(27, mono=True), fill="black")
    description = ["1 = included neighbor; 0 = ignored position",
                   "Each technique starts from the original.",
                   "Opening: erosion -> dilation",
                   "Closing: dilation -> erosion",
                   "Border: dilation=0, erosion=255",
                   "Saved PNGs keep the original resolution."]
    for row, text in enumerate(description):
        draw.text((left + 22, top + 444 + row * 37), text, font=_font(21), fill="black")
    sheet.save(path)


def run_experiment(image: np.ndarray, se: np.ndarray, name: str,
                   iterations: int, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(folder / "00_original.png")
    results = {}
    for index, (label, operation) in enumerate(OPERATIONS.items(), start=1):
        # The four operations are not chained; each starts from the original.
        results[label] = operation(image, se, iterations)
        Image.fromarray(results[label]).save(folder / f"0{index}_{label.lower()}.png")
    np.savetxt(folder / "structuring_element.txt", se, fmt="%d")
    save_comparison(image, results, se, name, iterations, folder / "comparison.png")
    lines = ["operation,nonzero_pixels,changed_pixels", f"Original,{np.count_nonzero(image)},0"]
    for label, result in results.items():
        lines.append(f"{label},{np.count_nonzero(result)},{np.count_nonzero(result != image)}")
    (folder / "metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Optional image path. Converted to grayscale; not thresholded.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--all-se", action="store_true", help="Also save the other four SE comparisons.")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Iterations of each erosion/dilation stage, default=1.")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    try:
        if args.input is None:
            image = create_test_image()
        else:
            with Image.open(args.input) as source:
                image = np.array(source.convert("L"), dtype=np.uint8)
        kernels = structuring_elements()
        run_experiment(image, kernels["Square_7x7"], "Square_7x7", args.iterations, args.output)
        if args.all_se:
            for name, se in kernels.items():
                if name != "Square_7x7":
                    run_experiment(image, se, name, args.iterations,
                                   args.output / "extra_structuring_elements" / name)
        print(f"Saved results: {args.output.resolve()}")
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
