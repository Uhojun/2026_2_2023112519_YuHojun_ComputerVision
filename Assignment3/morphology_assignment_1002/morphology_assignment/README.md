# Morphological Image Processing Assignment (October 2, 2026)

## Running

With Python 3.10 or later:

```bash
python -m pip install -r requirements.txt
python main.py
```

This saves the original image, the four operation results, a comparison figure, the structuring element array and the pixel counts to the `output` folder. The results of a previous run are already in this folder.

The main experiment uses a **7×7 square and one iteration per stage**. Opening is one erosion followed by one dilation; closing is one dilation followed by one erosion. All four results start independently from the same original image.

## Files

| File | Contents |
|---|---|
| `main.py` | The four morphology operations implemented from scratch, test image creation, saving results |
| `output/00_original.png` | Grayscale test image designed for this experiment |
| `output/01_dilation.png` | Dilation result |
| `output/02_erosion.png` | Erosion result |
| `output/03_opening.png` | Opening result |
| `output/04_closing.png` | Closing result |
| `output/comparison.png` | Original, the four results and the 0/1 array of the SE used |
| `comparison.txt` | Why the image was chosen, experimental conditions and a short comparison |

The original is not an arbitrary photograph. It is a test image made to show bright noise, dark holes, a narrow gap and a thin bridge. It includes gray objects and is not thresholded. To use a real photograph instead, use the `--input` option described below. **If you change the image, update the comparison text to match the new results.**

## Extra comparison: the five structuring elements from the earlier assignment

```bash
python main.py --all-se
```

The 7×7 square gives the main results. Results for Cross 5×5, Diamond 7×7, Vertical 7×1 and Horizontal 1×7 are saved to `output/extra_structuring_elements`. This repository keeps the comparison figure, metrics and SE array for each one; the command above also creates the individual images. The assignment does not require five SEs, so this experiment is optional.

These SEs differ in both size and shape, so differences in the results should not all be attributed to shape.

To use two iterations per stage:

```bash
python main.py --all-se --iterations 2 --output output_iterations2
```

`--iterations 2` does not repeat the whole opening twice. Opening becomes two erosions followed by two dilations, and closing becomes two dilations followed by two erosions. The included results and comparison text use one iteration.

## Using another image

```bash
python main.py --input "my_image.png" --output output_custom
```

The image is converted to grayscale but is not thresholded or resized for processing. The explanations treat bright regions as objects, so for dark text on a white background, dilation and erosion will appear to work in the opposite direction.

## What is implemented from scratch

- `dilation`: computes the maximum over the neighbors where the SE is 1.
- `erosion`: computes the minimum over the neighbors where the SE is 1.
- `opening`: runs the erosion and then the dilation above.
- `closing`: runs the dilation and then the erosion above.
- `main.py` does not use the morphology functions of OpenCV or SciPy.
- Images are `uint8` in the range 0-255, and a 0 in the SE marks an unused position.
- The neutral value outside the border is 0 for dilation and 255 for erosion.
- The input array is not overwritten.
- Only the comparison figure is resized, using nearest-neighbor interpolation. The individual PNGs keep the original resolution.

## Tests

```bash
python -m unittest -v test_morphology
```

Without OpenCV, the basic unit tests run and the one OpenCV comparison test is skipped. To run the comparison as well:

```bash
python -m pip install opencv-python
python -m unittest -v test_morphology
```

OpenCV is used only for checking. `validation_report.txt` contains the environment and test log from the original run.

## Concept references

- OpenCV, Morphological Transformations: https://docs.opencv.org/4.12.0/d9/d61/tutorial_py_morphological_ops.html
- SciPy, grey_dilation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_dilation.html
- SciPy, grey_erosion: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_erosion.html

The references were used to confirm the definitions of the operations. The test image and code were written for this assignment.
