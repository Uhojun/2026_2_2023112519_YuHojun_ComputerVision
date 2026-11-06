"""Run: python -m unittest -v test_morphology

The basic tests need only NumPy and Pillow.
If OpenCV is installed, 280 results are also compared with it pixel by pixel.
OpenCV is used only to check the results; main.py does not use it.
"""
import unittest

import numpy as np

from main import (OPERATIONS, closing, create_test_image, dilation, erosion,
                  opening, structuring_elements)

try:
    import cv2
except ImportError:
    cv2 = None


def reference_cases():
    corner = np.zeros((9, 11), dtype=np.uint8)
    corner[0, 0] = 255
    center = np.zeros((11, 13), dtype=np.uint8)
    center[5, 6] = 170
    ramp = ((np.arange(99).reshape(9, 11) * 7) % 256).astype(np.uint8)
    return {
        "assignment_image": create_test_image(),
        "all_black": np.zeros((7, 9), dtype=np.uint8),
        "all_white": np.full((7, 9), 255, dtype=np.uint8),
        "grayscale_ramp": ramp,
        "corner_impulse": corner,
        "center_impulse": center,
        "smaller_than_SE": np.array([[0, 170, 255], [255, 40, 0]], dtype=np.uint8),
    }


def opencv_reference(image, se, iterations):
    def d(a):
        return cv2.dilate(a, se, iterations=iterations,
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
    def e(a):
        return cv2.erode(a, se, iterations=iterations,
                         borderType=cv2.BORDER_CONSTANT, borderValue=255)
    return {"Dilation": d(image), "Erosion": e(image),
            "Opening": d(e(image)), "Closing": e(d(image))}


class MorphologyTests(unittest.TestCase):
    def setUp(self):
        self.image = create_test_image()
        self.se = np.ones((7, 7), dtype=np.uint8)

    def test_dilation_of_single_pixel(self):
        image = np.zeros((7, 7), dtype=np.uint8)
        image[3, 3] = 170
        expected = np.zeros_like(image)
        expected[2:5, 2:5] = 170
        np.testing.assert_array_equal(dilation(image, np.ones((3, 3), dtype=np.uint8)), expected)

    def test_erosion_of_rectangle(self):
        image = np.zeros((9, 9), dtype=np.uint8)
        image[2:7, 2:7] = 255
        expected = np.zeros_like(image)
        expected[3:6, 3:6] = 255
        np.testing.assert_array_equal(erosion(image, np.ones((3, 3), dtype=np.uint8)), expected)

    def test_opening_removes_six_bright_specks(self):
        result = opening(self.image, self.se)
        for y, x, size in [(18, 15, 3), (31, 138, 3), (67, 146, 5),
                           (105, 133, 3), (136, 21, 5), (125, 171, 3)]:
            self.assertTrue(np.all(result[y:y + size, x:x + size] == 0))
        self.assertTrue(np.all(result[34:118, 38:114] == 255))

    def test_closing_fills_three_small_holes(self):
        result = closing(self.image, self.se)
        for y, x in [(44, 242), (75, 280), (107, 328)]:
            self.assertTrue(np.all(result[y:y + 5, x:x + 5] == 255))

    def test_closing_preserves_large_hole(self):
        result = closing(self.image, self.se)
        self.assertTrue(np.all(result[88:103, 260:275] == 0))

    def test_closing_fills_five_pixel_gap(self):
        result = closing(self.image, self.se)
        self.assertTrue(np.all(result[180:254, 97:102] == 170))

    def test_opening_breaks_three_pixel_bridge(self):
        result = opening(self.image, self.se)
        self.assertTrue(np.all(result[216:219, 266:310] == 0))
        self.assertTrue(np.all(result[182:254, 228:266] == 255))
        self.assertTrue(np.all(result[182:254, 310:348] == 255))

    def test_extensivity_and_idempotence(self):
        for name, se in structuring_elements().items():
            for case, image in reference_cases().items():
                with self.subTest(se=name, case=case):
                    d, e = dilation(image, se), erosion(image, se)
                    o, c = opening(image, se), closing(image, se)
                    self.assertTrue(np.all(d >= image))
                    self.assertTrue(np.all(e <= image))
                    self.assertTrue(np.all(o <= image))
                    self.assertTrue(np.all(c >= image))
                    np.testing.assert_array_equal(opening(o, se), o)
                    np.testing.assert_array_equal(closing(c, se), c)

    def test_neutral_border_and_output_format(self):
        for value in (0, 170, 255):
            image = np.full((2, 3), value, dtype=np.uint8)
            for operation in OPERATIONS.values():
                result = operation(image, self.se)
                np.testing.assert_array_equal(result, image)
                self.assertEqual(result.dtype, np.uint8)
                self.assertEqual(result.shape, image.shape)

    def test_original_not_modified(self):
        before = self.image.copy()
        for operation in OPERATIONS.values():
            operation(self.image, self.se)
        np.testing.assert_array_equal(self.image, before)

    def test_asymmetric_SE_orientation(self):
        # Mathematical dilation: an SE extending right expands a point to the right.
        image = np.zeros((5, 7), dtype=np.uint8)
        image[2, 3] = 255
        se = np.array([[0, 1, 1]], dtype=np.uint8)
        expected = np.zeros_like(image)
        expected[2, 3:5] = 255
        np.testing.assert_array_equal(dilation(image, se), expected)

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            dilation(self.image.astype(float), self.se)
        with self.assertRaises(ValueError):
            erosion(self.image, np.ones((2, 2), dtype=np.uint8))
        with self.assertRaises(ValueError):
            opening(self.image, self.se, iterations=0)
        with self.assertRaises(ValueError):
            closing(self.image, np.zeros((3, 3), dtype=np.uint8))

    @unittest.skipUnless(cv2 is not None, "OpenCV is not installed; optional reference test skipped.")
    def test_matches_opencv_280_comparisons(self):
        count = 0
        for name, se in structuring_elements().items():
            for case, image in reference_cases().items():
                for iterations in (1, 2):
                    expected = opencv_reference(image, se, iterations)
                    for label, operation in OPERATIONS.items():
                        with self.subTest(se=name, case=case, n=iterations, operation=label):
                            actual = operation(image, se, iterations)
                            np.testing.assert_array_equal(actual, expected[label])
                            count += 1
        self.assertEqual(count, 280)


if __name__ == "__main__":
    unittest.main(verbosity=2)
