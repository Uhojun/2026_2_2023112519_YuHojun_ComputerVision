# 2026_2_2023112519_YuHojun_ComputerVision
This is the repository for the assignment submission of Yu Ho-jun, 2023112519, Computer Vision, 2nd Semester of the 2026 Academic Year.

## Assignments

| Folder | Topic | Script |
| --- | --- | --- |
| [Assignment1](Assignment1) | Comparing histogram equalization (HE) and adaptive histogram equalization (AHE) | `code.py` |
| [Assignment2](Assignment2) | Dilation with five structuring elements | `main.py` |
| [Assignment3](Assignment3) | Erosion, opening and closing | `main.py`, `morphology_assignment_1002/morphology_assignment/main.py` |
| [Assignment4](Assignment4) | KNN classification of CIFAR-10: values of k, distance metrics, 5-fold cross-validation | `knn_cifar10.py`, `input_representation.py` |

## Assignment 4 dataset

`cifar-10-python.tar.gz` (170MB) is not in this repository because it is over GitHub's 100MB file limit. To run the code, download it from the [CIFAR-10 page](https://cave.cs.toronto.edu/kriz/cifar.html) and put it in the `Assignment4` folder.

`Assignment4/results/cache/`, which is created while the code runs, is also not included. The measured results are in the CSV files in `Assignment4/results/`. See [Assignment4/README.md](Assignment4/README.md) for details.
