from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# 1. 기본 설정
# ==================================================
# main.py와 같은 폴더에 이미지 두 장을 넣으세요.
BASE_DIR = Path(__file__).resolve().parent
IMAGE_FILES = ["image.jpg", "image1.jpg"]
OUTPUT_DIR = BASE_DIR / "results"

# 기존 과제와 동일하게 각 기본 연산을 2회 적용합니다.
# 효과가 너무 강하면 1로 바꾸세요.
ITERATIONS = 2
SHOW_WINDOWS = True  # False: 창을 띄우지 않고 결과만 저장


# ==================================================
# 2. 구조 요소(SE) 정의: 기존 과제와 동일
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
# 3. 그레이스케일 이미지 읽기
# ==================================================
def read_grayscale(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {path}")

    # 한글이 포함된 경로에서도 읽을 수 있도록 처리합니다.
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"이미지 파일이 비어 있습니다: {path}")

    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return image


# ==================================================
# 4. 침식 / 열림 / 닫힘 수행
# ==================================================
def apply_morphology(image: np.ndarray) -> dict:
    results = {"Erosion": [], "Opening": [], "Closing": []}

    for kernel in KERNELS.values():
        # 침식: 원본 → 침식
        erosion = cv2.erode(image, kernel, iterations=ITERATIONS)

        # 열림: 원본 → 침식 → 팽창
        # 위의 침식 결과를 팽창의 입력으로 사용합니다.
        opening = cv2.dilate(erosion, kernel, iterations=ITERATIONS)

        # 닫힘: 원본 → 팽창 → 침식
        # 열림 결과가 아니라 원본에서 다시 시작합니다.
        dilation = cv2.dilate(image, kernel, iterations=ITERATIONS)
        closing = cv2.erode(dilation, kernel, iterations=ITERATIONS)

        results["Erosion"].append(erosion)
        results["Opening"].append(opening)
        results["Closing"].append(closing)

    return results


# ==================================================
# 5. 원본 + SE별 결과 + SE 배열 표시 및 저장
# ==================================================
def plot_comparison(original: np.ndarray, processed: list,
                    operation: str, image_name: str) -> None:
    titles = ["Original"] + list(KERNELS.keys())
    images = [original] + processed
    kernels = [None] + list(KERNELS.values())

    # 위쪽: 이미지 / 아래쪽: SE의 0·1 배열
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
        # 모든 결과를 같은 밝기 범위(0~255)로 비교합니다.
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
    print(f"저장 완료: {output_path}")

    if not SHOW_WINDOWS:
        plt.close(fig)


# ==================================================
# 6. 실행
# ==================================================
def main() -> None:
    if not isinstance(ITERATIONS, int) or ITERATIONS < 1:
        raise ValueError("ITERATIONS는 1 이상의 정수여야 합니다.")

    # 필요한 이미지를 먼저 확인하고 읽습니다.
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