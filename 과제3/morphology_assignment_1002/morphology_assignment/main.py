"""형태학 과제: Dilation / Erosion / Opening / Closing 직접 구현.

실행: python main.py
추가 구조 요소 5종 비교: python main.py --all-se
다른 이미지 사용: python main.py --input "my_image.png"

필요 패키지: numpy, Pillow
OpenCV, SciPy의 형태학 연산 함수는 사용하지 않습니다.
흑백/그레이스케일 모두 처리하며, 이진화나 크기 변경은 하지 않습니다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------- 1. 직접 구현한 형태학 연산 ----------
def _check(image: np.ndarray, se: np.ndarray) -> np.ndarray:
    """입력은 uint8 그레이스케일, SE는 중심이 1인 홀수 크기 0/1 배열."""
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
    """SE에서 1인 위치의 값들을 직접 비교해 최대/최솟값을 구합니다.

    각 SE 오프셋에 대해 영상 전체를 한 번에 비교하므로 픽셀 이중 반복문보다
    빠릅니다. np.maximum/minimum은 두 배열의 값 비교이며 형태학 전용 함수가
    아닙니다. 원본 배열은 덮어쓰지 않습니다.
    """
    mask = _check(image, se)
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)):
        raise TypeError("iterations must be an integer.")
    if iterations < 1:
        raise ValueError("iterations must be at least 1.")

    # 수학적 팽창에서는 SE를 180도 뒤집습니다.
    # 이번 5개 SE는 모두 중심 대칭이므로 뒤집어도 같은 모양입니다.
    if dilate:
        mask = mask[::-1, ::-1]
    ry, rx = mask.shape[0] // 2, mask.shape[1] // 2
    height, width = image.shape
    # 경계 밖 값은 연산에 영향을 주지 않도록 설정합니다.
    # uint8에서 max에는 0, min에는 255가 중립값입니다.
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
    """팽창: 이웃의 최댓값. 밝은 물체가 확장됩니다."""
    return _extreme_filter(image, se, dilate=True, iterations=iterations)


def erosion(image: np.ndarray, se: np.ndarray, iterations: int = 1) -> np.ndarray:
    """침식: 이웃의 최솟값. 밝은 물체가 축소됩니다."""
    return _extreme_filter(image, se, dilate=False, iterations=iterations)


def opening(image: np.ndarray, se: np.ndarray, iterations: int = 1) -> np.ndarray:
    """열림: 침식 → 팽창. 같은 SE로 작은 밝은 잡음/가는 연결을 제거합니다."""
    return dilation(erosion(image, se, iterations), se, iterations)


def closing(image: np.ndarray, se: np.ndarray, iterations: int = 1) -> np.ndarray:
    """닫힘: 팽창 → 침식. 같은 SE로 작은 어두운 구멍/좁은 틈을 채웁니다."""
    return erosion(dilation(image, se, iterations), se, iterations)


# ---------- 2. 목적에 맞게 설계한 원본 이미지 ----------
def create_test_image() -> np.ndarray:
    """무작위 이미지가 아닌, 네 가지 효과를 확인하는 384x288 테스트 영상.

    배경=0, 흰 물체=255, 회색 물체=170. 난수/이진화/보간을 사용하지 않습니다.
    좌상: 큰 사각형 + 3x3, 5x5 크기의 고립된 밝은 점 6개.
    우상: 큰 사각형 안에 작은 5x5 구멍 3개와 큰 15x15 구멍 1개.
    좌하: 밝기 170인 두 사각형 사이에 폭 5픽셀의 수직 틈.
    우하: 두 흰 사각형 사이에 두께 3픽셀의 수평 연결선.
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
    """1인 위치만 연산에 참여합니다. 0은 검은 픽셀이 아니라 미사용 위치입니다.

    Square: 모든 방향의 확장/축소를 비교하는 기본 SE.
    Cross: 십자 방향의 이웃을 사용.
    Diamond: 가로/세로와 가까운 대각선 이웃을 사용.
    Vertical / Horizontal: 각각 세로/가로 방향 효과를 비교.
    크기도 다르므로 추가 실험은 '모양만'의 공정한 비교는 아닙니다.
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


# ---------- 3. 결과 저장과 비교 이미지 ----------
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
    """각 영상은 같은 배율, 밝기 범위(0~255)로 표시합니다.

    비교용 그림만 최근접 보간으로 표시 크기를 조절합니다.
    개별 PNG는 처리 당시 원본 해상도를 그대로 유지합니다.
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
        # 순차적으로 네 연산을 누적하지 않습니다. 매번 원본에서 시작합니다.
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
