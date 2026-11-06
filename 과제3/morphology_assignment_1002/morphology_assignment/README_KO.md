# 형태학 영상처리 과제 — 2026년 10월 2일

## 실행

Python 3.10 이상에서 다음 명령을 실행합니다.

```bash
python -m pip install -r requirements.txt
python main.py
```

기본 실행만으로 원본, 네 연산 결과, 비교 그림, 구조 요소 배열과 픽셀 수가 `output` 폴더에 저장됩니다. 파일을 더블클릭하여 창을 띄우는 방식이 아니라, 결과 이미지를 저장하는 방식입니다. 이미 실행한 결과도 이 압축파일에 포함되어 있습니다.

기본 실험은 **Square 7×7, 각 단계 1회**입니다. 열림은 침식 1회 → 팽창 1회, 닫힘은 팽창 1회 → 침식 1회입니다. 네 가지 결과는 모두 같은 원본에서 각각 출발합니다.

## 제출에 사용할 파일

| 파일 | 내용 |
|---|---|
| `main.py` | 네 가지 형태학 연산 직접 구현, 원본 생성, 결과 저장 |
| `output/00_original.png` | 실험 목적에 맞춰 제작한 원본 그레이스케일 영상 |
| `output/01_dilation.png` | 팽창 결과 |
| `output/02_erosion.png` | 침식 결과 |
| `output/03_opening.png` | 열림 결과 |
| `output/04_closing.png` | 닫힘 결과 |
| `output/comparison.png` | 원본 + 네 결과 + 사용한 SE의 0/1 배열 |
| `comparison_KO.txt` | 이미지 선정 이유, 실험 조건, 간단한 비교 설명 |
| `comparison_EN.txt` | 영문 비교 설명 |

원본은 임의의 사진을 고른 것이 아니라, 밝은 잡음·어두운 구멍·좁은 틈·가는 연결선을 관찰하기 위해 직접 만든 테스트 이미지입니다. 회색 물체도 포함하고 있으며, 이진화하지 않습니다. 실제 사진을 사용해야 한다는 추가 지시가 있다면 아래 `--input` 기능으로 변경할 수 있습니다. **이미지를 바꾼 경우 비교 설명도 새 결과에 맞게 수정해야 합니다.**

## 추가 비교: 이전에 사용한 구조 요소 5종

```bash
python main.py --all-se
```

Square 7×7이 기본 결과이고, Cross 5×5 / Diamond 7×7 / Vertical 7×1 / Horizontal 1×7 결과는 `output/extra_structuring_elements`에 저장됩니다. 이 추가 결과도 이미 포함되어 있습니다. 이번 공지에는 SE를 5종 사용해야 한다는 조건은 없으므로, 추가 실험은 필수가 아닙니다.

추가 실험은 SE의 크기와 모양이 모두 달라서, 결과 차이를 전부 '모양의 영향'이라고 해석하면 안 됩니다.

각 단계 반복 횟수를 2회로 변경하려면 다음과 같이 실행합니다.

```bash
python main.py --all-se --iterations 2 --output output_iterations2
```

`--iterations 2`는 열림 전체를 두 번 반복한다는 뜻이 아닙니다. 침식 2회 후 팽창 2회이며, 닫힘은 팽창 2회 후 침식 2회입니다. 기본 제공 결과와 비교 설명은 1회 기준입니다.

## 다른 이미지 적용

```bash
python main.py --input "my_image.png" --output output_custom
```

읽을 때 그레이스케일로 변환하지만 임계값을 이용한 이진화나 처리용 크기 변경은 하지 않습니다. 밝은 영역을 물체로 보는 설명을 사용하므로, 어두운 글자가 흰 바탕에 있는 영상은 눈에 보이는 팽창·침식 방향을 반대로 느낄 수 있습니다.

## 직접 구현 범위

- `dilation`: SE에서 1인 위치의 이웃들 중 최댓값을 직접 계산합니다.
- `erosion`: SE에서 1인 위치의 이웃들 중 최솟값을 직접 계산합니다.
- `opening`: 위에서 구현한 침식 → 팽창을 실행합니다.
- `closing`: 위에서 구현한 팽창 → 침식을 실행합니다.
- OpenCV/SciPy의 형태학 함수는 `main.py`에서 사용하지 않습니다.
- 영상은 0~255의 `uint8`이고, SE의 0은 사용하지 않는 위치입니다.
- 경계 밖 중립값은 팽창 0, 침식 255입니다.
- 입력 배열을 덮어쓰지 않습니다.
- 비교용 그림만 최근접 보간으로 표시 크기를 맞추고, 개별 결과 PNG는 원본 해상도를 유지합니다.

## 테스트

```bash
python -m unittest -v test_morphology
```

OpenCV가 없으면 기본 단위 테스트를 실행하고, OpenCV 대조 테스트 하나는 건너뜁니다. 참조 구현 대조까지 재실행하려면 다음 명령을 사용합니다.

```bash
python -m pip install opencv-python
python -m unittest -v test_morphology
```

OpenCV는 검증에만 사용합니다. `validation_report.txt`에는 이번 실행 환경과 실제 테스트 로그가 들어 있습니다.

## 개념 참고 자료

- OpenCV, Morphological Transformations: https://docs.opencv.org/4.12.0/d9/d61/tutorial_py_morphological_ops.html
- SciPy, grey_dilation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_dilation.html
- SciPy, grey_erosion: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_erosion.html

참고 자료는 연산 정의 확인용이며, 제출 이미지와 구현 코드는 이 과제를 위해 새로 작성했습니다.
