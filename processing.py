"""Обработка поверхности с царапинами: Собель, порог, морфология и измерение длины."""
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class Scratch:
    """Измеренные характеристики одной выделенной царапины."""
    length: float
    area: int
    bbox: tuple[int, int, int, int]

@dataclass
class PipelineOutputs:
    gray: np.ndarray
    sobel: np.ndarray
    raw_mask: np.ndarray
    morph_mask: np.ndarray
    annotated: np.ndarray
    scratches: list[Scratch]

def apply_sobel(gray: np.ndarray, blur_size: int = 3) -> np.ndarray:
    """Сглаживает мелкий шум и вычисляет модуль градиента Собеля."""
    if blur_size < 1 or blur_size % 2 == 0:
        raise ValueError("Размер размытия должен быть нечётным и не меньше 1.")
    smooth = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    return cv2.convertScaleAbs(magnitude)

def threshold_edges(sobel: np.ndarray, threshold: int = 45) -> np.ndarray:
    """Преобразует градиент в бинарную маску."""
    if not 0 <= threshold <= 255:
        raise ValueError("Порог должен находиться в диапазоне 0..255.")
    _, mask = cv2.threshold(sobel, threshold, 255, cv2.THRESH_BINARY)
    return mask

def clean_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Удаляет одиночный шум открытием и соединяет близкие фрагменты закрытием."""
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Размер ядра должен быть нечётным и не меньше 1.")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

def measure_component_length(component: np.ndarray) -> float:
    """Оценивает длину объекта по проекции пикселей на главную ось (PCA)."""
    ys, xs = np.where(component > 0)
    if len(xs) < 2:
        return 0.0
    points = np.column_stack((xs, ys)).astype(np.float32)
    center = points.mean(axis=0)
    centered = points - center
    covariance = np.cov(centered, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, np.argmax(values)]
    projection = centered @ direction
    return float(projection.max() - projection.min() + 1.0)

def find_scratches(image: np.ndarray, mask: np.ndarray, min_area: int = 12):
    """Находит связные области, измеряет их длину и рисует результаты."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    annotated = image.copy()
    scratches = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        component = np.uint8(labels == label) * 255
        length = measure_component_length(component)
        scratches.append(Scratch(length, int(area), (int(x), int(y), int(w), int(h))))
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(annotated, f"{length:.0f}px", (x, max(15, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    return annotated, scratches

def run_pipeline(image: np.ndarray, threshold: int = 45,
                 kernel_size: int = 3, min_area: int = 12,
                 blur_size: int = 3) -> PipelineOutputs:
    """Полный конвейер обнаружения царапин."""
    if image is None or image.size == 0:
        raise ValueError("Передано пустое изображение.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    sobel = apply_sobel(gray, blur_size)
    raw_mask = threshold_edges(sobel, threshold)
    morph_mask = clean_mask(raw_mask, kernel_size)
    annotated, scratches = find_scratches(image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
                                          morph_mask, min_area)
    return PipelineOutputs(gray, sobel, raw_mask, morph_mask, annotated, scratches)
