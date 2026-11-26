"""Чтение, запись и генерация тестовых изображений."""
from pathlib import Path
import cv2
import numpy as np

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def load_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")
    return image

def save_image(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Не удалось сохранить: {path}")

def list_images(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"Нет каталога: {folder}")
    result = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not result:
        raise ValueError(f"В каталоге {folder} нет изображений.")
    return result

def generate_samples(folder: Path, seed: int = 42):
    """Создаёт три условных снимка поверхности с царапинами и шумом."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    h, w = 420, 600
    for index in range(1, 4):
        base = np.full((h, w), 175 + index * 8, dtype=np.float32)
        # Небольшая неоднородность поверхности
        yy, xx = np.mgrid[0:h, 0:w]
        base += 7 * np.sin(xx / 65) + 4 * np.cos(yy / 47)
        image = np.repeat(base[:, :, None], 3, axis=2)
        # Тонкие тёмные и светлые линии имитируют царапины
        lines = [
            ((70, 80 + index * 15), (500, 140 + index * 12), 2),
            ((120, 300), (430, 250 - index * 8), 3),
            ((250, 40), (280 + index * 25, 370), 1),
        ]
        for p1, p2, thickness in lines[:index + 1]:
            cv2.line(image, p1, p2, (45, 45, 45), thickness, cv2.LINE_AA)
            cv2.line(image, (p1[0], p1[1]-2), (p2[0], p2[1]-2),
                     (225, 225, 225), 1, cv2.LINE_AA)
        image += rng.normal(0, 3.5 + index, image.shape)
        image = np.clip(image, 0, 255).astype(np.uint8)
        save_image(folder / f"surface_{index}.png", image)
