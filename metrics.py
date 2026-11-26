"""Метрики MSE, PSNR и SSIM."""
import cv2
import numpy as np
from skimage.metrics import structural_similarity

def calculate_metrics(original, processed):
    """Сравнивает исходное изображение с обработанным."""
    a = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if original.ndim == 3 else original
    b = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) if processed.ndim == 3 else processed
    diff = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse))
    ssim = float(structural_similarity(a, b, data_range=255))
    return mse, psnr, ssim
