"""Запуск пакетной обработки изображений поверхности с царапинами."""
import argparse
import csv
import time
from pathlib import Path
import numpy as np
from io_utils import generate_samples, list_images, load_image, save_image
from processing import run_pipeline
from metrics import calculate_metrics

def main():
    parser = argparse.ArgumentParser(description="ЛР №18, вариант 11: измерение царапин.")
    parser.add_argument("--input", type=Path, default=Path("images"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--threshold", type=int, default=45)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--min-area", type=int, default=12)
    parser.add_argument("--blur-size", type=int, default=3)
    parser.add_argument("--generate-samples", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    if args.generate_samples and (not args.input.exists() or not any(args.input.iterdir())):
        generate_samples(args.input, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in list_images(args.input):
        start = time.perf_counter()
        image = load_image(path)
        result = run_pipeline(image, args.threshold, args.kernel_size,
                              args.min_area, args.blur_size)
        elapsed = time.perf_counter() - start
        stem = path.stem
        save_image(args.output / f"{stem}_01_original.png", image)
        save_image(args.output / f"{stem}_02_sobel.png", result.sobel)
        save_image(args.output / f"{stem}_03_threshold.png", result.raw_mask)
        save_image(args.output / f"{stem}_04_morphology.png", result.morph_mask)
        save_image(args.output / f"{stem}_05_detected.png", result.annotated)
        mse, psnr, ssim = calculate_metrics(image, result.annotated)
        lengths = ";".join(f"{s.length:.1f}" for s in result.scratches)
        rows.append({
            "filename": path.name, "threshold": args.threshold,
            "kernel_size": args.kernel_size, "min_area": args.min_area,
            "scratch_count": len(result.scratches), "lengths_px": lengths,
            "mse": round(mse, 3), "psnr_db": round(psnr, 2),
            "ssim": round(ssim, 4), "elapsed_sec": round(elapsed, 4)
        })
        print(f"{path.name}: найдено царапин={len(result.scratches)}, "
              f"длины (пикс.): {lengths or 'нет'}, PSNR={psnr:.2f} дБ, SSIM={ssim:.4f}")
    csv_path = args.output / "metrics_variant11.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Отчёт сохранён: {csv_path}")

if __name__ == "__main__":
    main()
